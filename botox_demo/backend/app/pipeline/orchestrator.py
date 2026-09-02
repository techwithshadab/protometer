"""The protected GraphRAG query pipeline: one turn, end to end.

    user message
      -> emergency check      urgent-symptom short-circuit on the RAW message (before tokenizing)
      -> protect()            tokenize the user's PII (never reaches retrieval/model/logs cleartext)
      -> retrieve()           GraphRAG: vector search for seed chunks + graph expansion to related
      -> generate()           grounded LLM answer over the retrieved context (model sees tokens)
      -> scan_reply()         egress guard: PII-leak + groundedness + medical-advice, fail closed
      -> reveal()             (role-gated), the public role reveals nothing, so the answer stays
                              free of the user's own PII; an internal role could see it
      -> Answer{answer, safety, refused, blocked, trace_id}

`answer()` returns the whole reply; `answer_stream()` yields it token-by-token (optimistic stream,
then the same guard runs on the full text and retracts to a safe fallback if it fails). Citations are
computed for the Langfuse trace (operator-facing provenance), not returned to the public UI.

Every branch fails safe: an emergency -> an urgent seek-help message; no retrieval hits -> a grounded
refusal; a blocked reply -> a safe redirect; the model never sees cleartext PII; the answer is built
only from retrieved chunks.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from app.obs import tracing
from app.pipeline import semantic_guard
from app.pipeline.guardrail import scan_reply
from app.pipeline.llm import LLMClient
from app.obs.prompt_manager import get_system_prompt
from app.protect.protector import WRAPPER_RE, ProtectionUnavailable, get_protector

_log = logging.getLogger("botox.orchestrator")

# The calm, compliant fallbacks, used whenever we cannot safely answer. Written in a natural,
# reassuring voice (no plumbing language), and they never echo the user's PII.
_REFUSAL = ("I don't have that information from the official BOTOX® site. For questions specific to "
            "your situation, your healthcare provider is the best person to ask.")
_BLOCKED = ("For anything involving your own treatment or dosing, your doctor is the right person to "
            "guide you. I'm happy to help with general questions about BOTOX®, feel free to ask.")
# Shown when PII protection cannot run (Protegrity unavailable). We fail closed rather than process
# an unprotected message, so the visitor gets a calm try-again, not an unprotected answer.
_PROTECTION_DOWN = ("Sorry, I can't take questions right now, our privacy protection service is "
                    "temporarily unavailable. Please try again in a little while.")

# Urgent-symptom message. BOTOX® carries a boxed warning for the distant spread of toxin effect
# (trouble swallowing, speaking, or breathing), which can appear hours to weeks after treatment. If
# a visitor describes such symptoms, we must NOT route them through the informational bot, we
# surface an unmistakable urgent-care message and stop, before any retrieval or model call.
_EMERGENCY = ("This may be a medical emergency. If you are having trouble breathing, swallowing, or "
              "speaking, or any severe or sudden symptoms, call your local emergency number (911 in "
              "the US) or get medical help right away. These can be signs of a serious reaction to "
              "BOTOX®, which can happen hours to weeks after treatment. Please don't wait for an "
              "answer here, contact emergency services or your doctor now.")

# Deterministic pre-retrieval emergency classifier (no model call). Matches descriptions of the
# boxed-warning symptoms and other acute distress. Deliberately broad: a false positive costs one
# extra "seek help now" message; a false negative could miss a real emergency.
_EMERGENCY_RE = re.compile(
    r"\b(can'?t|cannot|couldn'?t|trouble|difficulty|hard to|unable to|struggling to)\s+"
    r"(breath|breathe|breathing|swallow|swallowing|speak|speaking|talk|move)\b|"
    r"\b(choking|suffocat|gasping|wheezing)\b|"
    r"\bshortness of breath\b|\bcan'?t catch my breath\b|"
    r"\b(chest pain|heart attack|stroke)\b|"
    r"\b(severe|serious|allergic) (reaction|swelling|rash|symptoms?)\b|"
    r"\banaphyla|\bswelling of (my |the )?(face|throat|tongue|lips)\b|"
    r"\b(losing|loss of) (consciousness|vision)\b|\bblurred vision\b|"
    r"\b(drooping|paralyz|numb).*(spreading|worse|whole|all over)\b|"
    r"\b(emergency|urgent|911|ambulance)\b",
    re.I,
)


def _is_emergency(text: str) -> bool:
    return bool(_EMERGENCY_RE.search(text))


@dataclass
class Answer:
    answer: str
    citations: list[dict] = field(default_factory=list)   # [{title, url}]
    safety: bool = False
    refused: bool = False
    blocked: bool = False
    # True when the refusal is TRANSIENT (protection service down) and re-trying may succeed, so the
    # UI can offer a retry. A normal grounded refusal ("not on the official site") is NOT retryable.
    retryable: bool = False
    # The Langfuse trace id for this turn (None if tracing is off). Returned to the UI so a later
    # thumbs-up/down can attach a user-feedback score to this exact trace.
    trace_id: str | None = None
    # Internal signals (surfaced to /health-style debug + Prometheus metrics, not required by the UI):
    entities_protected: int = 0
    grounding_score: float = 0.0
    # Model + estimated token usage for the serving metrics (Langfuse carries the per-generation
    # detail; these let the /metrics endpoint aggregate cost/usage without re-plumbing the LLM call).
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    emergency: bool = False   # the emergency (urgent-symptom) check short-circuited this turn


class Orchestrator:
    def __init__(self, retriever) -> None:
        """`retriever` exposes .search(query, k) -> [chunk dict] and .graph_expand(ids) -> [chunk].
        Each chunk dict: {id, text, url, title, safety}. Injected so the app can wire the real
        GraphRAG store or a test double."""
        self.retriever = retriever
        # The SHARED process-wide Protector (one Protegrity login for the whole process). Raises
        # ProtectionUnavailable if Protegrity is down, so building the orchestrator at startup is
        # itself the hard protection gate.
        self.protector = get_protector()
        self.llm = LLMClient()

    def answer(self, message: str, k: int = 4, conversation_id: str = "",
               visitor_id: str | None = None) -> Answer:
        # Emergency check runs on the RAW message, before tokenization: it looks for symptom words
        # (trouble breathing/swallowing…), not PII, and must not be defeated if the PII tokenizer
        # happens to replace a word inside the symptom phrase. This is a symptom classifier, not a
        # data sink, it makes no external call and nothing here is logged/traced in the clear.
        emergency = _is_emergency(message)

        # An urgent symptom must surface even if protection is down: it runs on the RAW message, makes
        # no external call, and traces nothing in the clear. Handle it before touching Protegrity so a
        # protection outage can never swallow an emergency.
        if emergency:
            # The emergency message IS the safety response (a "call 911 now" instruction), so we do
            # NOT set safety=True, that would stack the milder generic "talk to your doctor" ISI
            # callout beneath it and dilute the urgency. The urgent text stands alone.
            tid = _trace_emergency(conversation_id, visitor_id)
            return Answer(answer=_EMERGENCY, safety=False, trace_id=tid, emergency=True)

        # 1. Protect the user's PII before anything downstream sees it. Only the PROTECTED text is
        # ever traced, the raw message and cleartext PII never reach the observability layer. If
        # Protegrity is unavailable, FAIL CLOSED: refuse the turn rather than let unprotected text
        # reach retrieval, the model, or the trace. (There is no mock fallback.)
        try:
            prot = self.protector.protect(message)
        except ProtectionUnavailable as exc:
            _log.warning("protection unavailable, refusing turn: %s", type(exc).__name__)
            return Answer(answer=_PROTECTION_DOWN, refused=True, retryable=True)
        query_for_retrieval = self.protector.strip_tags(prot.protected)
        user_tokens = set(prot.mapping.keys())
        topic = _classify_topic(query_for_retrieval)

        with tracing.trace_turn(user_input=prot.protected, conversation_id=conversation_id,
                                user_id=visitor_id, topic=topic,
                                entities_protected=prot.entities,
                                entity_types=sorted(set(prot.types))) as turn:
            turn.event("protect", entities=prot.entities, types=sorted(set(prot.types)),
                       protection_backend=self.protector.backend)
            # Safety/compliance + engagement KPIs available from the very start of the turn.
            turn.score("pii_entities_protected", prot.entities)
            turn.score("topic", topic)
            turn.score("protection_backend", self.protector.backend)

            # (The emergency short-circuit ran on the raw message before protection, above, so an
            # urgent symptom is handled even when Protegrity is down; it never reaches here.)

            # 2. GraphRAG retrieve: vector seeds + graph expansion.
            with turn.span("retrieve", query=query_for_retrieval, k=k) as sp:
                try:
                    seeds = self.retriever.search(query_for_retrieval, k=k) or []
                    seed_ids = [c["id"] for c in seeds]
                    expanded = self.retriever.graph_expand(seed_ids) if seed_ids else []
                except Exception as exc:  # noqa: BLE001, a retrieval fault must not leak or crash
                    _log.warning("retrieval failed: %s", type(exc).__name__)
                    seeds, expanded = [], []
                chunks = _dedupe(seeds + expanded)
                sp.set(seed_chunks=[_chunk_ref(c) for c in seeds],
                       expanded_chunks=[_chunk_ref(c) for c in expanded],
                       total_chunks=len(chunks))
            # Answer-quality KPIs: how much context we found, how close the best match was, and how
            # long retrieval took (span latency is finalised once the `with` block above exits).
            top_sim = max((c.get("score", 0.0) for c in seeds), default=0.0)
            turn.score("retrieval_hits", len(chunks))
            turn.score("top_similarity", round(float(top_sim), 4))
            turn.score("retrieve_ms", sp._latency_ms or 0)

            if not chunks:
                turn.score("outcome", "refused")
                turn.finish(output=_REFUSAL, outcome="refused", reason="no retrieval hits")
                return Answer(answer=_REFUSAL, refused=True, entities_protected=prot.entities,
                              trace_id=turn.trace_id)

            context = _build_context(chunks)

            # 3. Grounded generation, the model reasons over tokens and the retrieved context only.
            prompt = (f"CONTEXT:\n{context}\n\n"
                      f"USER QUESTION: {self.protector.strip_tags(prot.protected)}\n\n"
                      "Answer using only the context above. If it isn't there, say so.")
            sysprompt = get_system_prompt()  # Langfuse-managed when available, hardcoded fallback
            gen_sp = None
            # A first-class Langfuse GENERATION (model + usage + prompt-version link), not a plain
            # span. `prompt=sysprompt.handle` links this call to the Prompt-Management version when
            # the prompt came from Langfuse (handle is None on the hardcoded fallback -> no link).
            with turn.generation("generate", model=f"{self.llm.provider}:{self.llm.model}",
                                 prompt=sysprompt.handle,
                                 prompt_source=sysprompt.source,
                                 prompt_version=sysprompt.version) as sp:
                gen_sp = sp
                try:
                    raw = self.llm.complete(sysprompt.text, prompt)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("generation failed: %s", type(exc).__name__)
                    sp.set(error=type(exc).__name__)
                    turn.score("outcome", "error")
                    turn.finish(output=_REFUSAL, outcome="error", reason="generation failed")
                    return Answer(answer=_REFUSAL, refused=True, entities_protected=prot.entities,
                                  trace_id=turn.trace_id)
                # Token usage (~4 chars/token estimate; Ollama does not return usage uniformly) so the
                # generation carries input/output tokens for cost/usage charts.
                sp.set(output=raw)
                gen_model = f"{self.llm.provider}:{self.llm.model}"
                gen_in_tokens = max(1, len(sysprompt.text + prompt) // 4)
                gen_out_tokens = max(1, len(raw) // 4)
                sp.usage(input_tokens=gen_in_tokens, output_tokens=gen_out_tokens)
            turn.score("prompt_source", sysprompt.source)
            # Performance & cost KPIs. Token counts are an estimate (~4 chars/token), a real cost
            # figure needs the provider's usage, which local Ollama does not return uniformly.
            turn.score("generate_ms", gen_sp._latency_ms or 0)
            turn.score("answer_chars", len(raw))
            turn.score("est_answer_tokens", max(1, len(raw) // 4))
            turn.score("model", f"{self.llm.provider}:{self.llm.model}")

            # A small model sometimes echoes one of the user's OWN tokens back ("Hi NA1234..., here
            # are the side effects"). That is not a leak of anyone else's data, so we scrub the
            # user's own token strings (with their wrapper) from the draft for presentation, rather
            # than block an otherwise-good answer. We keep the ORIGINAL draft for the leak check.
            draft = raw
            raw = _scrub_own_tokens(raw, user_tokens)

            # 4. Egress guard, fail closed. The leak check runs on the ORIGINAL draft (raw_reply=draft)
            # so scrubbing the user's own echo cannot launder a foreign leaked token past the gate; the
            # groundedness/advice checks run on the presented text. A blocked/refused reply carries the
            # calm fallback (not a safety answer, so no ISI note, safety=False).
            verdict = scan_reply(raw, context=context, user_tokens=user_tokens, raw_reply=draft)
            turn.event("guard", outcome=verdict.outcome, grounding=round(verdict.grounding_score, 3),
                       safety_relevant=verdict.safety_relevant, reason=verdict.reason or None)
            # Safety/compliance + answer-quality KPIs (emitted for every terminal path).
            turn.score("grounding", round(verdict.grounding_score, 4))
            turn.score("guard_action", verdict.outcome)
            turn.score("safety_flagged", verdict.safety_relevant)
            turn.score("pii_leak_blocked", bool(verdict.leaked))
            if verdict.outcome == "blocked":
                turn.score("outcome", "blocked")
                turn.finish(output=_BLOCKED, outcome="blocked", reason=verdict.reason)
                return Answer(answer=_BLOCKED, blocked=True, entities_protected=prot.entities,
                              grounding_score=verdict.grounding_score, trace_id=turn.trace_id,
                              model=gen_model, input_tokens=gen_in_tokens, output_tokens=gen_out_tokens)
            if verdict.outcome == "refused":
                turn.score("outcome", "refused")
                turn.finish(output=_REFUSAL, outcome="refused", reason=verdict.reason)
                return Answer(answer=_REFUSAL, refused=True, entities_protected=prot.entities,
                              grounding_score=verdict.grounding_score, trace_id=turn.trace_id,
                              model=gen_model, input_tokens=gen_in_tokens, output_tokens=gen_out_tokens)

            # 4b. Protegrity Semantic Guardrail: a second, ML opinion on the outbound reply (when
            # wired). It can only ADD a block (e.g. an identifier the regex missed); it never
            # downgrades a reply the regex guard already passed.
            if _semantic_block(raw, turn):
                turn.score("outcome", "blocked")
                turn.score("guard_action", "blocked")
                turn.finish(output=_BLOCKED, outcome="blocked", reason="semantic guard rejected")
                return Answer(answer=_BLOCKED, blocked=True, entities_protected=prot.entities,
                              grounding_score=verdict.grounding_score, trace_id=turn.trace_id,
                              model=gen_model, input_tokens=gen_in_tokens, output_tokens=gen_out_tokens)

            # 5. Role-gated reveal. The PUBLIC chat path reveals NOTHING, so the answer carries no
            # user PII back (there should be none anyway, the guard just proved it). Re-identification
            # is available ONLY to an entitled support/internal caller via POST /api/support/reveal
            # (Protector.reveal_text), never on this public path, which stays zero-reveal.
            answer_text = raw  # already token-free per the guard; public reveal is a no-op

            # Citations are computed for the TRACE (operator-facing provenance) but not returned to
            # the public UI, the user sees a clean answer; the sources live in the trace.
            citations = _citations(chunks)
            turn.score("outcome", "answered")
            turn.score("source_pages", len(citations))
            turn.finish(output=answer_text, outcome="answered",
                        grounding=round(verdict.grounding_score, 3),
                        safety_relevant=verdict.safety_relevant,
                        sources=[c["url"] for c in citations])
            return Answer(answer=answer_text, citations=citations,
                          safety=verdict.safety_relevant, trace_id=turn.trace_id,
                          entities_protected=prot.entities, grounding_score=verdict.grounding_score)

    def answer_stream(self, message: str, k: int = 4, conversation_id: str = "",
                      visitor_id: str | None = None):
        """Streaming variant of answer(). Yields event dicts for an SSE endpoint:

            {"type": "token",  "text": "..."}                 a chunk of the draft, as it generates
            {"type": "final",  "answer","safety","refused",   the authoritative result after the
                     "blocked","trace_id","replaced"}          egress guard has run on the FULL text

        Design, OPTIMISTIC STREAM + RETRACT: we stream the model's draft live for instant feedback,
        but the egress guard is still authoritative and runs on the COMPLETE text. If the guard
        blocks/refuses, the `final` event carries the safe fallback with replaced=True, and the UI
        swaps out the streamed draft. The short-circuit paths (emergency, no-hits, gen error) emit no
        tokens, just a `final`. Every draft is validated.

        CONTAINMENT CAVEAT: this path is strictly weaker than answer() for CONTAINMENT. Because the
        draft is transmitted before the guard runs, a leaked token is briefly on the client before the
        retract arrives, retract can swap the displayed text but cannot un-send bytes already on the
        wire. answer() validates before anything leaves the server. For the strongest PII guarantee,
        prefer the non-streaming endpoint; streaming trades a small residual exposure for latency.
        """
        # Emergency first, on the raw message, before protection, so an urgent symptom surfaces even
        # when Protegrity is down. We open a PII-FREE trace (redacted placeholder input, never the raw
        # message) so this safety-critical event is still counted in observability.
        if _is_emergency(message):
            tid = _trace_emergency(conversation_id, visitor_id, streamed=True)
            yield {"type": "final", "answer": _EMERGENCY, "safety": False, "refused": False,
                   "blocked": False, "retryable": False, "trace_id": tid, "replaced": False}
            return

        # Protect, or FAIL CLOSED. No mock fallback: if Protegrity is unavailable, refuse the turn
        # rather than stream unprotected text.
        try:
            prot = self.protector.protect(message)
        except ProtectionUnavailable as exc:
            _log.warning("protection unavailable, refusing stream: %s", type(exc).__name__)
            yield {"type": "final", "answer": _PROTECTION_DOWN, "safety": False, "refused": True,
                   "blocked": False, "retryable": True, "trace_id": None, "replaced": False}
            return
        query_for_retrieval = self.protector.strip_tags(prot.protected)
        user_tokens = set(prot.mapping.keys())
        topic = _classify_topic(query_for_retrieval)

        with tracing.trace_turn(user_input=prot.protected, conversation_id=conversation_id,
                                user_id=visitor_id, topic=topic, streamed=True,
                                entities_protected=prot.entities,
                                entity_types=sorted(set(prot.types))) as turn:
            turn.event("protect", entities=prot.entities, types=sorted(set(prot.types)),
                       protection_backend=self.protector.backend)
            turn.score("pii_entities_protected", prot.entities)
            turn.score("topic", topic)
            turn.score("protection_backend", self.protector.backend)

            def final(answer, *, safety=False, refused=False, blocked=False, replaced=False,
                      retryable=False):
                return {"type": "final", "answer": answer, "safety": safety, "refused": refused,
                        "blocked": blocked, "retryable": retryable, "trace_id": turn.trace_id,
                        "replaced": replaced}

            # Retrieve.
            with turn.span("retrieve", query=query_for_retrieval, k=k) as sp:
                try:
                    seeds = self.retriever.search(query_for_retrieval, k=k) or []
                    expanded = self.retriever.graph_expand([c["id"] for c in seeds]) if seeds else []
                except Exception as exc:  # noqa: BLE001
                    _log.warning("retrieval failed: %s", type(exc).__name__)
                    seeds, expanded = [], []
                chunks = _dedupe(seeds + expanded)
                sp.set(total_chunks=len(chunks))
            top_sim = max((c.get("score", 0.0) for c in seeds), default=0.0)
            turn.score("retrieval_hits", len(chunks))
            turn.score("top_similarity", round(float(top_sim), 4))
            turn.score("retrieve_ms", sp._latency_ms or 0)
            if not chunks:
                turn.score("outcome", "refused")
                turn.finish(output=_REFUSAL, outcome="refused", reason="no retrieval hits")
                yield final(_REFUSAL, refused=True)
                return

            context = _build_context(chunks)
            prompt = (f"CONTEXT:\n{context}\n\n"
                      f"USER QUESTION: {query_for_retrieval}\n\n"
                      "Answer using only the context above. If it isn't there, say so.")

            # Stream generation, accumulating the full draft. Tokens go to the client optimistically.
            sysprompt = get_system_prompt()  # Langfuse-managed when available, hardcoded fallback
            parts: list[str] = []
            # First-class GENERATION (model + usage + prompt-version link) for the streamed call.
            with turn.generation("generate", model=f"{self.llm.provider}:{self.llm.model}",
                                 prompt=sysprompt.handle,
                                 prompt_source=sysprompt.source,
                                 prompt_version=sysprompt.version) as sp:
                try:
                    for piece in self.llm.stream(sysprompt.text, prompt):
                        parts.append(piece)
                        yield {"type": "token", "text": piece}
                except Exception as exc:  # noqa: BLE001
                    _log.warning("generation failed: %s", type(exc).__name__)
                    sp.set(error=type(exc).__name__)
                    turn.score("outcome", "error")
                    turn.finish(output=_REFUSAL, outcome="error", reason="generation failed")
                    # Nothing safe was validated, retract whatever streamed, show the fallback.
                    yield final(_REFUSAL, refused=True, replaced=bool(parts))
                    return
                _draft = "".join(parts)
                sp.set(output=_draft)
                sp.usage(input_tokens=max(1, len(sysprompt.text + prompt) // 4),
                         output_tokens=max(1, len(_draft) // 4))
            turn.score("generate_ms", sp._latency_ms or 0)

            draft = "".join(parts)
            raw = _scrub_own_tokens(draft, user_tokens)
            turn.score("answer_chars", len(raw))
            turn.score("est_answer_tokens", max(1, len(raw) // 4))
            turn.score("model", f"{self.llm.provider}:{self.llm.model}")

            # Egress guard on the COMPLETE text, authoritative, even though the draft already
            # streamed. The leak check runs on the ORIGINAL draft (raw_reply=draft) so scrubbing can't
            # launder a foreign token past it. On block/refuse we retract (replaced=True).
            verdict = scan_reply(raw, context=context, user_tokens=user_tokens, raw_reply=draft)
            turn.event("guard", outcome=verdict.outcome, grounding=round(verdict.grounding_score, 3),
                       safety_relevant=verdict.safety_relevant, reason=verdict.reason or None)
            turn.score("grounding", round(verdict.grounding_score, 4))
            turn.score("guard_action", verdict.outcome)
            turn.score("safety_flagged", verdict.safety_relevant)
            turn.score("pii_leak_blocked", bool(verdict.leaked))
            if verdict.outcome == "blocked":
                turn.score("outcome", "blocked")
                turn.finish(output=_BLOCKED, outcome="blocked", reason=verdict.reason)
                yield final(_BLOCKED, blocked=True, replaced=True)
                return
            if verdict.outcome == "refused":
                turn.score("outcome", "refused")
                turn.finish(output=_REFUSAL, outcome="refused", reason=verdict.reason)
                yield final(_REFUSAL, refused=True, replaced=True)
                return

            # Semantic Guardrail second opinion on the full streamed reply (when wired). A rejection
            # retracts the optimistic stream to the safe fallback (replaced=True), same as the regex
            # block path. It only ADDS a block; it never passes something the regex guard rejected.
            if _semantic_block(raw, turn):
                turn.score("outcome", "blocked")
                turn.score("guard_action", "blocked")
                turn.finish(output=_BLOCKED, outcome="blocked", reason="semantic guard rejected")
                yield final(_BLOCKED, blocked=True, replaced=True)
                return

            # Citations are computed for the TRACE (operator-facing provenance) only, they are NOT
            # returned to the public UI, same as the non-streaming /api/chat path. The visitor sees a
            # clean grounded answer; the per-chunk sources live in the Langfuse trace for operators.
            citations = _citations(chunks)
            turn.score("outcome", "answered")
            turn.score("source_pages", len(citations))
            turn.finish(output=raw, outcome="answered", grounding=round(verdict.grounding_score, 3),
                        safety_relevant=verdict.safety_relevant, sources=[c["url"] for c in citations])
            # The clean, guard-approved text. If _scrub_own_tokens changed it from what streamed,
            # replaced=True lets the UI resync to the canonical text.
            yield final(raw, safety=verdict.safety_relevant, replaced=(raw != "".join(parts)))


# Lightweight topic tags for engagement analytics, a coarse, deterministic keyword classifier
# (no model call). Order matters: the first matching bucket wins. "other" catches the rest.
_TOPIC_PATTERNS = [
    ("cost", re.compile(r"\b(cost|price|pay|afford|insurance|coverage|cover|savings|discount|copay)", re.I)),
    ("side_effects", re.compile(r"\b(side effect|adverse|risk|safe|safety|warning|reaction|swallow|breath)", re.I)),
    ("dosing", re.compile(r"\b(dose|dosage|units?|how much.*(take|inject)|how many units)", re.I)),
    ("conditions", re.compile(r"\b(treat|condition|migraine|dystonia|spasticit|bladder|sweat|indication|used for)", re.I)),
    ("how_it_works", re.compile(r"\b(how does|how long|last|work|procedure|expect|treatment|inject)", re.I)),
    ("find_provider", re.compile(r"\b(specialist|doctor|provider|near me|find a|appointment|consult)", re.I)),
]


def _classify_topic(text: str) -> str:
    for name, pat in _TOPIC_PATTERNS:
        if pat.search(text):
            return name
    return "other"


# Cap the retrieved context handed to the model. With the default k=4 seeds + up to 4 expanded
# chunks this is never hit, but it bounds the prompt if k, max_expand, or chunk size grows, so a
# retrieval-config change can't silently blow the model's context window. ~12k chars ~= 3k tokens.
_MAX_CONTEXT_CHARS = int(os.getenv("BOTOX_MAX_CONTEXT_CHARS", "12000"))


def _build_context(chunks: list[dict]) -> str:
    """Join retrieved chunks into the model's CONTEXT block, truncated to a char budget so an
    unusually large retrieval set can't overflow the context window. Chunks are added whole, in
    order, until the budget is reached."""
    parts: list[str] = []
    total = 0
    for c in chunks:
        piece = f"[{c['title']}] {c['text']}"
        if total + len(piece) > _MAX_CONTEXT_CHARS and parts:
            break
        parts.append(piece)
        total += len(piece) + 2   # +2 for the "\n\n" separator
    return "\n\n".join(parts)


def _trace_emergency(conversation_id: str, visitor_id: str | None, *, streamed: bool = False):
    """Open a PII-FREE trace for an emergency short-circuit so this safety-critical event is counted
    in observability. The raw message may carry PII and is NEVER traced, the trace input is a fixed
    redacted placeholder. Returns the trace_id (None when tracing is off). Never raises."""
    try:
        with tracing.trace_turn(user_input="[emergency: redacted]", conversation_id=conversation_id,
                                user_id=visitor_id, topic="emergency", streamed=streamed) as turn:
            turn.event("emergency", detected=True)
            turn.score("outcome", "emergency")
            turn.score("safety_flagged", True)
            turn.finish(output=_EMERGENCY, outcome="emergency", reason="urgent symptom detected")
            return turn.trace_id
    except Exception as exc:  # noqa: BLE001, observability must never break the safety path
        _log.debug("emergency trace failed: %s", type(exc).__name__)
        return None


def _semantic_block(reply: str, turn) -> bool:
    """Run Protegrity's Semantic Guardrail on the outbound reply (when wired) as a second opinion
    on top of the regex guard. Records the verdict on the trace and returns True if the service
    REJECTED the reply (a block). Never downgrades: an unavailable/disabled guard returns False and
    the regex guard's decision stands, so this can only ADD safety, never remove it. Any unexpected
    error (a malformed-but-decodable response) is swallowed to False, an OPTIONAL guard must never
    turn a reply the authoritative regex guard already approved into an error."""
    try:
        v = semantic_guard.scan_response(reply)
    except Exception as exc:  # noqa: BLE001, the optional layer must not break the request path
        _log.warning("semantic guard errored, ignoring: %s", type(exc).__name__)
        return False
    if not v.available:
        return False
    turn.event("semantic_guard", outcome=("rejected" if v.rejected else "approved"),
               score=round(v.score, 4), findings=v.findings or None)
    turn.score("semantic_guard_score", round(v.score, 4))
    turn.score("semantic_guard_action", "rejected" if v.rejected else "approved")
    return v.rejected


def _scrub_own_tokens(text: str, user_tokens: set[str]) -> str:
    """Remove the user's OWN surrogate tokens (and the wrapper immediately around them) that a model
    may have benignly echoed, e.g. "Hi [NAME]NA12ab[/NAME], here are..." -> "Hi, here are...". This
    ONLY targets this turn's own tokens; it deliberately does NOT strip arbitrary [TYPE] wrappers,
    that would launder a FOREIGN leaked token past the egress guard (which must still see and block
    it). The egress leak check runs on the PRE-scrub text, so scrubbing cannot hide a real leak.
    Trims the small grammatical debris a removed token leaves. The cleanup only runs when a token
    was actually removed, so a normal answer (e.g. "Hi, BOTOX is...") is never rewritten."""
    before = text
    for tok in user_tokens:
        if not tok:
            continue
        # Remove the token WITH an optional [TYPE]...[/TYPE] wrapper hugging it, in one shot.
        text = re.sub(rf"\[[A-Z_]+\]\s*{re.escape(tok)}\s*\[/[A-Z_]+\]", "", text)
        text = text.replace(tok, "")
    if text == before:
        return text                                          # nothing removed -> leave the reply as-is
    # Grammatical debris a removed greeting/name leaves: "Hello and,", "Hi ,", "Dear ,".
    text = re.sub(r"\b(Hello|Hi|Hey|Dear)\s+and\b[,:]?", r"\1,", text)
    text = re.sub(r"\b(Hello|Hi|Hey|Dear)\s*[,:]\s*(?=[A-Z])", r"\1! ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)               # space-before-punct from a removed token
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _dedupe(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in chunks:
        if c["id"] not in seen:
            seen.add(c["id"]); out.append(c)
    return out


def _chunk_ref(c: dict) -> dict:
    """A compact, trace-friendly reference to a retrieved chunk: which chunk, from which page, its
    score, enough for an operator to see what fed the answer, without dumping the full text."""
    return {"id": c.get("id"), "title": c.get("title"), "url": c.get("url"),
            "score": round(c["score"], 4) if isinstance(c.get("score"), (int, float)) else None,
            "safety": bool(c.get("safety"))}


def _citations(chunks: list[dict]) -> list[dict]:
    """One citation per distinct source page, in first-seen order."""
    seen, cites = set(), []
    for c in chunks:
        url = c.get("url")
        if url and url not in seen:
            seen.add(url)
            cites.append({"title": _clean_title(c.get("title") or url), "url": url})
    return cites


def _clean_title(t: str) -> str:
    t = re.sub(r"\s*[-|]\s*BOTOX.*$", "", t).strip()
    return t or "BOTOX® information"
