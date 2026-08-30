"""Turn-based serving: the same protection primitives, wrapped for a live chatbot or agents.

The measurement pipeline runs in batch (a corpus of tasks scored offline). A chatbot or a
multi-agent system needs the *same guarantees* on a *single live turn*: an inbound message may
carry PII, the model must reason over tokens only, and nothing identifying may reach the user
except what their role is entitled to see. Every one of those steps already exists as a tested
primitive, discovery + protect (ingest), the token-reasoning LLM client, the egress guardrail,
role-gated re-identification. This module is the thin, stateful loop that composes them into a
turn, so a serving layer never re-implements protection and never drifts from the batch path.

One turn:

    inbound text
      -> protect_text()         discover PII, tokenize it in place (wrapped tokens)
      -> [caller adds context]  retrieval / tools operate on the tokenized text
      -> llm.complete()         the model reasons over tokens, never plaintext
      -> guardrail.scan_response()   egress leak-check; fail closed
      -> reidentify(role)       plaintext only for entity types this role may see
    outbound text

The token-stability property is what makes this safe across a **multi-agent** system: a token
is deterministic per underlying value, so agents can pass pseudonymous references to each other
across hops, and only the final presentation boundary (one `reidentify` call, role-gated) turns
tokens back into plaintext. No agent in the middle ever holds a clear identifier.

Observability is per turn: each turn is one Langfuse generation in a session keyed by the
conversation id, so a live deployment gets the same three-plane telemetry the batch runs do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from amlguard.log import get_logger

_log = get_logger("serving")


def protect_text(
    text: str, protector: Any, domain: Any | None = None,
    roster: Any = None, scope: Any = None,
) -> tuple[str, int]:
    """Tokenize the PII in one message, in place, returning (protected_text, entity_count).

    Detects entities via the SAME hybrid detector the batch path uses (`detect_entities` =
    discovery UNION roster), protects each through the audited batch path ingest uses (no-op
    redaction included), and splices wrapped `[TYPE]token[/TYPE]` tags back by offset so
    re-identification can reverse them later. Reuses ingest's primitives verbatim, so a served
    turn is protected identically to a corpus document.

    The `roster` is load-bearing, not optional-for-convenience: discovery's ORGANIZATION recall
    is measured at zero across every threshold, and 45% of the corpus is
    organizations. Without the roster an org name an analyst types ("Sablefield Advisory
    Services") is NEVER tokenized and reaches the model as cleartext AND lands in Langfuse in the
    clear, breaking this module's "the model reasons over tokens, never plaintext" contract. A
    serving path that skips the roster is a leak, so callers should always pass one; `None` is
    kept only so unit tests can drive the discovery path in isolation.

    Entities the domain does not map, or the protector cannot protect, are redacted (`[REDACTED]`)
    rather than left clear (the audited path redacts a no-op).
    """
    from amlguard.ingest import (
        ENTITY_TO_ELEMENT,
        _expand_span,
        _splice,
        detect_entities,
        protect_batch_audited,
    )

    entities = detect_entities(text, roster=roster, scope=scope)
    if not entities:
        return text, 0

    # Group by data element for one batched protect call per element, exactly as ingest does.
    # An entity whose type has no tokenization element is REDACTED, never left clear: discovery
    # flagged it as PII, so passing it verbatim to the model would be a serving-path leak (the
    # batch path redacts a no-op the same way). Fail closed, do not fail open on an unmapped type.
    by_element: dict[str, list[tuple[int, int, str, str]]] = {}
    unmapped: list[tuple[int, int, str]] = []
    for ent in entities:
        entity_type = ent.get("entity_type")
        element = ENTITY_TO_ELEMENT.get(entity_type)
        start, end = _expand_span(text, ent["start"], ent["end"])
        if element is None:
            unmapped.append((start, end, f"[{entity_type or 'PII'}][REDACTED][/{entity_type or 'PII'}]"))
            continue
        by_element.setdefault(element, []).append((start, end, entity_type, text[start:end]))

    replacements: list[tuple[int, int, str]] = list(unmapped)
    count = len(unmapped)
    for element, spans in by_element.items():
        tokens = protect_batch_audited(protector, [s[3] for s in spans], element)
        for (start, end, entity_type, _value), token in zip(spans, tokens):
            replacements.append((start, end, f"[{entity_type}]{token}[/{entity_type}]"))
            count += 1

    return _splice(text, replacements), count


@dataclass
class TurnResult:
    """One turn's audit record, everything a caller needs to trust the turn."""

    reply: str                       # role-gated, re-identified reply shown to the user
    raw_completion: str              # the model's tokenized output, before re-identification
    protected_input: str             # the inbound message with PII tokenized
    entities_protected: int          # PII items tokenized on the way in
    revealed: int                    # identifiers re-identified for this role on the way out
    egress_blocked: bool             # true if the guardrail withheld the reply
    # The guardrail's rich verdict on the reply: per-processor score/explanation and the
    # conversation-level batch score. Empty when no guardrail was supplied. Surfaced so a UI
    # can show WHY the egress guard passed/blocked, not just a boolean.
    egress_detail: dict = field(default_factory=dict)
    error: str = ""
    out_of_scope: int = 0            # tokens the role could see but this turn's scope withheld
    canary_hits: int = 0             # revealed values that tripped the canary tripwire

    @property
    def ok(self) -> bool:
        return not self.error and not self.egress_blocked


@dataclass
class ConversationSession:
    """A stateful, protection-preserving conversation for a chatbot or an agent.

    Holds the per-conversation identity (protector, LLM, role, optional guardrail and domain)
    and runs one protected turn at a time. Deliberately does NOT own retrieval or tool-calling:
    the caller decides what context to add to the tokenized message, because that varies by
    application, while the protection boundary does not. The session guarantees the boundary.

    Give each session its OWN `LLMClient` (and its own `Protector` under concurrency): the
    session configures its client for serving, response caching OFF and this conversation's id
    as the trace session, and a client shared across sessions would carry one conversation's
    caching/session state into another. `turn()` re-applies both every call so a shared client
    is at least corrected to the current caller, but per-session clients are the intended shape.
    """

    protector: Any
    llm: Any
    # An opaque session key (e.g. a UUID), NOT PII. It becomes the Langfuse session id and is
    # written to telemetry unredacted, so it must not contain a customer name/email/account.
    conversation_id: str
    role: Any = None                 # a reidentify.Role; None -> AUDITOR (reveals nothing)
    guardrail: Any = None            # optional egress guard; when set, load-bearing
    domain: Any = None               # selects the system prompt
    # The hybrid roster (ORGANIZATION + identifier values discovery misses). Load-bearing on the
    # inbound protect step: without it, org names an analyst types are never tokenized and reach
    # the model / Langfuse in the clear. A live/UI session MUST supply one; None is for tests
    # that drive the discovery-only path deliberately.
    roster: Any = None
    scope: Any = None                # ProtectionScope the roster is filtered against (blocking)
    # Fail-closed policy for a MISSING guardrail on the analyst/live path. When True (the serving
    # default), a turn with no guardrail withholds the reply rather than returning it unscanned:
    # architecture.md's "on the analyst path the guard fails closed" must hold even when the guard
    # could not be *built* (e.g. the sidecar 500s under memory pressure), not only when its
    # scan_response raises. Tests that deliberately drive an unguarded path set this False.
    require_guardrail: bool = True
    system_prompt: str | None = None  # explicit override; else resolved from the domain
    history: list[dict] = field(default_factory=list, repr=False)
    # Where each turn's audit record goes, called with the TurnResult after EVERY turn,
    # including failed and egress-blocked ones. A live deployment (e.g. an AML assistant with
    # a SAR-supporting-documentation obligation) passes a durable sink; None keeps the in-memory
    # history only. The record is metadata (tokens protected, revealed, egress outcome, error),
    # never clear PII, the reply is already role-gated.
    audit_sink: Any = None
    # Optional detokenization defenses (see reveal_ledger.py), all no-ops when None:
    #  * ledger    — a RevealLedger; one hash-chained record per reveal (metadata only).
    #  * tripwire  — a CanaryTripwire; revealed values scanned for canaries.
    #  * scope_bound — when True, a turn may only re-identify the subject entities of THIS turn's
    #    inbound message (the tokens the caller's own question produced), not every subject that
    #    happened to surface in the model's reply. This shrinks an authorized-but-injected
    #    session's blast radius: an injected instruction to "reveal all parties" yields tokens.
    ledger: Any = None
    tripwire: Any = None
    scope_bound: bool = False

    def _audit(self, result: "TurnResult", turn_index: int) -> None:
        """Emit one turn's audit record: to the caller's sink, and to Langfuse as a score."""
        if self.audit_sink is not None:
            try:
                self.audit_sink({
                    "conversation_id": self.conversation_id,
                    "turn": turn_index,
                    "entities_protected": result.entities_protected,
                    "revealed": result.revealed,
                    "egress_blocked": result.egress_blocked,
                    "ok": result.ok,
                    "error": result.error,
                })
            except Exception as exc:  # noqa: BLE001, auditing must not break the turn
                _log.warning("audit sink failed: %s: %s", type(exc).__name__, exc)
        # A per-turn quality signal on the conversation's Langfuse session, so failed and
        # blocked turns are visible in telemetry, not only successful generations.
        try:
            from amlguard.observability import record_score

            record_score(
                "serving.turn_ok", 1.0 if result.ok else 0.0,
                comment=result.error or "ok",
            )
        except Exception:  # noqa: BLE001, telemetry is never a dependency
            pass

    def _system(self) -> str:
        if self.system_prompt is not None:
            return self.system_prompt
        from amlguard.domains import get_domain
        from amlguard.observability import managed_prompt

        return managed_prompt((self.domain or get_domain()).investigation_prompt)

    def turn(self, message: str, context: str = "", max_tokens: int | None = None) -> TurnResult:
        """Run one protected turn, then audit it.

        `message` is the raw user text (may contain PII); `context` is any additional material
        the caller has already tokenized (retrieved chunks, tool output) to append to the
        prompt. The reply is re-identified for `self.role` and, when a guardrail is set,
        scanned before it is returned, fail closed. EVERY outcome (success, failure, block) is
        passed to `_audit`, so a live system has a complete per-turn record, not only the
        successful generations the LLM client traces.
        """
        turn_index = len(self.history)
        result = self._run_turn(message, context, max_tokens)
        self._audit(result, turn_index)
        return result

    def _run_turn(self, message: str, context: str, max_tokens: int | None) -> TurnResult:
        from amlguard.reidentify import AUDITOR, reidentify

        role = self.role or AUDITOR
        # This conversation's client is configured for serving, once. Two invariants a live
        # deployment needs and the batch client does not default to:
        #   * response caching OFF, or one user's completion is served to another user asking
        #     the same question (the cache key is prompt-only, not conversation-scoped);
        #   * this conversation's id as the Langfuse session, so telemetry groups by
        #     conversation and does not mis-attribute later conversations to the first.
        # A `ConversationSession` therefore requires its OWN `LLMClient` (do not share one
        # across sessions); this sets both fields to THIS conversation every turn, so even a
        # mistakenly shared client is at least corrected to the caller that is using it now.
        try:
            self.llm.enable_cache = False
            self.llm.trace_session = self.conversation_id
            if getattr(self.llm, "trace_component", "") == "":
                self.llm.trace_component = "serving"
        except Exception:  # noqa: BLE001, a read-only client must not break the turn
            pass
        try:
            protected_input, n_entities = protect_text(
                message, self.protector, self.domain, roster=self.roster, scope=self.scope,
            )
        except Exception as exc:  # noqa: BLE001, a protection failure must not leak the raw text
            _log.warning("inbound protection failed: %s: %s", type(exc).__name__, exc)
            return TurnResult(
                reply="", raw_completion="", protected_input="", entities_protected=0,
                revealed=0, egress_blocked=False,
                error=f"inbound-protection-failed: {type(exc).__name__}",
            )

        prompt = protected_input if not context else f"{protected_input}\n\n{context}"
        try:
            completion = self.llm.complete(self._system(), prompt, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            return TurnResult(
                reply="", raw_completion="", protected_input=protected_input,
                entities_protected=n_entities, revealed=0, egress_blocked=False,
                error=f"generation-failed: {type(exc).__name__}",
            )

        # Egress leak-check BEFORE re-identification, so it flags what the model leaked, not the
        # identifiers the role is entitled to see. Fail closed on BOTH a block AND an error: if
        # the guard is unreachable (GuardrailUnavailable) or raises, the reply is withheld, never
        # shown unscanned. This mirrors the two steps above rather than letting an exception
        # escape turn() and dropping the caller onto an unguarded path.
        egress_detail: dict = {}
        # Fail closed when the guard is REQUIRED but ABSENT (not just when scan_response raises).
        # A None guardrail on the analyst path means the egress leak-check cannot run, so the
        # reply must be withheld — returning it unscanned would let a model-emitted clear
        # identifier reach the analyst, the exact fail-open the "analyst path fails closed"
        # invariant forbids. A caller that intends an unguarded path sets require_guardrail=False.
        if self.guardrail is None and self.require_guardrail:
            _log.warning("egress guard required but unavailable; withholding reply (fail closed)")
            return TurnResult(
                reply="[withheld: egress check unavailable]",
                raw_completion=completion, protected_input=protected_input,
                entities_protected=n_entities, revealed=0, egress_blocked=True,
                error="egress-unavailable: no guardrail",
            )
        if self.guardrail is not None:
            try:
                verdict = self.guardrail.scan_response(completion or "")
            except Exception as exc:  # noqa: BLE001, an unreachable guard must fail closed
                _log.warning("egress scan failed: %s: %s", type(exc).__name__, exc)
                return TurnResult(
                    reply="[withheld: egress check unavailable]",
                    raw_completion=completion, protected_input=protected_input,
                    entities_protected=n_entities, revealed=0, egress_blocked=True,
                    error=f"egress-unavailable: {type(exc).__name__}",
                )
            # Capture the rich verdict for the UI: per-processor score/explanation + the
            # conversation-level batch score. Discarding this was the leverage gap.
            egress_detail = {
                "outcome": getattr(verdict, "outcome", ""),
                "score": round(getattr(verdict, "score", 0.0) or 0.0, 4),
                "batch_outcome": getattr(verdict, "batch_outcome", ""),
                "batch_score": round(getattr(verdict, "batch_score", 0.0) or 0.0, 4),
                "processors": [
                    {"name": getattr(f, "processor", ""),
                     "score": round(getattr(f, "score", 0.0) or 0.0, 4),
                     "explanation": getattr(f, "explanation", "")}
                    for f in getattr(verdict, "findings", []) or []
                ],
                "leaked_values": len(getattr(verdict, "leaked_values", ()) or ()),
            }
            if verdict.blocked:
                reason = (
                    f"{len(verdict.leaked_values)} forbidden value(s)"
                    if verdict.leaked_values else verdict.outcome
                )
                return TurnResult(
                    reply="[withheld: response failed the egress check]",
                    raw_completion=completion, protected_input=protected_input,
                    entities_protected=n_entities, revealed=0, egress_blocked=True,
                    egress_detail=egress_detail, error=f"egress-blocked: {reason}",
                )

        # Re-identify last. A failure here must not surface the raw (possibly tag-bearing)
        # completion; withhold rather than leak.
        #
        # Scope-bound reveal: the authorized scope for this turn is the set of tokens the caller's
        # OWN inbound message produced (the subject they asked about). Binding the reveal to those
        # means a reply that surfaced other subjects' tokens, e.g. via an injected "list everyone"
        # instruction, re-identifies only the intended subject; the rest stay protected.
        scope_tokens = None
        if self.scope_bound:
            from amlguard.reidentify import find_tokens
            scope_tokens = frozenset(tok for _e, _el, tok in find_tokens(protected_input))
        try:
            reidentified = reidentify(
                completion, self.protector, role,
                scope_tokens=scope_tokens, ledger=self.ledger, tripwire=self.tripwire,
                actor=self.conversation_id, purpose="live-turn",
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("re-identification failed: %s: %s", type(exc).__name__, exc)
            return TurnResult(
                reply="[withheld: re-identification failed]", raw_completion=completion,
                protected_input=protected_input, entities_protected=n_entities,
                revealed=0, egress_blocked=False,
                error=f"reidentify-failed: {type(exc).__name__}",
            )
        self.history.append({"user": protected_input, "assistant": completion})
        return TurnResult(
            reply=reidentified.text, raw_completion=completion,
            protected_input=protected_input, entities_protected=n_entities,
            revealed=reidentified.revealed, egress_blocked=False,
            egress_detail=egress_detail,
            out_of_scope=reidentified.out_of_scope, canary_hits=reidentified.canary_hits,
        )
