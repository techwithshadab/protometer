"""Pin the protection boundary and the egress guards. Offline, deterministic, no services."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.guardrail import scan_reply  # noqa: E402
from app.pipeline.orchestrator import Orchestrator  # noqa: E402
from app.protect.protector import Protector  # noqa: E402


# ── Protection boundary ───────────────────────────────────────────────────────────────────────

def test_pii_is_tokenized_and_never_cleartext():
    p = Protector()
    msg = "my name is Jane Doe, email jane@example.com, doctor is Dr. Smith, call 415-555-1234"
    r = p.protect(msg)
    model_sees = p.strip_tags(r.protected)
    assert r.entities >= 4
    for pii in ("Jane Doe", "jane@example.com", "Dr. Smith", "415-555-1234"):
        assert pii not in model_sees, f"{pii} leaked to the model"


def test_repeated_pii_gets_a_stable_token():
    p = Protector()
    r = p.protect("email a@b.com ... again a@b.com")
    # same value -> same token, so it's one entity across the message
    tokens = [t for t in r.mapping if r.mapping[t] == "a@b.com"]
    assert len(tokens) == 1


def test_no_pii_is_a_noop():
    p = Protector()
    r = p.protect("What is BOTOX used to treat?")
    assert r.entities == 0 and "[" not in r.protected


# ── Protegrity Discovery detection ──────────────────────────────────────────────────────────────

def test_discovery_spans_drive_tokenization():
    # Detection comes from Discovery spans; a bare name Discovery finds is tokenized and never
    # reaches the model in the clear.
    p = Protector()

    class _StubDiscovery:
        def discover(self, text):
            i = text.index("Ana Rey")
            return [{"type": "NAME", "start": i, "end": i + len("Ana Rey"),
                     "value": "Ana Rey", "score": 0.95}]

    p._discovery = _StubDiscovery()
    r = p.protect("Ana Rey here, tell me about BOTOX side effects")
    model_sees = p.strip_tags(r.protected)
    assert r.entities == 1 and "Ana Rey" not in model_sees


def test_discovery_failure_fails_closed():
    # A Discovery failure must raise ProtectionUnavailable (fail closed): undetected PII is
    # unprotected PII; there is no regex/mock to silently fall back to.
    from app.protect.discovery import DiscoveryError
    from app.protect.protector import ProtectionUnavailable
    p = Protector()

    class _BrokenDiscovery:
        def discover(self, text):
            raise DiscoveryError("provider down")

    p._discovery = _BrokenDiscovery()
    try:
        p.protect("my email is x@y.com")
    except ProtectionUnavailable:
        return
    assert False, "Discovery failure must fail closed (ProtectionUnavailable)"


def test_tokenization_is_batched_and_deduped():
    # A repeated value must cost ONE protect call (dedup + cache), and tokenization must batch by
    # element rather than one round-trip per span. We count calls and payload sizes on a spy.
    p = Protector()
    calls = {"n": 0, "sizes": []}

    class _SpyDiscovery:
        def discover(self, text):
            # two DOCTOR spans with the SAME value, plus one EMAIL span
            out = []
            for val, etype in (("Dr. Smith", "DOCTOR"), ("Dr. Smith", "DOCTOR"),
                               ("a@b.com", "EMAIL")):
                i = text.index(val)
                out.append({"type": etype, "start": i, "end": i + len(val),
                            "value": val, "score": 0.99})
                text = text[:i] + ("#" * len(val)) + text[i + len(val):]  # avoid re-finding
            return out

    class _SpyTokenizer:
        def protect_values(self, values, element):
            calls["n"] += 1
            calls["sizes"].append(len(values))
            return ["TK" + str(abs(hash((element, v))))[:8] for v in values]

    p._discovery = _SpyDiscovery()
    p._pty = _SpyTokenizer()
    r = p.protect("Dr. Smith and Dr. Smith, email a@b.com")
    # 3 spans detected, but "Dr. Smith" dedups to 1 -> 2 distinct values across 2 elements
    assert r.entities == 3, r.types
    # one batched call per element (DOCTOR->string, EMAIL->string are the SAME element 'string'),
    # so a single round-trip with the 2 distinct values
    assert calls["n"] == 1, f"expected one batched round-trip, got {calls['n']}"
    assert calls["sizes"] == [2], f"expected 2 distinct values in the batch, got {calls['sizes']}"


def test_noop_protection_is_redacted_not_leaked():
    # If the tokenizer returns every value unchanged (a measured Protegrity no-op) under every
    # element, the value cannot be emitted as cleartext. Rather than fail the whole turn closed, the
    # per-value fallback REDACTS the value (a one-way surrogate): no cleartext leaks, and one odd
    # value doesn't refuse an otherwise-fine message.
    p = Protector()

    class _NoopTokenizer:
        def protect_values(self, values, element):
            return list(values)  # unchanged == no-op under any element

    class _OneSpan:
        def discover(self, text):
            return [{"type": "EMAIL", "start": 0, "end": len("x@y.com"),
                     "value": "x@y.com", "score": 0.99}]

    p._discovery = _OneSpan()
    p._pty = _NoopTokenizer()
    r = p.protect("x@y.com asking about BOTOX")
    model_sees = p.strip_tags(r.protected)
    assert "x@y.com" not in model_sees, "cleartext must never survive, even a no-op"
    assert r.entities == 1                      # the value was accounted for (redacted), turn stands


def test_one_unprotectable_value_does_not_refuse_the_turn():
    # A value that fails under its mapped element AND under `string` is redacted; a sibling value in
    # the same message still tokenizes normally. One bad value must not fail-close the whole turn.
    p = Protector()

    class _PartialTokenizer:
        def protect_values(self, values, element):
            # "BADVAL" can never be protected (always raises); everything else tokenizes fine.
            if any(v == "BADVAL" for v in values):
                raise RuntimeError("44, invalid input")
            return ["TK" + str(abs(hash((element, v))))[:8] for v in values]

    class _TwoSpans:
        def discover(self, text):
            out = []
            for sub, ty in (("BADVAL", "SSN"), ("john@example.com", "EMAIL")):
                i = text.index(sub)
                out.append({"type": ty, "start": i, "end": i + len(sub), "value": sub, "score": 0.99})
            out.sort(key=lambda d: d["start"]); return out

    p._discovery = _TwoSpans()
    p._pty = _PartialTokenizer()
    r = p.protect("id BADVAL, email john@example.com")
    model_sees = p.strip_tags(r.protected)
    assert r.entities == 2                       # both spans handled (one redacted, one tokenized)
    assert "BADVAL" not in model_sees and "john@example.com" not in model_sees


def test_truncated_numeric_span_is_expanded():
    # Discovery returns truncated numeric spans for some formats ("-555-7023" within "312-555-7023");
    # expansion must recover the whole identifier so the prefix ("312") doesn't leak. A name span must
    # NOT expand into adjacent digits.
    from app.protect.discovery import _expand_span
    t = "call 312-555-7023 now"
    i = t.index("-555-7023")
    s, e = _expand_span(t, i, i + len("-555-7023"))
    assert t[s:e] == "312-555-7023"
    t2 = "John Smith 42"
    j = t2.index("John Smith")
    s2, e2 = _expand_span(t2, j, j + len("John Smith"))
    assert t2[s2:e2] == "John Smith"       # letters aren't identifier chars; no over-expansion


def test_multibyte_offsets_splice_correctly():
    # Discovery returns CHARACTER offsets (verified live), so the splice using text[start:end] is
    # correct even when a multibyte char (®, accents, emoji) precedes the PII span. This pins that:
    # a NAME span after "BOTOX® 🎉" must tokenize exactly, leaving the multibyte chars intact.
    p = Protector()
    TEXT = "BOTOX® 🎉 José wants info"

    class _CharOffsetDiscovery:
        def discover(self, text):
            i = text.index("José")                    # character index, as the live service returns
            return [{"type": "NAME", "start": i, "end": i + len("José"), "value": "José",
                     "score": 0.9}]

    class _Tok:
        def protect_values(self, values, element):
            return ["TK" + str(abs(hash((element, v))))[:8] for v in values]

    p._discovery = _CharOffsetDiscovery()
    p._pty = _Tok()
    r = p.protect(TEXT)
    model = p.strip_tags(r.protected)
    assert "BOTOX®" in model and "🎉" in model      # multibyte chars untouched
    assert "José" not in model and r.entities == 1  # the name tokenized, nothing adjacent corrupted


def test_separator_sensitive_value_is_normalised_before_protect():
    # A ccn rejects separators, so a spaced/dashed card must be normalised (separators stripped)
    # before it's sent to protect. We capture what the tokenizer actually receives.
    p = Protector()
    seen = {}

    class _CaptureTokenizer:
        def protect_values(self, values, element):
            seen[element] = list(values)
            return ["TK" + str(abs(hash((element, v))))[:8] for v in values]

    class _CardSpan:
        def discover(self, text):
            sub = "4111 1111 1111 1111"; i = text.index(sub)
            return [{"type": "CREDITCARD", "start": i, "end": i + len(sub), "value": sub, "score": 0.99}]

    p._discovery = _CardSpan()
    p._pty = _CaptureTokenizer()
    p.protect("my card is 4111 1111 1111 1111 ok")
    assert seen.get("ccn") == ["4111111111111111"], f"ccn value not normalised: {seen}"


# ── Egress guards ─────────────────────────────────────────────────────────────────────────────

CTX = ("BOTOX is used to treat chronic migraine and cervical dystonia. Common side effects include "
       "headache, neck pain, and injection-site bruising. Important Safety Information: distant "
       "spread of toxin effect has been reported.")


def test_grounded_reply_passes_and_flags_safety():
    reply = ("BOTOX is used to treat chronic migraine and cervical dystonia. Common side effects "
             "include headache and neck pain. Important safety information applies.")
    v = scan_reply(reply, context=CTX, user_tokens=set())
    assert v.ok and v.outcome == "passed" and v.safety_relevant


def test_ungrounded_reply_is_refused():
    reply = "BOTOX cures baldness and grows hair overnight in every patient guaranteed forever."
    v = scan_reply(reply, context=CTX, user_tokens=set())
    assert not v.ok and v.outcome == "refused"


def test_safety_note_reflects_the_answer_not_the_context():
    # The context carries safety text, but an answer that says nothing about risk must NOT get the
    # Important Safety Information note just because the context did. (The "ISI on everything" bug.)
    reply = "BOTOX is used to treat chronic migraine and cervical dystonia."
    v = scan_reply(reply, context=CTX, user_tokens=set())
    assert v.ok and not v.safety_relevant, "an indications answer should not flag safety"
    # An answer that DOES discuss side effects still flags it.
    reply2 = "Common side effects include headache and neck pain."
    v2 = scan_reply(reply2, context=CTX, user_tokens=set())
    assert v2.ok and v2.safety_relevant


def test_meta_language_echo_is_refused():
    # The model reciting its instructions instead of answering must be refused, not shown.
    for bad in (
        "I can only share general information from the official site. Never use outside knowledge.",
        "Based on the context provided, I don't have that in my sources.",
        "As an AI, I am instructed to only use the context above.",
        "I can only answer from my sources.",
    ):
        v = scan_reply(bad, context=CTX, user_tokens=set())
        assert not v.ok and v.outcome == "refused", f"should refuse meta-echo: {bad!r}"


def test_meta_guard_does_not_refuse_legit_answers():
    # "context" and "sources" are ordinary words, a grounded answer that uses them naturally must
    # NOT be tripped by the META guard. (Regression: the meta-leak guard once matched the bare
    # nouns.) We assert specifically that the meta pattern doesn't fire, independent of grounding.
    from app.pipeline.guardrail import _META_LEAK
    for good in (
        "In the context of chronic migraine, BOTOX is a treatment option.",
        "Common side effects and their sources are discussed with your doctor.",
        "BOTOX is used to treat chronic migraine and cervical dystonia.",
    ):
        assert _META_LEAK.search(good) is None, f"meta guard should not fire on: {good!r}"


def test_fabricated_numbers_are_refused_but_grounded_numbers_pass():
    # Lexical overlap alone let a fabricated statistic ride along on on-topic words. A reply that
    # asserts a number absent from the context must be refused (anti-hallucination on specifics);
    # a reply whose numbers ARE in the context still passes.
    ctx = ("BOTOX is used to treat chronic migraine. In studies, 155 units were used across "
           "31 injection sites.")
    bad = scan_reply("BOTOX shows a 92 percent success rate within three days.",
                     context=ctx, user_tokens=set())
    assert not bad.ok and bad.outcome == "refused"
    good = scan_reply("In studies, 155 units were injected across 31 sites.",
                      context=ctx, user_tokens=set())
    assert good.ok and good.outcome == "passed"
    # A reply with no numbers is unaffected by the numeric check.
    none = scan_reply("BOTOX is used to treat chronic migraine.", context=ctx, user_tokens=set())
    assert none.ok


def test_numeric_grounding_is_unit_aware():
    # A dose number is only grounded by the SAME unit: "200 mg" is not grounded by "200 units"
    # (a real dosing-safety difference the digit-only check missed). Non-dosage units are lenient
    # (compound units like "injection sites"), and bare word-numbers ("one option") are not claims.
    ctx = ("In studies, 200 units of BOTOX were injected across 31 injection sites. "
           "Relief lasts 3 to 4 months.")
    # dosage-unit mismatch -> refused
    assert scan_reply("BOTOX uses 200 mg total.", context=ctx, user_tokens=set()).outcome == "refused"
    # exact dose -> passed
    assert scan_reply("About 200 units were injected.", context=ctx, user_tokens=set()).ok
    # compound-unit number match -> passed (not over-strict)
    assert scan_reply("It was given across 31 sites.", context=ctx, user_tokens=set()).ok
    # fabricated percent -> refused
    assert not scan_reply("It has a 92% success rate.", context=ctx, user_tokens=set()).ok
    # bare word-number is NOT a numeric claim (no false refusal)
    ctx2 = "BOTOX is one treatment option a doctor may discuss for chronic migraine."
    assert scan_reply("BOTOX is one option a doctor may discuss for chronic migraine.",
                      context=ctx2, user_tokens=set()).ok


def test_medical_advice_is_blocked():
    reply = "You should take 50 units of BOTOX and stop taking your other medication."
    v = scan_reply(reply, context=CTX, user_tokens=set())
    assert not v.ok and v.outcome == "blocked"


def test_advice_paraphrases_are_blocked_but_informational_passes():
    # The medical-advice block must catch common directive paraphrases, not just "you should",
    # while leaving informational second-person statements alone.
    ctx = "BOTOX is used to treat chronic migraine. Common side effects include headache."
    for bad in ("You'll need to take 100 units.", "You can increase your dose.",
                "Make sure to inject it weekly.", "I would recommend you stop taking it."):
        assert scan_reply(bad, context=ctx, user_tokens=set()).outcome == "blocked", bad
    for ok in ("You can experience headache as a side effect.",
               "You may notice results within days.",
               "BOTOX is used to treat chronic migraine."):
        assert scan_reply(ok, context=ctx, user_tokens=set()).outcome != "blocked", ok


def test_scrub_does_not_rewrite_a_clean_reply():
    # Regression: the grammatical-cleanup ran on every reply, rewriting "Hi, BOTOX..." -> "Hi! BOTOX".
    # It must only run when a token was actually removed.
    from app.pipeline.orchestrator import _scrub_own_tokens
    clean = "Hi, BOTOX is used to treat chronic migraine. Dr. visits help too."
    assert _scrub_own_tokens(clean, set()) == clean
    assert _scrub_own_tokens(clean, {"TKnotpresent"}) == clean


def test_pii_or_token_leak_is_blocked():
    v1 = scan_reply("Your email EM1234abcd00 is noted.", context=CTX, user_tokens={"EM1234abcd00"})
    assert not v1.ok and v1.outcome == "blocked" and v1.leaked == ["EM1234abcd00"]
    v2 = scan_reply("Here is [EMAIL]x[/EMAIL] for you.", context=CTX, user_tokens=set())
    assert not v2.ok and v2.outcome == "blocked"


def test_legit_bracketed_acronym_is_not_scrubbed_or_blocked():
    # A legitimate acronym like "[FDA]" in an answer must not be deleted (strip/scrub) or blocked
    # (guard). Only OUR PII wrapper types may be treated as tokens by strip_tags.
    from app.pipeline.orchestrator import _scrub_own_tokens
    assert _scrub_own_tokens("See the [FDA] label.", set()) == "See the [FDA] label."
    assert Protector.strip_tags("see [FDA] guidance") == "see [FDA] guidance"
    ctx = "BOTOX is FDA-approved to treat chronic migraine."
    v = scan_reply("BOTOX is FDA-approved; see the [FDA] label for chronic migraine.",
                   context=ctx, user_tokens=set())
    assert v.ok and v.outcome == "passed"


def test_foreign_pii_wrapper_is_blocked_not_laundered():
    # A FOREIGN PII wrapper ([EMAIL]x[/EMAIL]) that is NOT one of this turn's own tokens must be
    # LEFT by the scrub (so it isn't laundered) and BLOCKED by the guard. (Regression: the scrub
    # used to strip all wrappers, hiding a foreign leak from the guard.)
    from app.pipeline.orchestrator import _scrub_own_tokens
    draft = "Contact [EMAIL]x9f8e7d6c5b[/EMAIL] now; BOTOX treats migraine."
    scrubbed = _scrub_own_tokens(draft, set())          # no own tokens -> wrapper survives
    assert "[EMAIL]" in scrubbed, "foreign wrapper must NOT be scrubbed away"
    v = scan_reply(scrubbed, context="BOTOX treats chronic migraine.", user_tokens=set(),
                   raw_reply=draft)
    assert not v.ok and v.outcome == "blocked"


def test_bare_redaction_token_is_blocked_but_legit_ids_pass():
    # A bare REDACTION token (RD + 12 hex) that lost its wrapper is a leak and must block.
    from app.protect.protector import _redaction_token
    red = _redaction_token("SSN", "x")   # e.g. RD<12hex>
    v = scan_reply(f"BOTOX treats migraine; ref {red} noted.",
                   context="BOTOX treats chronic migraine.", user_tokens=set())
    assert not v.ok and v.outcome == "blocked"
    # A legitimate pharma identifier (a trial ID) that merely LOOKS token-ish must NOT be blocked as a
    # surrogate (regression: the broad heuristic false-blocked NCT IDs / product codes). It's grounded
    # here because the context contains it.
    ctx = "BOTOX was studied in trial NCT01234567 for chronic migraine."
    v2 = scan_reply("The pivotal trial NCT01234567 studied BOTOX for chronic migraine.",
                    context=ctx, user_tokens=set())
    assert v2.ok and v2.outcome == "passed"


# ── Orchestrator fail-safe paths (no model, no retrieval) ───────────────────────────────────────

class _EmptyRetriever:
    def search(self, q, k=4):
        return []

    def graph_expand(self, ids):
        return []


def test_no_retrieval_hits_refuses_safely():
    orch = Orchestrator(retriever=_EmptyRetriever())
    a = orch.answer("my email is x@y.com, tell me about BOTOX")
    assert a.refused and not a.blocked
    assert "x@y.com" not in a.answer            # PII never echoed even on the refusal path
    assert a.entities_protected == 1


# ── Retrieval similarity floor ──────────────────────────────────────────────────────────────────

class _ScoredRetriever:
    """A retriever whose search() honours a min-similarity floor, like the real vectorstore. An
    off-topic query returns nothing, so the orchestrator must refuse rather than borrow context."""
    def __init__(self, floor=0.30):
        self.floor = floor

    def search(self, q, k=4):
        # crude: only "botox" queries clear the floor; anything else is off-corpus
        score = 0.80 if "botox" in q.lower() else 0.05
        if score < self.floor:
            return []
        return [{"id": "c1", "text": "BOTOX treats chronic migraine.", "url": "u",
                 "title": "BOTOX", "safety": False, "score": score}]

    def graph_expand(self, ids):
        return []


def test_offtopic_query_refuses_without_borrowing_context():
    # Regression: search() had no similarity floor, so it always returned k chunks and the
    # "no retrieval hits -> refuse" path never fired; an off-corpus question got answered from the
    # least-irrelevant chunks. With a floor, an off-topic query returns no chunks and is refused.
    orch = Orchestrator(retriever=_ScoredRetriever())
    a = orch.answer("what is the capital of France")
    assert a.refused and not a.blocked, "off-topic query must be refused, not answered"


# ── Fail-closed when Protegrity is unavailable ──────────────────────────────────────────────────

def _orch_with_down_protection():
    """An Orchestrator whose protector raises ProtectionUnavailable, built WITHOUT constructing the
    real (Protegrity-requiring) Protector."""
    from app.pipeline.llm import LLMClient
    from app.protect.protector import ProtectionUnavailable

    class _DownProtector:
        backend = "protegrity"
        detector = "discovery"

        def protect(self, text):
            raise ProtectionUnavailable("discovery down")

        def strip_tags(self, t):
            return t

    o = Orchestrator.__new__(Orchestrator)
    o.retriever = _EmptyRetriever()
    o.protector = _DownProtector()
    o.llm = LLMClient()
    return o


def test_turn_fails_closed_when_protection_unavailable():
    # No mock fallback: if Protegrity can't protect the message, the turn must REFUSE (no unprotected
    # text reaches retrieval/model/trace), in both the sync and streaming paths.
    from app.pipeline.orchestrator import _PROTECTION_DOWN
    o = _orch_with_down_protection()
    a = o.answer("my email is jane@x.com, tell me about botox")
    assert a.refused and a.answer == _PROTECTION_DOWN

    events = list(o.answer_stream("my email is jane@x.com, tell me about botox"))
    finals = [e for e in events if e.get("type") == "final"]
    assert finals and finals[0]["refused"] and finals[0]["answer"] == _PROTECTION_DOWN
    assert not any(e.get("type") == "token" for e in events), "no tokens may stream before protection"


def test_protection_down_refusal_is_retryable():
    # A protection outage refusal is TRANSIENT: it must carry retryable=True so the UI can offer a
    # retry (unlike a normal grounded refusal, which is not retryable).
    from app.pipeline.orchestrator import _PROTECTION_DOWN
    o = _orch_with_down_protection()
    a = o.answer("tell me about botox")
    assert a.refused and a.retryable and a.answer == _PROTECTION_DOWN
    # streaming path carries the flag too
    ev = [e for e in o.answer_stream("tell me about botox") if e.get("type") == "final"][0]
    assert ev["refused"] and ev.get("retryable") is True


def test_emergency_survives_protection_outage():
    # An urgent symptom runs on the raw message before protection, so it must still surface even when
    # Protegrity is down (safety must never depend on the protection service being up).
    from app.pipeline.orchestrator import _EMERGENCY
    o = _orch_with_down_protection()
    a = o.answer("I can't breathe after my BOTOX injection")
    # Emergency does NOT set safety=True (the urgent message is itself the safety response; the
    # generic ISI callout would only dilute it), but it must NOT be a refusal/block.
    assert a.answer == _EMERGENCY and not a.refused and not a.blocked


# ── Emergency short-circuit ─────────────────────────────────────────────────────────────────────

def test_emergency_short_circuits_before_retrieval():
    from app.pipeline.orchestrator import _EMERGENCY, _is_emergency
    # Urgent boxed-warning symptoms are detected...
    for urgent in ("I'm having trouble breathing after my botox",
                   "my throat is swelling and I can't swallow",
                   "severe allergic reaction"):
        assert _is_emergency(urgent), urgent
    # ...and ordinary questions are not.
    for normal in ("what are the side effects", "how much does botox cost",
                   "does it help migraine"):
        assert not _is_emergency(normal), normal
    # End to end: an emergency returns the urgent message WITHOUT touching the (empty) retriever,
    # and is not a normal refusal/block.
    orch = Orchestrator(retriever=_EmptyRetriever())
    a = orch.answer("I can't breathe after my BOTOX injection")
    assert a.answer == _EMERGENCY and not a.refused and not a.blocked

    # Regression: the emergency check must run on the RAW message, BEFORE PII tokenization, here
    # the NAME tokenizer over-matches "I am having" and would otherwise erase "having trouble",
    # hiding the emergency. It must still fire.
    a2 = orch.answer("I am having trouble breathing after my botox injection")
    assert a2.answer == _EMERGENCY, "emergency must survive a PII tokenizer false-positive"


# ── Protegrity Semantic Guardrail (egress second opinion) ───────────────────────────────────────

class _OneChunkRetriever:
    def search(self, q, k=4):
        return [{"id": "c1", "text": "BOTOX treats chronic migraine and cervical dystonia.",
                 "url": "u", "title": "BOTOX", "safety": False, "score": 0.8}]

    def graph_expand(self, ids):
        return []


class _StubLLM:
    provider = "stub"
    model = "stub"

    def __init__(self, reply):
        self._reply = reply

    def complete(self, system, prompt):
        return self._reply


def test_semantic_guard_blocks_when_service_rejects(monkeypatch):
    # When the Semantic Guardrail rejects an otherwise-grounded reply (e.g. it flags a leaked
    # identifier the regex guard missed), the reply must be BLOCKED and swapped for the safe text.
    from app.pipeline import orchestrator as orch_mod
    from app.pipeline.semantic_guard import SemanticVerdict
    monkeypatch.setattr(orch_mod.semantic_guard, "scan_response",
                        lambda reply: SemanticVerdict(available=True, rejected=True, score=0.97,
                                                      findings=["EMAIL_ADDRESS : [0, 5]"]))
    orch = Orchestrator(retriever=_OneChunkRetriever())
    orch.llm = _StubLLM("BOTOX treats chronic migraine and cervical dystonia.")
    a = orch.answer("what does botox treat")
    assert a.blocked and not a.refused


def test_semantic_guard_unavailable_does_not_change_verdict(monkeypatch):
    # When the guard is off/unreachable (available=False), the regex guard's PASS stands, no
    # downgrade, no spurious block.
    from app.pipeline import orchestrator as orch_mod
    from app.pipeline.semantic_guard import SemanticVerdict
    monkeypatch.setattr(orch_mod.semantic_guard, "scan_response",
                        lambda reply: SemanticVerdict(available=False, reason="disabled"))
    orch = Orchestrator(retriever=_OneChunkRetriever())
    orch.llm = _StubLLM("BOTOX treats chronic migraine and cervical dystonia.")
    a = orch.answer("what does botox treat")
    assert not a.blocked and not a.refused


# ── Feedback anti-forgery ───────────────────────────────────────────────────────────────────────

def test_feedback_trace_signature_roundtrip():
    from app.api.main import _sign_trace, _verify_trace
    handle = _sign_trace("trace-xyz")
    assert _verify_trace(handle) == "trace-xyz"          # valid signature verifies
    assert _verify_trace("trace-xyz.deadbeef") is None   # forged/wrong signature rejected
    assert _verify_trace("no-signature") is None         # malformed handle rejected


# ── Role-gated reveal ───────────────────────────────────────────────────────────────────────────

def test_reveal_serves_only_issued_tokens_no_oracle():
    # reveal_text must serve ONLY from the issued-token registry (tokens THIS process actually
    # produced). It must NOT unprotect caller-supplied/guessed tokens, that would be a decryption
    # oracle for fishing other users' PII.
    p = Protector()
    # Issue tokens by protecting a real message (the stub discovery detects "Jane Doe" + email).
    prot = p.protect("Jane Doe, email jane@example.com, asks about BOTOX.")
    # 1) An issued transcript reveals back to the originals.
    revealed = p.reveal_text(prot.protected)
    assert "Jane Doe" in revealed and "jane@example.com" in revealed

    # 2) A GUESSED token we never issued must NOT reveal cleartext, it returns a marker and never
    # calls unprotect (the stub tokenizer has no unprotect, so a call would even error).
    guessed = "hi [SSN]314-16-0607[/SSN] and [PHONE]000-000-0000[/PHONE]"
    out = p.reveal_text(guessed)
    assert "[unknown token]" in out
    assert "314-16-0607" not in out and "000-000-0000" not in out

    # 3) A redaction token (RD-prefixed, never issued) is marked unrevealable, not guessed.
    from app.protect.protector import _redaction_token
    red = _redaction_token("SSN", "x")
    out2 = p.reveal_text(f"ssn [SSN]{red}[/SSN]")
    assert "unrevealable" in out2


def test_reveal_registry_survives_restart(tmp_path):
    # The issued-token allowlist is PERSISTED (token -> element, no cleartext), so a FRESH Protector
    # (a "restart" / a separate support process) can still reveal a transcript issued by an earlier
    # instance, by looking the token up and unprotecting on demand.
    from app.protect.protector import TokenRegistry
    reg_path = tmp_path / "reg.json"

    # First "process": protect a message; its registry persists to reg_path.
    p1 = Protector()
    p1._registry = TokenRegistry(path=reg_path)
    prot = p1.protect("Jane Doe, email jane@example.com about BOTOX.")
    shared_back = p1._pty._back        # the stub's deterministic token->value store (mimics Protegrity)

    # Second "process": a brand-new Protector reading the SAME persisted allowlist, and a tokenizer
    # that reverses the same deterministic tokens (as real Protegrity would).
    p2 = Protector()
    p2._registry = TokenRegistry(path=reg_path)   # reloads the persisted allowlist from disk
    p2._pty._back = shared_back                    # same deterministic unprotect mapping
    revealed = p2.reveal_text(prot.protected)
    assert "Jane Doe" in revealed and "jane@example.com" in revealed

    # A guessed token still isn't in the persisted allowlist -> unknown, no oracle across restarts.
    assert "[unknown token]" in p2.reveal_text("[SSN]999-99-9999[/SSN]")


def test_support_reveal_endpoint_auth(monkeypatch):
    # The reveal endpoint is DISABLED without a token (404), 401 on a bad token, and only reveals
    # with the correct bearer token.
    import importlib
    monkeypatch.setenv("SUPPORT_API_TOKEN", "secret-xyz")
    import app.api.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    class _P:
        def reveal_text(self, t):
            return t.replace("[NAME]TOK[/NAME]", "Jane")
    monkeypatch.setattr(main, "get_protector", lambda: _P(), raising=False)
    # patch the lazily-imported get_protector used inside the endpoint
    import app.protect.protector as P
    monkeypatch.setattr(P, "get_protector", lambda: _P())

    c = TestClient(main.app)
    body = {"protected_text": "hi [NAME]TOK[/NAME]"}
    assert c.post("/api/support/reveal", json=body).status_code == 401  # no header
    assert c.post("/api/support/reveal", json=body,
                  headers={"Authorization": "Bearer nope"}).status_code == 401
    r = c.post("/api/support/reveal", json=body, headers={"Authorization": "Bearer secret-xyz"})
    assert r.status_code == 200 and r.json()["revealed"] == "hi Jane"

    # Disabled when no token is configured.
    monkeypatch.delenv("SUPPORT_API_TOKEN", raising=False)
    importlib.reload(main)
    c2 = TestClient(main.app)
    assert c2.post("/api/support/reveal", json=body,
                   headers={"Authorization": "Bearer secret-xyz"}).status_code == 404


# ── Dynamic discovery threshold (multi-PII name leak) ─────────────────────────────────────────

def test_dynamic_threshold_keeps_a_low_scoring_name_in_a_multi_pii_message(monkeypatch):
    """MEASURED LEAK the fix closes: in a name+email+phone sentence Discovery scores the PERSON
    span BELOW the flat 0.6 (~0.57), so a fixed threshold DROPS it and the name flows to the model +
    trace in the CLEAR. The count-aware acceptance bar eases with the number of distinct candidates,
    so a 3-candidate message accepts the 0.57 name — while the SAME name alone (1 candidate) at 0.57
    is still rejected as too weak (a lone word must be clearly PII)."""
    import json as _json
    from app.protect import discovery as D

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def __init__(self, payload): self._p = payload
        def json(self): return self._p

    def _make_client(payload):
        c = D.DiscoveryClient()
        captured = {}
        class _Sess:
            def post(self, url, params=None, headers=None, data=None, timeout=None):
                captured["threshold"] = params.get("score_threshold")
                return _Resp(payload)
        c._session = _Sess()
        return c, captured

    # Multi-PII: PERSON @0.57 (under 0.6), EMAIL/PHONE @0.99. 3 distinct candidates -> bar eased.
    text = "I'm Jane Smith, jane@x.com, 312-555-7023"
    def span(v, t, s):
        i = text.index(v); return {"location": {"start_index": i, "end_index": i + len(v)}, "score": s, "_t": t}
    multi = {"classifications": {
        "PERSON": [span("Jane Smith", "PERSON", 0.57)],
        "EMAIL_ADDRESS": [span("jane@x.com", "EMAIL_ADDRESS", 0.99)],
        "PHONE_NUMBER": [span("312-555-7023", "PHONE_NUMBER", 0.99)],
    }}
    c, cap = _make_client(multi)
    out = c.discover(text)
    vals = {s["value"] for s in out}
    assert "Jane Smith" in vals, f"name dropped despite 3 candidates: {vals}"
    assert cap["threshold"] <= 0.4  # queried at the low floor, not 0.6

    # The SAME name ALONE at 0.57 (1 candidate) must NOT be accepted (bar stays at the flat 0.6).
    solo_text = "Jane Smith"
    solo = {"classifications": {"PERSON": [
        {"location": {"start_index": 0, "end_index": len(solo_text)}, "score": 0.57}]}}
    c2, _ = _make_client(solo)
    out2 = c2.discover(solo_text)
    assert not out2, f"a lone weak span should be rejected, got {out2}"


def test_dynamic_threshold_off_restores_flat_behaviour(monkeypatch):
    """PROTEGRITY_DISCOVERY_DYNAMIC=off pins the flat threshold and queries at it directly."""
    import importlib
    monkeypatch.setenv("PROTEGRITY_DISCOVERY_DYNAMIC", "off")
    from app.protect import discovery as D
    importlib.reload(D)
    try:
        assert D._acceptance_threshold(5) == D.DISCOVERY_THRESHOLD  # no easing when off
    finally:
        monkeypatch.delenv("PROTEGRITY_DISCOVERY_DYNAMIC", raising=False)
        importlib.reload(D)
