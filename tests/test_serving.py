"""The turn-based serving seam: one live turn keeps the batch pipeline's guarantees.

Drives ConversationSession with fakes for discovery, the protector, and the LLM (no live
services), and asserts the protection boundary holds on a single turn: inbound PII is
tokenized before it reaches the model, the model sees tokens not plaintext, the egress guard
can withhold a reply, and re-identification is role-gated on the way out.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class FakeProtector:
    """Deterministic, reversible fake: TOK<n> per distinct value, remembers the mapping."""

    def __init__(self):
        self._fwd, self._rev, self._n = {}, {}, 0

    def protect_values(self, values, element):
        out = []
        for v in values:
            if v not in self._fwd:
                self._n += 1
                tok = f"TOK{self._n}"
                self._fwd[v], self._rev[(element, tok)] = tok, v
            out.append(self._fwd[v])
        return out

    def protect_value(self, value, element):
        return self.protect_values([value], element)[0]

    def unprotect_values(self, tokens, element):
        return [self._rev.get((element, t), t) for t in tokens]


def _fake_discovery(monkeypatch, entities):
    import protometer.ingest as ingest
    monkeypatch.setattr(ingest, "discover_entities", lambda text, **kw: entities)


def test_protect_text_tokenizes_inbound_pii(monkeypatch):
    from protometer.serving import protect_text
    text = "Contact John Smith about the case."
    start = text.index("John Smith")
    _fake_discovery(monkeypatch, [
        {"entity_type": "PERSON", "start": start, "end": start + len("John Smith")}
    ])
    protected, n = protect_text(text, FakeProtector())
    assert n == 1
    assert "John Smith" not in protected
    assert "[PERSON]" in protected and "[/PERSON]" in protected


def test_protect_text_redacts_unmapped_pii_type_never_leaks_it(monkeypatch):
    """A discovered PII type with no tokenization element must be REDACTED, not left clear.

    Leaving it clear (the old `continue`) was a serving-path leak: discovery flagged it as PII,
    so it must never reach the model verbatim. Fail closed."""
    from protometer.ingest import ENTITY_TO_ELEMENT
    from protometer.serving import protect_text
    assert ENTITY_TO_ELEMENT.get("MADE_UP_TYPE") is None  # precondition: truly unmapped
    text = "The secret code is BANANA-42 today."
    start = text.index("BANANA-42")
    _fake_discovery(monkeypatch, [
        {"entity_type": "MADE_UP_TYPE", "start": start, "end": start + len("BANANA-42")}
    ])
    protected, n = protect_text(text, FakeProtector())
    assert n == 1
    assert "BANANA-42" not in protected          # the clear PII is gone
    assert "[REDACTED]" in protected             # replaced by an explicit redaction marker


def test_protect_text_uses_roster_for_organization_discovery_misses(monkeypatch):
    """The serving path must tokenize an ORGANIZATION discovery misses, via the hybrid roster.

    Discovery's ORGANIZATION recall is ~0. Without the roster the org name is left in
    the clear and reaches the model / Langfuse — the exact serving-path leak this fix closes."""
    from protometer.roster import roster_from_parties
    from protometer.serving import protect_text

    text = "Look into Sablefield Advisory Services about the wire."
    # Discovery finds NOTHING (its real behaviour for organizations).
    _fake_discovery(monkeypatch, [])
    roster = roster_from_parties([
        {"full_name": "Sablefield Advisory Services", "party_type": "organization"},
    ])

    # Without the roster: the org leaks through un-tokenized (the bug).
    leaky, n0 = protect_text(text, FakeProtector())
    assert n0 == 0 and "Sablefield Advisory Services" in leaky

    # With the roster: the org is tokenized, never reaches the model in the clear.
    protected, n1 = protect_text(text, FakeProtector(), roster=roster)
    assert n1 == 1
    assert "Sablefield Advisory Services" not in protected
    assert "[ORGANIZATION]" in protected and "[/ORGANIZATION]" in protected


def test_turn_protects_inbound_model_never_sees_plaintext(monkeypatch):
    from protometer.reidentify import INVESTIGATOR
    from protometer.serving import ConversationSession

    text = "Look into John Smith please."
    start = text.index("John Smith")
    _fake_discovery(monkeypatch, [
        {"entity_type": "PERSON", "start": start, "end": start + len("John Smith")}
    ])

    seen = {}

    class FakeLLM:
        def complete(self, system, prompt, max_tokens=None):
            seen["prompt"] = prompt
            # echo the token back tagged, as a well-behaved model would
            import re
            tok = re.search(r"\[PERSON\](\w+)\[/PERSON\]", prompt).group(1)
            return f"Reviewing [PERSON]{tok}[/PERSON] now."

    sess = ConversationSession(
        protector=FakeProtector(), llm=FakeLLM(), conversation_id="c1",
        role=INVESTIGATOR, system_prompt="sys", require_guardrail=False,
    )
    result = sess.turn(text)
    assert "John Smith" not in seen["prompt"], "model saw plaintext PII"
    assert result.entities_protected == 1
    assert result.ok
    # investigator role re-identifies on the way out
    assert "John Smith" in result.reply
    assert result.revealed == 1


def test_turn_fails_closed_on_egress_block(monkeypatch):
    from protometer.reidentify import INVESTIGATOR
    from protometer.serving import ConversationSession

    _fake_discovery(monkeypatch, [])

    class FakeLLM:
        def complete(self, system, prompt, max_tokens=None):
            return "here is a leaked value"

    class BlockingGuard:
        def scan_response(self, content, extra_tokens=None):
            class V:
                blocked = True
                leaked_values = ("secret",)
                outcome = "rejected"
            return V()

    sess = ConversationSession(
        protector=FakeProtector(), llm=FakeLLM(), conversation_id="c2",
        role=INVESTIGATOR, guardrail=BlockingGuard(), system_prompt="sys",
    )
    result = sess.turn("hello")
    assert result.egress_blocked
    assert not result.ok
    assert "leaked value" not in result.reply


def test_auditor_role_reveals_nothing(monkeypatch):
    from protometer.serving import ConversationSession

    _fake_discovery(monkeypatch, [])

    class FakeLLM:
        def complete(self, system, prompt, max_tokens=None):
            return "Reviewing [PERSON]TOK1[/PERSON]."

    # default role is AUDITOR (may_unprotect empty)
    sess = ConversationSession(
        protector=FakeProtector(), llm=FakeLLM(), conversation_id="c3", system_prompt="sys",
        require_guardrail=False,
    )
    result = sess.turn("hi")
    assert result.revealed == 0
    assert "TOK1" in result.reply  # stays tokenized for an auditor


def test_turn_fails_closed_when_egress_scan_raises(monkeypatch):
    """An unreachable guardrail must withhold the reply, not let the exception escape turn()
    onto an unguarded path."""
    from protometer.reidentify import INVESTIGATOR
    from protometer.serving import ConversationSession

    _fake_discovery(monkeypatch, [])

    class FakeLLM:
        enable_cache = True
        trace_session = ""
        trace_component = ""

        def complete(self, system, prompt, max_tokens=None):
            return "some reply"

    class ExplodingGuard:
        def scan_response(self, content, extra_tokens=None):
            raise RuntimeError("guardrail unreachable")

    sess = ConversationSession(
        protector=FakeProtector(), llm=FakeLLM(), conversation_id="c-egress-fail",
        role=INVESTIGATOR, guardrail=ExplodingGuard(), system_prompt="sys",
    )
    result = sess.turn("hi")
    assert result.egress_blocked and not result.ok
    assert "some reply" not in result.reply
    assert "egress-unavailable" in result.error


def test_turn_fails_closed_when_guardrail_absent_on_analyst_path(monkeypatch):
    """A MISSING guardrail (not just a raising one) must withhold the reply when required.

    The UI builds the guardrail per domain; when the sidecar 500s under memory pressure the build
    fails and the session gets guardrail=None. On the analyst path that must fail CLOSED — the
    old code returned the reply unscanned (fail open), letting a model-emitted clear identifier
    reach the analyst."""
    from protometer.reidentify import INVESTIGATOR
    from protometer.serving import ConversationSession

    _fake_discovery(monkeypatch, [])

    class FakeLLM:
        def complete(self, system, prompt, max_tokens=None):
            return "here is 999-00-1234 in the clear"

    # guardrail=None, require_guardrail defaults True -> must withhold.
    sess = ConversationSession(
        protector=FakeProtector(), llm=FakeLLM(), conversation_id="c-noguard",
        role=INVESTIGATOR, system_prompt="sys",
    )
    result = sess.turn("look into a case")
    assert result.egress_blocked and not result.ok
    assert "999-00-1234" not in result.reply
    assert result.error == "egress-unavailable: no guardrail"


def test_session_disables_response_cache_on_its_client(monkeypatch):
    """A serving session must turn off the batch client's response cache, or one user's reply
    is served to another user asking the same thing."""
    from protometer.serving import ConversationSession

    _fake_discovery(monkeypatch, [])

    class FakeLLM:
        enable_cache = True   # batch default
        trace_session = ""
        trace_component = ""

        def complete(self, system, prompt, max_tokens=None):
            return "reply"

    llm = FakeLLM()
    sess = ConversationSession(protector=FakeProtector(), llm=llm,
                               conversation_id="c-cache", system_prompt="sys",
                               require_guardrail=False)
    sess.turn("hi")
    assert llm.enable_cache is False
    assert llm.trace_session == "c-cache"


def test_audit_sink_receives_every_turn_including_failures(monkeypatch):
    """A live system needs a per-turn audit record for EVERY outcome. The sink must fire on a
    successful turn and on a failed one (here: generation failure), carrying metadata only."""
    from protometer.serving import ConversationSession

    _fake_discovery(monkeypatch, [])
    records = []

    class OkLLM:
        enable_cache = True
        trace_session = ""
        trace_component = ""

        def complete(self, system, prompt, max_tokens=None):
            return "fine"

    class FailLLM(OkLLM):
        def complete(self, system, prompt, max_tokens=None):
            raise RuntimeError("model down")

    sess = ConversationSession(protector=FakeProtector(), llm=OkLLM(),
                               conversation_id="c-audit", system_prompt="sys",
                               audit_sink=records.append, require_guardrail=False)
    sess.turn("hello")
    assert len(records) == 1 and records[0]["ok"] and records[0]["turn"] == 0

    sess.llm = FailLLM()
    sess.turn("again")
    assert len(records) == 2 and not records[1]["ok"]
    assert "generation-failed" in records[1]["error"]
    # audit records carry metadata only, never the reply text
    assert "reply" not in records[0]


def test_turn_captures_egress_detail(monkeypatch):
    """The turn must surface the guardrail's per-processor verdict + batch score, not just a
    boolean. Drives a fake guardrail returning a rich verdict."""
    from protometer.serving import ConversationSession

    _fake_discovery(monkeypatch, [])

    class FakeLLM:
        enable_cache = True; trace_session = ""; trace_component = ""
        def complete(self, system, prompt, max_tokens=None): return "a clean reply"

    class Finding:
        processor = "pii"; score = 0.1; explanation = "clean"

    class Verdict:
        blocked = False; outcome = "approved"; score = 0.1
        batch_outcome = "approved"; batch_score = 0.15
        leaked_values = (); findings = [Finding()]

    class FakeGuard:
        def scan_response(self, content, extra_tokens=None): return Verdict()

    sess = ConversationSession(protector=FakeProtector(), llm=FakeLLM(),
                               conversation_id="c-egress-detail", system_prompt="sys",
                               guardrail=FakeGuard())
    r = sess.turn("hi")
    assert r.ok
    assert r.egress_detail["batch_score"] == 0.15
    assert r.egress_detail["processors"][0]["explanation"] == "clean"
