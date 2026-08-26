"""The egress leak-check's digit-matching: high-sensitivity ids caught even when embedded.

Pins the finding that an account number embedded in a longer digit run evaded the check (the
boundary rule that prevents amount/decimal false positives also let `978024684` hide
`78024684`). High-sensitivity digit identifiers (SSN/account/card/tax id) now match on any
occurrence; ordinary digit values keep the boundary rule so decimal collisions stay quiet.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard.guardrail import (
    Guardrail,
    high_sensitivity_values_from_parties,
)


def _guard(**kw):
    return Guardrail(enabled=False, **kw)


def test_embedded_high_sensitivity_id_is_caught():
    g = _guard(
        forbidden_values=frozenset({"78024684"}),
        high_sensitivity_values=frozenset({"78024684"}),
    )
    # embedded inside a longer digit run — the bypass the boundary rule left open
    assert g._leaked("ref 978024684 on file") == ("78024684",)
    # and on its own boundary
    assert g._leaked("account 78024684") == ("78024684",)


def test_ordinary_digit_value_keeps_boundary_rule():
    """A non-high-sensitivity digit value must NOT match inside an unrelated digit run,
    so decimal/SHAP-contribution collisions stay quiet (the false-positive class)."""
    g = _guard(forbidden_values=frozenset({"78024684"}))  # NOT high-sensitivity
    assert g._leaked("contribution 978024684 shown") == ()
    assert g._leaked("id 78024684 alone") == ("78024684",)


def test_high_sensitivity_set_covers_only_id_fields():
    parties = [{
        "ssn": "123456789", "account_number": "987654321", "tax_id": "555443322",
        "amount": "712000.00", "phone": "5551234567", "full_name": "Jane Roe",
    }]
    hs = high_sensitivity_values_from_parties(parties)
    assert "123456789" in hs and "987654321" in hs and "555443322" in hs
    # amounts/phones/names are NOT any-occurrence matched (they collide with unrelated digits)
    assert "712000.00" not in hs and "5551234567" not in hs and "Jane Roe" not in hs


def test_scan_result_captures_conversation_batch_verdict(monkeypatch):
    """The scan must capture the conversation-level batch score/outcome, not only per-message
    (we discarded this signal once before). Drives _scan with a fake service response."""
    import amlguard.guardrail as gm

    fake_body = {
        "messages": [{"outcome": "approved", "score": 0.12,
                      "processors": [{"name": "pii", "score": 0.12, "explanation": "clean"}]}],
        "batch": {"outcome": "approved", "score": 0.18},
    }

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return fake_body

    monkeypatch.setattr(gm._GUARDRAIL_SESSION, "post", lambda *a, **k: FakeResp())
    g = Guardrail(enabled=True)
    r = g._scan("some text", "pii", ("ai", "user"))
    assert r.batch_outcome == "approved"
    assert abs(r.batch_score - 0.18) < 1e-9
    assert r.findings and r.findings[0].explanation == "clean"


def test_protection_token_flagged_as_pii_is_discounted_not_blocked():
    """A Protegrity protection token the service mislabels (e.g. as PASSWORD) must be
    DISCOUNTED, not block a safe reply, while a REAL corpus value still hard-blocks. This was
    a live serving-path false-positive: tokens like '4oB93 T7MdI3' scored 0.95 and
    blocked otherwise-safe replies. Safety invariant: leaked_values (real values) still blocks.
    """
    from amlguard.guardrail import Guardrail

    g = Guardrail(enabled=False, forbidden_values=frozenset({"Leila Rahman"}))
    # a reply mentioning a protection-token-shaped value the service typed as PASSWORD
    content = "Reviewing token TOKENABC123 for the case."
    # inject the token into the protection-token set so the discount recognises it
    from amlguard.guardrail import _normalize_for_match
    object.__setattr__(g, "_token_cache", frozenset({_normalize_for_match("TOKENABC123")}))
    idx = content.index("TOKENABC123")
    expl = f"['PASSWORD : [{idx}, {idx+11}]']"
    assert g._is_surrogate_only(content, ("PASSWORD",), expl) is True

    # a REAL clear value in the same shape is NOT discounted (leaked_values path hard-blocks)
    content2 = "The subject is Leila Rahman."
    assert g._leaked(content2) == ("Leila Rahman",)  # real value -> hard block regardless


def test_leakable_typed_tokens_discount_and_per_word_real_names_do_not():
    """The discount is decided by the SPANS, not the entity TYPES. The service labels a
    protection token by the type of the value it replaced (a tokenized SSN is still typed
    SOCIAL_SECURITY_ID), and it splits a multi-word PERSON token into per-word spans. Both must
    still discount when every span is a verified token; a real cleartext name-part must not.

    Pins the fix for the serving-path false-positive where a chat reply written entirely over
    tokens was blocked because SOCIAL_SECURITY_ID/PERSON are in LEAKABLE_ENTITIES — even though
    every flagged span was a protection token."""
    from amlguard.guardrail import Guardrail, _normalize_for_match

    g = Guardrail(enabled=False, forbidden_values=frozenset({"Hassan Delacroix"}))
    # Token set holds a multi-word PERSON token, its words, and a tokenized SSN. The forbidden real
    # name and ITS words are subtracted (rail 2), so "Hassan"/"Delacroix" can never be a "token".
    object.__setattr__(g, "_token_cache", frozenset({
        _normalize_for_match("2S3y A47Vmilfi"), _normalize_for_match("2S3y"),
        _normalize_for_match("A47Vmilfi"), _normalize_for_match("997-02-4771"),
    }))

    # Reply over tokens: PERSON token split into two per-word spans + a SSN-typed token span.
    content = "Note on 2S3y A47Vmilfi, SSN 997-02-4771, flagged."
    p1 = content.index("2S3y"); p2 = content.index("A47Vmilfi"); p3 = content.index("997-02-4771")
    expl = (f"['PERSON : [{p1}, {p1+4}]', 'PASSWORD : [{p2}, {p2+9}]', "
            f"'SOCIAL_SECURITY_ID : [{p3}, {p3+11}]']")
    assert g._is_surrogate_only(content, ("PERSON", "PASSWORD", "SOCIAL_SECURITY_ID"), expl) is True

    # A real cleartext name-part flagged as PERSON must NOT be discounted (rail 2 on words).
    content_r = "The subject is Hassan, per the file."
    pr = content_r.index("Hassan")
    expl_r = f"['PERSON : [{pr}, {pr+6}]']"
    assert g._is_surrogate_only(content_r, ("PERSON",), expl_r) is False


def test_protection_tokens_never_contain_a_real_clear_value():
    """SECURITY REGRESSION GUARD (review r3, HIGH): the protection-token discount must never
    treat a real CLEAR corpus value as a token — that would let the service's mislabeling
    discount a genuine leak. Two rails: cleartext scopes (none/_anon-monetary/quasi-yearclear)
    are excluded, and forbidden_values are subtracted. Verified against the real corpus."""
    import json
    from pathlib import Path

    from amlguard.guardrail import (
        Guardrail,
        _normalize_for_match,
        forbidden_values_from_parties,
    )

    none_parties = Path("data/protected/none/parties.json")
    if not none_parties.exists():
        import pytest
        pytest.skip("requires an ingested corpus")

    parties = json.loads(none_parties.read_text())
    g = Guardrail(enabled=False, forbidden_values=forbidden_values_from_parties(parties))
    tokens = g._protection_tokens()

    # every forbidden (clear) value is provably absent from the token set
    for v in list(g.forbidden_values)[:200]:
        assert _normalize_for_match(v) not in tokens, (
            f"clear value {v!r} leaked into the protection-token set (discount could pass a leak)"
        )
    # the cleartext baseline's first party name is not a token
    assert _normalize_for_match(parties[0]["full_name"]) not in tokens


def test_aml_high_sensitivity_fields_single_source_of_truth():
    """The domain=None path (eval/hybrid guard) and the AML domain path (live UI guard) must
    seed the SAME high-sensitivity field set, or measurement and deployment disagree on what a
    hard leak is. They now both derive from the AML domain; this pins that they never drift."""
    from amlguard.domains import get_domain
    from amlguard.guardrail import _high_sensitivity_fields

    assert _high_sensitivity_fields() == get_domain("aml").high_sensitivity_fields

    # And the resulting value sets match on a sample corpus.
    parties = [{"ssn": "111223333", "account_number": "9900112233",
                "credit_card": "4111111111111111", "tax_id": "55667788", "bank_account": "12345678"}]
    via_default = high_sensitivity_values_from_parties(parties)
    via_domain = get_domain("aml").high_sensitivity_values(parties)
    assert via_default == via_domain
