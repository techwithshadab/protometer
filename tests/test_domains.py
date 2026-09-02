"""The domain seam: one pipeline, many use cases, with the AML default unchanged.

Pins the invariants that make multi-domain safe: the AML domain reproduces ingest's field maps
exactly (so the default path is untouched), every registered domain resolves to real prompt
files, and every domain's guardrail model is one the vendor actually exposes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard.domains import DEFAULT_DOMAIN, domain_names, get_domain


def test_default_domain_is_aml():
    assert DEFAULT_DOMAIN == "aml"
    assert get_domain().name == "aml"
    assert get_domain(None).name == "aml"


def test_aml_domain_matches_ingest_field_maps_exactly():
    """The AML domain's record_fields must equal PARTY_FIELDS + TRANSACTION_FIELDS, or the
    default path would tokenize a different field set than ingest does."""
    from amlguard.ingest import PARTY_FIELDS, TRANSACTION_FIELDS

    expected = {**PARTY_FIELDS, **TRANSACTION_FIELDS}
    assert get_domain("aml").record_fields == expected


def test_unknown_domain_is_a_loud_error():
    with pytest.raises(KeyError):
        get_domain("nope")


def test_every_domain_resolves_to_real_prompt_files():
    from amlguard.prompts import load_prompt

    for name in domain_names():
        d = get_domain(name)
        for role in ("investigation_prompt", "rationale_prompt", "judge_prompt"):
            text = load_prompt(getattr(d, role))
            assert text, f"{name}.{role} is empty"


def test_every_domain_uses_a_vendor_guardrail_model():
    # The Semantic-Guardrail exposes exactly these prompt-scoring domain models.
    vendor_models = {"customer-support", "financial", "healthcare"}
    for name in domain_names():
        assert get_domain(name).injection_processor in vendor_models
        assert get_domain(name).pii_processor == "pii"


def test_high_sensitivity_values_are_domain_specific():
    hc = get_domain("healthcare")
    records = [{"mrn": "MRN0012345", "ssn": "123456789", "patient_name": "Jane Roe"}]
    hs = hc.high_sensitivity_values(records)
    assert "MRN0012345" in hs and "123456789" in hs
    assert "Jane Roe" not in hs  # a name is not a digit-only high-sensitivity id


def test_every_domain_entity_type_is_tokenizable():
    """Every entity type a domain declares MUST be in ingest.ENTITY_TO_ELEMENT, or that field
    is discovered but silently NOT tokenized, a plaintext leak. This caught the healthcare
    domain mapping `mrn` to a non-existent MEDICAL_RECORD_NUMBER type."""
    from amlguard.ingest import ENTITY_TO_ELEMENT

    for name in domain_names():
        d = get_domain(name)
        for field_name, entity_type in d.record_fields.items():
            assert entity_type in ENTITY_TO_ELEMENT, (
                f"domain {name!r} field {field_name!r} -> {entity_type!r} is not in "
                f"ENTITY_TO_ELEMENT; it would be discovered but not tokenized (leak)"
            )


def test_billing_can_reveal_the_mrn_entity_type_it_bills_against():
    """The healthcare `billing` role's default story is 'sees name + MRN to bill'. The MRN
    tokenizes as HEALTH_CARE_ID (domains.py maps `mrn` -> HEALTH_CARE_ID), so billing MUST permit
    that exact entity type — otherwise it sees the name but never the MRN (fails closed, breaks the
    demo beat). SSN / insurance_id stay masked (distinct entity types), so this is minimum-necessary,
    not over-reveal."""
    from amlguard.domains import get_domain
    from amlguard.reidentify import ROLES

    billing = ROLES["billing"]
    hc = get_domain("healthcare")
    mrn_type = hc.record_fields["mrn"]
    assert billing.permits(mrn_type), (
        f"billing cannot reveal the MRN's actual entity type {mrn_type!r}")
    # minimum-necessary: SSN and insurance_id are NOT revealable by billing
    assert not billing.permits(hc.record_fields["ssn"])
    assert not billing.permits(hc.record_fields["insurance_id"])


def test_forbidden_values_use_domain_fields():
    """The egress backstop must forbid a NON-AML domain's identifiers, not only AML party
    fields, or a healthcare/support corpus's names/MRNs pass the check (cross-domain leak)."""
    from amlguard.guardrail import forbidden_values_from_parties

    hc = get_domain("healthcare")
    # Real corpus rows store the name under `full_name` (NOT patient_name); the domain's
    # record_fields must use that same key or the backstop misses the name (the cross-domain leak
    # this test guards). `mrn` is the healthcare-specific identifier.
    records = [{"full_name": "Jane Roe", "mrn": "MRN0012345", "amount": "unused"}]
    # default (AML fields) knows full_name but NOT the mrn -> misses the healthcare identifier
    aml_default = forbidden_values_from_parties(records)
    assert "MRN0012345" not in aml_default
    # domain fields DO forbid both the name and the domain-specific mrn
    with_domain = forbidden_values_from_parties(records, tuple(hc.record_fields))
    assert "Jane Roe" in with_domain and "MRN0012345" in with_domain


def test_scope_element_override_for_datetime_yearclear():
    """The quasi-yearclear scope must protect dates with datetime_yc (year in clear), while
    every other entity keeps the default element and the default curve is unchanged."""
    from amlguard.ingest import ENTITY_TO_ELEMENT
    from amlguard.scopes import CURVE_ORDER, get_scope

    yc = get_scope("quasi-yearclear")
    assert yc.element_for("DATETIME", ENTITY_TO_ELEMENT.get("DATETIME")) == "datetime_yc"
    assert yc.element_for("DATE_OF_BIRTH", ENTITY_TO_ELEMENT.get("DATE_OF_BIRTH")) == "datetime_yc"
    # non-date entities fall through to the global default
    assert yc.element_for("AMOUNT", ENTITY_TO_ELEMENT.get("AMOUNT")) == "number"
    assert yc.element_for("PERSON", ENTITY_TO_ELEMENT.get("PERSON")) == "string"
    # quasi (no override) still uses full datetime
    assert get_scope("quasi").element_for("DATETIME", "datetime") == "datetime"
    # opt-in scope: NOT in the committed curve, so it can't force a re-bill
    assert "quasi-yearclear" not in CURVE_ORDER
