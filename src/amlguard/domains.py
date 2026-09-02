"""Domain configuration: the seam that lets one protection pipeline serve many use cases.

The protect -> embed -> infer -> re-identify -> observe machinery is domain-agnostic; what is
domain-specific is a small, enumerable set of choices: which schema fields carry which entity
types, which Semantic-Guardrail domain model scores prompts, which prompt set the LLM uses, and
how a run is labelled in telemetry. This module makes those choices *data*, one `Domain` object,
selected by name, instead of constants scattered across `ingest`, `guardrail`, and `pipeline`.

The AML/financial domain is the default and reproduces the exact constants the pipeline shipped
with, so nothing changes for the existing measurement work. Healthcare and customer-support are
provided as first-class alternatives to demonstrate the seam is real, each a plug-in of the same
five choices, not a fork of the pipeline.

Deliberately *not* in scope here: the labelled training data (a corpus generator is its own
concern per domain) and the ML metric names. Those are domain content, not protection
configuration; a domain that needs a trained model brings its own corpus and labels. This module
covers what the *protection and serving* path needs to run correctly on a new domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Domain:
    """Everything the protection/serving path needs to know that varies by use case.

    A `Domain` is the whole domain-coupling surface in one object: change these fields and the
    same pipeline runs on a different use case, with no edits to `ingest`/`guardrail`/`pipeline`.
    """

    name: str
    label: str
    # Schema field -> discovery entity type. What `protect_structured` tokenizes, and what the
    # egress guard treats as forbidden. AML ships a party schema and a transaction schema; other
    # domains ship whatever their records look like (a patient record, a support ticket).
    record_fields: dict[str, str] = field(default_factory=dict)
    # The Semantic-Guardrail domain model used to score *prompts* for manipulation. The vendor
    # exposes customer-support / financial / healthcare; the response PII scan always uses `pii`.
    injection_processor: str = "customer-support"
    pii_processor: str = "pii"
    # Prompt registry names (config/prompts/<name>.txt, versioned in Langfuse). A domain points
    # at its own system/rationale/judge prompts, so re-framing the assistant needs no code change.
    investigation_prompt: str = "amlguard-investigation-system"
    rationale_prompt: str = "amlguard-rationale-system"
    judge_prompt: str = "amlguard-judge-system"
    # The field names whose values are high-sensitivity digit-only identifiers, matched on ANY
    # occurrence by the egress guard (SSN/account/card shapes). Domain-specific: a healthcare MRN
    # is high-sensitivity, a support-ticket order number may not be.
    high_sensitivity_fields: tuple[str, ...] = ()
    # Whether this domain supports a LIVE chatbot turn. Live chat protects each turn against the
    # domain's own party corpus (AML's full corpus; support/healthcare party masters built by
    # scripts/build_domain_corpus.py and loaded by the container entrypoint). All shipped domains
    # support it; a fork adding a domain without a corpus flips this off. Single source of truth:
    # the UI offers Live from it, and /chat/turn fails with a precise 503 when it is off.
    supports_live_chat: bool = False
    # Re-identification roles for THIS domain, most-restrictive first. The live chat and the
    # role-view cards offer only these, so a support turn can't be run as an AML `investigator`
    # (which would reveal the customer in full — the exact inversion of the "agent masked,
    # supervisor full" story). `default_live_role` is the low-privilege default a live turn uses
    # when the request names no domain-valid role, so the masked view is what you see first.
    live_roles: tuple[str, ...] = ()
    default_live_role: str = ""

    def role_for(self, requested: str | None) -> str:
        """The effective role name for a live turn: the requested role IF it is valid for this
        domain, else this domain's default. Prevents cross-domain role bleed (e.g. an AML
        `investigator` role leaking full PII on a customer-support turn)."""
        if requested and requested in self.live_roles:
            return requested
        return self.default_live_role or (self.live_roles[0] if self.live_roles else "auditor")

    def high_sensitivity_values(self, records: list[dict]) -> frozenset[str]:
        """Clear high-sensitivity identifier values in these records, for the egress guard."""
        return frozenset(
            value
            for record in records
            for name in self.high_sensitivity_fields
            if len(value := str(record.get(name) or "")) >= 5
        )


# --- The registry -----------------------------------------------------------------------------
#
# AML reproduces the shipped constants exactly. The party and transaction field maps below are
# the union of ingest.PARTY_FIELDS and ingest.TRANSACTION_FIELDS (kept here as the domain's view;
# ingest remains the source of truth for the AML path itself, and a test pins that they agree).

_AML = Domain(
    name="aml",
    label="Anti-money-laundering investigation",
    record_fields={
        "full_name": "PERSON",
        "account_number": "ACCOUNT_NUMBER",
        "bank_account": "BANK_ACCOUNT",
        "address": "ADDRESS",
        "city": "LOCATION",
        "email": "EMAIL_ADDRESS",
        "phone": "PHONE_NUMBER",
        "ssn": "SOCIAL_SECURITY_ID",
        "date_of_birth": "DATE_OF_BIRTH",
        "tax_id": "TAX_ID",
        "credit_card": "CREDIT_CARD",
        "amount": "AMOUNT",
        "value_date": "DATETIME",
    },
    injection_processor="customer-support",
    live_roles=("auditor", "analyst", "investigator"),
    default_live_role="analyst",   # partial view: orgs/locations revealed, identities masked
    investigation_prompt="amlguard-investigation-system",
    rationale_prompt="amlguard-rationale-system",
    judge_prompt="amlguard-judge-system",
    high_sensitivity_fields=("ssn", "account_number", "bank_account", "credit_card", "tax_id"),
    supports_live_chat=True,  # full party corpus under data/corpus/
)

_HEALTHCARE = Domain(
    name="healthcare",
    label="Clinical record assistant",
    record_fields={
        # Party rows store the name under `full_name` (like AML); a `patient_name` key here would be
        # inert — ingest-protection, the egress forbidden-value hard-block, and trace redaction all
        # read party rows BY this key, so a mismatch silently leaves patient names unprotected in
        # traces and unblocked at egress. Must match the corpus row key.
        "full_name": "PERSON",
        # HEALTH_CARE_ID is the entity type ingest actually maps (to the `number` element);
        # MEDICAL_RECORD_NUMBER is not in ENTITY_TO_ELEMENT and would silently pass unprotected.
        "mrn": "HEALTH_CARE_ID",
        "ssn": "SOCIAL_SECURITY_ID",
        "date_of_birth": "DATE_OF_BIRTH",
        "address": "ADDRESS",
        "phone": "PHONE_NUMBER",
        "email": "EMAIL_ADDRESS",
        "insurance_id": "NATIONAL_ID",
        "provider_name": "PERSON",
    },
    injection_processor="healthcare",
    live_roles=("researcher", "billing", "clinician"),
    default_live_role="billing",   # sees name + MRN to bill; the rest stays masked
    investigation_prompt="healthcare-investigation-system",
    rationale_prompt="healthcare-rationale-system",
    judge_prompt="healthcare-judge-system",
    high_sensitivity_fields=("mrn", "ssn", "insurance_id"),
    supports_live_chat=True,  # ships a patient master under data/corpus/healthcare/
)

_SUPPORT = Domain(
    name="customer-support",
    label="Customer-support assistant",
    record_fields={
        # Party rows store the name under `full_name` (see healthcare note above); `customer_name`
        # here would be inert and leave customer names unprotected in traces / unblocked at egress.
        "full_name": "PERSON",
        "email": "EMAIL_ADDRESS",
        "phone": "PHONE_NUMBER",
        "address": "ADDRESS",
        "account_number": "ACCOUNT_NUMBER",
        "credit_card": "CREDIT_CARD",
        "order_id": "ACCOUNT_NUMBER",
    },
    injection_processor="customer-support",
    live_roles=("support_agent", "supervisor"),
    default_live_role="support_agent",   # front-line agent: customer masked (the demo's default view)
    investigation_prompt="support-investigation-system",
    rationale_prompt="support-rationale-system",
    judge_prompt="support-judge-system",
    high_sensitivity_fields=("account_number", "credit_card"),
    supports_live_chat=True,  # ships a customer master under data/corpus/customer-support/
)

_REGISTRY: dict[str, Domain] = {d.name: d for d in (_AML, _HEALTHCARE, _SUPPORT)}

# The default domain. Selecting nothing gives the AML/financial behaviour the pipeline shipped
# with, so every existing call site and test is unchanged.
DEFAULT_DOMAIN = "aml"


def get_domain(name: str | None = None) -> Domain:
    """Resolve a domain by name; `None` (or unset) yields the default AML domain."""
    key = name or DEFAULT_DOMAIN
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown domain {key!r}. Known: {', '.join(sorted(_REGISTRY))}"
        ) from None


def domain_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
