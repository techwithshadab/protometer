"""Protection scope definitions, the independent variable of the whole study.

Each scope is a subset of the entity types Protegrity's discovery service can detect,
mapped to the data elements it protects them with. By design, a curve point must be a
*configuration* of one mechanism, never a separate implementation: if adding a point ever
requires pipeline code, the policy layer is too weak and that is the bug to fix.

The four scopes deliberately widen along the axis a real institution would widen along -
direct identifiers first (uncontroversial), then quasi-identifiers (where re-identification
risk actually lives), then everything detected (maximally safe, probably unusable).

The knee is expected between DIRECT and QUASI: aggregation checkpoints depend on AMOUNT
surviving, and QUASI is where AMOUNT and DATETIME start being tokenized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Entities naming a party outright. Protecting these is the minimum any bank would deploy,
# and the maximum most would consider uncontroversial.
DIRECT_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "PERSON",
        "SOCIAL_SECURITY_ID",
        "NATIONAL_ID",
        "TAX_ID",
        "PASSPORT",
        "DRIVER_LICENSE",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "BANK_ACCOUNT",
        "ACCOUNT_NUMBER",
        "ACCOUNT_NAME",
        "CRYPTO_ADDRESS",
        "USERNAME",
        "USER_NAME",
        # An organization's name identifies the party outright, it is a direct identifier
        # by definition, not a quasi-identifier. It sits here rather than in QUASI because
        # counterparty organizations are the parties under investigation in AML, and leaving
        # them in the clear would defeat the architecture's central claim.
        "ORGANIZATION",
        # A street address is a direct identifier, by exactly the argument above. Classifying
        # it as quasi meant that at `direct`, the scope this project calls the realistic
        # minimum a bank would deploy, and the scope every adversarial attack is scored
        # against, all 600 party addresses sat in the clear beside their tokenized names,
        # every one unique:
        #
        #   "full_name": "UvBUFEcNme G6J66CDwBa A5XA"
        #   "address":   "5477 Sycamore Row, Tampa, FL 33602"
        #
        # Re-identification was therefore 100% at `direct` by reading the adjacent column,
        # with no attack required. A protected narrative leaked it too: the name was
        # tokenized and the address that identifies the person sat four words later.
        "ADDRESS",
        "LOCATION",
    }
)

# Entities that do not name anyone alone but re-identify in combination, the classic
# quasi-identifier problem. This is where protection starts costing analytical utility:
# AMOUNT and DATETIME are what aggregation and typology detection actually run on.
QUASI_IDENTIFIERS: frozenset[str] = frozenset(
    {
        # ADDRESS and LOCATION were here and are now DIRECT, see the note above.
        "DATETIME",
        "DOB",
        "DATE_OF_BIRTH",
        "AMOUNT",
        "AGE",
        "GENDER",
        "TITLE",
        "IP_ADDRESS",
        "MAC_ADDRESS",
        "URL",
        "HEALTH_CARE_ID",
    }
)

# Remaining detectable types, swept in only at maximum scope.
RESIDUAL: frozenset[str] = frozenset(
    {
        "CURRENCY",
        "CURRENCY_CODE",
        "CURRENCY_NAME",
        "CURRENCY_SYMBOL",
        "PASSWORD",
        "NRP",
        "IN_VEHICLE_REGISTRATION",
        "IN_VOTER",
        "IN_GSTIN",
        "KR_RRN",
        "TH_TNIN",
    }
)


@dataclass(frozen=True)
class ProtectionScope:
    """One point on the utility-vs-scope curve."""

    name: str
    entities: frozenset[str]
    description: str
    # Measurement confirmed external_iv breaks determinism. Setting this true is the ablation:
    # same scope, no stable tokens, so cross-document entity resolution should collapse.
    break_determinism: bool = False
    # Per-scope data-element overrides: entity type -> element, taking precedence over the
    # global ENTITY_TO_ELEMENT. Lets a scope protect the same entity with a *different* element,
    # e.g. dates with `datetime_yc` (year-in-clear) instead of `datetime` (year scrambled), so
    # the utility of a partial-protection choice becomes its own measured curve point.
    element_overrides: "dict[str, str]" = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "-")

    def protects(self, entity_type: str) -> bool:
        return entity_type in self.entities

    def element_for(self, entity_type: str, default: str | None) -> str | None:
        """The data element to protect `entity_type` with under this scope, override first."""
        return self.element_overrides.get(entity_type, default)


# Quasi-identifiers split by what they cost analytically, so the cliff between `direct` and
# `quasi` can be attributed rather than merely observed.
#
# Measured: `direct` retains 100% of baseline utility while `quasi` retains 36%. Everything
# added between them is in these two sets, but the aggregate cannot say which is responsible -
# and the answer matters operationally, since an institution can choose to protect one and not
# the other.
TEMPORAL_QUASI: frozenset[str] = frozenset({"DATETIME", "DOB", "DATE_OF_BIRTH", "AGE"})

MONETARY_QUASI: frozenset[str] = frozenset({"AMOUNT"})

# The remainder: locational, demographic and network quasi-identifiers, which carry
# re-identification risk without participating in the arithmetic.
CONTEXTUAL_QUASI: frozenset[str] = QUASI_IDENTIFIERS - TEMPORAL_QUASI - MONETARY_QUASI


SCOPES: dict[str, ProtectionScope] = {
    "none": ProtectionScope(
        name="none",
        entities=frozenset(),
        description="Baseline. Clear text throughout. The reference every delta is measured against.",
    ),
    "direct": ProtectionScope(
        name="direct",
        entities=DIRECT_IDENTIFIERS,
        description="Direct identifiers only, the realistic minimum a bank would deploy.",
    ),
    # -- cliff resolution ---------------------------------------------------------------
    # Three points between `direct` (100% utility) and `quasi` (36%), each adding one class
    # of quasi-identifier, so the collapse can be attributed to a specific data class.
    "direct-plus-context": ProtectionScope(
        name="direct-plus-context",
        entities=DIRECT_IDENTIFIERS | CONTEXTUAL_QUASI,
        description="Direct plus locational and demographic quasi-identifiers. Amounts and dates stay clear.",
    ),
    "direct-plus-temporal": ProtectionScope(
        name="direct-plus-temporal",
        entities=DIRECT_IDENTIFIERS | CONTEXTUAL_QUASI | TEMPORAL_QUASI,
        description="Adds dates. Isolates whether temporal reasoning drives the utility cliff.",
    ),
    "direct-plus-monetary": ProtectionScope(
        name="direct-plus-monetary",
        entities=DIRECT_IDENTIFIERS | CONTEXTUAL_QUASI | MONETARY_QUASI,
        description="Adds amounts but not dates. Isolates whether arithmetic drives the cliff.",
    ),
    "quasi": ProtectionScope(
        name="quasi",
        entities=DIRECT_IDENTIFIERS | QUASI_IDENTIFIERS,
        description="Direct plus quasi-identifiers. Measured location of the utility knee.",
    ),
    # A partial-protection variant of `quasi`: same entities, but dates are protected with
    # `datetime_yc` (year in the clear) instead of `datetime` (year scrambled). This measures
    # the utility an institution recovers by choosing year-granularity date protection, the
    # temporal signal the classifier's day-index/temporal features and year-level aggregation
    # need, while still hiding the exact date. Verified live: 2020-08-06 -> 2020-04-04 keeps the
    # year, vs `datetime` -> 2278-02-13. Not in CURVE_ORDER: an opt-in comparison scope, so
    # adding it does not force a re-run of the committed eight-scope curve.
    "quasi-yearclear": ProtectionScope(
        name="quasi-yearclear",
        entities=DIRECT_IDENTIFIERS | QUASI_IDENTIFIERS,
        description="Like quasi, but dates keep their year (datetime_yc). Measures the utility "
                    "of year-granularity date protection.",
        element_overrides={"DATETIME": "datetime_yc", "DOB": "datetime_yc",
                           "DATE_OF_BIRTH": "datetime_yc"},
    ),
    "all": ProtectionScope(
        name="all",
        entities=DIRECT_IDENTIFIERS | QUASI_IDENTIFIERS | RESIDUAL,
        description="Everything the discovery service detects. Maximum protection.",
    ),
    # The determinism ablation run. Holds scope fixed at `direct` and varies only determinism, so
    # any delta against `direct` is attributable to token stability alone.
    "direct-nondeterministic": ProtectionScope(
        name="direct-nondeterministic",
        entities=DIRECT_IDENTIFIERS,
        description="Ablation: direct scope with external_iv, breaking cross-document token stability.",
        break_determinism=True,
    ),
}

# Order matters for reporting: monotonically widening, ablation last.
CURVE_ORDER: tuple[str, ...] = (
    "none",
    "direct",
    "direct-plus-context",
    "direct-plus-temporal",
    "direct-plus-monetary",
    "quasi",
    "all",
    "direct-nondeterministic",
)


def get_scope(name: str) -> ProtectionScope:
    try:
        return SCOPES[name]
    except KeyError:
        raise KeyError(f"Unknown scope {name!r}. Known: {', '.join(SCOPES)}") from None
