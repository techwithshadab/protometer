"""Re-identification at the presentation boundary.

This is the only place in the system where tokens become plaintext again, and it runs
*after* the LLM has produced its answer. That ordering is the architecture's central claim:
the model reasons over tokens, and real identifiers appear only in the analyst's view.

Round-tripping relies on the wrapper tags ingestion emits, `[PERSON]token[/PERSON]`. The
tag carries the entity type, which determines the data element to unprotect with, so the
tags are not decoration but the state that makes reversal possible. Text stripped of its
tags cannot be re-identified through this path.

Two consequences worth stating plainly:

  * Re-identification is **best-effort on model output**. If the LLM paraphrases a token or
    drops its tags, that token cannot be reversed, a real limitation of tokenizing text a
    generative model then rewrites, and one this module reports rather than hides.
  * Unprotection is **application-gated, not policy-gated**. Developer Edition's
    `check_access()` is a stub returning `True` unconditionally and every sample user maps
    to `superuser`, so the role checks here are enforced by this application and must be
    labelled as such wherever they are shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from amlguard.ingest import ENTITY_TO_ELEMENT
from amlguard.protect import Protector, is_auth_limit

# Matches [ENTITY_TYPE]token[/ENTITY_TYPE], with an OPTIONAL `|element` in the open tag:
# [DATETIME|datetime_yc]tok[/DATETIME]. The element records how the token was PROTECTED so
# re-identification reverses it under the SAME element — a scope with element_overrides (e.g.
# quasi-yearclear -> datetime_yc) protects a date under an element the global ENTITY_TO_ELEMENT
# map does not name, and unprotecting under the wrong element does not round-trip. The element
# is optional so every existing untagged corpus still parses; the close tag carries only the
# type (a back-reference), so the open/close still agree on type.
TAG_PATTERN = re.compile(
    # Inner content = runs of non-'[' chars, plus any '[' that does NOT start a tag. Each
    # alternative is anchored and mutually exclusive, and the '[^\[]' run is possessive (*+), so
    # there is no ambiguous overlap to backtrack over: matching is linear in the input length
    # (avoids the polynomial ReDoS a tempered-dot `(?:(?!...).)*` invites, which matters because
    # reidentify runs over attacker-influenceable model output).
    r"\[([A-Z_]+)(?:\|([a-z0-9_]+))?\]((?:[^\[]*+|\[(?!/?[A-Z_]+\]))*)\[/\1\]",
    re.DOTALL,
)


@dataclass(frozen=True)
class Role:
    """An application-defined view over protected data.

    **Not** a Protegrity policy role. Developer Edition provides no local policy
    configuration and its access check is a stub, so these roles are enforced here, in
    application code. Every surface that displays role-differentiated output must say so.
    """

    name: str
    label: str
    # Entity types this role may see in the clear. Everything else stays tokenized.
    may_unprotect: frozenset[str]
    description: str

    def permits(self, entity_type: str) -> bool:
        return entity_type in self.may_unprotect


ANALYST = Role(
    name="analyst",
    label="Junior Analyst",
    may_unprotect=frozenset({"ORGANIZATION", "LOCATION", "ADDRESS"}),
    description="Reviews behaviour and counterparty structure; individuals stay tokenized.",
)

INVESTIGATOR = Role(
    name="investigator",
    label="Senior Investigator",
    may_unprotect=frozenset(ENTITY_TO_ELEMENT),
    description="Full re-identification for SAR preparation and escalation.",
)

AUDITOR = Role(
    name="auditor",
    label="Auditor",
    may_unprotect=frozenset(),
    description="Verifies process without ever seeing identities; nothing is unprotected.",
)

# Customer-support roles. A front-line agent handles the case with
# identifiers masked; a supervisor may fully re-identify. Same application-enforced Gate-2
# mechanism as the AML roles, retargeted for the support "agent sees masked / supervisor sees
# all" pattern from the reference chatbot (vendor Orchestrators-BankingPortalChatbot).
SUPPORT_AGENT = Role(
    name="support_agent",
    label="Support Agent",
    # Non-identifying context only: the org and rough location help route/resolve a case,
    # while name, email, phone, card and account stay tokenized on the agent's screen.
    may_unprotect=frozenset({"ORGANIZATION", "LOCATION"}),
    description="Front-line agent; resolves cases with customer identifiers masked.",
)

SUPERVISOR = Role(
    name="supervisor",
    label="Support Supervisor",
    may_unprotect=frozenset(ENTITY_TO_ELEMENT),
    description="Escalation supervisor; full re-identification for identity verification.",
)

# Healthcare roles. A clinical access model, not the AML investigation roles:
# a treating clinician needs the whole record; a researcher works on a de-identified dataset; a
# billing clerk works to the "minimum necessary" — enough to bill the right patient/record, no more.
TREATING_CLINICIAN = Role(
    name="clinician",
    label="Treating Clinician",
    may_unprotect=frozenset(ENTITY_TO_ELEMENT),
    description="Direct care; sees the full record because treatment requires the patient's identity.",
)

RESEARCHER = Role(
    name="researcher",
    label="Researcher",
    # Nothing identifying: research runs on the de-identified dataset, so every identifier stays a token.
    may_unprotect=frozenset(),
    description="Works on the de-identified dataset; no identifier is ever re-identified.",
)

BILLING = Role(
    name="billing",
    label="Billing / Claims",
    # Minimum necessary: the patient name and medical-record/beneficiary number to attach a claim to
    # the right record — clinical detail and contact identifiers stay tokenized. HEALTH_CARE_ID is
    # the entity type the MRN actually tokenizes as (domains.py maps `mrn` -> HEALTH_CARE_ID); the
    # legacy MED_REC/MEDICAL_RECORD_NUMBER names never appear as tag types, so without HEALTH_CARE_ID
    # billing could see the name but not the MRN it exists to bill against (fails closed, no leak).
    may_unprotect=frozenset({"PERSON", "HEALTH_CARE_ID", "MED_REC", "MEDICAL_RECORD_NUMBER"}),
    description="Minimum-necessary access; sees who and which record to bill, not clinical detail.",
)

ROLES: dict[str, Role] = {
    r.name: r for r in (ANALYST, INVESTIGATOR, AUDITOR, SUPPORT_AGENT, SUPERVISOR,
                        TREATING_CLINICIAN, RESEARCHER, BILLING)
}


@dataclass
class ReidentificationResult:
    text: str
    revealed: int = 0
    withheld: int = 0          # role not permitted to see this entity type
    failed: int = 0
    out_of_scope: int = 0      # role permits the type, but the token is outside the reveal scope
    canary_hits: int = 0       # revealed values that matched a registered canary (tripwire)
    # Token -> plaintext, for building the side-by-side view without a second API round.
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        return (f"revealed={self.revealed} withheld={self.withheld} "
                f"out_of_scope={self.out_of_scope} failed={self.failed}")


def find_tokens(text: str) -> list[tuple[str, str, str]]:
    """Return (entity_type, element, token) for every wrapper tag present.

    `element` is the tag's explicit protection element, or "" when the tag omits it (a legacy
    corpus, or a scope with no element override) — callers fall back to ENTITY_TO_ELEMENT then.
    """
    return [(m.group(1), m.group(2) or "", m.group(3)) for m in TAG_PATTERN.finditer(text)]


def reidentify(
    text: str,
    protector: Protector,
    role: Role = AUDITOR,
    strip_tags: bool = True,
    *,
    scope_tokens: "frozenset[str] | None" = None,
    ledger: "object | None" = None,
    tripwire: "object | None" = None,
    actor: str = "app",
    purpose: str = "presentation",
) -> ReidentificationResult:
    """Replace tokens with plaintext for entity types this role may see.

    Tokens the role may not see are left protected. Unprotect calls are batched per data
    element, an analyst view over a long document would otherwise cost one round-trip per
    token, against an API with a measured burst limit.

    Three optional defenses layer on the same boundary, all no-ops when unset:

    * ``scope_tokens`` — SCOPE-bound reveal. When given, only these token strings may be
      detokenized even if the role permits the entity type; every other token stays protected
      and is counted as ``out_of_scope``. This shrinks an authorized-but-injected session's blast
      radius from "every subject in the reply" to "the subject this turn is actually about".
    * ``ledger`` — a RevealLedger; one hash-chained record is appended per reveal (metadata only).
    * ``tripwire`` — a CanaryTripwire; revealed values are scanned for canaries and the count is
      recorded on the result and the ledger.
    """
    result = ReidentificationResult(text=text)
    tagged = find_tokens(text)
    if not tagged:
        return result

    # Group the tokens this role is permitted to see, by the element that reverses them. The
    # element comes from the TAG when present (so an override element like datetime_yc reverses
    # correctly), falling back to the global map for legacy/un-annotated tags.
    by_element: dict[str, list[str]] = {}
    for entity_type, tag_element, token in tagged:
        if not role.permits(entity_type):
            result.withheld += 1
            continue
        if scope_tokens is not None and token not in scope_tokens:
            # Scope-bound reveal: the role could see this entity type, but this specific token is
            # outside the caller's authorized scope (a different subject in the retrieved context).
            result.out_of_scope += 1
            continue
        element = tag_element or ENTITY_TO_ELEMENT.get(entity_type)
        if element is None:
            result.failed += 1
            continue
        by_element.setdefault(element, []).append(token)

    recovered: dict[tuple[str, str], str] = {}
    for element, tokens in by_element.items():
        unique = list(dict.fromkeys(tokens))
        try:
            values = protector.unprotect_values(unique, element)
        except Exception as exc:  # noqa: BLE001, a failed element must not lose the document
            # ...except an auth-throttle, which is an infrastructure fault, not a per-document
            # reveal failure: folding it into the silent `failed` counter hides the one
            # condition (the login 429 the SDK misreports as bad credentials) an operator most
            # needs to see, and every subsequent document would fail identically. Fail loud.
            if is_auth_limit(exc):
                raise
            # Any other element-level failure degrades: count it at OCCURRENCE grain (not
            # unique-token grain) so `failed` is on the same scale as `revealed`/withheld,
            # which are counted per occurrence in `replace`.
            result.failed += len(tokens)
            continue
        for token, value in zip(unique, values):
            recovered[(element, token)] = value

    def replace(match: re.Match[str]) -> str:
        entity_type, tag_element, token = match.group(1), match.group(2), match.group(3)
        if not role.permits(entity_type):
            # Keep it tokenized, and keep the tag so the withholding stays visible.
            return match.group(0)
        if scope_tokens is not None and token not in scope_tokens:
            return match.group(0)  # out of scope: stays protected, already counted above
        element = tag_element or ENTITY_TO_ELEMENT.get(entity_type)
        value = recovered.get((element, token)) if element else None
        if value is None:
            return match.group(0)
        result.revealed += 1
        result.mapping[token] = value
        return value if strip_tags else f"[{entity_type}]{value}[/{entity_type}]"

    result.text = TAG_PATTERN.sub(replace, text)

    # Tripwire + ledger on the values actually recovered this call. Both best-effort: a telemetry
    # fault must never lose a legitimate reveal, but a tripped canary is surfaced on the result.
    if tripwire is not None and result.mapping:
        try:
            result.canary_hits = len(tripwire.scan(result.mapping.values()))
        except Exception:  # noqa: BLE001
            pass
    if ledger is not None and (result.revealed or result.canary_hits):
        try:
            counts: dict[str, int] = {}
            for etype, _te, tok in tagged:
                if tok in result.mapping:
                    counts[etype] = counts.get(etype, 0) + 1
            ledger.append(
                actor=actor, role=role.name, purpose=purpose,
                entity_counts=counts,
                scope=("bound" if scope_tokens is not None else "role"),
                canary_hits=result.canary_hits,
            )
        except Exception:  # noqa: BLE001, ledger write must not break a reveal
            pass
    return result


def strip_tags(text: str) -> str:
    """Remove wrapper tags, leaving bare tokens.

    This is what the LLM should receive: the tags are internal plumbing, and leaving them in
    the prompt both wastes context and hints at structure the model does not need. Stripping
    them is irreversible for re-identification, so it is applied to the prompt copy only -
    never to the stored corpus.
    """
    return TAG_PATTERN.sub(lambda m: m.group(3), text)
