"""Roster matching, deterministic detection for entities the model does not emit.

Protegrity's discovery service returns **zero** ORGANIZATION entities on this corpus at
every score threshold from 0.6 down to 0.0, despite `ORGANIZATION` appearing in the SDK's
own `DATA_ELEMENT_MAPPING`. Measured, not assumed.

That gap is fatal to the architecture's central claim. This corpus is 45% organizations, and
the layering and round-tripping typologies are built entirely from org-to-org chains, so
relying on discovery alone would leave counterparty names sitting in plaintext in the very
payload we claim contains no real identifiers.

The fallback is deliberately *not* another statistical model. Every party in this system is
already enumerated in `parties.json`, so the entities discovery misses are not unknown -
they are simply unrecognised. Exact matching against a known roster is therefore both more
accurate than an NER model and fully explainable, at the cost of only detecting parties
already on file.

That limitation is honest and worth stating plainly: this catches known counterparties, not
novel ones. In a real deployment the roster is the customer master, which is exactly the
population an institution needs to protect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RosterMatch:
    entity_type: str
    start: int
    end: int
    text: str
    score: float = 1.0  # exact match, no probabilistic judgement involved

    def as_entity(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "text": self.text,
            "source": "roster",
        }


class Roster:
    """Finds known party names in free text by exact, word-bounded match.

    Longer names are matched first so that an organization like `Meridian Holdings Ltd` is
    not shadowed by a shorter roster entry it contains.
    """

    def __init__(self, names_by_type: dict[str, list[str]]) -> None:
        # (entity_type, name) ordered longest-first to prevent shorter names shadowing longer.
        self._entries: list[tuple[str, str, re.Pattern[str]]] = []
        for entity_type, names in names_by_type.items():
            for name in sorted(set(names), key=len, reverse=True):
                if not name.strip():
                    continue
                # Guard boundaries only where the value itself starts/ends with a word
                # character. Values like emails and addresses begin or end with punctuation,
                # and a `\w` lookaround against punctuation never matches, which would
                # silently exclude exactly the identifiers this roster exists to catch.
                prefix = r"(?<!\w)" if name[0].isalnum() else ""
                suffix = r"(?!\w)" if name[-1].isalnum() else ""
                pattern = re.compile(rf"{prefix}{re.escape(name)}{suffix}")
                self._entries.append((entity_type, name, pattern))
        self._entries.sort(key=lambda e: len(e[1]), reverse=True)

    def find(self, text: str, taken: list[tuple[int, int]] | None = None) -> list[RosterMatch]:
        """Find roster names, skipping spans already claimed by another detector.

        `taken` carries the offsets discovery already matched. Respecting them is what keeps
        the two detectors composable: the roster fills gaps rather than competing for spans
        the model already handled.
        """
        claimed: list[tuple[int, int]] = list(taken or [])
        matches: list[RosterMatch] = []

        for entity_type, name, pattern in self._entries:
            # C-speed containment prefilter before any regex work. Almost every roster entry
            # is absent from any given note, ~24k compiled patterns against a ~500-character
            # narrative meant ~33 million finditer scans per ingest scope, and this one line
            # removes >99% of them (measured: 20.6s -> sub-second per 100 narratives).
            # Iteration order over the survivors is unchanged, so longest-first claim
            # priority, the semantics that stop `Meridian Holdings` shadowing
            # `Meridian Holdings Ltd`, is untouched.
            if name not in text:
                continue
            for found in pattern.finditer(text):
                start, end = found.start(), found.end()
                if any(start < c_end and end > c_start for c_start, c_end in claimed):
                    continue
                claimed.append((start, end))
                matches.append(RosterMatch(entity_type, start, end, found.group()))

        return sorted(matches, key=lambda m: m.start)


# Party field -> entity type, for values the roster matches literally.
#
# These are included because the discovery model is **not reliable enough to carry the
# invariant alone**. Measured on this corpus: the same phone-number format is detected in
# some narratives and missed entirely in others (`Contact on file: 415-555-6159.` returned
# no PHONE_NUMBER span at all), and phone numbers are frequently misclassified as
# SOCIAL_SECURITY_ID. A statistical detector that misses an identifier leaves it in the
# clear, and "usually detected" is not a security property.
#
# Every one of these values is already known from `parties.json`, so exact matching is both
# complete over the known population and fully explainable.
ROSTER_VALUE_FIELDS: dict[str, str] = {
    "phone": "PHONE_NUMBER",
    "ssn": "SOCIAL_SECURITY_ID",
    "email": "EMAIL_ADDRESS",
    "account_number": "ACCOUNT_NUMBER",
    "bank_account": "BANK_ACCOUNT",
    "tax_id": "TAX_ID",
    "credit_card": "CREDIT_CARD",
    "address": "ADDRESS",
    # City is a separate party field and was omitted, so when discovery missed a city inside
    # an address string, detecting the street parts but not the city, nothing caught it.
    # Measured: 64-65 leaked values per LOCATION-protecting scope on the scaled corpus.
    "city": "LOCATION",
    # Dates of birth reach narrative prose the same way names do, and the discovery service is
    # unreliable on date formats (measured: three of twenty variants undetected). Their
    # absence here meant a scope claiming to protect DATE_OF_BIRTH protected it in the party
    # record and not in the text.
    "date_of_birth": "DATE_OF_BIRTH",
}


def roster_from_narratives(narratives: list[dict]) -> dict[str, list[str]]:
    """Collect sensitive values each narrative declares it contains in prose form.

    Detection is format-sensitive: prose dates, written-out amounts and European
    decimal notation are not detected at any threshold, so a scope claiming to protect AMOUNT
    and DATETIME silently protects nothing in a sentence written the way people write.

    The corpus records the exact strings it emitted, so those values can be matched literally.
    This is the same hybrid strategy already used for organizations, deterministic lookup
    where the statistical detector is blind, and it carries the same honest limitation: it
    covers values the system already knows about, not arbitrary prose.
    """
    values_by_type: dict[str, list[str]] = {}
    for narrative in narratives:
        for entity_type, values in (narrative.get("narrative_values") or {}).items():
            for value in values:
                if value:
                    values_by_type.setdefault(entity_type, []).append(str(value))
    return values_by_type


def roster_from_parties(
    parties: list[dict], extra_values: dict[str, list[str]] | None = None
) -> Roster:
    """Build a roster of every known party name and identifier value.

    Names cover the class discovery misses entirely (ORGANIZATION). Identifier values cover
    the class discovery misses *intermittently*, which is the more dangerous failure, since
    partial coverage looks like success.

    `extra_values` folds in values from other sources, notably narrative-declared dates and
    amounts whose prose renderings the detector cannot see at all.
    """
    names_by_type: dict[str, list[str]] = {"ORGANIZATION": [], "PERSON": []}
    for entity_type, values in (extra_values or {}).items():
        names_by_type.setdefault(entity_type, []).extend(values)
    for entity_type in ROSTER_VALUE_FIELDS.values():
        names_by_type.setdefault(entity_type, [])

    for party in parties:
        name = (party.get("full_name") or "").strip()
        if name:
            key = "ORGANIZATION" if party.get("party_type") == "organization" else "PERSON"
            names_by_type[key].append(name)

        for field_name, entity_type in ROSTER_VALUE_FIELDS.items():
            value = str(party.get(field_name) or "").strip()
            if value:
                names_by_type[entity_type].append(value)

    return Roster(names_by_type)
