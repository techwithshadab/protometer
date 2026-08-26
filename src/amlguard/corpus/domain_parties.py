"""Party corpora for the non-AML live-chat domains (customer-support, healthcare).

AML ships a full transaction/alert corpus; support and healthcare only need a *party roster*
so a live chatbot turn can protect the entities a user types. The roster's job is to
catch known names and identifiers the discovery service misses — for a person-centric domain that
is the customer master (support) or the patient master (healthcare).

These parties are shaped to satisfy TWO consumers with no per-domain special-casing downstream:
  * `roster.roster_from_parties` reads `full_name`, `party_type`, and the standard identifier
    fields in `roster.ROSTER_VALUE_FIELDS` (phone, ssn, email, account_number, address, city,
    date_of_birth, …). Every party here fills those, so the roster protects them out of the box.
  * The domain's own `record_fields` (domains.py) names the domain-specific identifiers a
    structured record carries (healthcare `mrn`/`insurance_id`, support `order_id`). Those ride
    along as extra columns; the roster folds them in via `extra_values` (see ui `_get_roster`),
    so a typed MRN or insurance id is tokenized too, not just the generic PII shapes.

Deterministic by construction (seeded RNG, fiction-safe identifier ranges), mirroring
`corpus/parties.py`, so a rebuild is byte-identical and `check_determinism` stays green.
"""

from __future__ import annotations

import random

from amlguard.corpus.parties import (
    CITIES,
    FIRST_NAMES,
    LAST_NAMES,
    STREETS,
    _phone,
    _ssn,
)

# Insurers and clinics are drawn from small fixed pools so names recur (a real master has repeat
# providers/insurers), which is exactly what makes the ORGANIZATION-style roster entries meaningful.
INSURERS = (
    "Meridian Health Plan", "Blue Harbor Mutual", "Northgate Assurance",
    "Cobalt Care Network", "Halcyon Benefit Group", "Verdant Health Cooperative",
)
CLINICS = (
    "Dr. Priya Nakamura", "Dr. Marcus Hill", "Dr. Leila Farouk", "Dr. Devon Reyes",
    "Dr. Anika Rahman", "Dr. Rafael Almeida",
)
EMAIL_HOSTS = ("mailbox.com", "inboxly.net", "quickpost.org")


def _email(rng: random.Random, first: str, last: str) -> str:
    return f"{first.lower()}.{last.lower()}@{rng.choice(EMAIL_HOSTS)}"


def _person_base(rng: random.Random, index: int, prefix: str) -> dict:
    """The fields common to a support customer and a healthcare patient.

    Every value here is one the roster reads directly (`ROSTER_VALUE_FIELDS`), so the roster
    protects it with no domain-specific wiring. `party_type='individual'` keys the name as PERSON.
    """
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    city, state, postcode = rng.choice(CITIES)
    street = f"{rng.randint(100, 9899)} {rng.choice(STREETS)}"
    return {
        "party_id": f"{prefix}{index:05d}",
        "party_type": "individual",
        "full_name": f"{first} {last}",
        "email": _email(rng, first, last),
        "phone": _phone(rng),
        "address": f"{street}, {city}, {state} {postcode}",
        "city": city,
        "ssn": _ssn(rng),
        "date_of_birth": f"{rng.randint(1945, 2005)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        # Carried for name generation; not a protected field itself.
        "_first": first,
        "_last": last,
    }


def generate_support_parties(rng: random.Random, count: int) -> list[dict]:
    """Customer master for the customer-support domain: individuals with an account + a recent order."""
    parties: list[dict] = []
    for i in range(1, count + 1):
        p = _person_base(rng, i, "CS")
        p.pop("_first"); p.pop("_last")
        p.update({
            "account_number": f"{rng.randint(10**9, 10**10 - 1)}",
            # Some customers have a card on file; order ids are alphanumeric so they exercise the
            # ACCOUNT_NUMBER entity mapping the support record_fields declare for order_id.
            "credit_card": ("4111" + "".join(str(rng.randint(0, 9)) for _ in range(12)))
                           if rng.random() < 0.5 else "",
            "order_id": f"ORD-{rng.randint(100000, 999999)}",
        })
        parties.append(p)
    return parties


def generate_healthcare_parties(rng: random.Random, count: int) -> list[dict]:
    """Patient master for the healthcare domain: individuals with an MRN, insurer, and provider."""
    parties: list[dict] = []
    for i in range(1, count + 1):
        p = _person_base(rng, i, "HC")
        p.pop("_first"); p.pop("_last")
        p.update({
            # MRN and insurance id are the domain's high-sensitivity identifiers (domains.py).
            "mrn": f"MRN{rng.randint(1000000, 9999999)}",
            "insurance_id": f"{rng.choice(('A', 'B', 'C', 'X', 'Z'))}{rng.randint(100000000, 999999999)}",
            "provider_name": rng.choice(CLINICS),
            "insurer": rng.choice(INSURERS),
        })
        parties.append(p)
    return parties


# Registry of the non-AML party generators, keyed by domain name. Each takes (rng, count).
DOMAIN_PARTY_GENERATORS = {
    "customer-support": generate_support_parties,
    "healthcare": generate_healthcare_parties,
}


def generate_domain_parties(domain: str, count: int, seed: int) -> list[dict]:
    """Deterministic party corpus for a non-AML domain. Raises KeyError for an unknown domain."""
    gen = DOMAIN_PARTY_GENERATORS[domain]
    return gen(random.Random(seed), count)
