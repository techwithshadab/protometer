"""Party generation, the entities whose PII the pipeline protects.

Every sensitive field generated here must satisfy two constraints:

  1. It must be *detectable* by Protegrity's discovery service, since the unstructured path
     relies on discovery finding it. Values are therefore generated in conventional formats
     rather than arbitrary strings.
  2. It must be *acceptable* to the data element that protects it. Measurement recorded `ccn`
     rejecting space-separated card numbers with error 44, so card numbers are emitted as
     unseparated digits.

Values are synthetic. SSNs use the 900-999 area range, which the Social Security
Administration never issues, and card numbers use the 4111 test BIN, so nothing here can
collide with a real identity even by accident.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

FIRST_NAMES = (
    "Allison", "Marcus", "Priya", "Devon", "Yuki", "Rafael", "Nadia", "Tobias",
    "Imani", "Sergei", "Lucia", "Hassan", "Bianca", "Kwame", "Freya", "Diego",
    "Anika", "Mateo", "Chidi", "Rowan", "Sana", "Oskar", "Leila", "Bruno",
)
LAST_NAMES = (
    "Hill", "Okafor", "Nakamura", "Vasquez", "Petrov", "Almeida", "Choudhury",
    "Lindqvist", "Mwangi", "Delacroix", "Reyes", "Farouk", "Kowalski", "Santoro",
    "Iversen", "Baptiste", "Rahman", "Novak", "Adeyemi", "Sokolov",
)
ORG_HEADS = (
    "Meridian", "Alderpoint", "Northgate", "Cobalt Ridge", "Halcyon", "Verdant",
    "Ironwood", "Blue Harbor", "Sablefield", "Kestrel", "Latimer", "Thornbury",
    "Greycastle", "Windermere", "Solstice", "Pemberton",
)
ORG_TAILS = (
    "Holdings Ltd", "Trading Co", "Capital Partners", "Logistics Group", "Consulting LLC",
    "Ventures Inc", "Import Export SA", "Property Group", "Advisory Services",
    "Commodities BV", "Management GmbH",
)
CITIES = (
    ("Newark", "NJ", "07102"), ("Tampa", "FL", "33602"), ("Phoenix", "AZ", "85004"),
    ("Reno", "NV", "89501"), ("Buffalo", "NY", "14202"), ("Mobile", "AL", "36602"),
    ("Fresno", "CA", "93721"), ("Toledo", "OH", "43604"), ("Boise", "ID", "83702"),
    ("Shreveport", "LA", "71101"),
)
STREETS = (
    "Bellweather Ave", "Canal St", "Sycamore Row", "Harbour Way", "Pinecrest Dr",
    "Old Mill Rd", "Fairmount Blvd", "Kingsley Ct", "Riverbend Ln", "Ashcombe St",
)
# Jurisdictions that carry analytical weight in AML review.
HIGH_RISK_JURISDICTIONS = ("PA", "CY", "AE", "MT", "VG", "SC", "LB")
DOMESTIC_JURISDICTION = "US"


@dataclass(frozen=True)
class Party:
    """A person or organization. Field names double as discovery entity hints."""

    party_id: str
    party_type: str  # "individual" | "organization"
    full_name: str
    account_number: str
    bank_account: str
    jurisdiction: str
    address: str
    city: str
    email: str
    phone: str
    # Individuals only; empty for organizations.
    ssn: str = ""
    date_of_birth: str = ""
    # Organizations only.
    tax_id: str = ""
    credit_card: str = ""
    # Risk annotations, unprotected metadata, useful for demonstrating that non-sensitive
    # fields remain available for filtering even at maximum protection scope.
    risk_rating: str = "low"
    is_pep: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sensitive_fields(self) -> dict[str, str]:
        """Field -> discovery entity type, for the structured protection path.

        Only non-empty fields are returned, so individuals and organizations each yield
        their own applicable subset.
        """
        mapping = {
            "full_name": "PERSON" if self.party_type == "individual" else "ORGANIZATION",
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
        }
        return {f: e for f, e in mapping.items() if getattr(self, f)}


def _ssn(rng: random.Random) -> str:
    # 900-999 area numbers are never issued by the SSA, safe by construction.
    return f"{rng.randint(900, 999)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"


def _card(rng: random.Random) -> str:
    # Unseparated: `ccn` rejects spaces with error 44 (measured).
    return "4111" + "".join(str(rng.randint(0, 9)) for _ in range(12))


def _phone(rng: random.Random) -> str:
    # 555 exchange is reserved for fiction.
    return f"{rng.choice((415, 212, 312, 702, 305))}-555-{rng.randint(1000, 9999):04d}"


def generate_party(rng: random.Random, index: int, party_type: str) -> Party:
    city, state, postcode = rng.choice(CITIES)
    address = f"{rng.randint(100, 9899)} {rng.choice(STREETS)}"

    # A minority sit in high-risk jurisdictions, which is what makes jurisdiction a
    # meaningful analytical signal rather than noise.
    jurisdiction = (
        rng.choice(HIGH_RISK_JURISDICTIONS) if rng.random() < 0.22 else DOMESTIC_JURISDICTION
    )
    risk = "high" if jurisdiction != DOMESTIC_JURISDICTION else rng.choice(("low", "low", "medium"))

    if party_type == "individual":
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}@{rng.choice(('mailbox.com', 'inboxly.net', 'quickpost.org'))}"
        return Party(
            party_id=f"P{index:05d}",
            party_type="individual",
            full_name=name,
            account_number=f"{rng.randint(10**9, 10**10 - 1)}",
            bank_account=f"{rng.randint(10**7, 10**8 - 1)}",
            jurisdiction=jurisdiction,
            address=f"{address}, {city}, {state} {postcode}",
            city=city,
            email=email,
            phone=_phone(rng),
            ssn=_ssn(rng),
            date_of_birth=f"{rng.randint(1955, 2000)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            risk_rating=risk,
            is_pep=rng.random() < 0.06,
        )

    head, tail = rng.choice(ORG_HEADS), rng.choice(ORG_TAILS)
    name = f"{head} {tail}"
    slug = head.lower().replace(" ", "")
    return Party(
        party_id=f"P{index:05d}",
        party_type="organization",
        full_name=name,
        account_number=f"{rng.randint(10**9, 10**10 - 1)}",
        bank_account=f"{rng.randint(10**7, 10**8 - 1)}",
        jurisdiction=jurisdiction,
        address=f"{address}, {city}, {state} {postcode}",
        city=city,
        email=f"accounts@{slug}-group.com",
        phone=_phone(rng),
        tax_id=f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}",
        credit_card=_card(rng) if rng.random() < 0.4 else "",
        risk_rating=risk,
        is_pep=False,
    )


def generate_parties(rng: random.Random, count: int, org_ratio: float = 0.45) -> list[Party]:
    """Generate a party population with a realistic mix of individuals and organizations."""
    parties: list[Party] = []
    for i in range(count):
        party_type = "organization" if rng.random() < org_ratio else "individual"
        parties.append(generate_party(rng, i + 1, party_type))
    return parties
