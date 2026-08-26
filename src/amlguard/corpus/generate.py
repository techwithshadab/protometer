"""Corpus assembly, parties, transactions, planted typologies, narratives, ground truth.

Deterministic under a fixed seed, so the corpus is reproducible and every evaluation run
across every protection scope sees identical underlying data. Without that, differences
between curve points would confound protection scope with corpus variation.

Original sizing: ~300 parties, ~2000 transactions, ~300 narratives, large enough
that retrieval and entity resolution are non-trivial, small enough to debug by reading and
to protect within the hosted API's undocumented rate limits.

Narratives matter disproportionately. They are unstructured investigator prose containing
PII inline, which is where Protegrity's discovery service earns its place and where
[[Semantic Erasure]] is observable. Structured records alone would make protection trivial
and the finding uninteresting.
"""

from __future__ import annotations

import collections
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from amlguard.corpus.parties import Party, generate_parties
from amlguard.corpus.typologies import (
    CTR_THRESHOLD,
    PlantedTransactions,
    TypologyInstance,
    plant_funnel_account,
    plant_layering,
    plant_round_tripping,
    plant_structuring,
    plant_trade_based,
)
from amlguard.corpus.vocabulary import benign_family, channel_for, memo_for

CORPUS_START = date(2025, 1, 6)
CORPUS_DAYS = 300

# Most typology instances any one party may participate in.
#
# Two is the smallest cap that still lets a party recur across cases, which real launderers
# do, and which the cross-document identity-resolution tasks depend on, while preventing the
# single party that previously appeared in 54 separate typologies. See the reasoning at the
# `usage` counter in `_plant_all`.
MAX_INSTANCES_PER_PARTY = 2

# Fraction of alerts that escalate to a SAR filing. Bank Policy Institute, *Getting to
# Effectiveness* (2018): ~16M alerts across 19 institutions produced >640,000 SARs.
#
# The widely quoted "99% false positive" figure belongs to sanctions/name screening, a
# different control with far lower true-match rates; for transaction monitoring ~4% conversion
# is the defensible number.
TARGET_SAR_CONVERSION_RATE = 0.04

# Rendering variants for dates and amounts as they appear in *narrative* text.
#
# We measured Protegrity's discovery service against these: ISO, US and European
# slash-delimited dates are detected; prose dates ("15 April 2025", "April 15, 2025"),
# written-out amounts and European decimal notation are **not detected at all**. A corpus that
# emits only ISO dates and bare decimals is therefore testing the best case for detection and
# overstating how much protection succeeds on realistic records.
#
# Structured ledger fields stay canonical, an institution controls its own schema, while
# narrative prose varies, because case notes are written by people.

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

MAGNITUDES = (
    (1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"),
)


def format_date_variant(iso_date: str, rng: random.Random) -> str:
    """Render an ISO date the way a person might write it in a case note."""
    year, month, day = (int(part) for part in iso_date.split("-"))
    month_name = MONTHS[month - 1]
    return rng.choice(
        (
            iso_date,                              # 2025-04-15   detected
            f"{month:02d}/{day:02d}/{year}",       # 04/15/2025   detected
            f"{day:02d}/{month:02d}/{year}",       # 15/04/2025   detected
            f"{day} {month_name} {year}",          # 15 April 2025    NOT detected
            f"{month_name} {day}, {year}",         # April 15, 2025   NOT detected
        )
    )


def _spell_amount(value: Decimal) -> str:
    """Approximate an amount in words, as an investigator would summarise it."""
    amount = int(value)
    for size, name in MAGNITUDES:
        if amount >= size:
            whole = amount // size
            remainder = (amount % size) * 10 // size
            fraction = f" point {remainder}" if remainder else ""
            return f"{whole}{fraction} {name} dollars"
    return f"{amount} dollars"


def format_amount_variant(value: Decimal, rng: random.Random) -> str:
    """Render an amount the way it might appear in narrative text."""
    return rng.choice(
        (
            f"{value:.2f}",                        # 712500.00        detected
            f"${value:,.2f}",                      # $712,500.00      detected (symbol dropped)
            f"USD {value / 1000:.1f}k",            # USD 712.5k       detected
            _spell_amount(value),                  # seven hundred... NOT detected
            f"{value:,.2f} EUR".replace(",", "\x00").replace(".", ",").replace("\x00", "."),
        )                                          # 712.500,00 EUR   NOT detected
    )


# Benign memo and channel vocabularies now live in `corpus.vocabulary`, shared with every
# typology generator. Keeping private word lists on each side is what let the two drift apart
# until a twelve-word lookup separated the classes with zero false positives, see that
# module's docstring for the three leaks this arrangement prevents.


@dataclass
class Corpus:
    parties: list[Party]
    transactions: list[dict]
    narratives: list[dict]
    typologies: list[TypologyInstance]
    alerts: list[dict] = field(default_factory=list)

    def party_by_id(self, party_id: str) -> Party:
        return next(p for p in self.parties if p.party_id == party_id)


def _weighted_party_sampler(rng: random.Random, parties: list[Party]):
    """Draw parties with a power-law skew, as real counterparty activity is distributed.

    Uniform sampling produces a flat entity distribution, and flatness silently flatters the
    adversarial evaluation: frequency analysis works by matching the *k*-th most common token
    to the *k*-th most common plaintext, so a corpus where nothing recurs offers no rank signal
    and the attack reports ~0% recovery. That result would be a property of the synthetic data
    rather than of the protection scheme.

    Real institutional data is heavily skewed. A small head of counterparties, payroll
    providers, utilities, correspondent banks, appears in a large share of transactions, while
    a long tail appears once. Zipf-like weights reproduce that shape, making the frequency
    attack a fair test.
    """
    ranked = list(parties)
    rng.shuffle(ranked)
    # Zipf weights: the most active party appears ~len(parties) times as often as the least.
    weights = [1.0 / (rank + 1) for rank in range(len(ranked))]
    return lambda: rng.choices(ranked, weights=weights, k=1)[0]


def _benign_transactions(
    rng: random.Random, parties: list[Party], count: int, start_index: int
) -> list[dict]:
    """Background noise. Without a realistic base rate, every typology is trivially
    visible as the only activity in the corpus, and detection becomes meaningless."""
    txns: list[dict] = []
    draw = _weighted_party_sampler(rng, parties)
    for i in range(count):
        origin = draw()
        beneficiary = draw()
        # Redraw rather than allowing self-payment, which no ledger would contain.
        attempts = 0
        while beneficiary.party_id == origin.party_id and attempts < 8:
            beneficiary = draw()
            attempts += 1
        if beneficiary.party_id == origin.party_id:
            continue
        # Log-ish distribution: most payments small, a few large.
        # A deliberate share of benign activity sits in the 7.5k-10k band that structuring
        # occupies. Without it, "amount just under the reporting threshold" is definitionally
        # equivalent to "suspicious", flagged transactions had a minimum of $7,521 against a
        # benign minimum of $51, so a single threshold separated the classes and any model
        # trained on the corpus learned the generator.
        #
        # Real ledgers are full of ordinary payments in that band: rent, invoices, payroll
        # runs. Including them is both more faithful and what makes the typology's *pattern*
        #, repetition, timing, one counterparty, the only real signal.
        amount = Decimal(rng.choice((
            rng.randrange(50, 5_000),
            rng.randrange(500, 25_000),
            rng.randrange(5_000, 180_000),
            rng.randrange(7_500, 9_999),
        )))
        # Round-thousand values on a realistic share of benign traffic. Round values are a
        # genuine FATF trade-based indicator and are kept as a typology signature, but while
        # only planted transactions produced them (78 positive against 6 benign), the
        # signature was a near-pure label. Contract and settlement payments are routinely
        # round in real ledgers, so benign traffic should produce them too.
        if rng.random() < 0.06:
            amount = (amount // 1000 + 1) * 1000
        # Sub-dollar cents on a realistic share of the rest. Whole-dollar benign amounts
        # against typology amounts carrying true cents (the residue of commission arithmetic:
        # `amount * 0.93`) made "has non-zero cents" a pure separator, 140 positive, 0
        # negative. Real ledgers are full of fractional amounts, so this is both the fix and
        # the more faithful choice.
        elif rng.random() < 0.55:
            amount += Decimal(rng.randrange(1, 100)) / Decimal(100)
        amount = amount.quantize(Decimal("0.01"))

        family = benign_family(rng)
        txns.append({
            "transaction_id": f"TXN{start_index + i:06d}",
            "origin_party_id": origin.party_id,
            "beneficiary_party_id": beneficiary.party_id,
            "amount": str(amount),
            "currency": "USD",
            "value_date": (CORPUS_START + timedelta(days=rng.randint(0, CORPUS_DAYS))).isoformat(),
            "channel": channel_for(rng, family),
            "memo": memo_for(rng, family),
        })
    return txns


def _camouflage_transactions(
    rng: random.Random,
    parties: list[Party],
    typology_party_ids: set[str],
    transactions: list[dict],
    start_index: int,
) -> list[dict]:
    """Give typology participants ordinary activity matching their peers.

    Without this, participation in a planted pattern is inferable from **activity volume
    alone**. Benign transactions are drawn with Zipf weights, so most parties sit in a
    low-activity tail, but typology parties are sampled uniformly and then given a handful of
    pattern transactions each. Measured on the previous corpus: `beneficiary_in_degree` reached
    an AUC of 0.293 against the label (strongly inverted), and typology parties had a median
    activity of 8 against 5 for everyone else.

    A classifier learning "unusual degree implies suspicious" is learning the generator, not
    laundering. The fix is to top each typology party up to a realistic activity level with
    ordinary transactions, so degree carries no label information.
    """
    activity: collections.Counter[str] = collections.Counter()
    for txn in transactions:
        activity[txn["origin_party_id"]] += 1
        activity[txn["beneficiary_party_id"]] += 1

    # Target the median of parties *not* involved in any typology, so participants blend into
    # the population rather than standing out in either direction.
    others = [activity[p.party_id] for p in parties if p.party_id not in typology_party_ids]
    target = sorted(others)[len(others) // 2] if others else 6

    filler: list[dict] = []

    for party_id in sorted(typology_party_ids):
        deficit = target - activity[party_id]
        # Jitter so every participant does not land on exactly the median, which would itself
        # be a signature.
        deficit += rng.randint(-2, 3)
        for _ in range(max(0, deficit)):
            counterparty = rng.choice(parties)
            if counterparty.party_id == party_id:
                continue
            outbound = rng.random() < 0.5
            amount = Decimal(rng.choice((
                rng.randrange(50, 5_000),
                rng.randrange(500, 25_000),
                rng.randrange(5_000, 180_000),
            )))
            if rng.random() < 0.55:
                amount += Decimal(rng.randrange(1, 100)) / Decimal(100)
            amount = amount.quantize(Decimal("0.01"))
            family = benign_family(rng)
            filler.append({
                "transaction_id": f"TXN{start_index + len(filler):06d}",
                "origin_party_id": party_id if outbound else counterparty.party_id,
                "beneficiary_party_id": counterparty.party_id if outbound else party_id,
                "amount": str(amount),
                "currency": "USD",
                "value_date": (
                    CORPUS_START + timedelta(days=rng.randint(0, CORPUS_DAYS))
                ).isoformat(),
                "channel": channel_for(rng, family),
                "memo": memo_for(rng, family),
            })

    return filler


def _renumber_transactions(
    transactions: list[dict], instances: list[TypologyInstance]
) -> list[TypologyInstance]:
    """Assign every transaction an id from one namespace, erasing the typology from the key.

    Planted transactions were previously numbered by their generator, `STR011-T02`,
    `LAY003-H01`, `FUN017-D010`, while benign traffic was `TXN000043`. The prefix *was* the
    label: 762 of 6,843 ids carried one, and that set was exactly the flagged set.

    That mattered because the id is not an internal detail. `hybrid.py` puts it at the top of
    the rationale prompt (`ITEM {item_id}, ranked ...`), so every escalated item handed the
    model a string reading "structuring instance 11" before asking whether the activity looked
    like structuring. The classifier never consumed ids, so average precision was unaffected -
    but the analyst-facing artifact, the one the submission presents as its product, was
    generated with the answer in view.

    Renumbering happens after the chronological sort so ids ascend with `value_date`, which is
    what a real ledger's sequence numbers do. The typology mapping survives in
    `ground_truth.json`, where it belongs: available for scoring, absent from the data.

    Returns rebuilt instances, because `TypologyInstance` is frozen and its `transaction_ids`
    are the answer key, rewriting the ledger without rewriting the key in the same step would
    silently break every checkpoint that references a transaction.
    """
    mapping: dict[str, str] = {}
    for i, txn in enumerate(transactions):
        new_id = f"TXN{i:06d}"
        mapping[txn["transaction_id"]] = new_id
        txn["transaction_id"] = new_id

    return [
        replace(
            instance,
            transaction_ids=tuple(mapping[t] for t in instance.transaction_ids),
        )
        for instance in instances
    ]


def _plant_all(
    rng: random.Random,
    parties: list[Party],
    n_structuring: int,
    n_layering: int,
    n_round: int,
    n_funnel: int = 0,
    n_trade: int = 0,
) -> list[PlantedTransactions]:
    """Plant typologies over distinct party sets so instances stay independently scorable."""
    planted: list[PlantedTransactions] = []
    # Organizations make more plausible layering intermediaries than individuals.
    #
    # Participants are drawn **Zipf-weighted**, matching how benign traffic is distributed.
    # Uniform selection put typology transactions on ordinary parties while benign traffic
    # concentrated on high-degree hubs, so origin out-degree separated the classes at the
    # transaction level, median 9 for flagged against 24 for benign, even after party-level
    # activity was equalised. Drawing participants the same way benign counterparties are drawn
    # removes the difference at its source.
    orgs = [p for p in parties if p.party_type == "organization"]
    individuals = [p for p in parties if p.party_type == "individual"]
    draw_org = _weighted_party_sampler(rng, orgs)
    draw_individual = _weighted_party_sampler(rng, individuals)

    # How many typology instances each party already participates in. Zipf-weighted draws
    # concentrate on the same hubs, so 122 instances needing ~513 party-slots put one party
    # into 54 separate typologies and made 196 of 600 parties (32.7%) illicit.
    #
    # Two things follow from that density, and both corrupt the measurement:
    #
    #   * Real AML populations are far under 1% illicit. At 32.7%, "touches a known-illicit
    #     party" is *genuinely* predictive of a benign transaction 43.5% of the time, so the
    #     GuiltyWalker feature cannot be separated from the label by any train/test split -
    #     splitting by instance left a 65-point gap, and splitting by connected component was
    #     impossible because 119 of 122 instances formed a single component.
    #   * A party in 54 typologies is not a launderer, it is a generator artifact.
    #
    # Participation is still Zipf-weighted, because analysis established that drawing typology
    # participants uniformly while benign traffic follows a power law makes degree a label.
    # The weighting is preserved and reuse is capped: a candidate already in
    # `MAX_INSTANCES_PER_PARTY` instances is passed over.
    usage: collections.Counter[str] = collections.Counter()

    def draw_capped(draw, pool: list[Party], exclude: set[str] | None = None) -> Party:
        """Zipf-weighted draw that skips parties already saturated with typologies."""
        excluded = exclude or set()
        for _ in range(64):
            candidate = draw()
            if (
                candidate.party_id not in excluded
                and usage[candidate.party_id] < MAX_INSTANCES_PER_PARTY
            ):
                return candidate
        # Fall back to the least-used party in the pool rather than looping forever, so a
        # small pool degrades gracefully instead of hanging.
        available = [p for p in pool if p.party_id not in excluded] or pool
        return min(available, key=lambda p: usage[p.party_id])

    def sample_orgs(count: int) -> list[Party]:
        """Distinct organizations, weighted like ordinary counterparty activity."""
        chosen: list[Party] = []
        seen: set[str] = set()
        while len(chosen) < count and len(seen) < len(orgs):
            candidate = draw_capped(draw_org, orgs, seen)
            if candidate.party_id in seen:
                break
            seen.add(candidate.party_id)
            chosen.append(candidate)
        for party in chosen:
            usage[party.party_id] += 1
        return chosen

    for i in range(n_structuring):
        subject = draw_capped(draw_individual, individuals)
        beneficiary = draw_capped(draw_org, orgs, {subject.party_id})
        usage[subject.party_id] += 1
        usage[beneficiary.party_id] += 1
        planted.append(plant_structuring(
            rng, f"STR{i:03d}", subject.party_id, beneficiary.party_id,
            CORPUS_START + timedelta(days=rng.randint(0, CORPUS_DAYS - 30)),
        ))

    for i in range(n_layering):
        chain = sample_orgs(rng.randint(4, 6))
        planted.append(plant_layering(
            rng, f"LAY{i:03d}", [p.party_id for p in chain],
            CORPUS_START + timedelta(days=rng.randint(0, CORPUS_DAYS - 60)),
        ))

    for i in range(n_round):
        circuit = sample_orgs(rng.randint(3, 5))
        planted.append(plant_round_tripping(
            rng, f"RTP{i:03d}", [p.party_id for p in circuit],
            CORPUS_START + timedelta(days=rng.randint(0, CORPUS_DAYS - 60)),
        ))

    # Funnel accounts: many individual depositors converging on one collector, withdrawn
    # elsewhere. Individuals deposit; an organization collects (FIN-2014-A005).
    for i in range(n_funnel):
        # Pick the collector first, then draw depositors whose city differs from the funnel's,
        # so the corpus realizes FIN-2014-A005's defining geographic-dispersion signature (cash
        # gathered in a *different* area from where the account is domiciled), not only the
        # temporal convergence. The dispersion is recorded in the instance, so a detector can be
        # credited for it. A short bounded retry keeps a small/degenerate city pool from hanging.
        funnel, destination = sample_orgs(2)
        depositors: list[Party] = []
        for _ in range(rng.randint(3, 6)):
            picked = None
            for _attempt in range(8):
                candidate = draw_capped(
                    draw_individual, individuals, {p.party_id for p in depositors}
                )
                picked = candidate
                if candidate.city != funnel.city:
                    break  # dispersed away from the funnel's domicile, as the advisory requires
            depositors.append(picked)
            usage[picked.party_id] += 1
        planted.append(plant_funnel_account(
            rng, f"FUN{i:03d}", [p.party_id for p in depositors],
            funnel.party_id, destination.party_id,
            CORPUS_START + timedelta(days=rng.randint(0, CORPUS_DAYS - 30)),
            depositor_cities=[p.city for p in depositors],
            funnel_city=funnel.city,
        ))

    # Trade-based: an importer over-paying against one invoice, settled partly to third
    # parties unrelated to the trade (FATF/Egmont 2021).
    for i in range(n_trade):
        importer, exporter, *intermediaries = sample_orgs(rng.randint(4, 5))
        planted.append(plant_trade_based(
            rng, f"TBM{i:03d}", importer.party_id, exporter.party_id,
            [p.party_id for p in intermediaries],
            CORPUS_START + timedelta(days=rng.randint(0, CORPUS_DAYS - 90)),
        ))

    return planted


def _narrative_for_typology(
    rng: random.Random, corpus_parties: dict[str, Party], instance: TypologyInstance, doc_id: str
) -> dict:
    """Investigator prose naming parties in the clear.

    PII appears inline and unstructured, this is the text the discovery service must find
    entities in, and where tokenization visibly degrades semantic retrieval.
    """
    subject = corpus_parties[instance.subject_party_id]
    others = [corpus_parties[p] for p in instance.party_ids if p != instance.subject_party_id]
    other_names = ", ".join(p.full_name for p in others[:4])

    contact = f"{subject.email} / {subject.phone}"
    identifier = f"SSN {subject.ssn}" if subject.ssn else f"tax ID {subject.tax_id}"

    openings = (
        f"Alert review for {subject.full_name} (account {subject.account_number}).",
        f"Case note: {subject.full_name}, {identifier}, flagged by transaction monitoring.",
        f"Escalation summary concerning {subject.full_name} of {subject.address}.",
    )
    closings = (
        "Recommend escalation to a SAR filing.",
        "Referred to senior investigator for disposition.",
        "Further review of counterparty relationships requested.",
        "No adverse media identified at this stage; monitoring continues.",
    )

    # Dates and amounts rendered as a person would write them, not as the ledger stores them.
    # The emitted strings are recorded so the leak check can assert they were protected -
    # otherwise a format the detector cannot see passes through silently.
    activity_date = format_date_variant(
        (CORPUS_START + timedelta(days=rng.randint(0, CORPUS_DAYS))).isoformat(), rng
    )
    activity_amount = format_amount_variant(instance.total_amount, rng)

    body = (
        f"{rng.choice(openings)} "
        f"Subject is domiciled in {subject.jurisdiction} with an internal risk rating of "
        f"{subject.risk_rating}. Reachable at {contact}. "
        f"Monitoring identified {instance.narrative_hint}. "
        f"Activity was first reported on {activity_date}, with exposure assessed at "
        f"{activity_amount}. "
        f"Counterparties involved: {other_names}. "
        f"{rng.choice(closings)}"
    )

    return {
        "document_id": doc_id,
        "document_type": "investigation_note",
        "subject_party_id": subject.party_id,
        "typology_id": instance.typology_id,
        "text": body,
        # Sensitive values this note contains in narrative form, for leak verification.
        "narrative_values": {"DATETIME": [activity_date], "AMOUNT": [activity_amount]},
        # Entities named in this note, in order of appearance. Ground truth for the
        # adversarial evaluation: scoring a re-identification attack requires knowing which
        # plaintext each token replaced.
        "plaintext_entities": {
            "PERSON": [subject.full_name] if subject.party_type == "individual" else [],
            "ORGANIZATION": (
                ([subject.full_name] if subject.party_type == "organization" else [])
                + [p.full_name for p in others if p.party_type == "organization"]
            ),
        },
    }


def _benign_narrative(rng: random.Random, party: Party, doc_id: str) -> dict:
    """KYC/onboarding prose for parties with no planted typology.

    These matter: if only suspicious parties had narratives, document existence alone would
    leak the answer and no reasoning would be required.
    """
    identifier = f"SSN {party.ssn}" if party.ssn else f"tax ID {party.tax_id}"
    templates = (
        f"KYC refresh completed for {party.full_name} ({identifier}), residing at "
        f"{party.address}. Contact details verified: {party.email}, {party.phone}. "
        f"Source of funds documented as salaried employment. Risk rating "
        f"{party.risk_rating}; no escalation required.",
        f"Periodic review of {party.full_name}, account {party.account_number}. "
        f"Jurisdiction {party.jurisdiction}. Correspondence address {party.address}. "
        f"Transaction activity consistent with declared business purpose. "
        f"Contact on file: {party.phone}.",
        f"Onboarding file for {party.full_name}. Primary contact {party.email}. "
        f"Beneficial ownership documentation received and verified. "
        f"Registered address {party.address}. No sanctions or PEP matches returned.",
    )
    return {
        "document_id": doc_id,
        "document_type": "kyc_note",
        "subject_party_id": party.party_id,
        "typology_id": None,
        "text": rng.choice(templates),
    }


# Monitoring scenarios that fire on benign activity. Each names a rule an institution really
# runs and the ordinary explanation an investigator would find on review.
#
# Real transaction monitoring converts roughly 4% of alerts into SARs (BPI, *Getting to
# Effectiveness*, 2018: ~16M alerts across 19 institutions produced >640,000 SARs). A corpus in
# which every alert is a true positive measures pattern identification while omitting the
# discipline that dominates the actual job, clearing benign activity.
BENIGN_ALERT_SCENARIOS: tuple[tuple[str, str, str], ...] = (
    ("LARGE_CASH_DEPOSIT", "Cash deposit exceeding profile threshold",
     "Proceeds of documented property sale; completion statement on file."),
    ("VELOCITY_CHANGE", "Transaction velocity deviates from customer profile",
     "Seasonal business cycle consistent with prior years."),
    ("HIGH_RISK_JURISDICTION", "Funds transfer involving higher-risk jurisdiction",
     "Established supplier relationship, invoices and shipping documents verified."),
    ("ROUND_AMOUNT", "Repeated round-value transfers",
     "Fixed monthly rent payment under a signed lease."),
    ("DORMANT_REACTIVATION", "Activity on previously dormant account",
     "Account reactivated following probate; inheritance documented."),
    ("STRUCTURING_THRESHOLD", "Multiple deposits below reporting threshold",
     "Retail takings deposited daily; volumes consistent with declared turnover."),
    ("RAPID_MOVEMENT", "Funds withdrawn shortly after deposit",
     "Payroll float; outbound salary payments match inbound funding."),
    ("NEW_COUNTERPARTY", "First transaction with a new counterparty",
     "New supplier onboarded; contract and due diligence on file."),
    ("PEP_ACTIVITY", "Activity involving a politically exposed person",
     "Salary credit from a public-sector employer; source of funds consistent."),
    ("CROSS_BORDER_VOLUME", "Aggregate cross-border volume exceeds threshold",
     "Import business; trade documentation reviewed and consistent."),
)

DISPOSITIONS = ("closed_no_action", "closed_documented", "closed_false_positive")

# What each scenario actually fires on. A monitoring rule is a predicate over activity, so an
# alert whose triggering transaction does not satisfy its own scenario is not an alert, it is
# a label attached to an unrelated row.
#
# Scenarios not listed here fire on party attributes or aggregates rather than on one
# transaction's shape (PEP_ACTIVITY on the customer record, CROSS_BORDER_VOLUME and
# VELOCITY_CHANGE on windowed totals), and take any transaction of the matching party.
SCENARIO_PREDICATES: dict[str, Callable[[dict], bool]] = {
    "LARGE_CASH_DEPOSIT": lambda t: (
        t["channel"] in ("cash_deposit", "branch_deposit", "atm_deposit")
        and Decimal(t["amount"]) > Decimal("8000")
    ),
    "STRUCTURING_THRESHOLD": lambda t: (
        t["channel"] in ("cash_deposit", "branch_deposit", "atm_deposit")
        and Decimal("7000") < Decimal(t["amount"]) < CTR_THRESHOLD
    ),
    "ROUND_AMOUNT": lambda t: Decimal(t["amount"]) % 1000 == 0,
    "RAPID_MOVEMENT": lambda t: t["channel"] in ("wire", "swift", "correspondent"),
    "HIGH_RISK_JURISDICTION": lambda t: t["channel"] in ("swift", "correspondent"),
    "NEW_COUNTERPARTY": lambda t: True,
    "DORMANT_REACTIVATION": lambda t: True,
}


def _match_scenario(
    rng: random.Random,
    scenario_id: str,
    parties: list[Party],
    by_party: dict[str, list[dict]],
) -> tuple[Party, dict | None]:
    """Find a party and a transaction that genuinely satisfy `scenario_id`.

    Falls back to any transaction of a random party when the scenario has no transaction-level
    predicate, or when no matching activity exists, a monitoring system does occasionally
    fire on aggregates rather than on a single movement, and forcing a match would distort the
    population rather than improve it.
    """
    predicate = SCENARIO_PREDICATES.get(scenario_id)
    if predicate is not None:
        # Sample a few parties rather than scanning the whole ledger: the corpus is large
        # enough that a match is found quickly, and an exhaustive scan per alert would make
        # generation quadratic.
        for _ in range(24):
            party = rng.choice(parties)
            matches = [t for t in by_party.get(party.party_id, []) if predicate(t)]
            if matches:
                return party, rng.choice(matches)

    party = rng.choice(parties)
    candidates = by_party.get(party.party_id, [])
    return party, (rng.choice(candidates) if candidates else None)


def _benign_score(rng: random.Random, trigger: dict | None, prior_alerts: int) -> int:
    """Rule score as a function of the activity that fired it.

    Magnitude and repeat-alerting drive the score, as they do in a real scoring rule, with a
    small random component for the residual the model cannot see. Benign scores still overlap
    escalating ones, a clean separation would make triage a lookup, but the overlap now
    comes from genuinely ambiguous activity rather than from two disjoint uniform draws.
    """
    base = 30
    if trigger is not None:
        amount = Decimal(trigger["amount"])
        # Roughly logarithmic in amount, so a $200k movement scores well above a $2k one
        # without the relationship being a straight line an analyst could invert.
        base += min(22, int(amount / Decimal("9000")))
        if trigger["channel"] in ("cash_deposit", "branch_deposit", "atm_deposit"):
            base += 6
    base += min(10, prior_alerts * 3)
    return max(30, min(72, base + rng.randint(-6, 6)))


def _benign_alerts(
    rng: random.Random,
    parties: list[Party],
    transactions: list[dict],
    count: int,
    start_index: int,
) -> list[dict]:
    """Alerts on ordinary activity, which an investigator must clear rather than escalate.

    These carry the same schema as true-positive alerts, including prior-match counts and
    linked cases, because an alert's escalation weight comes largely from its history: a third
    alert on one subject is a different object from a first.
    """
    by_party: dict[str, list[dict]] = {}
    for txn in transactions:
        by_party.setdefault(txn["origin_party_id"], []).append(txn)

    # Prior-alert history accumulates as alerts are generated, so counts are internally
    # consistent rather than sampled independently.
    history: collections.Counter[str] = collections.Counter()
    scenario_history: collections.Counter[tuple[str, str]] = collections.Counter()
    alerts: list[dict] = []

    for i in range(count):
        # The scenario is chosen first and the alert is raised on activity that actually
        # matches it. Drawing a random party and a random one of their transactions produced
        # alerts with no causal relationship to the activity that supposedly triggered them -
        # a LARGE_CASH_DEPOSIT alert could sit on a party whose largest movement was a card
        # payment. An analyst's first step is to look at the triggering activity, so an alert
        # that does not point at matching activity does not model the work.
        scenario_id, reason, rationale = rng.choice(BENIGN_ALERT_SCENARIOS)
        party, trigger = _match_scenario(rng, scenario_id, parties, by_party)
        raised_floor = (
            date.fromisoformat(trigger["value_date"]) if trigger else CORPUS_START
        )
        raised = raised_floor + timedelta(days=rng.randint(0, 3))

        alerts.append(
            {
                "alert_id": f"ALERT{start_index + i:04d}",
                "subject_party_id": party.party_id,
                "typology_id": None,
                "scenario_id": scenario_id,
                "raised_on": raised.isoformat(),
                "reason": reason,
                # Score is a function of the triggering activity, not an independent draw.
                # Real rule scores rise with the magnitude and repetition that fired them, so
                # an independently-sampled score makes the number decorative, and makes
                # ranking on it a lookup rather than an inference.
                "score": _benign_score(rng, trigger, history[party.party_id]),
                # Open, and without its answer. Every benign alert was previously
                # pre-dispositioned with the exculpatory explanation stated in plain English
                # ("Proceeds of documented property sale; completion statement on file"), so
                # the corpus contained no alert that actually required investigating. The
                # rationale is retained under a held-out key for scoring, never on the record
                # the model reads.
                "status": "open",
                "disposition": None,
                "held_out_disposition": rng.choice(DISPOSITIONS),
                "held_out_rationale": rationale,
                "triggering_transaction_id": (
                    trigger["transaction_id"] if trigger else None
                ),
                "prior_match_count_same_scenario": scenario_history[(party.party_id, scenario_id)],
                "prior_match_count_all": history[party.party_id],
                "linked_alert_ids": [],
                "escalated": False,
            }
        )
        history[party.party_id] += 1
        scenario_history[(party.party_id, scenario_id)] += 1

    return alerts


def generate_corpus(
    seed: int = 20260811,
    n_parties: int = 300,
    n_benign_transactions: int = 2000,
    n_structuring: int = 8,
    n_layering: int = 6,
    n_round_tripping: int = 5,
    n_funnel_account: int = 5,
    n_trade_based: int = 4,
    n_benign_narratives: int = 280,
) -> Corpus:
    rng = random.Random(seed)

    parties = generate_parties(rng, n_parties)
    by_id = {p.party_id: p for p in parties}

    planted = _plant_all(
        rng, parties, n_structuring, n_layering, n_round_tripping,
        n_funnel_account, n_trade_based,
    )
    planted_txns = [t for p in planted for t in p.transactions]
    instances = [p.instance for p in planted if p.instance]

    transactions = _benign_transactions(rng, parties, n_benign_transactions, start_index=1)
    transactions.extend(planted_txns)

    # Camouflage: top typology participants up to normal activity so degree carries no label.
    typology_parties = {p for inst in instances for p in inst.party_ids}
    transactions.extend(
        _camouflage_transactions(
            rng, parties, typology_parties, transactions,
            start_index=n_benign_transactions + 10_000,
        )
    )
    transactions.sort(key=lambda t: (t["value_date"], t["transaction_id"]))
    instances = _renumber_transactions(transactions, instances)

    narratives: list[dict] = []
    for i, instance in enumerate(instances):
        narratives.append(_narrative_for_typology(rng, by_id, instance, f"DOC{i:04d}"))

    # Benign narratives drawn from parties not involved in any planted typology.
    flagged = {pid for inst in instances for pid in inst.party_ids}
    clean = [p for p in parties if p.party_id not in flagged]
    rng.shuffle(clean)
    for i, party in enumerate(clean[:n_benign_narratives]):
        narratives.append(_benign_narrative(rng, party, f"DOC{len(instances) + i:04d}"))

    # Follow-up notes, so some parties are discussed across several documents.
    #
    # Cross-document identity resolution needs parties that actually recur: with one note per
    # party, recognising "the same party in two notes" is untestable, and the determinism
    # ablation has nothing to degrade. A real case file accumulates notes over time, so this
    # is also the more faithful corpus.
    followup_templates = (
        "Follow-up review for {name}. Previously reported activity remains unresolved. "
        "Contact attempted on {phone}; no response received. Account {account} remains "
        "under enhanced monitoring.",
        "Second-line review of {name}. Escalation from the initial alert stands. "
        "Correspondence sent to {email} has not been acknowledged. Recommend continued "
        "restriction pending further evidence.",
        "Quarterly reassessment of {name}, registered at {address}. Risk rating unchanged. "
        "Transaction volumes remain inconsistent with the declared business profile.",
    )

    # Skewed rather than one-per-subject: a few subjects accumulate many notes while most get
    # one. This is how real case files grow, and it is also what gives the frequency-analysis
    # attack a rank signal to exploit, without it the attack is untestable.
    followup_subjects: list[str] = []
    for rank, instance in enumerate(instances):
        # Zipf-ish: the first subject gets several follow-ups, later ones progressively fewer.
        note_count = max(1, 6 // (rank + 1))
        followup_subjects.extend([instance.subject_party_id] * note_count)

    for _i, party_id in enumerate(followup_subjects):
        party = by_id[party_id]
        template = rng.choice(followup_templates)
        narratives.append(
            {
                "document_id": f"DOC{len(narratives):04d}",
                "document_type": "followup_note",
                "subject_party_id": party_id,
                "typology_id": None,
                "text": template.format(
                    name=party.full_name,
                    phone=party.phone,
                    email=party.email,
                    address=party.address,
                    account=party.account_number,
                ),
                # Ground truth for the adversarial evaluation. Follow-up notes carry the
                # recurring entities, so omitting these would hide exactly the frequency
                # signal the attack exists to measure.
                "plaintext_entities": {
                    "PERSON": [party.full_name] if party.party_type == "individual" else [],
                    "ORGANIZATION": (
                        [party.full_name] if party.party_type == "organization" else []
                    ),
                },
            }
        )

    # One alert per planted typology, the investigator's entry point into a case.
    # Escalating alerts, sharing the schema of the benign population so triage cannot be
    # solved by spotting a structural difference between the two.
    scenario_for = {
        "structuring": ("STRUCTURING_THRESHOLD", "Multiple deposits below reporting threshold"),
        "layering": ("RAPID_MOVEMENT", "Funds withdrawn shortly after deposit"),
        "round_tripping": ("CIRCULAR_FLOW", "Funds returned to originating party via intermediaries"),
    }

    true_alerts: list[dict] = []
    date_of_txn = {t["transaction_id"]: t["value_date"] for t in transactions}
    for i, inst in enumerate(instances):
        scenario_id, reason = scenario_for.get(
            inst.typology, ("UNUSUAL_PATTERN", "Automated transaction monitoring threshold breach")
        )
        true_alerts.append(
            {
                "alert_id": f"ALERT{i:04d}",
                "subject_party_id": inst.subject_party_id,
                "typology_id": inst.typology_id,
                "scenario_id": scenario_id,
                # Raised a few days after the typology's activity completes, the same
                # causal rule benign alerts already follow. A random date produced alerts
                # raised up to 208 days *before* their own activity finished (54 of 122), and
                # under a temporal evaluation split that decorative date becomes load-bearing:
                # an alert whose evidence sits in the training window was being ranked by a
                # model that only scores the test window, dragging queue precision to 0.10 for
                # a reason that had nothing to do with the model. Monitoring fires after the
                # pattern it fires on; 1-5 days is a realistic batch-detection lag.
                "raised_on": (
                    max(
                        date.fromisoformat(date_of_txn[txn_id])
                        for txn_id in inst.transaction_ids
                        if txn_id in date_of_txn
                    )
                    + timedelta(days=rng.randint(1, 5))
                ).isoformat(),
                "reason": reason,
                # Overlapping the benign range deliberately: real scores do not separate
                # cleanly, and a corpus where they do would make triage trivial. The overlap
                # is partial rather than total, an escalating alert is usually, but not
                # always, scored above a benign one, so ranking is learnable without being
                # a lookup.
                "score": rng.randint(68, 95),
                "status": "open",
                # Never "SAR filed". 31 CFR 1020.320(e) makes both a SAR and *the fact of its
                # existence* confidential, so a field recording it is the one thing that must
                # not be indexed, embedded, or emitted to a model identifies the
                # rule and the corpus previously violated it. The underlying facts are not
                # privileged, which is what makes an investigative copilot lawful at all; the
                # filing decision is. "Escalated to investigation" is the disposition an
                # analyst records before that line is crossed.
                "disposition": "escalated_to_investigation",
                "disposition_rationale": inst.narrative_hint,
                "triggering_transaction_id": inst.transaction_ids[0],
                # Repeat alerting is the real escalation driver, a third alert on one subject
                # is a different object from a first, so escalating alerts carry heavier
                # history than the benign population, which tops out at a handful.
                "prior_match_count_same_scenario": rng.randint(2, 5),
                "prior_match_count_all": rng.randint(4, 11),
                "linked_alert_ids": [],
                "escalated": True,
            }
        )

    # Benign alerts sized to a realistic conversion rate. BPI measured ~4% across 19
    # institutions, so 19 true positives imply roughly 475 alerts in total.
    benign_count = max(0, round(len(instances) / TARGET_SAR_CONVERSION_RATE) - len(instances))
    alerts = true_alerts + _benign_alerts(
        rng, parties, transactions, benign_count, start_index=len(true_alerts)
    )
    alerts.sort(key=lambda a: (a["raised_on"], a["alert_id"]))

    # Link alerts sharing a subject, which is what drives real escalation decisions.
    by_subject: dict[str, list[dict]] = {}
    for alert in alerts:
        by_subject.setdefault(alert["subject_party_id"], []).append(alert)
    for subject_alerts in by_subject.values():
        if len(subject_alerts) < 2:
            continue
        for alert in subject_alerts:
            alert["linked_alert_ids"] = [
                other["alert_id"] for other in subject_alerts if other is not alert
            ]

    return Corpus(
        parties=parties,
        transactions=transactions,
        narratives=narratives,
        typologies=instances,
        alerts=alerts,
    )


def write_corpus(corpus: Corpus, out_dir: Path) -> dict[str, Path]:
    """Write the corpus as JSON. Ground truth is written separately and must never be
    ingested into the retrieval index, it is the answer key, not evidence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "parties": out_dir / "parties.json",
        "transactions": out_dir / "transactions.json",
        "narratives": out_dir / "narratives.json",
        "alerts": out_dir / "alerts.json",
        "ground_truth": out_dir / "ground_truth.json",
    }

    paths["parties"].write_text(json.dumps([p.to_dict() for p in corpus.parties], indent=2))
    paths["transactions"].write_text(json.dumps(corpus.transactions, indent=2))
    paths["narratives"].write_text(json.dumps(corpus.narratives, indent=2))
    paths["alerts"].write_text(json.dumps(corpus.alerts, indent=2))
    paths["ground_truth"].write_text(
        json.dumps([t.as_ground_truth() for t in corpus.typologies], indent=2)
    )
    return paths
