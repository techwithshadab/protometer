"""AML typologies, the planted patterns that give the evaluation its ground truth.

Every suspicious pattern in the corpus is generated deliberately, and every generator
returns both the transactions it created and a `TypologyInstance` recording exactly which
parties and transactions constitute it. That record *is* the answer key: an evaluation
checkpoint asking "which accounts show structuring?" has an exact expected set, not a
judged one.

The three typologies are chosen because they stress different reasoning under protection:

  STRUCTURING, amount reasoning near a reporting threshold. Survives tokenization of
                  names; collapses if AMOUNT is protected. Isolates the quasi-identifier
                  boundary.
  LAYERING, multi-hop chains through intermediaries. Depends entirely on stable
                  cross-document tokens, so it is the typology the determinism ablation
                  should break.
  ROUND_TRIPPING- value returning to origin via a circuit. Requires recognising the same
                  party at both ends, the sharpest test of deterministic tokenization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum

from protometer.corpus.vocabulary import channel_for, invoice_reference, memo_for


class Typology(str, Enum):
    STRUCTURING = "structuring"
    LAYERING = "layering"
    ROUND_TRIPPING = "round_tripping"
    # Both are first-class FinCEN SAR subtypes rather than inventions: Money Laundering
    # category 36 assigns code 824 to "Funnel account" and 823 to "Trade Based Money
    # Laundering/Black Market Peso Exchange" (FinCEN SAR XML Schema User Guide v1.6, 2021).
    FUNNEL_ACCOUNT = "funnel_account"
    TRADE_BASED = "trade_based"


# FinCEN SAR subtype codes, so a planted typology maps to what an institution would actually
# file rather than to a label invented for this corpus. All five codes are from one verified
# source, the FinCEN SAR XML Schema User Guide v1.6 (2021), which enumerates the checkbox codes
# of FinCEN Report 111: Structuring is category 32 and Money Laundering is category 36. An
# earlier version used a bare "114"/"1999", which are not codes in that schema; corrected here.
SAR_SUBTYPE_CODES: dict[str, str] = {
    # Category 32, Structuring: 32c "Transaction(s) below CTR threshold" is the defining box.
    "structuring": "32c",
    # Category 36, Money Laundering: 36z "Other" (no dedicated layering/round-tripping box).
    "layering": "36z",
    "round_tripping": "36z",
    "funnel_account": "824",    # 36-series "Funnel account"
    "trade_based": "823",       # 36-series "Trade Based Money Laundering / Black Market Peso"
}


# US Bank Secrecy Act currency transaction reporting threshold. Structuring is defined by
# deliberately staying under it, so the number is load-bearing rather than decorative.
CTR_THRESHOLD = Decimal("10000")


@dataclass(frozen=True)
class TypologyInstance:
    """Ground truth for one planted pattern.

    `party_ids` and `transaction_ids` are the expected answers for evaluation checkpoints.
    They are recorded at generation time precisely so that no post-hoc detection step is
    needed to know what the right answer is.
    """

    typology_id: str
    typology: Typology
    party_ids: tuple[str, ...]
    transaction_ids: tuple[str, ...]
    # The party an analyst would name as the subject of a SAR, origin of the funds.
    subject_party_id: str
    total_amount: Decimal
    narrative_hint: str

    def as_ground_truth(self) -> dict:
        return {
            "typology_id": self.typology_id,
            "typology": self.typology.value,
            "subject_party_id": self.subject_party_id,
            "party_ids": sorted(self.party_ids),
            "transaction_ids": sorted(self.transaction_ids),
            "total_amount": str(self.total_amount),
        }


@dataclass
class PlantedTransactions:
    """A typology's output: the transactions plus the answer key that describes them."""

    transactions: list[dict] = field(default_factory=list)
    instance: TypologyInstance | None = None


def _txn(
    txn_id: str,
    origin: str,
    beneficiary: str,
    amount: Decimal,
    when: date,
    channel: str,
    memo: str,
) -> dict:
    return {
        "transaction_id": txn_id,
        "origin_party_id": origin,
        "beneficiary_party_id": beneficiary,
        "amount": str(amount),
        "currency": "USD",
        "value_date": when.isoformat(),
        "channel": channel,
        "memo": memo,
    }


def plant_structuring(
    rng: random.Random,
    typology_id: str,
    subject_id: str,
    beneficiary_id: str,
    start: date,
) -> PlantedTransactions:
    """Many deposits deliberately just under the CTR threshold, over a short window.

    The signal is statistical, not individual: no single transaction is remarkable, and the
    pattern only exists in aggregate. That makes AMOUNT the load-bearing field, which is
    exactly why this typology should degrade sharply once quasi-identifiers are protected.
    """
    count = rng.randint(6, 11)
    txns: list[dict] = []
    total = Decimal("0")

    for i in range(count):
        # Just under threshold, varied enough not to look scripted.
        amount = CTR_THRESHOLD - Decimal(rng.randrange(50, 1200))
        when = start + timedelta(days=rng.randint(0, 18))
        txn = _txn(
            f"{typology_id}-T{i:02d}",
            subject_id,
            beneficiary_id,
            amount,
            when,
            channel_for(rng, "deposit"),
            memo_for(rng, "deposit"),
        )
        txns.append(txn)
        total += amount

    return PlantedTransactions(
        transactions=txns,
        instance=TypologyInstance(
            typology_id=typology_id,
            typology=Typology.STRUCTURING,
            party_ids=(subject_id, beneficiary_id),
            transaction_ids=tuple(t["transaction_id"] for t in txns),
            subject_party_id=subject_id,
            total_amount=total,
            narrative_hint=(
                f"{count} cash deposits totalling ${total:,.2f}, each below the "
                f"${CTR_THRESHOLD:,.0f} reporting threshold, within a three-week window"
            ),
        ),
    )


def plant_layering(
    rng: random.Random,
    typology_id: str,
    chain_party_ids: list[str],
    start: date,
) -> PlantedTransactions:
    """Funds moved through a chain of intermediaries, shedding value at each hop.

    Detecting this requires recognising that the beneficiary of hop N is the originator of
    hop N+1, across separate records. Under tokenization that recognition depends entirely
    on the same party yielding the same token every time, so this typology is the one the
    determinism ablation is expected to destroy.
    """
    assert len(chain_party_ids) >= 3, "layering needs at least three parties"

    # Drawn to overlap the benign amount distribution (p10 1.2k, p50 12k, p90 132k) rather
    # than always being large. Planted chains starting at 180k+ made "large transaction"
    # nearly equivalent to "suspicious": every positive sat above the benign minimum, so a
    # single amount threshold separated the classes.
    # Capped at the benign maximum. Chains starting above it left 78 positives beyond
    # anything benign traffic could produce, so "amount exceeds the benign ceiling" was a
    # pure, if low-recall, separator, the same defect as the memo and cents leaks.
    amount = Decimal(rng.choice((
        rng.randrange(3_000, 40_000, 500),
        rng.randrange(20_000, 150_000, 1_000),
        rng.randrange(100_000, 179_000, 1_000),
    )))
    origin_amount = amount
    txns: list[dict] = []
    when = start

    for hop, (origin, beneficiary) in enumerate(
        zip(chain_party_ids, chain_party_ids[1:])
    ):
        # Each hop skims a commission, the decreasing amounts are themselves a signal.
        amount = (amount * Decimal(rng.randrange(88, 97)) / Decimal(100)).quantize(Decimal("0.01"))
        when = when + timedelta(days=rng.randint(1, 6))
        txns.append(
            _txn(
                f"{typology_id}-H{hop:02d}",
                origin,
                beneficiary,
                amount,
                when,
                channel_for(rng, "wire"),
                memo_for(rng, "wire"),
            )
        )

    return PlantedTransactions(
        transactions=txns,
        instance=TypologyInstance(
            typology_id=typology_id,
            typology=Typology.LAYERING,
            party_ids=tuple(chain_party_ids),
            transaction_ids=tuple(t["transaction_id"] for t in txns),
            subject_party_id=chain_party_ids[0],
            total_amount=origin_amount,
            narrative_hint=(
                f"${origin_amount:,.2f} moved through {len(chain_party_ids)} parties in "
                f"{len(txns)} hops, each hop shedding a commission"
            ),
        ),
    )


def plant_funnel_account(
    rng: random.Random,
    typology_id: str,
    depositor_ids: list[str],
    funnel_id: str,
    withdrawal_id: str,
    start: date,
    depositor_cities: list[str] | None = None,
    funnel_city: str | None = None,
) -> PlantedTransactions:
    """Sub-threshold cash deposits from many geographies into one account, withdrawn elsewhere.

    Signature taken from FinCEN advisory FIN-2014-A005: a single account receives multiple cash
    deposits below the reporting threshold **in a different geographic area from where it is
    domiciled**, and the funds are withdrawn elsewhere with **little time elapsing between
    deposits and withdrawals**.

    The geographic-dispersion half of that signature is realized by the caller, which draws
    depositors whose city differs from the funnel's domicile, and recorded here (distinct
    depositor cities vs the funnel city) so a detector can be credited for it rather than the
    ground truth capturing only the temporal convergence. `depositor_cities`/`funnel_city` are
    optional so the function still runs for callers that do not supply geography.

    Distinct from structuring in shape, which is why it is worth planting separately: structuring
    is one depositor splitting one sum, while a funnel account is *many* depositors converging on
    one collector. A detector tuned to per-party deposit counts sees nothing.
    """
    assert len(depositor_ids) >= 3, "a funnel needs several depositors to be a funnel"

    txns: list[dict] = []
    total = Decimal("0")
    when = start

    for i, depositor in enumerate(depositor_ids):
        # Each depositor makes one or two deposits, all below the threshold.
        for j in range(rng.randint(1, 2)):
            amount = CTR_THRESHOLD - Decimal(rng.randrange(200, 2500))
            when = start + timedelta(days=rng.randint(0, 9))
            txns.append(
                _txn(
                    f"{typology_id}-D{i:02d}{j}",
                    depositor,
                    funnel_id,
                    amount,
                    when,
                    channel_for(rng, "deposit"),
                    memo_for(rng, "deposit"),
                )
            )
            total += amount

    # Withdrawal follows quickly, the defining temporal signature.
    #
    # Anchored to the *latest* deposit rather than to the loop variable. Deposit dates are
    # drawn independently within the window, so the last one generated is not the last one
    # chronologically, and using the loop variable placed the sweep before deposits it was
    # supposed to sweep, which made the pattern undetectable by its own signature.
    withdrawal_date = max(
        date.fromisoformat(t["value_date"]) for t in txns
    ) + timedelta(days=rng.randint(1, 4))
    withdrawn = (total * Decimal(rng.randrange(90, 98)) / Decimal(100)).quantize(Decimal("0.01"))
    txns.append(
        _txn(
            f"{typology_id}-W00",
            funnel_id,
            withdrawal_id,
            withdrawn,
            withdrawal_date,
            channel_for(rng, "wire"),
            memo_for(rng, "wire"),
        )
    )

    return PlantedTransactions(
        transactions=txns,
        instance=TypologyInstance(
            typology_id=typology_id,
            typology=Typology.FUNNEL_ACCOUNT,
            party_ids=(*depositor_ids, funnel_id, withdrawal_id),
            transaction_ids=tuple(t["transaction_id"] for t in txns),
            subject_party_id=funnel_id,
            total_amount=total,
            narrative_hint=(
                f"{len(txns) - 1} cash deposits from {len(depositor_ids)} separate depositors "
                f"into a single account totalling ${total:,.2f}, each below the "
                f"${CTR_THRESHOLD:,.0f} threshold, withdrawn within days"
                + (
                    f", deposited across {len({c for c in depositor_cities if c != funnel_city})}"
                    f" cities away from the account's domicile of {funnel_city}"
                    if depositor_cities and funnel_city
                    else ""
                )
            ),
        ),
    )


def plant_trade_based(
    rng: random.Random,
    typology_id: str,
    importer_id: str,
    exporter_id: str,
    intermediary_ids: list[str],
    start: date,
) -> PlantedTransactions:
    """Over-invoiced trade payments routed through intermediaries.

    Signature from FATF/Egmont, *Trade-Based Money Laundering: Risk Indicators* (2021): payments
    materially exceeding plausible commodity value, settled through third parties unrelated to
    the trade, with invoice references that repeat across supposedly distinct shipments.

    The detectable pattern in a ledger is **repeated large round-value payments against the same
    invoice reference, paid to parties other than the counterparty named on the trade**, value
    moving under cover of commerce rather than the tight temporal chain of layering.
    """
    assert intermediary_ids, "trade-based laundering needs a third-party settler"

    # Drawn from the shared vocabulary so the reference is byte-identical in shape to the ones
    # benign commercial traffic carries. The FATF indicator is the *repetition of one
    # reference across supposedly distinct shipments*, not the presence of a reference, but
    # while `INV-` appeared only here, the five-character prefix was a perfect classifier for
    # this typology (73 positive, 0 negative). The pattern survives; the token no longer
    # identifies it.
    invoice = invoice_reference(rng)
    txns: list[dict] = []
    total = Decimal("0")
    when = start

    for i in range(rng.randint(3, 5)):
        # Round values are themselves a FATF indicator.
        # Round values are a FATF indicator, but the *magnitude* should not be one, spread
        # across the benign range so size alone carries no signal.
        # Top band capped below the benign ceiling (180k), as in layering and round-tripping:
        # 28 trade-based payments previously sat above anything benign traffic could produce.
        amount = Decimal(rng.choice((rng.randrange(5, 40), rng.randrange(30, 120),
                                     rng.randrange(100, 179))) * 1000)
        when = when + timedelta(days=rng.randint(4, 21))
        beneficiary = rng.choice([exporter_id, *intermediary_ids])
        txns.append(
            _txn(
                f"{typology_id}-T{i:02d}",
                importer_id,
                beneficiary,
                amount,
                when,
                channel_for(rng, "wire"),
                # The repeated reference across shipments is the giveaway.
                f"payment {invoice}",
            )
        )
        total += amount

    return PlantedTransactions(
        transactions=txns,
        instance=TypologyInstance(
            typology_id=typology_id,
            typology=Typology.TRADE_BASED,
            party_ids=(importer_id, exporter_id, *intermediary_ids),
            transaction_ids=tuple(t["transaction_id"] for t in txns),
            subject_party_id=importer_id,
            total_amount=total,
            narrative_hint=(
                f"{len(txns)} round-value payments totalling ${total:,.2f} against a single "
                f"invoice reference ({invoice}), settled partly to third parties unrelated to "
                f"the underlying trade"
            ),
        ),
    )


def plant_round_tripping(
    rng: random.Random,
    typology_id: str,
    circuit_party_ids: list[str],
    start: date,
) -> PlantedTransactions:
    """Value leaves a party and returns to it through a circuit of intermediaries.

    The defining feature is that the first originator and the final beneficiary are the
    same party. An analyst must resolve identity across the whole circuit to see it, which
    makes it the sharpest test of deterministic tokenization in the corpus.
    """
    assert len(circuit_party_ids) >= 3, "round-tripping needs at least three parties"

    subject = circuit_party_ids[0]
    circuit = [*circuit_party_ids, subject]  # closes the loop back to the subject
    # Same reasoning as layering: overlap the benign distribution so magnitude is not a label.
    # Capped at the benign maximum, as in layering above.
    amount = Decimal(rng.choice((
        rng.randrange(5_000, 50_000, 500),
        rng.randrange(30_000, 179_000, 1_000),
        rng.randrange(100_000, 179_000, 1_000),
    )))
    origin_amount = amount
    txns: list[dict] = []
    when = start

    for hop, (origin, beneficiary) in enumerate(zip(circuit, circuit[1:])):
        amount = (amount * Decimal(rng.randrange(92, 99)) / Decimal(100)).quantize(Decimal("0.01"))
        when = when + timedelta(days=rng.randint(2, 11))
        txns.append(
            _txn(
                f"{typology_id}-R{hop:02d}",
                origin,
                beneficiary,
                amount,
                when,
                channel_for(rng, "wire"),
                memo_for(rng, "wire"),
            )
        )

    return PlantedTransactions(
        transactions=txns,
        instance=TypologyInstance(
            typology_id=typology_id,
            typology=Typology.ROUND_TRIPPING,
            party_ids=tuple(circuit_party_ids),
            transaction_ids=tuple(t["transaction_id"] for t in txns),
            subject_party_id=subject,
            total_amount=origin_amount,
            narrative_hint=(
                f"${origin_amount:,.2f} left the subject and returned via "
                f"{len(circuit_party_ids) - 1} intermediaries over {len(txns)} transfers"
            ),
        ),
    )
