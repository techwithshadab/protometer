"""Shared ledger vocabulary, the fields a typology must *not* be identifiable by.

Every transaction in the corpus, planted or benign, draws its `memo` and `channel` from the
vocabularies defined here. That is the point of the module: when the two generation paths
own their own word lists, they drift apart, and the drift becomes the label.

## Why this exists

Three separate leaks were found in the generated corpus, all the same defect wearing
different clothes:

  * `channel`, deposit channels appeared only in planted typologies, so a classifier reached
    AP 0.998 by learning "deposit implies suspicious". Fixed by adding deposits to
    the benign vocabulary, but only the *deposit* family. `swift`, `correspondent` and
    `intra_group` stayed positive-only: 193 transactions, zero benign.
  * `memo`, the five typology generators drew from five private word lists and benign drew
    from a sixth. The vocabularies were disjoint except for the empty string, so a rule of the
    form *"is this memo absent from the benign vocabulary?"* scored **TP 640, FP 0** at recall
    0.840 with no model and no learning. That single rule reproduced the reported
    `precision@50 = 1.000` at every protection scope.
  * `INV-` references, emitted only by the trade-based generator, while the benign memo was
    `invoice 4471` with no hyphen. `memo_has_invoice_ref` was a pure separator: 73 positive,
    0 negative.

The reason all three survived the single-feature-AUC sweep is that each is a
**pure but low-recall** indicator. High precision, modest AUC. A marginal-AUC scan is blind
to exactly this shape, which is why `tests/test_invariants.py` now screens on *precision
purity* as well.

## The rule this module enforces

A typology must be detectable by its **pattern**, repetition, timing, topology, the
relationship between transactions, and never by the presence of a token that benign traffic
cannot produce. Both paths draw from the same pools here, so any future vocabulary added to
one is available to the other by construction.

Memos and channels are correlated with each other (a cash deposit does not carry an
`intercompany loan` memo), because uncorrelated draws are their own giveaway: a wire memo on
an ATM deposit is a signature no real ledger produces.
"""

from __future__ import annotations

import random

# Channel families, each with the memos that plausibly accompany it. Real ledgers correlate
# the two, so drawing them independently would itself be a generator artifact.
#
# Every family is available to *both* benign and planted transactions. A typology selects the
# family its mechanism requires, structuring genuinely is cash deposits, layering genuinely
# is wires, and benign traffic populates all of them, so the family narrows the candidate set
# without ever determining the label.
CHANNEL_FAMILIES: dict[str, tuple[str, ...]] = {
    "deposit": ("cash_deposit", "branch_deposit", "atm_deposit"),
    "wire": ("wire", "swift", "correspondent", "intra_group"),
    "retail": ("ach", "card", "direct_debit", "standing_order"),
}

# Memos by channel family. Deliberately overlapping across families where a phrase is
# genuinely generic ("invoice settlement" appears on both wires and retail payments), because
# an exclusive phrase is a label by another name.
MEMOS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "deposit": (
        "deposit", "cash deposit", "counter deposit", "cash", "takings",
        "daily receipts", "till float", "branch credit", "",
    ),
    "wire": (
        "consulting fee", "invoice settlement", "advisory services", "management fee",
        "trade settlement", "supplier settlement", "goods payment", "invoice",
        "intercompany loan", "capital injection", "loan repayment", "equity subscription",
        "shareholder loan", "professional fees", "contract milestone", "freight charges",
        "monthly transfer", "",
    ),
    "retail": (
        "payroll", "rent", "supplier payment", "monthly transfer", "insurance premium",
        "utility payment", "subscription renewal", "refund", "invoice", "professional fees",
        "contract milestone", "",
    ),
}


def invoice_reference(rng: random.Random) -> str:
    """A structured invoice reference, in the format both paths use.

    Exposed as a function rather than inlined so benign traffic and the trade-based typology
    produce byte-identical shapes. `INV-` previously appeared only in planted transactions
    while benign memos said `invoice 4471`, which made a five-character substring a perfect
    classifier for one entire typology class.
    """
    return f"INV-{rng.randrange(10000, 99999)}"


def memo_for(rng: random.Random, family: str, invoice_rate: float = 0.18) -> str:
    """Draw a memo appropriate to `family`, sometimes carrying an invoice reference.

    `invoice_rate` applies to **both** generation paths. Commercial payment traffic carries
    references at a high rate in reality; what matters for the measurement is only that the
    rate is the same on both sides of the label, so the reference narrows nothing.
    """
    if family in ("wire", "retail") and rng.random() < invoice_rate:
        return f"payment {invoice_reference(rng)}"
    return rng.choice(MEMOS_BY_FAMILY[family])


def channel_for(rng: random.Random, family: str) -> str:
    """Draw a channel from one family."""
    return rng.choice(CHANNEL_FAMILIES[family])


def benign_family(rng: random.Random) -> str:
    """Pick a channel family for background traffic.

    Weighted toward retail and wire, with a substantial deposit share. The deposit share is
    load-bearing: it is what stops "cash deposit" from implying "structuring", and it was the
    fix for the first of the three vocabulary leaks.
    """
    return rng.choices(("retail", "wire", "deposit"), weights=(0.45, 0.35, 0.20), k=1)[0]
