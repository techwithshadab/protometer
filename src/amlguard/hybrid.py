"""Hybrid triage, the classifier ranks, the model reasons about what reaches an analyst.

The obvious justification for a hybrid is cost: run the cheap model, escalate to the expensive
one only when uncertain. On this system that argument is weak. The classifier makes no API calls
at all, reaches high precision at the review depths an analyst actually works, and the LLM's
measured contribution over a copy-the-detector baseline is confined to task types the detector
does not cover. Cost is not the binding constraint.

**The real justification is regulatory.** FFIEC requires a designated decision maker able to
justify a filing; supporting documentation must be reconstructable for five years; and examiners
judge *process, not individual calls*. A score is not a justification. The hybrid's
job is therefore to produce **defensible reasoning where a decision needs one**, not to save
money, which is also the deployment shape the research identified as having the lowest
regulatory friction.

**Why ranking rather than a score threshold.** The natural design is "escalate when the score
crosses 0.6", and it is unsound here for a reason that does not depend on the score being badly
calibrated (measured ECE is a low **0.021**): review capacity is a headcount, so the queue is
worked top-down to whatever depth the team can reach. A fixed threshold either floods a small
team or idles a large one, while rank adapts to capacity and is the quantity an examiner can be
shown directly. Calibration good enough to threshold on is not the same as thresholding being
the right operational design.

Rank sidesteps this entirely, and matches how AML review actually works: capacity is a headcount,
so the queue is worked top-down to whatever depth the team can reach. `review_capacity` is that
depth, and it is a staffing parameter rather than a tuned hyperparameter.

**What must not be built.** FFIEC examination procedure #12 states alert volume "should not be
tailored solely to meet existing staffing levels", and TD Bank was penalised for rejecting
scenarios because they would raise volume. So this component **orders** a queue and never
suppresses it: every alert below the capacity line remains open and reviewable, and nothing here
closes an alert. The LLM assists the analyst who reviews; it does not decide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np


@dataclass
class TriageDecision:
    """One item's place in the queue, and whether it warrants model-assisted review."""

    item_id: str
    score: float
    rank: int
    # Above the capacity line: an analyst will review this, so a rationale is worth generating.
    escalated: bool
    rationale: str = ""
    attributions: list[dict[str, Any]] = field(default_factory=list)
    # Figures/party-ids the rationale asserts without basis in its evidence, persisted so an
    # auditor can see which assertions are the model's own, not the record's.
    ungrounded: list[str] = field(default_factory=list)
    # The human's half of the record. The system orders and explains; the analyst decides -
    # and without a field for that decision there is no override rate, no feedback loop, and
    # no evidence a human ever looked. Written by whatever review surface fronts this queue;
    # empty means "not yet reviewed", which is itself information.
    reviewed_by: str = ""
    analyst_disposition: str = ""   # e.g. escalate / close_no_action / needs_more_information
    analyst_note: str = ""
    # Model, verbatim prompt, raw completion, timestamp, set by generate_rationale.
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": round(self.score, 4),
            "rank": self.rank,
            "escalated": self.escalated,
            "rationale": self.rationale,
            "attributions": self.attributions[:5],
            "ungrounded": self.ungrounded,
            "reviewed_by": self.reviewed_by,
            "analyst_disposition": self.analyst_disposition,
            "analyst_note": self.analyst_note,
            "provenance": self.provenance,
        }


@dataclass
class HybridResult:
    """A ranked queue plus the cost of reasoning over its head."""

    decisions: list[TriageDecision] = field(default_factory=list)
    review_capacity: int = 0
    # Populated by the caller that owns the LLM client (`scripts/run_hybrid.py`), because
    # ranking is deliberately model-free: `rank_queue` never makes a call, so this module has
    # no cost to report. They stay on the result object rather than in the script so the
    # published artifact carries the queue and its cost together.
    llm_calls: int = 0
    llm_cost_usd: float = 0.0
    # Egress-scan outcomes over the escalated head, populated by the caller for the same
    # reason as the cost fields. Present on BOTH grains' result types: the transaction-grain
    # result lacking them meant a `hasattr` guard silently dropped real block counts and
    # MLflow recorded 0 while the console printed the truth.
    egress_blocked: int = 0
    egress_discounted: int = 0

    @property
    def escalated(self) -> list[TriageDecision]:
        return [d for d in self.decisions if d.escalated]

    def precision_at_capacity(self, labels: dict[str, int]) -> float:
        """Share of the reviewed head that is genuinely suspicious.

        The operational metric: of the items an analyst actually opens, how many were worth
        opening. Everything below the line is unreviewed, not dismissed.
        """
        head = self.escalated
        if not head:
            return 0.0
        return sum(labels.get(d.item_id, 0) for d in head) / len(head)

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_capacity": self.review_capacity,
            "queue_length": len(self.decisions),
            "llm_calls": self.llm_calls,
            "llm_cost_usd": round(self.llm_cost_usd, 4),
            "egress_blocked": self.egress_blocked,
            "egress_discounted": self.egress_discounted,
            "decisions": [d.to_dict() for d in self.decisions],
        }


def rank_queue(
    item_ids: list[str], scores: np.ndarray, review_capacity: int
) -> list[TriageDecision]:
    """Order the queue by score and mark the head an analyst will reach.

    Ties break by item id so the ordering is deterministic, an arbitrary tiebreak would make
    the queue irreproducible, and reconstructability is a five-year obligation.
    """
    order = sorted(
        range(len(item_ids)), key=lambda i: (-float(scores[i]), item_ids[i])
    )
    return [
        TriageDecision(
            item_id=item_ids[position],
            score=float(scores[position]),
            rank=rank + 1,
            escalated=rank < review_capacity,
        )
        for rank, position in enumerate(order)
    ]



# Plain-language meaning of every classifier feature, with its direction.
#
# This table exists because its absence produced false case notes. The prompt used to pass raw
# internal names, `b_gw_mean_length = 4.571`, and the model guessed at the letters: `gw_`
# became "gatekeeper" in some rationales and "geographic watchlist" in others, and one shipped
# case note directed the analyst to verify sanctions-watchlist matches in a system that has no
# watchlist. 19 of 50 rationales in the queue invented a control that does not exist.
#
# Direction matters as much as meaning: `gw_mean_length` is HOPS TO a known-illicit party, with
# an unreached sentinel of 11.0, so high is CLEAN, the opposite of what a model assumes about
# a "risk feature". Each entry states what a high value indicates so the narration cannot
# invert it.
FEATURE_GLOSSARY: dict[str, str] = {
    "amount": "transaction amount in dollars",
    "amount_below_threshold": "1 if the amount sits just under the $10,000 cash-reporting threshold",
    "amount_log": "log-scaled transaction amount",
    "day_index": "when the transaction happened (day number)",
    "is_deposit_channel": "1 if this is a cash/branch/ATM deposit",
    "is_wire_channel": "1 if this is a wire/SWIFT/correspondent transfer",
    "origin_out_degree": "how many payments the sending party has made (higher = more active sender)",
    "beneficiary_in_degree": "how many payments the receiving party has received",
    "counterparty_shared_count": "times this exact sender-receiver pair has transacted",
    "memo_has_invoice_ref": "1 if the memo carries an invoice reference",
    "out_degree": "payments sent by this party",
    "in_degree": "payments received by this party",
    "succ_out_mean": "average payment count sent by the parties this party pays (a degree count, not dollars)",
    "succ_out_min": "least-active party this party pays",
    "succ_out_max": "most-active party this party pays",
    "succ_in_mean": "average payment count received by the parties this party pays (a degree count, not dollars)",
    "succ_in_min": "lowest payment count among parties this party pays",
    "succ_in_max": "highest payment count among parties this party pays",
    "pred_out_mean": "average payment count sent by the parties paying this party (a degree count, not dollars)",
    "pred_out_min": "lowest payment count among parties paying this party",
    "pred_out_max": "highest payment count among parties paying this party",
    "pred_in_mean": "average payment count received by the parties paying this party (a degree count, not dollars)",
    "pred_in_min": "least-active party paying this party",
    "pred_in_max": "most-active party paying this party",
    "gw_hit_rate": (
        "fraction of random walks from this party that reach a KNOWN-ILLICIT party "
        "(higher = closer to known illicit activity)"
    ),
    "gw_mean_length": (
        "average hops for a walk to reach a known-illicit party; 11.0 means NEVER reached, "
        "and also encodes a party absent from the model's training history "
        "(HIGHER = CLEANER, lower = closer to known illicit activity)"
    ),
    "gw_min_length": (
        "shortest path a walk took to a known-illicit party; 11.0 means never reached "
        "(HIGHER = CLEANER)"
    ),
    "gw_distinct_illicit": "how many distinct known-illicit parties walks reached (higher = more exposure)",
    "cycle_count": "number of payment cycles (length <= 6) this party sits on, circular fund flow",
    "min_cycle_length": "shortest payment cycle through this party",
    "scatter_width": "distinct recipients this party fans funds out to",
    "gather_width": "distinct senders converging funds into this party",
    "pagerank": "network importance of this party in the payment graph",
    "kcore": "how densely interconnected this party's neighbourhood is",
    "clustering": "how much this party's counterparties also transact with each other",
    "betweenness": "how often this party sits on paths between other parties (intermediary role)",
}

_SIDE = {"o": "sending party", "b": "receiving party"}


def describe_feature(name: str) -> str:
    """Resolve a raw feature name into words an analyst, and the LLM, can use safely."""
    side = ""
    base = name
    if len(name) > 2 and name[1] == "_" and name[0] in _SIDE:
        side = f"{_SIDE[name[0]]}: "
        base = name[2:]
    meaning = FEATURE_GLOSSARY.get(base)
    return f"{side}{meaning}" if meaning else name


# RATIONALE_SYSTEM moved to config/prompts/amlguard-rationale-system.txt (loaded via observability.managed_prompt).




def _extract_amounts(text: str) -> list[tuple[str, Decimal]]:
    """(raw, value) for every numeric assertion in `text`, K/M/bn suffixes expanded.

    The previous tokenizer read `$707k` as `707`, then flagged it as ungrounded against an
    evidence value of 707078.56, every one of the quasi run's four flags was this false
    positive. Meanwhile `10k` became `10`, fell under a smallness floor, and the model's
    genuinely invented "$10k reporting threshold" passed unflagged in five shipped rationales.
    The suffix is part of the number; parsing it makes both failures impossible and removes
    the need for the floor that created the blind spot.
    """
    from decimal import InvalidOperation

    out: list[tuple[str, Decimal]] = []
    # IGNORECASE: the letter suffixes already matched both cases via character class, but
    # the word suffixes did not, "5 Million" extracted as bare `5`, fell under the prose
    # floor, and an invented figure passed the gate that "5 million" would have caught.
    for match in re.finditer(
        r"\d[\d,]*(?:\.\d+)?\s*(?:[km]\b|bn\b|million\b|thousand\b|billion\b)?",
        text,
        re.IGNORECASE,
    ):
        raw = match.group(0).strip()
        try:
            digits = re.match(r"[\d,\.]+", raw).group(0)
            value = Decimal(digits.replace(",", ""))
        except (InvalidOperation, AttributeError):
            continue
        suffix = raw[len(digits):].strip().lower()
        scale = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000,
                 "bn": 1_000_000_000, "billion": 1_000_000_000}.get(suffix, 1)
        out.append((raw, value * scale))
    return out


# Integer counts this small are prose, not claims, "two sentences", "one of three
# intermediaries". Applied only to UNSUFFIXED integers: `10k` expands to 10,000 and is
# checked in full, which closes the blind spot the old token-level floor created.
_GROUNDING_FLOOR = 12


def ungrounded_terms(rationale: str, evidence: str) -> list[str]:
    """Figures and party ids the rationale asserts but the evidence does not contain.

    **Scope, stated plainly: this checks figures and party identifiers only.** A reversed
    payment direction, a wrong date written in prose, or a qualitative fabrication
    ("previously convicted", "sanctioned entity") passes this gate, those classes are
    unguarded, and the limitations section says so. What it does check, it now checks with
    magnitude awareness: `$707k` grounds against `707078.56`, `98%` against `0.9821`, and an
    invented `$10k threshold` is flagged because the suffix expands before comparison.

    Deliberately a **marker, not a blocker**, FFIEC order-never-suppress applies to our own
    guards too. Terms are persisted per decision for the analyst and the auditor.
    """
    evidence_amounts = [value for _, value in _extract_amounts(evidence)]
    flagged: list[str] = []

    def grounds(cited: "Decimal") -> bool:
        exponent = cited.as_tuple().exponent
        places = -exponent if isinstance(exponent, int) and exponent < 0 else 0
        quantum = Decimal(1).scaleb(-places)
        for value in evidence_amounts:
            if value == cited or value.quantize(quantum) == cited:
                return True
            scaled = value * 100  # percent rendering: 0.9821 -> "98%"
            if scaled == cited or scaled.quantize(quantum) == cited:
                return True
            # Magnitude-rounded rendering: 707078.56 -> "$707k" (expanded to 707000).
            #
            # The rounding unit is the cited figure's own precision, its trailing zeros -
            # not a fixed list. A fixed 1k/1M list could not express "710k" (a legitimate
            # nearest-10k rendering of 707,078) and gave "707k" a full-unit tolerance that
            # grounded it against 707,999, which renders as 708k. Half a unit is rounding's
            # actual definition.
            #
            # The relative bound exists because half-unit alone still grounds "$1M" against
            # 707,078, that IS the nearest-million rounding, but a rendering that distorts
            # the underlying value by 41% is a different number wearing a rounded costume.
            # 10% admits honest renderings ("about $1M" for $1.08M) and flags the
            # invented-round-figure class the gate exists to catch; as everywhere in this
            # gate, a flag is a marker for the analyst, never a blocker.
            # Any integral citation >= 1000 qualifies, not only whole thousands: "135.9k"
            # expands to 135,900, whose precision is the hundreds place, requiring
            # cited % 1000 == 0 gave decimal-suffix renderings no rounding tolerance at
            # all, and two legitimate rationale figures (135.9k for 135,934.60; 90.8k for
            # 90,791.12) shipped flagged. Small integers stay exact-match-only above.
            if (
                value != 0
                and cited >= 1_000
                and cited == cited.to_integral_value()
            ):
                digits_str = str(int(cited))
                unit = Decimal(10) ** (len(digits_str) - len(digits_str.rstrip("0")))
                if (
                    abs(value - cited) <= unit / 2
                    and abs(value - cited) / value <= Decimal("0.10")
                ):
                    return True
        return False

    # Party ids first, and their digits are excluded from the numeric pass, `P02386` used to
    # flag twice, once as an id and once as the number 2386.
    party_ids = set(re.findall(r"\bP\d{4,6}\b", rationale))
    numeric_text = re.sub(r"\bP\d{4,6}\b", " ", rationale)
    for party in sorted(party_ids):
        if party not in evidence:
            flagged.append(party)

    for raw, cited in _extract_amounts(numeric_text):
        is_bare_int = cited == cited.to_integral_value() and not re.search(
            r"[km]\b|thousand|million|billion|bn\b", raw, re.IGNORECASE
        )
        if is_bare_int and cited < _GROUNDING_FLOOR:
            continue
        if not grounds(cited):
            flagged.append(raw)

    seen: set[str] = set()
    return [t for t in flagged if not (t in seen or seen.add(t))]




def generate_rationale(
    llm: Any,
    decision: TriageDecision,
    evidence: str,
    attributions: list[dict[str, Any]] | None = None,
) -> TriageDecision:
    """Attach a reasoned rationale to one escalated item.

    Model attributions are supplied to the prompt where available, so the narrative explains the
    *model's* basis rather than inventing an independent one. That keeps the rationale tied to
    what actually drove the score, which is what an examiner reconstructing the decision needs.

    A failure here degrades to the attributions alone rather than raising: an item must still
    reach the analyst if the language model is unavailable.
    """
    import time

    from amlguard.llm import extract_json

    attribution_text = ""
    if attributions:
        attribution_text = "\n".join(
            f"- {describe_feature(a['feature'])} = {a['value']:.4g} "
            f"({a.get('direction', 'affects')} the score by {abs(a['contribution']):.3f})"
            for a in attributions[:5]
        )

    # Resolve the rationale system prompt once from the Langfuse registry (fallback to the
    # code constant). Resolved ONCE so the pacing key, the actual call, and the provenance
    # hash all reference the same text, a per-call re-resolve could desync the hash.
    from amlguard.observability import domain_project, managed_prompt

    # Hybrid rationale runs on the AML queue (AML-only path), so the AML rationale prompt + project.
    system = managed_prompt("amlguard-rationale-system", project=domain_project("aml"))

    prompt = (
        f"ITEM {decision.item_id}, ranked {decision.rank} in the review queue "
        f"(model score {decision.score:.2f}).\n\n"
        f"WHAT DROVE THE MODEL SCORE\n{attribution_text or '(not available)'}\n\n"
        f"CASE EVIDENCE\n{evidence}\n\n"
        f'Respond as: {{"supports_suspicion": "...", "undermines_suspicion": "...", '
        f'"what_to_check_next": "..."}}'
    )

    # Paced: rationales are issued in a tight loop over the review head, and Bedrock throttles
    # rapid sequential calls. A short floor between calls costs seconds across a queue and
    # avoids losing answers to throttling entirely. The client owns the cache-key knowledge;
    # a fake without `would_bill` paces every call, the safe default.
    would_bill = getattr(llm, "would_bill", None)
    if would_bill is None or would_bill(system, prompt, 768):
        # A cache hit makes no network call, and sleeping before one taxed every re-run
        # 0.4s x 50 rationales for nothing.
        time.sleep(0.4)
    try:
        # max_tokens=768: a runaway guard, not a length control. The prompt's ~40-words-per-
        # field instruction is what shapes the answer (8/50 parsed before it existed, 50/50
        # after, at half the cost); the ceiling is ~3x the observed maximum output (p50 198,
        # p95 225, max 246 across 100 rationales) so a repeating model cannot block the queue,
        # while a genuinely long case still fits.
        completion = llm.complete(system, prompt, max_tokens=768)
        answer = extract_json(completion)
        decision.rationale = " ".join(
            f"{key.replace('_', ' ').capitalize()}: {value}"
            for key, value in answer.items()
            if value
        )
    except Exception as exc:  # noqa: BLE001, the item must still reach the analyst
        # Degrading is correct, but the reason must survive: a bare exception name hid that
        # 42 of 50 calls were being throttled while the run reported success.
        completion = ""
        decision.rationale = f"(rationale unavailable: {type(exc).__name__}: {str(exc)[:120]})"

    decision.attributions = attributions or []
    # Provenance for the five-year reconstruction obligation: the exact prompt, the raw
    # completion, the model that answered, and when. `pipeline.py` has always retained its
    # prompt verbatim as "the evidence for the central claim"; this path, the one whose
    # output reaches an analyst, discarded all four, so a decision could not be reproduced
    # without re-running a paid call and hoping the cache still held.
    from datetime import datetime, timezone

    decision.provenance = {
        "model": getattr(llm, "name", "unknown"),
        "prompt": prompt,
        "system_prompt_sha256": __import__("hashlib").sha256(
            system.encode()
        ).hexdigest()[:16],
        "completion": completion,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return decision
