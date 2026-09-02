"""Alert-grain triage, the queue an analyst actually works.

The classifier scores *transactions*, and an earlier version of this pipeline put those
transactions straight into the review queue. No AML operation works that way, and the output
showed why: of 50 queued items, **17 were distinct subjects** and one party appeared **ten
times**. An analyst opening that queue opens the same case ten times.

The unit of work is the **alert**, a monitoring scenario firing on a subject, which rolls up
to a case on that subject. `data/corpus/alerts.json` already carried the right schema (3,050
alerts, 4.0% conversion, prior-match counts, linked alerts, a triggering transaction); nothing
consumed it. This module is the missing join: transaction-level model scores aggregate to the
subject, and the alert inherits that evidence.

## Why rank on more than the score

Two things dominate a real queue that a pure score ignores, and both are regulatory rather
than statistical:

**The filing clock.** 31 CFR 1020.320(b)(3) gives 30 calendar days to file a SAR (60 where no
subject is identified), and the clock starts at *initial detection*, the point at which a
review concludes the activity is suspicious, not at alert generation. This queue does not know
that determination date (it is the analyst's, made downstream), so it anchors urgency to the
**alert date** as a deliberately conservative internal SLA: treating an aging alert as closer
to its deadline over-prioritizes rather than under-prioritizes, which is the safe direction. An
alert at SLA-day 27 outranks a higher-scoring alert at day 2, because ranking on score alone
will eventually let an old alert sit while a fresh high-scorer jumps it, and an examiner asks
how the queue guards against exactly that. Stated as an internal SLA, not a regulatory
deadline, because equating alert date with the regulatory clock would misstate the rule.

**Repeat alerting.** A third alert on one subject is a different object from a first. The
corpus models this (`prior_match_count_all`, `linked_alert_ids`) because escalation weight in
practice comes substantially from history, not from any single firing.

## What this deliberately does not do

It does not close alerts, and it does not drop them from the queue. FFIEC examination
procedure #12 states alert volume "should not be tailored solely to meet existing staffing
levels", and TD Bank was penalised for suppressing scenarios that raised volume. So capacity
marks how far an analyst will *reach*, and every alert below the line stays open and
reviewable, the same rule `hybrid.rank_queue` follows for transactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

# 31 CFR 1020.320(b)(3). Sixty days applies where no subject has been identified; every alert
# here names a subject, so thirty is the operative figure.
SAR_FILING_DAYS = 30

# Days remaining at which deadline pressure begins to dominate the model score. Inside this
# window an alert climbs regardless of score, because the clock is the binding constraint.
URGENCY_WINDOW_DAYS = 10


@dataclass
class AlertDecision:
    """One alert's place in the queue, and why it sits there."""

    alert_id: str
    subject_party_id: str
    scenario_id: str
    raised_on: str
    days_remaining: int
    # Aggregated model evidence for the subject over the alert's lookback window.
    model_score: float
    transaction_count: int
    prior_alerts: int
    linked_alerts: int
    # The composite this queue is ordered by.
    priority: float
    rank: int
    escalated: bool
    # The transactions that carry the subject's evidence, for the case file.
    evidence_transaction_ids: tuple[str, ...] = ()
    rationale: str = ""
    attributions: list[dict[str, Any]] = field(default_factory=list)
    # See hybrid.ungrounded_terms, assertions with no basis in the evidence, marked not blocked.
    ungrounded: list[str] = field(default_factory=list)
    # The human's half of the record. The system orders and explains; the analyst decides -
    # and without a field for that decision there is no override rate, no feedback loop, and
    # no evidence a human ever looked. Written by whatever review surface fronts this queue;
    # empty means "not yet reviewed", which is itself information.
    reviewed_by: str = ""
    analyst_disposition: str = ""   # e.g. escalate / close_no_action / needs_more_information
    analyst_note: str = ""
    # Model, verbatim prompt, raw completion, timestamp, set by hybrid.generate_rationale.
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def overdue(self) -> bool:
        return self.days_remaining < 0

    # `hybrid.generate_rationale` is shared across both grains, and reads `item_id`, `score`
    # and `rank`. Exposing them here keeps one rationale path rather than two that can drift
    # apart, the prompt, the word-count instruction and the JSON contract are the parts most
    # expensive to get right, and duplicating them is how they diverge.
    @property
    def item_id(self) -> str:
        return self.alert_id

    @property
    def score(self) -> float:
        return self.model_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "subject_party_id": self.subject_party_id,
            "scenario_id": self.scenario_id,
            "raised_on": self.raised_on,
            "days_remaining": self.days_remaining,
            "overdue": self.overdue,
            "model_score": round(self.model_score, 4),
            "transaction_count": self.transaction_count,
            "prior_alerts": self.prior_alerts,
            "linked_alerts": self.linked_alerts,
            "priority": round(self.priority, 4),
            "rank": self.rank,
            "escalated": self.escalated,
            "evidence_transaction_ids": list(self.evidence_transaction_ids[:8]),
            "rationale": self.rationale,
            "attributions": self.attributions[:5],
            "ungrounded": self.ungrounded,
            "reviewed_by": self.reviewed_by,
            "analyst_disposition": self.analyst_disposition,
            "analyst_note": self.analyst_note,
            "provenance": self.provenance,
        }


def subject_scores(
    transactions: list[dict],
    item_ids: list[str],
    scores: "Any",
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Roll transaction scores up to their originating party.

    The subject's score is the **maximum** over their scored transactions, not the mean. A
    party with one clearly suspicious movement among many ordinary ones is exactly the case an
    analyst must see; averaging would bury it under the ordinary traffic, which is the
    behaviour that makes rule-based systems miss layering.
    """
    by_id = {t["transaction_id"]: t for t in transactions}
    best: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}

    for item_id, score in zip(item_ids, scores):
        txn = by_id.get(item_id)
        if txn is None:
            continue
        value = float(score)
        for party in (txn["origin_party_id"], txn["beneficiary_party_id"]):
            if value > best.get(party, -1.0):
                best[party] = value
            evidence.setdefault(party, []).append(item_id)

    # Strongest evidence first, so a case file leads with the transaction that drove the
    # score. The lookup is built once: the previous form reconstructed a full
    # `dict(zip(item_ids, scores))` inside the sort key, once per element per party, ~8M
    # dict builds at this corpus size and O(N²) growth with it, in the path that runs on
    # every queue construction.
    score_of = {item_id: float(score) for item_id, score in zip(item_ids, scores)}
    for items in evidence.values():
        items.sort(key=lambda i: -score_of.get(i, 0.0))

    return best, evidence


def _days_remaining(raised_on: str, as_of: date) -> int:
    """Days left on the filing clock for an alert raised on `raised_on`."""
    try:
        raised = date.fromisoformat(raised_on)
    except (TypeError, ValueError):
        return SAR_FILING_DAYS
    return SAR_FILING_DAYS - (as_of - raised).days


def _priority(model_score: float, days_remaining: int, prior_alerts: int) -> float:
    """Composite ordering key: model evidence, deadline pressure, and repeat history.

    Deliberately a transparent formula rather than a second learned model. An examiner asks
    why *this* alert was reviewed before *that* one, and "a model decided" is not an answer a
    designated decision maker can defend. Every term here is explainable in one sentence.

    Deadline pressure is the dominant term inside the urgency window and negligible outside
    it, which reproduces how a real queue behaves: score orders the work until the clock
    starts to bind, and then the clock wins.
    """
    # Urgency rises as the deadline approaches and then **saturates** rather than growing
    # without bound once overdue.
    #
    # An unbounded term is wrong on a static corpus and wrong in production for the same
    # reason. Here, alerts span ten months with no arrival-rate model, so 93% are already past
    # 30 days when the ledger is replayed in one sitting; letting overdue-ness accumulate made
    # every one of them saturate the term, and the queue degenerated to "oldest first" with
    # the model score contributing nothing (measured: precision@50 fell to 0.000).
    #
    # In a live queue the same shape is what you want anyway: among alerts that are *all* past
    # due, age is no longer the discriminator, the evidence is. Capping urgency lets the
    # deadline dominate inside the window, where it should, and hands the ordering back to the
    # model score once everything in view is equally late.
    urgency = 0.0
    if days_remaining <= URGENCY_WINDOW_DAYS:
        elapsed = min(URGENCY_WINDOW_DAYS - days_remaining + 1, URGENCY_WINDOW_DAYS + 1)
        urgency = elapsed / (URGENCY_WINDOW_DAYS + 1)

    # Repeat alerting saturates: the step from one prior alert to three is meaningful, from
    # ten to twelve is not.
    history = min(prior_alerts, 6) / 6.0

    return model_score + 1.5 * urgency + 0.25 * history


def rank_alerts(
    alerts: list[dict],
    transactions: list[dict],
    item_ids: list[str],
    scores: "Any",
    review_capacity: int,
    as_of: date | None = None,
    window_start: str | None = None,
) -> list[AlertDecision]:
    """Order the alert queue and mark the head an analyst will reach.

    Only alerts whose subject appears in the scored (test-fold) transactions are queued -
    scoring an alert whose evidence the model never saw would be reporting a number with
    nothing behind it.

    `window_start` restricts the queue to alerts raised in the scoring window, and passing it
    is what makes the measurement deployment-faithful. The classifier scores one period of
    activity (the temporal test fold); an alert raised months earlier fires on evidence that
    sits entirely outside that period, so ranking it by the subject's *current* score is a
    category error, measured, it dragged precision@50 from 0.96 to 0.26, not because the
    model degraded but because 76 of 122 genuinely-escalating alerts could not be found by a
    model that never saw their evidence. A live queue has this property automatically: the
    model scores today's activity, and today's alerts fired on today's activity.

    Ties break by alert id so the ordering is reproducible; reconstructability is a five-year
    obligation, and an arbitrary tiebreak makes a queue impossible to reproduce for an
    examiner.
    """
    best, evidence = subject_scores(transactions, item_ids, scores)

    # The review date is the **most recent alert's** raise date, not the ledger's last day.
    #
    # A static corpus spans ten months, so measuring the clock from the end of the data makes
    # 93% of alerts historically overdue, every one of them saturates the urgency term and the
    # model score stops mattering. That is an artifact of replaying a year of history in one
    # sitting, not a property of a real queue, which is worked as alerts arrive.
    #
    # Anchoring to the newest alert reproduces the live situation: recent alerts have most of
    # their clock left, older ones are genuinely against the deadline, and the mix is what an
    # analyst opens on any given morning. It is also deterministic, unlike `date.today()`,
    # which would make every rerun of a historical corpus produce a different queue.
    if as_of is None:
        latest_alert = max((a.get("raised_on", "") for a in alerts), default="")
        as_of = (
            date.fromisoformat(latest_alert)
            if latest_alert
            else date.today()
        )

    candidates: list[AlertDecision] = []
    for alert in alerts:
        if window_start and str(alert.get("raised_on", "")) < window_start:
            continue
        subject = alert.get("subject_party_id")
        if subject not in best:
            continue
        days_left = _days_remaining(str(alert.get("raised_on", "")), as_of)
        model_score = best[subject]
        prior = int(alert.get("prior_match_count_all") or 0)
        candidates.append(
            AlertDecision(
                alert_id=str(alert.get("alert_id", "")),
                subject_party_id=str(subject),
                scenario_id=str(alert.get("scenario_id", "")),
                raised_on=str(alert.get("raised_on", "")),
                days_remaining=days_left,
                model_score=model_score,
                transaction_count=len(evidence.get(subject, ())),
                prior_alerts=prior,
                linked_alerts=len(alert.get("linked_alert_ids") or ()),
                priority=_priority(model_score, days_left, prior),
                rank=0,
                escalated=False,
                evidence_transaction_ids=tuple(evidence.get(subject, ())[:8]),
            )
        )

    candidates.sort(key=lambda d: (-d.priority, d.alert_id))
    for position, decision in enumerate(candidates):
        decision.rank = position + 1
        decision.escalated = position < review_capacity
    return candidates


@dataclass
class AlertQueueResult:
    """The ranked alert queue plus what reviewing its head cost."""

    decisions: list[AlertDecision] = field(default_factory=list)
    review_capacity: int = 0
    llm_calls: int = 0
    llm_cost_usd: float = 0.0
    # Egress-guard outcome and the queue's own precision, persisted rather than printed.
    # Both were previously logged only to stdout, which made every claim about them
    # unfalsifiable from the committed artifact, an audit could not tell 17 discounted from
    # 13, or 0.960 precision from 1.000, without rerunning a paid job.
    precision_at_capacity_value: float = 0.0
    egress_blocked: int = 0
    egress_discounted: int = 0
    # Identity of the classifier build that scored this queue (training.ClassifierBundle),
    # so "reproduce this ranking" starts from a named model rather than a re-fit and a hope.
    classifier_hash: str = ""

    @property
    def escalated(self) -> list[AlertDecision]:
        return [d for d in self.decisions if d.escalated]

    def precision_at_capacity(self, escalating_alert_ids: set[str]) -> float:
        """Share of the reviewed head that genuinely warranted escalation.

        Measured against the alert's own ground truth, not the transaction's: the question is
        whether the *case* was worth opening.
        """
        head = self.escalated
        if not head:
            return 0.0
        return sum(1 for d in head if d.alert_id in escalating_alert_ids) / len(head)

    def distinct_subjects(self) -> int:
        """Subjects in the reviewed head. At transaction grain this collapsed 50 items to 17."""
        return len({d.subject_party_id for d in self.escalated})

    def to_dict(self) -> dict[str, Any]:
        return {
            "grain": "alert",
            "review_capacity": self.review_capacity,
            "queue_length": len(self.decisions),
            "distinct_subjects_in_head": self.distinct_subjects(),
            "overdue_in_head": sum(1 for d in self.escalated if d.overdue),
            "precision_at_capacity": round(self.precision_at_capacity_value, 4),
            "classifier_hash": self.classifier_hash,
            "llm_calls": self.llm_calls,
            "llm_cost_usd": round(self.llm_cost_usd, 4),
            "egress_blocked": self.egress_blocked,
            "egress_discounted": self.egress_discounted,
            "decisions": [d.to_dict() for d in self.decisions],
        }
