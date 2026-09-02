"""Investigation tasks and their checkpoints, the measurement instrument.

By design: [[Investigation Task]]s provide the realism frame (an investigator works a
case, rather than answering trivia about a corpus) while [[Checkpoint]]s inside them carry
the measurement. Tasks are the container; checkpoints are the score.

Checkpoints are **stratified**, because protection does not degrade all reasoning equally:

  IDENTITY, resolving a party across documents. Depends entirely on token stability,
                  so the determinism ablation should destroy this stratum and nothing else.
  AGGREGATION, arithmetic over amounts. Survives `direct` untouched and should collapse
                  at `quasi`, where AMOUNT starts being tokenized. Isolates the quasi boundary.
  TYPOLOGY, classifying a laundering pattern. The reasoning is structural, so it should
                  be the most protection-resistant stratum.
  NARRATIVE, comprehension of free-text case notes. The only stratum needing a judge.

Reporting a blended average across these would measure the task mix rather than the
protection, which is why per-stratum reporting is mandatory.

Ground truth comes from the planted typologies, so IDENTITY, AGGREGATION and
TYPOLOGY checkpoints are scored by **exact match against known-correct answers**, verifiable
by a sceptical reviewer rather than requiring trust in a rubric.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class Stratum(str, Enum):
    IDENTITY = "identity"
    AGGREGATION = "aggregation"
    TYPOLOGY = "typology"
    NARRATIVE = "narrative"


class Scoring(str, Enum):
    EXACT = "exact"          # equality against ground truth
    NUMERIC = "numeric"      # within a relative tolerance
    SET = "set"              # F1 over a set of expected items
    JUDGE = "judge"          # LLM-as-judge; narrative only
    RANKED = "ranked"        # precision@k over an ordered list; order is the answer


@dataclass
class Checkpoint:
    """One verifiable assertion inside a task."""

    checkpoint_id: str
    stratum: Stratum
    scoring: Scoring
    # Key to read from the model's JSON answer.
    answer_key: str
    expected: Any
    description: str
    # Relative tolerance for NUMERIC. 0.02 accepts rounding, not a different answer.
    tolerance: float = 0.02
    # For JUDGE checkpoints: what the judge is asked to verify.
    judge_criterion: str = ""

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "stratum": self.stratum.value,
            "scoring": self.scoring.value,
            "answer_key": self.answer_key,
            "expected": str(self.expected),
            "description": self.description,
        }


@dataclass
class InvestigationTask:
    """A realistic investigator workflow, decomposed into checkpoints."""

    task_id: str
    question: str
    response_shape: str
    checkpoints: list[Checkpoint]
    # Entry point into the case. Party ids are surrogate keys, never protected, semantic
    # search cannot find a party by name once tokenized, so this is how an
    # investigator actually opens a case.
    party_id: str | None = None
    typology_id: str | None = None
    top_k: int = 5

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "party_id": self.party_id,
            "typology_id": self.typology_id,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }


# -- task construction -----------------------------------------------------------------
#
# Tasks are generated from planted ground truth rather than hand-written, so expected values
# cannot drift from the corpus. Regenerating the corpus regenerates consistent tasks.


def _typology_task(
    instance: dict, transactions_by_id: dict[str, dict], index: int
) -> InvestigationTask:
    """Classify the typology and identify its participants, the core AML workflow."""
    typology_id = instance["typology_id"]
    subject = instance["subject_party_id"]
    parties = set(instance["party_ids"])
    txn_ids = instance["transaction_ids"]

    # Expected total is derived from the transaction rows themselves, never from the
    # generator's notional figure.
    #
    # For chains, `TypologyInstance.total_amount` records the amount *before* the first
    # commission was deducted, a number that appears in no transaction row and is therefore
    # unobservable from the ledger. Scoring against it would fail every correct answer.
    #
    # Structuring sums its deposits, since the pattern is the aggregate. Chains report the
    # first hop, which is the value that actually entered the chain and the figure an
    # investigator would cite.
    pattern_txns = [transactions_by_id[t] for t in txn_ids if t in transactions_by_id]
    pattern_txns.sort(key=lambda t: (t["value_date"], t["transaction_id"]))

    if instance["typology"] == "structuring":
        total = sum(Decimal(t["amount"]) for t in pattern_txns)
    else:
        total = Decimal(pattern_txns[0]["amount"]) if pattern_txns else Decimal("0")

    return InvestigationTask(
        task_id=f"T{index:02d}-{typology_id}",
        typology_id=typology_id,
        party_id=subject,
        # The question must be unambiguous about scope. An early version asked only to
        # "review their transactions", and the model correctly summed the party's *entire*
        # ledger while the checkpoints expected only the suspicious subset, scoring the
        # model wrong for answering the question actually asked. The instruction below
        # confines every figure to the transactions forming the pattern.
        question=(
            f"Party {subject} has been flagged by transaction monitoring. Their full "
            f"transaction history is provided, and most of it is ordinary business activity. "
            f"Identify the SUSPICIOUS SUBSET of transactions that together form a single "
            f"money-laundering pattern, and report on that subset ONLY.\n\n"
            f"Ordinary activity, payroll, rent, supplier payments, insurance, utilities, "
            f"subscriptions, is not part of the pattern and must be excluded from every "
            f"figure you report.\n\n"
            f"DETECTED CANDIDATE PATTERNS lists patterns found by deterministic rules, with "
            f"exact figures. When a candidate matches your conclusion, copy its "
            f"total_amount, transaction_count and counterparties EXACTLY rather than "
            f"recomputing them from the ledger. Use the transaction network and case notes "
            f"to decide WHICH candidate is correct and to explain why.\n\n"
            f"Look for one of these patterns:\n"
            f"- structuring: repeated cash or branch deposits by the same party, each just "
            f"below a $10,000 reporting threshold, clustered in a short window\n"
            f"- layering: a chain of transfers where the beneficiary of one transfer is the "
            f"originator of the next, with the amount shrinking slightly at each hop\n"
            f"- round_tripping: value that leaves this party and returns to it through "
            f"intermediaries, so this party is both the first originator and final beneficiary"
        ),
        response_shape=(
            '{"typology": "structuring|layering|round_tripping|none", '
            '"counterparties": ["party ids in the suspicious pattern, excluding the subject"], '
            '"total_amount": <sum of ONLY the suspicious transactions, digits only>, '
            '"transaction_count": <count of ONLY the suspicious transactions>, '
            '"reasoning": "one or two sentences citing the specific evidence"}'
        ),
        checkpoints=[
            Checkpoint(
                checkpoint_id=f"{typology_id}-typology",
                stratum=Stratum.TYPOLOGY,
                scoring=Scoring.EXACT,
                answer_key="typology",
                expected=instance["typology"],
                description="Correct laundering typology identified",
            ),
            Checkpoint(
                checkpoint_id=f"{typology_id}-counterparties",
                stratum=Stratum.IDENTITY,
                scoring=Scoring.SET,
                answer_key="counterparties",
                expected=sorted(parties - {subject}),
                description="Counterparties in the pattern resolved correctly",
            ),
            Checkpoint(
                checkpoint_id=f"{typology_id}-total",
                stratum=Stratum.AGGREGATION,
                scoring=Scoring.NUMERIC,
                answer_key="total_amount",
                expected=float(total),
                description="Total value moved through the pattern",
            ),
            Checkpoint(
                checkpoint_id=f"{typology_id}-count",
                stratum=Stratum.AGGREGATION,
                scoring=Scoring.NUMERIC,
                answer_key="transaction_count",
                expected=len(txn_ids),
                description="Number of transactions in the pattern",
                tolerance=0.0,
            ),
            Checkpoint(
                checkpoint_id=f"{typology_id}-reasoning",
                stratum=Stratum.NARRATIVE,
                scoring=Scoring.JUDGE,
                answer_key="reasoning",
                expected=instance["typology"],
                description="Reasoning cites evidence consistent with the typology",
                judge_criterion=(
                    f"Does the reasoning describe behaviour consistent with "
                    f"{instance['typology']}, citing specific evidence such as amounts, "
                    f"timing, or counterparty structure?"
                ),
            ),
        ],
    )


def _entity_resolution_task(
    instance: dict, index: int
) -> InvestigationTask:
    """Count a party's distinct counterparties.

    Deliberately pure IDENTITY: answering requires recognising the same party across
    separate records, which is exactly what deterministic tokenization enables and what the
    ablation removes.
    """
    subject = instance["subject_party_id"]
    return InvestigationTask(
        task_id=f"T{index:02d}-{instance['typology_id']}-entity",
        typology_id=instance["typology_id"],
        party_id=subject,
        # Scoped explicitly to DIRECT counterparties: the prompt now carries a two-hop
        # network, so "counterparties" without qualification would be ambiguous and the
        # expected values, computed from direct transactions only, would not match.
        question=(
            f"Consider ONLY transactions where party {subject} is itself the originator or "
            f"the beneficiary. Ignore transactions between other parties.\n\n"
            f"How many distinct parties does {subject} transact with directly, and which "
            f"single one appears in the most transactions with {subject}?"
        ),
        response_shape=(
            '{"distinct_counterparties": <number>, '
            '"most_frequent_counterparty": "<party id>", '
            '"reasoning": "one sentence"}'
        ),
        checkpoints=[],  # filled by build_tasks, which has the transaction data
    )


def _discrimination_task(
    instance: dict,
    decoys: list[dict],
    transactions_by_id: dict[str, dict],
    index: int,
) -> InvestigationTask | None:
    """Choose between competing candidates, a judgement the detector cannot make.

    The typology task's copyable checkpoints let transcription outscore reasoning:
    the prompt hands the model exact figures and instructs it to copy them, so a program that
    copies the top candidate beats the LLM. Any measurement built on those checkpoints is
    measuring the detectors.

    This task removes the shortcut structurally. The detectors surface **several** candidate
    patterns for a subject, and the question is which one the *narrative evidence* supports.
    Copying the top-ranked candidate is now a specific wrong answer roughly as often as it is
    right, because the detector ranks by chain length and value while the case notes describe
    what actually happened.

    The model must therefore do the thing the detector cannot: read the prose, weigh it against
    the structural candidates, and pick. There is no field to copy.
    """
    if not decoys:
        return None

    subject = instance["subject_party_id"]
    correct = instance["typology"]

    return InvestigationTask(
        task_id=f"T{index:02d}-{instance['typology_id']}-discriminate",
        typology_id=instance["typology_id"],
        party_id=subject,
        top_k=6,
        question=(
            f"Transaction monitoring has surfaced SEVERAL candidate patterns for party "
            f"{subject}. They cannot all be correct, the detection rules fire on structural "
            f"shape alone and do not read the case notes.\n\n"
            f"Your job is to decide which candidate the NARRATIVE EVIDENCE actually supports, "
            f"and to say why the others do not fit.\n\n"
            f"Do not simply take the first or highest-ranked candidate. The ranking reflects "
            f"chain length and transaction value, not evidential support. Read the case notes "
            f"and let them decide.\n\n"
            f"State which candidate the case notes corroborate, and identify one candidate the "
            f"evidence contradicts."
        ),
        response_shape=(
            '{"supported_typology": "structuring|layering|round_tripping|funnel_account|'
            'trade_based|none", '
            '"contradicted_typology": "the candidate the notes argue against, or none", '
            '"evidence_quote": "the phrase from the case notes that decided it", '
            '"reasoning": "one or two sentences"}'
        ),
        checkpoints=[
            Checkpoint(
                checkpoint_id=f"{instance['typology_id']}-discriminate",
                stratum=Stratum.TYPOLOGY,
                scoring=Scoring.EXACT,
                answer_key="supported_typology",
                expected=correct,
                description="Correct candidate chosen against competing alternatives",
            ),
            Checkpoint(
                checkpoint_id=f"{instance['typology_id']}-grounding",
                stratum=Stratum.NARRATIVE,
                scoring=Scoring.JUDGE,
                answer_key="evidence_quote",
                expected=correct,
                description="Choice grounded in an actual phrase from the case notes",
                judge_criterion=(
                    f"Does the quoted evidence come from case-note prose (not from a "
                    f"transaction table or a candidate summary), and does it genuinely support "
                    f"{correct} rather than merely restating the label?"
                ),
            ),
        ],
    )


def _counterfactual_task(
    instance: dict, transactions_by_id: dict[str, dict], index: int
) -> InvestigationTask | None:
    """Ask what would change the conclusion, unanswerable by copying anything.

    A detector emits *what it found*. It cannot say what evidence would overturn the finding,
    because that requires a model of why the pattern is suspicious rather than merely that it
    matches a shape.

    This is also the question a compliance officer actually asks before escalating, and the one
    an examiner asks afterwards: what would have made you decide otherwise? A copilot that
    cannot answer it is not doing investigative reasoning.
    """
    subject = instance["subject_party_id"]
    typology = instance["typology"]

    # What would legitimately explain each pattern away. These are the real defences an
    # investigator hears, which is why they make a fair test.
    exculpatory = {
        "structuring": "documented cash-intensive business with declared turnover matching the deposits",
        "layering": "genuine trade documentation for each hop showing goods actually moved",
        "round_tripping": "an intercompany loan agreement with a repayment schedule",
        "funnel_account": "a payroll or collections mandate authorising deposits from multiple parties",
        "trade_based": "shipping documents and customs records matching the invoiced values",
    }

    return InvestigationTask(
        task_id=f"T{index:02d}-{instance['typology_id']}-counterfactual",
        typology_id=instance["typology_id"],
        party_id=subject,
        top_k=6,
        question=(
            f"Party {subject} shows a pattern consistent with money laundering.\n\n"
            f"Before escalating, state what evidence would DISPROVE the suspicion, what a "
            f"reviewer could find that would legitimately explain this activity and justify "
            f"closing the alert.\n\n"
            f"Then state which single piece of currently-available information most supports "
            f"escalation, and what is missing from the file that a reviewer would need next."
        ),
        # `analytical_next_step`, not a filing recommendation. The system deliberately never
        # recommends filing or closing (the filing decision is a human's, 31 CFR 1020.320);
        # this eval field measures whether the model reaches the analytically-correct next
        # action, and `close` is not among the options for the same reason the analyst path
        # forbids it, a copilot that recommends closing an alert is one compliance rejects.
        response_shape=(
            '{"exculpatory_evidence": "what would explain this activity legitimately", '
            '"strongest_indicator": "the single strongest reason to escalate", '
            '"missing_information": "what the file lacks", '
            '"analytical_next_step": "escalate|request_information"}'
        ),
        checkpoints=[
            Checkpoint(
                checkpoint_id=f"{instance['typology_id']}-counterfactual",
                stratum=Stratum.NARRATIVE,
                scoring=Scoring.JUDGE,
                answer_key="exculpatory_evidence",
                expected=exculpatory.get(typology, ""),
                description="Identifies evidence that would legitimately explain the pattern",
                judge_criterion=(
                    f"For a suspected {typology} pattern, does the answer describe evidence "
                    f"that would genuinely exculpate, something like "
                    f"'{exculpatory.get(typology, 'a legitimate business explanation')}', "
                    f"rather than restating the suspicion or naming generic documents?"
                ),
            ),
            Checkpoint(
                checkpoint_id=f"{instance['typology_id']}-next-step",
                stratum=Stratum.TYPOLOGY,
                scoring=Scoring.EXACT,
                answer_key="analytical_next_step",
                # Every planted instance is a genuine typology, so escalation is the correct
                # analytical next step. Requesting information is the defensible alternative
                # and is not credited: an analyst who cannot commit on this evidence has not
                # concluded. This is not a filing recommendation, see the response shape.
                expected="escalate",
                description="Reaches the correct analytical next step",
            ),
        ],
    )


def _cross_document_identity_task(
    narratives: list[dict], parties_by_id: dict[str, dict], index: int
) -> InvestigationTask | None:
    """Resolve a party across narrative documents by its tokenized name alone.

    Every other task routes identity through party ids, unprotected surrogate keys, so token
    stability is never exercised and the determinism ablation measured nothing.
    This task deliberately removes that route: no party id is given, and the answer depends
    entirely on recognising that the *same token* denotes the same party in several documents.

    Under deterministic tokenization the token repeats and the task is answerable. Under the
    ablation each occurrence carries a different token, so the party becomes several apparent
    strangers and the task should collapse. That contrast is the ablation's whole purpose.
    """
    # Pick a party appearing in several narratives, the more documents, the sharper the test.
    appearances: dict[str, list[str]] = {}
    for narrative in narratives:
        subject = narrative.get("subject_party_id")
        if subject:
            appearances.setdefault(subject, []).append(narrative["document_id"])

    candidates = sorted(
        ((pid, docs) for pid, docs in appearances.items() if len(docs) >= 2),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    if not candidates:
        return None

    party_id, documents = candidates[0]
    party = parties_by_id.get(party_id)
    if not party:
        return None

    return InvestigationTask(
        task_id=f"T{index:02d}-crossdoc-identity",
        party_id=None,  # deliberately withheld: the token is the only identity handle
        typology_id=None,
        top_k=12,
        question=(
            "The case notes below concern several different parties. Party names have been "
            "replaced by pseudonymous tokens.\n\n"
            "Exactly one party is discussed in MORE THAN ONE case note. Identify that party by "
            "quoting its name token EXACTLY as it appears, and list the document ids of every "
            "note that discusses it.\n\n"
            "Two notes concern the same party only when the name token is character-for-character "
            "identical. Different tokens denote different parties, however similar the notes look."
        ),
        response_shape=(
            '{"repeated_party_token": "<the name token, copied exactly>", '
            '"document_ids": ["DOC...", "DOC..."], '
            '"document_count": <number>, '
            '"reasoning": "one sentence"}'
        ),
        checkpoints=[
            Checkpoint(
                checkpoint_id="crossdoc-count",
                stratum=Stratum.IDENTITY,
                scoring=Scoring.NUMERIC,
                answer_key="document_count",
                expected=len(documents),
                description="Number of documents discussing the repeated party",
                tolerance=0.0,
            ),
            Checkpoint(
                checkpoint_id="crossdoc-documents",
                stratum=Stratum.IDENTITY,
                scoring=Scoring.SET,
                answer_key="document_ids",
                expected=sorted(documents),
                description="Documents resolved to the same party across notes",
            ),
        ],
    )


def _triage_task(
    alerts: list[dict], index: int, batch_size: int = 12
) -> InvestigationTask | None:
    """Rank a batch of alerts by escalation priority, the work that dominates the real job.

    Every other task in this suite assumes an alert is worth investigating. Real transaction
    monitoring converts roughly 4% of alerts into SARs, so an analyst's day is
    mostly spent *clearing* ordinary activity: the property sale, the payroll float, the
    inheritance. A corpus of true positives only cannot measure that.

    The task is deliberately framed as **ordering, not suppression**. FFIEC examination
    procedure #12 states that alert volume "should not be tailored solely to meet existing
    staffing levels", and TD Bank was penalised for rejecting scenarios that would raise
    volume. A copilot that recommends closing alerts is one a compliance officer would refuse;
    a copilot that recommends what to look at first is not.

    Scoring uses precision@k rather than classification accuracy, because ranking is the
    decision being made and a analyst works down a queue.
    """
    if not alerts:
        return None

    # A batch with a realistic mix: mostly benign, a couple of genuine escalations.
    escalating = [a for a in alerts if a.get("escalated")][:2]
    benign = [a for a in alerts if not a.get("escalated")][: batch_size - len(escalating)]
    batch = sorted(escalating + benign, key=lambda a: a["alert_id"])
    if not escalating:
        return None

    lines = []
    for alert in batch:
        lines.append(
            f"- {alert['alert_id']} | subject {alert['subject_party_id']} "
            f"| scenario {alert['scenario_id']} | score {alert['score']} "
            f"| prior alerts on subject: {alert['prior_match_count_all']} "
            f"(same scenario: {alert['prior_match_count_same_scenario']}) "
            f"| linked alerts: {len(alert['linked_alert_ids'])} "
            f"| reason: {alert['reason']}"
        )

    return InvestigationTask(
        task_id=f"T{index:02d}-triage",
        party_id=None,
        typology_id=None,
        top_k=8,
        question=(
            "The following transaction-monitoring alerts are awaiting review.\n\n"
            + "\n".join(lines)
            + "\n\nRank them by investigative priority, which should an analyst examine "
            "first. Do NOT recommend closing or suppressing any alert; every alert will be "
            "reviewed regardless. You are ordering a queue, not filtering it.\n\n"
            "Weigh the rule score, the subject's prior alert history (repeat alerting on one "
            "subject is a stronger signal than any single alert), how many alerts are linked "
            "to the same subject, and whether the scenario describes a pattern or a "
            "one-off event."
        ),
        response_shape=(
            '{"ranked_alert_ids": ["highest priority first, all alerts included"], '
            '"top_priority_alert_id": "<alert id>", '
            '"reasoning": "one or two sentences on what drove the ordering"}'
        ),
        checkpoints=[
            # The former `triage-top-priority` checkpoint was dropped: with two genuine
            # escalations and no ground-truth priority BETWEEN them, `escalating[0]` was
            # first-in-file-order, an arbitrary label that scored the model wrong at every
            # scope and injected constant noise. The ranked checkpoint below carries real,
            # order-aware ground truth (both escalations must reach the top 3) and is kept.
            Checkpoint(
                checkpoint_id="triage-precision-at-3",
                stratum=Stratum.TYPOLOGY,
                scoring=Scoring.RANKED,
                answer_key="ranked_alert_ids",
                expected=[a["alert_id"] for a in escalating],
                description="Escalating alerts ranked in the top 3",
                # Set-scoring would ignore order entirely, a ranking that buries both
                # escalations at the bottom would score identically to one that surfaces them.
                # Order is the decision being made, so only the head of the list counts.
                tolerance=3,
            ),
        ],
    )


def build_tasks(
    corpus_dir: Path,
    max_typologies: int = 8,
    visible_transactions: Callable[[str], list[dict]] | None = None,
) -> list[InvestigationTask]:
    """Construct the task set from planted ground truth.

    One typology task per planted instance, plus entity-resolution tasks whose expected
    answers are computed directly from the transaction ledger. Every expected value is
    derived from the corpus, so tasks and data cannot drift apart.

    `visible_transactions` returns the ledger a given subject's prompt will actually contain.
    Supplying it keeps entity-resolution expectations aligned with what the model can see;
    omitting it falls back to the full corpus, which is correct only for an unfiltered prompt.
    """
    ground_truth = json.loads((corpus_dir / "ground_truth.json").read_text())
    transactions = json.loads((corpus_dir / "transactions.json").read_text())
    by_id = {t["transaction_id"]: t for t in transactions}

    # A subject can host more than one planted pattern. Scoring against one specific instance
    # would then fail a model that correctly identified the *other* real pattern for that
    # subject, measuring which instance the detector surfaced first rather than whether the
    # reasoning was right. Subjects with overlapping instances are excluded so every task has
    # exactly one correct answer.
    subject_counts: dict[str, int] = {}
    for instance in ground_truth:
        subject_counts[instance["subject_party_id"]] = (
            subject_counts.get(instance["subject_party_id"], 0) + 1
        )
    unambiguous = [g for g in ground_truth if subject_counts[g["subject_party_id"]] == 1]

    # Spread across typology kinds rather than taking the first N, so no single kind
    # dominates the score.
    by_kind: dict[str, list[dict]] = {}
    for instance in unambiguous:
        by_kind.setdefault(instance["typology"], []).append(instance)

    selected: list[dict] = []
    while len(selected) < max_typologies and any(by_kind.values()):
        for kind in list(by_kind):
            if by_kind[kind] and len(selected) < max_typologies:
                selected.append(by_kind[kind].pop(0))

    tasks: list[InvestigationTask] = []
    for i, instance in enumerate(selected, start=1):
        tasks.append(_typology_task(instance, by_id, i))

    # Entity-resolution tasks, scoped to the transactions the prompt actually contains.
    #
    # Expected values must be computed over the *visible* ledger, not the full one. An earlier
    # version counted every transaction in the corpus while the prompt carried a filtered
    # network view, so a model answering correctly about what it could see was scored wrong -
    # measuring the prompt-construction strategy rather than entity resolution.
    #
    # `visible_transactions` is supplied by the caller and mirrors what the pipeline builds.
    for _offset, instance in enumerate(selected[:2]):
        subject = instance["subject_party_id"]
        ledger = visible_transactions(subject) if visible_transactions else transactions
        involved = [
            t for t in ledger
            if subject in (t["origin_party_id"], t["beneficiary_party_id"])
        ]
        counterparties: dict[str, int] = {}
        for txn in involved:
            other = (
                txn["beneficiary_party_id"]
                if txn["origin_party_id"] == subject
                else txn["origin_party_id"]
            )
            counterparties[other] = counterparties.get(other, 0) + 1

        if not counterparties:
            continue
        most_frequent = max(counterparties.items(), key=lambda kv: (kv[1], kv[0]))[0]

        task = _entity_resolution_task(instance, len(tasks) + 1)
        task.checkpoints = [
            Checkpoint(
                checkpoint_id=f"{instance['typology_id']}-distinct-cp",
                stratum=Stratum.IDENTITY,
                scoring=Scoring.NUMERIC,
                answer_key="distinct_counterparties",
                expected=len(counterparties),
                description="Distinct counterparty count",
                tolerance=0.0,
            ),
            Checkpoint(
                checkpoint_id=f"{instance['typology_id']}-top-cp",
                stratum=Stratum.IDENTITY,
                scoring=Scoring.EXACT,
                answer_key="most_frequent_counterparty",
                expected=most_frequent,
                description="Most frequent counterparty resolved",
            ),
        ]
        tasks.append(task)

    # Reasoning tasks the detectors cannot answer by construction. These carry the
    # measurement now: the copyable checkpoints remain for continuity but are no longer the
    # basis of any claim about model contribution.
    for _offset, instance in enumerate(selected[:3]):
        decoys = [g for g in ground_truth if g["typology"] != instance["typology"]][:2]
        discriminate = _discrimination_task(instance, decoys, by_id, len(tasks) + 1)
        if discriminate:
            tasks.append(discriminate)

    for instance in selected[:3]:
        counterfactual = _counterfactual_task(instance, by_id, len(tasks) + 1)
        if counterfactual:
            tasks.append(counterfactual)

    # One triage task. This is the only task exercising the benign-alert population, and it
    # measures the discipline that dominates real investigation: ordering a queue that is ~96%
    # ordinary activity.
    alerts_path = corpus_dir / "alerts.json"
    if alerts_path.exists():
        triage = _triage_task(json.loads(alerts_path.read_text()), len(tasks) + 1)
        if triage:
            tasks.append(triage)

    # The cross-document identity task was removed after review. Its premise ("exactly one
    # party is discussed in more than one note") was false, 112 parties recur, so the ground
    # truth was wrong and it failed at every scope including `none`, unable to discriminate
    # deterministic from nondeterministic tokens as it was meant to. Even reframed to "the
    # most-repeated party", dense top-k retrieval over 752 notes cannot reliably surface one
    # party's seven documents, so the task could not measure token stability through the one
    # retrieval mode blind to it. The determinism ablation's operational value (801 vs 37,629
    # API calls) stands on its own; the analytical arm is honestly reported as inconclusive
    # rather than propped up by a broken instrument.
    # `_cross_document_identity_task` is retained in the module for provenance, uncalled.

    return tasks
