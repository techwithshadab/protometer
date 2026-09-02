"""Copy-the-detector baseline, what score is achievable without any reasoning.

Three of five typology checkpoints (`counterparties`, `total_amount`, `transaction_count`) are
scored on figures the prompt explicitly instructs the model to copy verbatim from the detected
candidates. A model that copies the top candidate and does no reasoning at all should therefore
score well on them.

Without measuring that, the evaluation cannot say whether the LLM contributes anything. This
baseline makes the model's marginal contribution explicit: **LLM score minus copy score** is the
part attributable to reasoning rather than transcription.

The baseline makes no API calls and involves no model. It picks the highest-ranked candidate and
emits its figures, the cheapest possible "answer".

**It is deliberately identical at every protection scope**, and that is a property rather than a
defect. Since that correction, detection runs on the *clear* ledger for every scope so that candidate
input is constant and the curve isolates the model. The baseline consumes only that candidate
output, so it necessarily produces the same score everywhere.

The consequence is that it is a **fixed reference line, not a per-scope control**. It answers
"what does transcription alone achieve on these tasks?", a constant, and the model's distance
above it is what varies. If detection were run on each scope's protected ledger instead
(`--detection-on-protected`), the baseline would move with scope and would measure something
different: how much a copying strategy degrades as the detectors themselves degrade.
"""

from __future__ import annotations

from typing import Any

from protometer.eval.scoring import TaskScore, score_checkpoint
from protometer.eval.tasks import InvestigationTask
from protometer.pipeline import InvestigationPipeline


def copy_baseline_answer(
    pipeline: InvestigationPipeline, task: InvestigationTask
) -> dict[str, Any]:
    """The answer a program that only copies the top candidate would produce."""
    if task.party_id is None:
        # Triage and cross-document tasks have no detector output to copy, so the baseline
        # has nothing to say. Returning empty scores them zero, which is the honest floor.
        return {}

    candidates = pipeline.candidate_patterns(task.party_id)
    if not candidates:
        return {}

    top = candidates[0]
    return {
        "typology": top["pattern"],
        "counterparties": top["counterparties"],
        "total_amount": top["total_amount"],
        "transaction_count": top["transaction_count"],
        # Deliberately mechanical: the baseline does not reason, so its "reasoning" is the
        # detector's own evidence string. A judge should mark this down where genuine
        # narrative comprehension is required.
        "reasoning": top["evidence"],
    }


def run_baseline(
    pipeline: InvestigationPipeline, tasks: list[InvestigationTask]
) -> list[TaskScore]:
    """Score the copy baseline over the same tasks and checkpoints as the model."""
    scores: list[TaskScore] = []
    for task in tasks:
        answer = copy_baseline_answer(pipeline, task)
        task_score = TaskScore(task_id=task.task_id)
        task_score.checkpoints = [
            # No judge: the baseline must not be credited for narrative reasoning it did not
            # do, and a judge call here would cost money to grade a fixed string.
            score_checkpoint(checkpoint, answer, None)
            for checkpoint in task.checkpoints
        ]
        scores.append(task_score)
    return scores


def marginal_contribution(model_mean: float, baseline_mean: float) -> float:
    """Share of the model's score not obtainable by copying.

    Zero means the model added nothing over transcription. Negative means it did worse than
    copying, which is itself informative.
    """
    if model_mean <= 0:
        return 0.0
    return (model_mean - baseline_mean) / model_mean
