"""Checkpoint scoring, deterministic where possible, judged only where necessary.

Scoring credibility is the submission's foundation: a reviewer can verify an exact-match
number but must *trust* a rubric. Three of the four strata are therefore scored by direct
comparison against planted ground truth, and only NARRATIVE goes to a judge.

Two design rules follow from wanting the measurement to be about protection rather than
about formatting:

  * **Normalise generously, compare strictly.** `"$66,733.00"`, `"66733"` and `66733.0` are
    the same answer; `66734` is not. Punishing a model for currency symbols would confound
    output formatting with the reasoning degradation being measured.

  * **Never award partial credit for a missing answer.** A key the model omitted scores zero
    rather than being skipped, because silently dropping unanswered checkpoints would inflate
    scores exactly where protection makes answering hardest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from amlguard.eval.tasks import Checkpoint, Scoring, Stratum

# Judge prompt kept terse and binary. A graded rubric would invite the judge to reward
# fluency, which is not what is being measured.
# JUDGE_SYSTEM moved to config/prompts/amlguard-judge-system.txt (loaded via observability.managed_prompt).


@dataclass
class CheckpointScore:
    checkpoint_id: str
    stratum: Stratum
    scoring: Scoring
    passed: bool
    score: float           # 1.0/0.0 for binary, F1 in [0,1] for SET
    expected: str
    actual: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "stratum": self.stratum.value,
            "scoring": self.scoring.value,
            "passed": self.passed,
            "score": round(self.score, 4),
            "expected": self.expected,
            "actual": self.actual,
            "detail": self.detail,
        }


def normalise_number(value: Any) -> Decimal | None:
    """Parse a number from whatever shape the model produced.

    Handles currency symbols, thousands separators, and stray prose. Returns None when no
    number is present, which is itself a result at wide protection scopes, where amounts are
    tokenized and arithmetic becomes impossible.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))

    text = str(value).strip()
    if not text:
        return None
    # First numeric run, not every digit in the string. Stripping all non-digits concatenated
    # separate numbers: "66733 across 7 transactions" became 667337, turning a correct answer
    # into a relative error of 9.0 and scoring it wrong.
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def normalise_text(value: Any) -> str:
    """Lowercase, collapse whitespace, drop surrounding punctuation."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower()).strip(".,;:'\"")


def normalise_identifier(value: Any) -> str:
    """Normalise a party or document id, stripped of adornment.

    Models sometimes return `"Party P00255"`, `"p00255."` or `"[DOC0042]"`, the same answer,
    differently dressed. Document ids are matched before party ids so that a note reference is
    not mistaken for a party reference.
    """
    text = str(value or "").strip().upper()

    # Zero-padded ids are normalised on their numeric part rather than their literal form.
    # Models reproduce `ALERT0016` as `ALERT016` or `ALERT00234`, a transcription slip, not a
    # different answer, and scoring that as a reasoning failure would confound output
    # formatting with the degradation being measured.
    for prefix in ("ALERT", "DOC"):
        match = re.search(rf"{prefix}0*(\d+)", text)
        if match:
            return f"{prefix}{int(match.group(1))}"

    match = re.search(r"P0*(\d+)", text)
    if match:
        return f"P{int(match.group(1))}"
    return re.sub(r"[^A-Z0-9]", "", text)


def _score_exact(checkpoint: Checkpoint, actual: Any) -> tuple[bool, str]:
    expected = checkpoint.expected
    # Party ids get identifier normalisation; everything else is text.
    if isinstance(expected, str) and re.fullmatch(r"P\d{4,6}", expected):
        return normalise_identifier(actual) == normalise_identifier(expected), ""
    return normalise_text(actual) == normalise_text(expected), ""


def _score_numeric(checkpoint: Checkpoint, actual: Any) -> tuple[bool, str]:
    parsed = normalise_number(actual)
    if parsed is None:
        return False, "no parseable number in answer"

    expected = Decimal(str(checkpoint.expected))
    if expected == 0:
        return parsed == 0, ""

    relative_error = abs(parsed - expected) / abs(expected)
    passed = relative_error <= Decimal(str(checkpoint.tolerance))
    return passed, f"relative error {float(relative_error):.4f}"


def _score_set(checkpoint: Checkpoint, actual: Any) -> tuple[float, str]:
    """F1 over expected vs. returned items.

    F1 rather than exact set equality because partial credit is the informative signal here:
    at wide protection scopes a model typically finds *some* counterparties, and collapsing
    that to pass/fail would discard the shape of the degradation.
    """
    expected = {normalise_identifier(v) for v in (checkpoint.expected or [])}
    if isinstance(actual, str):
        actual_items = re.split(r"[,;]\s*", actual)
    elif isinstance(actual, (list, tuple, set)):
        actual_items = list(actual)
    else:
        actual_items = []

    got = {normalise_identifier(v) for v in actual_items if normalise_identifier(v)}
    if not expected:
        return (1.0 if not got else 0.0), "expected empty set"
    if not got:
        return 0.0, "no items returned"

    true_positives = len(expected & got)
    if true_positives == 0:
        return 0.0, f"0/{len(expected)} matched"

    precision = true_positives / len(got)
    recall = true_positives / len(expected)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, f"precision {precision:.2f} recall {recall:.2f}"


def _score_ranked(checkpoint: Checkpoint, actual: Any) -> tuple[float, str]:
    """Precision@k over an ordered list.

    Used where the ordering *is* the answer, alert triage ranks a queue, and a set-based
    score would treat a ranking that buries the escalations at the bottom as equivalent to one
    that surfaces them first. `tolerance` carries k.
    """
    k = int(checkpoint.tolerance) or 3
    expected = {normalise_identifier(v) for v in (checkpoint.expected or [])}
    if isinstance(actual, str):
        items = re.split(r"[,;]\s*", actual)
    elif isinstance(actual, (list, tuple)):
        items = list(actual)
    else:
        items = []

    ranked = [normalise_identifier(v) for v in items if normalise_identifier(v)]
    if not ranked or not expected:
        return 0.0, "no ranking returned"

    # Deduplicated, order preserved. Counting a repeated id twice let a model that named one
    # escalating alert three times score a perfect precision@3 while missing the other.
    head: list[str] = []
    for item in ranked:
        if item not in head:
            head.append(item)
        if len(head) == k:
            break
    hits = sum(1 for item in head if item in expected)
    # Denominator is how many *could* have appeared in the head, so a k larger than the number
    # of expected items does not cap the score below 1.0.
    return hits / min(k, len(expected)), f"{hits}/{min(k, len(expected))} in top {k}"


def score_checkpoint(
    checkpoint: Checkpoint, answer: dict, judge: Any | None = None
) -> CheckpointScore:
    """Score one checkpoint against a model answer."""
    actual = answer.get(checkpoint.answer_key) if isinstance(answer, dict) else None
    actual_repr = str(actual)[:160]

    # A missing key is a failure, never a skip, skipping would inflate scores precisely
    # where protection makes answering hardest.
    if actual is None:
        return CheckpointScore(
            checkpoint_id=checkpoint.checkpoint_id,
            stratum=checkpoint.stratum,
            scoring=checkpoint.scoring,
            passed=False,
            score=0.0,
            expected=str(checkpoint.expected)[:160],
            actual="<missing>",
            detail=f"answer has no key {checkpoint.answer_key!r}",
        )

    if checkpoint.scoring is Scoring.RANKED:
        score, detail = _score_ranked(checkpoint, actual)
        passed = score >= 0.999
    elif checkpoint.scoring is Scoring.SET:
        score, detail = _score_set(checkpoint, actual)
        # A set checkpoint counts as passed only when it is essentially complete; the
        # fractional score still carries the partial-credit signal.
        passed = score >= 0.999
    elif checkpoint.scoring is Scoring.NUMERIC:
        passed, detail = _score_numeric(checkpoint, actual)
        score = 1.0 if passed else 0.0
    elif checkpoint.scoring is Scoring.JUDGE:
        passed, detail = _score_judge(checkpoint, actual, judge)
        score = 1.0 if passed else 0.0
    else:
        passed, detail = _score_exact(checkpoint, actual)
        score = 1.0 if passed else 0.0

    return CheckpointScore(
        checkpoint_id=checkpoint.checkpoint_id,
        stratum=checkpoint.stratum,
        scoring=checkpoint.scoring,
        passed=bool(passed),
        score=float(score),
        expected=str(checkpoint.expected)[:160],
        actual=actual_repr,
        detail=detail,
    )


def _score_judge(checkpoint: Checkpoint, actual: Any, judge: Any) -> tuple[bool, str]:
    """Grade a narrative checkpoint with an LLM judge.

    Reported separately from exact-match scores so a verifiable number is never diluted by a
    judged one. Judge failures score zero rather than raising: a run must not abort because
    one grading call failed.
    """
    if judge is None:
        return False, "no judge configured"

    from amlguard.llm import extract_json

    text = str(actual).strip()
    if not text:
        return False, "empty reasoning"

    from amlguard.observability import managed_prompt

    judge_system = managed_prompt("amlguard-judge-system")
    try:
        verdict = extract_json(
            judge.complete(
                judge_system,
                f"CRITERION\n{checkpoint.judge_criterion}\n\nREASONING\n{text}",
                max_tokens=200,
            )
        )
    except Exception as exc:  # noqa: BLE001, grading failure must not abort the run
        return False, f"judge error: {str(exc)[:80]}"

    return bool(verdict.get("pass")), str(verdict.get("why", ""))[:120]


@dataclass
class TaskScore:
    """Every checkpoint in one task, plus its completion verdict."""

    task_id: str
    checkpoints: list[CheckpointScore] = field(default_factory=list)
    error: str = ""
    # The model that actually answered, and how long the task took end-to-end. Both exist
    # because the published curve was once a silent three-model mix: Bedrock throttling
    # tripped the fallback chain mid-run, local models answered ~half the tasks, and nothing
    # in the artifact recorded the substitution. A per-task model stamp makes that class of
    # contamination visible in the result file itself rather than only in cache forensics.
    model: str = ""
    seconds: float = 0.0

    @property
    def completed(self) -> bool:
        """[[Task Completion Rate]] is strict: every checkpoint must pass.

        Deliberately harsher than mean checkpoint score. The two figures diverging is a
        finding to explain, not a contradiction to hide.
        """
        return bool(self.checkpoints) and all(c.passed for c in self.checkpoints)

    @property
    def mean_score(self) -> float:
        if not self.checkpoints:
            return 0.0
        return sum(c.score for c in self.checkpoints) / len(self.checkpoints)

    def by_stratum(self) -> dict[str, list[CheckpointScore]]:
        grouped: dict[str, list[CheckpointScore]] = {}
        for checkpoint in self.checkpoints:
            grouped.setdefault(checkpoint.stratum.value, []).append(checkpoint)
        return grouped

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "completed": self.completed,
            "mean_score": round(self.mean_score, 4),
            "error": self.error,
            "model": self.model,
            "seconds": round(self.seconds, 2),
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }
