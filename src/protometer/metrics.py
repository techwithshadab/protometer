"""Metrics chosen for how AML models are actually judged, not for how ML papers report.

Most default classification metrics are misleading on this problem, and it is worth stating why
rather than reporting them alongside better ones.

**Accuracy is useless.** The positive class is ~8% here and well under 1% in production, so
predicting "never suspicious" scores >99% and detects nothing.

**ROC-AUC is nearly useless.** It averages over every threshold including ones no institution
would ever operate at. A model can post 0.99 AUC while its top 100 alerts are all false
positives. It is reported here only because reviewers expect it, and it is deliberately not the
headline.

**Precision@k is the operational metric.** An AML team has fixed review capacity, a set number
of alerts per analyst per day, so what matters is the hit rate in the top *k* the team will
actually open. That is the number a compliance officer would ask for, and it is what changes when
protection degrades ranking.

**Recall matters more than precision, asymmetrically.** A missed SAR is a regulatory failure; a
false positive costs analyst time. FFIEC examination procedure #12 explicitly forbids tuning
alert volume to staffing levels, so a model that improves precision by suppressing alerts is one
an examiner would criticise. Recall@k is therefore reported alongside precision@k.

**Calibration matters because scores drive decisions.** If a model says 0.8, roughly 80% of such
cases should be suspicious, otherwise thresholds set on its scores mean nothing, and every
downstream tuning decision inherits the error. Expected Calibration Error captures this.

**Alert review rate and lift** translate the model into the language of a monitoring programme:
what fraction of the book must be reviewed, and how much better than random is the queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class AMLMetrics:
    """Metrics for a scored, ranked alert queue."""

    # Operational: what the team sees at its actual review depth.
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    lift_at_k: dict[int, float] = field(default_factory=dict)

    # Threshold-free summaries.
    average_precision: float = 0.0
    roc_auc: float = 0.0

    # Reliability of the scores themselves.
    expected_calibration_error: float = 0.0

    # At the operating threshold that maximises F1.
    operating_threshold: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    false_positive_rate: float = 0.0
    alert_review_rate: float = 0.0

    n_samples: int = 0
    n_positive: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "average_precision": round(self.average_precision, 4),
            "roc_auc": round(self.roc_auc, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "precision_at_k": {str(k): round(v, 4) for k, v in self.precision_at_k.items()},
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "lift_at_k": {str(k): round(v, 4) for k, v in self.lift_at_k.items()},
            "operating_threshold": round(self.operating_threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "alert_review_rate": round(self.alert_review_rate, 4),
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
        }

    def summary(self) -> str:
        # Report the deepest review depth actually computed rather than assuming 50 exists.
        # `.get(50, 0)` printed "P@50=0.000" on any test set smaller than 50 rows, a
        # catastrophic-looking number for a depth that was never evaluated.
        if not self.precision_at_k:
            return f"AP={self.average_precision:.3f} (no review depths evaluated)"
        depth = max(self.precision_at_k)
        return (
            f"AP={self.average_precision:.3f} "
            f"P@{depth}={self.precision_at_k[depth]:.3f} "
            f"R@{depth}={self.recall_at_k.get(depth, 0):.3f} "
            f"lift@{depth}={self.lift_at_k.get(depth, 0):.1f}x "
            f"ECE={self.expected_calibration_error:.3f}"
        )


def _expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, bins: int = 10
) -> float:
    """Mean gap between predicted confidence and observed frequency, weighted by bin size.

    Zero means a score of 0.8 corresponds to an 80% chance of being suspicious. Anything else
    means thresholds set on these scores do not mean what they appear to.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        # First bin is closed on the left. With `>` on every edge a score of exactly 0.0 fell
        # into no bin at all, and `RandomForestClassifier.predict_proba` emits exact 0.0
        # routinely at an ~8% positive rate, on a representative run 80% of scores were
        # dropped and the bin weights summed to 0.2 instead of 1.0. The docstring's "weighted
        # by bin size" was then false, and the reported ECE, the entire justification for
        # ranking rather than thresholding, was biased low by an unknown amount.
        in_bin = (scores >= low) & (scores <= high) if i == 0 else (
            (scores > low) & (scores <= high)
        )
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(labels[in_bin].mean() - scores[in_bin].mean())
    return float(error)


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """The F1-maximising decision threshold for these labels/scores.

    Exposed so the operating point can be selected on the TRAINING fold and then applied
    unchanged to the test fold, rather than selected and scored on the same test fold (which
    overfits the threshold and biases every thresholded metric upward). Returns 0.5 when the
    curve is degenerate (one class, or no thresholds).
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(
        2 * precisions * recalls, precisions + recalls,
        out=np.zeros_like(precisions), where=(precisions + recalls) > 0,
    )
    best = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return float(thresholds[best]) if len(thresholds) else 0.5


def compute(
    labels: np.ndarray,
    scores: np.ndarray,
    review_depths: tuple[int, ...] = (10, 25, 50, 100),
    operating_threshold: float | None = None,
) -> AMLMetrics:
    """Compute the full metric set for one scored queue.

    `review_depths` are the alert counts a team might actually work through. They are absolute
    rather than proportional because review capacity is a headcount, not a percentage.

    `operating_threshold`, when given, is applied as-is (the caller selected it on a *separate*
    fold, so the reported precision/recall/FPR carry no threshold-overfitting bias). When None
    it is F1-selected on the passed labels/scores, the standalone default, which the caller is
    responsible for knowing is optimistic if labels/scores are the evaluation set.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    positives = int(labels.sum())
    base_rate = positives / len(labels) if len(labels) else 0.0

    metrics = AMLMetrics(n_samples=len(labels), n_positive=positives)
    if positives == 0 or positives == len(labels):
        return metrics

    metrics.average_precision = float(average_precision_score(labels, scores))
    metrics.roc_auc = float(roc_auc_score(labels, scores))
    metrics.expected_calibration_error = _expected_calibration_error(labels, scores)

    # Stable sort. Random-forest probabilities are heavily tied, multiples of 1/200 at
    # `n_estimators=200`, so quicksort's arbitrary ordering within a tie made a large share
    # of the top-k head irreproducible run to run. `hybrid.py` already breaks ties explicitly
    # with the note that "reconstructability is a five-year obligation"; the two modules
    # ranked the same scores with opposite guarantees.
    order = np.argsort(-scores, kind="stable")
    for depth in review_depths:
        k = min(depth, len(labels))
        top = labels[order[:k]]
        precision = float(top.mean())
        # Recorded under the depth actually evaluated. Storing a clamped depth under its
        # requested label published `precision_at_k[100]` computed over 40 rows on a 40-row
        # test set, and `summary()`'s `.get(50, 0)` then fabricated a 0.000 for a depth that
        # was never computed.
        metrics.precision_at_k[k] = precision
        metrics.recall_at_k[k] = float(top.sum() / positives)
        # Lift: how much better than working the queue in random order.
        metrics.lift_at_k[k] = precision / base_rate if base_rate else 0.0

    # Operating point: use the caller-supplied threshold (selected on a separate fold, so the
    # reported precision/recall carry no overfitting bias) or F1-select it here. Recorded
    # explicitly either way, so a reviewer sees which threshold the reported figures belong to.
    if operating_threshold is not None:
        metrics.operating_threshold = float(operating_threshold)
    else:
        metrics.operating_threshold = select_f1_threshold(labels, scores)

    predicted = (scores >= metrics.operating_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    metrics.precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    metrics.recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    metrics.false_positive_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
    # Share of the book an analyst must open at this threshold, the staffing question.
    metrics.alert_review_rate = float(predicted.mean())

    return metrics
