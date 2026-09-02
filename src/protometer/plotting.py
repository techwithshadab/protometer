"""Figures for the training run: evaluation curves and the SHAP plot families.

The metrics answer "how good and what does it rely on" as numbers; a reviewer also needs to
*see* them. This module renders, from one classifier bundle, the graphs that make the numbers
legible and logs nothing itself, it returns `{artifact_name: figure}` and the caller decides
where they go (MLflow, in practice). Keeping it pure keeps it testable and keeps MLflow out of
the plotting logic.

Two families:

- **Evaluation curves** — precision-recall (whose area is the headline AP), ROC (whose area is
  ROC-AUC), and the score histogram by class (does the model separate illicit from licit).
- **SHAP plots** — the taxonomy the SHAP library itself distinguishes, each answering a
  different question:
    * `beeswarm`  — global: every feature's value-vs-impact distribution across the test set;
    * `bar`       — global: mean |SHAP|, the ranked reliance the model actually uses;
    * `waterfall` — local: how one high-scoring case was built up from the base rate, the
                    view an analyst's case note is denominated in;
    * `scatter`   — dependence: the top feature's SHAP value against its own value, showing
                    whether the relationship is monotone or has a threshold.

Every figure is deterministic (the SHAP sample is seeded) so the same corpus produces the same
plots. matplotlib runs headless (`Agg`) so this works with no display.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from protometer.log import get_logger

_log = get_logger("plotting")


def _mpl():
    import matplotlib

    matplotlib.use("Agg")  # headless: no display in CI or a container
    import matplotlib.pyplot as plt

    return plt


def evaluation_figures(labels: np.ndarray, scores: np.ndarray, scope: str) -> dict[str, Any]:
    """PR curve, ROC curve, and score-by-class histogram for one scope's test fold."""
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    plt = _mpl()
    figs: dict[str, Any] = {}

    # Degenerate fold (one class only) makes these metrics undefined; skip rather than crash.
    if len(np.unique(labels)) < 2:
        _log.warning("scope %s: single-class test fold, skipping curves", scope)
        return figs

    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#1f77b4", lw=2)
    ax.axhline(labels.mean(), ls="--", color="#888", lw=1, label=f"base rate {labels.mean():.3f}")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"{scope}: precision-recall (AP={ap:.3f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    figs["plots/precision_recall.png"] = fig

    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#d62728", lw=2)
    ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1, label="chance")
    ax.set(xlabel="false positive rate", ylabel="true positive rate", xlim=(0, 1), ylim=(0, 1.02),
           title=f"{scope}: ROC (AUC={auc:.3f})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    figs["plots/roc.png"] = fig

    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.linspace(0, 1, 31)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, color="#2ca02c", label="licit", density=True)
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, color="#d62728", label="illicit", density=True)
    ax.set(xlabel="model score", ylabel="density", title=f"{scope}: score separation by class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["plots/score_histogram.png"] = fig

    return figs


def shap_figures(explanation: Any, scope: str, top_case: int | None = None) -> dict[str, Any]:
    """The four SHAP plot families from a prepared `shap.Explanation`.

    `top_case` selects which row the local waterfall explains; when None the highest-impact
    row (largest summed |SHAP|) is used, which is the case most worth showing.
    """
    import shap

    plt = _mpl()
    figs: dict[str, Any] = {}

    # SHAP's plot functions draw onto the *current* matplotlib axes and do not reliably open a
    # fresh figure; calling them back to back overlays every plot onto one canvas (a real bug
    # this module hit: beeswarm + bar + waterfall + histogram all superimposed). The discipline
    # that fixes it: close all state, open exactly one figure, let SHAP draw into it, title and
    # capture THAT figure, and never touch `gcf()` again until the next plot has reset state.
    def _render(draw, title: str, key: str) -> None:
        try:
            plt.close("all")  # start from a clean slate, no accumulated axes
            draw()            # SHAP opens its own figure and draws into it
            fig = plt.gcf()   # capture exactly that figure, nothing carried over
            fig.set_size_inches(7, 5)
            fig.suptitle(title, fontsize=10)
            fig.tight_layout()
            figs[key] = fig
        except Exception as exc:  # noqa: BLE001, one plot failing must not lose the others
            _log.warning("%s failed for %s: %s", key, scope, exc)
            plt.close("all")

    if top_case is None:
        top_case = int(np.abs(explanation.values).sum(axis=1).argmax())
    top_feat = int(np.abs(explanation.values).mean(axis=0).argmax())
    feat_name = explanation.feature_names[top_feat]

    _render(lambda: shap.plots.beeswarm(explanation, show=False, max_display=15),
            f"{scope}: SHAP beeswarm (global impact)", "plots/shap_beeswarm.png")
    _render(lambda: shap.plots.bar(explanation, show=False, max_display=15),
            f"{scope}: SHAP bar (mean |value| reliance)", "plots/shap_bar.png")
    _render(lambda: shap.plots.waterfall(explanation[top_case], show=False, max_display=12),
            f"{scope}: SHAP waterfall (highest-impact case)", "plots/shap_waterfall.png")
    _render(lambda: shap.plots.scatter(explanation[:, top_feat], show=False),
            f"{scope}: SHAP dependence — {feat_name}", "plots/shap_dependence.png")

    return figs
