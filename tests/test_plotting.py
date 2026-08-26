"""The training-plot seam: distinct figures, no overlay, graceful degradation.

Pins the overlay bug that actually occurred: SHAP's plot functions draw onto the current
axes and, called back to back without resetting state, superimpose every plot onto one
canvas (all four SHAP PNGs came out byte-identical and unreadable). These tests assert the
figures are distinct objects with distinct content, and that a degenerate fold is skipped
rather than crashing the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _fig_bytes(fig) -> int:
    import io

    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=60)
    plt.close(fig)
    return len(buf.getvalue())


def test_evaluation_figures_are_three_distinct_plots():
    from amlguard.plotting import evaluation_figures
    rng = np.random.default_rng(0)
    labels = np.array([0] * 80 + [1] * 20)
    scores = np.clip(labels * 0.4 + rng.normal(0.3, 0.2, 100), 0, 1)
    figs = evaluation_figures(labels, scores, "test")
    assert set(figs) == {"plots/precision_recall.png", "plots/roc.png",
                         "plots/score_histogram.png"}
    sizes = [_fig_bytes(f) for f in figs.values()]
    assert len(set(sizes)) == 3, f"figures not distinct: {sizes}"


def test_single_class_fold_is_skipped_not_crashed():
    from amlguard.plotting import evaluation_figures
    labels = np.zeros(50, dtype=int)  # degenerate: one class only
    scores = np.random.default_rng(1).random(50)
    assert evaluation_figures(labels, scores, "degenerate") == {}


def test_shap_figures_do_not_overlay():
    """The regression guard: four SHAP plots must be four DISTINCT figures."""
    import shap
    from sklearn.ensemble import RandomForestClassifier

    from amlguard.explain import explanation_for
    from amlguard.plotting import shap_figures

    rng = np.random.default_rng(2)
    X = rng.random((120, 6))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    names = [f"f{i}" for i in range(6)]
    expl = explanation_for(model, X, names, background=X, sample=60)
    assert isinstance(expl, shap.Explanation)

    figs = shap_figures(expl, "test")
    assert set(figs) == {"plots/shap_beeswarm.png", "plots/shap_bar.png",
                         "plots/shap_waterfall.png", "plots/shap_dependence.png"}
    sizes = [_fig_bytes(f) for f in figs.values()]
    # The overlay bug made these identical; distinct sizes prove separate canvases.
    assert len(set(sizes)) == len(sizes), f"SHAP figures overlaid (equal sizes): {sizes}"
