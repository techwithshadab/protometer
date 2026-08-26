"""Explainability, why the model flagged this transaction, and what protection cost that.

Explainability is not optional decoration in AML. An institution must be able to articulate why
an alert was raised: FFIEC requires a designated decision maker able to justify a filing, SARs
demand a narrative that stands up to examination, and an unexplainable model is one a validation
team will refuse to approve. A copilot whose reasoning cannot be reconstructed five years later
fails the reconstructability requirement outright.

There is also a question specific to this project that only SHAP can answer: **does protection
change what the model relies on?** Global accuracy could hold steady while the model quietly
shifts from amount-based reasoning to structural proxies. That would be a materially different
model wearing the same score, and an institution would want to know.

Two views are produced:

  * **Global**, mean absolute SHAP per feature, i.e. what the model relies on overall. Compared
    across protection scopes, this shows whether protection shifted the model's basis.
  * **Local**, the contribution of each feature to one decision, which is the artifact an
    investigator would attach to a case file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Background rows passed to TreeExplainer. The SHAP documentation recommends 100-1000 rows
# for interventional expectations, and TreeExplainer's masker itself subsamples anything
# larger than 100, with ITS own rng. Capping at 100 here keeps the subsample under this
# module's fixed seed, so the same model and data always produce the same attributions.
_BACKGROUND_CAP = 100
_BACKGROUND_SEED = 20260818

# One explainer per (model, background) pair. Rebuilding a TreeExplainer for every escalated
# alert repeated identical setup 50x per hybrid run; the strong reference to the model keys
# the cache safely (id() alone could be reused after garbage collection).
_EXPLAINER_CACHE: dict[int, tuple[Any, Any, int]] = {}


def _build_explainer(model: Any, background: np.ndarray) -> Any:
    """The single TreeExplainer construction seam, interventional, probability units.

    Both choices follow the SHAP documentation rather than the defaults. Without `data`,
    TreeExplainer silently runs `tree_path_dependent`, which conditions on the tree's own
    split distribution, with this model's heavily correlated feature pairs (origin/
    beneficiary mirrors, the degree family), that spreads credit along correlations instead
    of measuring each feature against the training distribution. `model_output="probability"`
    (which *requires* interventional mode) makes the narrative's "raises the score by 0.034"
    literally denominated in the score's own units instead of coincidentally so.

    An earlier version accepted a `background` parameter and ignored it, running
    tree_path_dependent while callers believed they were conditioning on the training
    distribution. The cache below is also why this function exists: one construction path
    means one place to get these choices right.
    """
    import shap

    original_rows = int(background.shape[0])
    cached = _EXPLAINER_CACHE.get(id(model))
    if cached is not None and cached[0] is model and cached[2] == original_rows:
        return cached[1]

    if original_rows > _BACKGROUND_CAP:
        rng = np.random.default_rng(_BACKGROUND_SEED)
        background = background[
            rng.choice(original_rows, _BACKGROUND_CAP, replace=False)
        ]
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )
    _EXPLAINER_CACHE[id(model)] = (model, explainer, original_rows)
    return explainer


@dataclass
class FeatureAttribution:
    """One feature's contribution to a single prediction."""

    feature: str
    value: float
    contribution: float

    @property
    def direction(self) -> str:
        return "raises" if self.contribution > 0 else "lowers"


@dataclass
class Explanation:
    """Why the model scored one transaction the way it did."""

    score: float
    base_value: float
    attributions: list[FeatureAttribution] = field(default_factory=list)

    def top(self, n: int = 5) -> list[FeatureAttribution]:
        return sorted(self.attributions, key=lambda a: -abs(a.contribution))[:n]

    def narrative(self) -> str:
        """Plain-language rationale, in the shape a case note would carry.

        Deliberately mechanical rather than generated: an explanation produced by a language
        model would itself need explaining, and a hallucinated justification attached to a SAR
        is a false statement to a federal database.
        """
        parts = [f"Model score {self.score:.2f} (baseline {self.base_value:.2f})."]
        for attribution in self.top(3):
            parts.append(
                f"{attribution.feature} = {attribution.value:.4g} "
                f"{attribution.direction} the score by {abs(attribution.contribution):.3f}."
            )
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "base_value": round(self.base_value, 4),
            "narrative": self.narrative(),
            "attributions": [
                {
                    "feature": a.feature,
                    "value": round(a.value, 4),
                    "contribution": round(a.contribution, 4),
                }
                for a in self.top(10)
            ],
        }


def global_importance(
    model: Any,
    features: np.ndarray,
    feature_names: list[str],
    sample: int = 300,
    background: np.ndarray | None = None,
) -> dict[str, float]:
    """Mean absolute SHAP value per feature, what the model relies on overall.

    `features` are the rows explained; `background` is the reference distribution the
    interventional explainer conditions on, and it should be the TRAINING fold, the same
    distribution `explain_prediction` uses, so the global and per-decision views agree.
    An earlier version used `features[:300]` as both explicand and background, a
    ledger-ordered head of the test fold, which contradicted the module's own "relative to
    the training distribution" framing. Both are now sampled deterministically at random.
    """
    rng = np.random.default_rng(_BACKGROUND_SEED)

    def _sample(arr: np.ndarray) -> np.ndarray:
        if len(arr) <= sample:
            return arr
        return arr[rng.choice(len(arr), sample, replace=False)]

    subset = _sample(features)
    bg = _sample(background if background is not None else features)
    explainer = _build_explainer(model, bg)
    values = explainer.shap_values(subset)

    # Binary classifiers return per-class arrays in some SHAP versions; take the positive class.
    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]

    means = np.abs(values).mean(axis=0)
    total = means.sum() or 1.0
    return {
        name: float(value / total)
        for name, value in sorted(
            zip(feature_names, means), key=lambda pair: -pair[1]
        )
    }


def explain_prediction(
    model: Any, row: np.ndarray, feature_names: list[str], background: np.ndarray
) -> Explanation:
    """Attribute one prediction to its features, relative to the training distribution.

    `background` is the training-fold feature matrix: the attribution answers "compared to a
    typical training row, what moved this score", the framing an analyst's case note needs.
    """
    explainer = _build_explainer(model, background)
    values = explainer.shap_values(row.reshape(1, -1))
    base = explainer.expected_value

    if isinstance(values, list):
        values, base = values[1], base[1]
    elif values.ndim == 3:
        values, base = values[:, :, 1], base[1] if np.ndim(base) else base

    contributions = np.asarray(values).reshape(-1)
    score = float(model.predict_proba(row.reshape(1, -1))[0, 1])

    return Explanation(
        score=score,
        base_value=float(np.asarray(base).reshape(-1)[0]),
        attributions=[
            FeatureAttribution(feature=name, value=float(row[i]), contribution=float(contributions[i]))
            for i, name in enumerate(feature_names)
        ],
    )


def explanation_for(
    model: Any,
    features: np.ndarray,
    feature_names: list[str],
    background: np.ndarray,
    sample: int = 200,
) -> Any:
    """A `shap.Explanation` over a deterministic sample of `features`, for plotting.

    This is the object the modern SHAP plot families (beeswarm, bar, waterfall, scatter)
    consume: SHAP values, the data they explain, the base value, and feature names, all
    aligned. It goes through the same cached interventional/probability explainer as every
    other view here, so a plot and a case-note attribution never disagree about what the model
    relies on. Sampling is seeded so the same corpus produces the same plots.
    """
    import shap

    rng = np.random.default_rng(_BACKGROUND_SEED)
    subset = features
    if len(features) > sample:
        subset = features[rng.choice(len(features), sample, replace=False)]

    explainer = _build_explainer(model, background)
    values = explainer.shap_values(subset)
    base = explainer.expected_value
    # Positive class for binary output, across the SHAP return-shape variants.
    if isinstance(values, list):
        values, base = values[1], base[1]
    elif np.ndim(values) == 3:
        values = values[:, :, 1]
        base = base[1] if np.ndim(base) else base
    base_arr = np.full(len(subset), float(np.asarray(base).reshape(-1)[0]))
    return shap.Explanation(
        values=np.asarray(values),
        base_values=base_arr,
        data=subset,
        feature_names=feature_names,
    )


def basis_shift(
    importance_a: dict[str, float], importance_b: dict[str, float]
) -> dict[str, float]:
    """How much the model's reliance moved between two protection scopes.

    Positive means the feature matters *more* under scope B. A large shift with unchanged
    accuracy is the interesting case: the model is compensating, and an institution should know
    it is now leaning on structural proxies rather than the values it was designed around.
    """
    features = set(importance_a) | set(importance_b)
    return {
        feature: round(importance_b.get(feature, 0.0) - importance_a.get(feature, 0.0), 4)
        for feature in sorted(
            features,
            key=lambda f: -abs(importance_b.get(f, 0.0) - importance_a.get(f, 0.0)),
        )
    }
