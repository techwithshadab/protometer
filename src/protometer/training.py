"""Training stage, a classifier fitted on protected data.

The hackathon's primary challenge names four stages: *ingestion, training, embedding, and
inference*. Training is the one almost no submission will attempt, because it is the stage
where protection is hardest to reason about: an LLM can be told "these tokens are stable
pseudonyms", but a gradient-based learner has no such affordance. It sees whatever the feature
extractor produces and nothing else.

That makes it the sharpest available test of the project's central question. A model trained on
tokenized data either learns something or it does not, and the answer is a number rather than a
judgement.

**What is being measured.** The classifier predicts whether a transaction belongs to a planted
laundering typology. It is fitted **per protection scope**, on that scope's protected ledger,
and evaluated against the same held-out split every time. The delta between scopes is what
protection costs a trained model.

**Why this is not the LLM measurement in miniature.** The evaluation harness suffers a
confound: its checkpoints score figures the prompt instructs the model to copy, so
transcription outscores reasoning. Training has no such escape, there is no prompt to copy
from, and a feature that has been tokenized is genuinely unavailable. Whatever signal survives
is signal the protection left behind.

**A negative result is a result.** If a classifier trained on tokenized amounts learns nothing,
that is a finding about where protection stops being free, and it is reported as such rather
than tuned away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from protometer import graph_features
from protometer import metrics as aml_metrics
from protometer.explain import global_importance
from protometer.log import get_logger

_log = get_logger("training")

# Fixed so the split is identical across every scope. Without this, scopes would differ in
# their train/test partition as well as in protection, confounding the comparison.
SPLIT_SEED = 20260813
TEST_FRACTION = 0.3


@dataclass
class TrainingResult:
    """What one scope's classifier achieved."""

    scope_name: str
    n_train: int
    n_test: int
    n_positive: int
    # Average precision is the headline: the class is ~8% positive, so accuracy is
    # uninformative and ROC-AUC is optimistic on imbalanced data.
    average_precision: float
    roc_auc: float
    # Standard deviation of AP across RandomForest seeds (split and features fixed, only
    # random_state varies). Reported because single-seed point estimates invited over-reading
    # of ~1-point scope differences: an effect smaller than this SD is seed noise, not
    # protection cost. 0.0 until computed.
    average_precision_seed_std: float = 0.0
    # Mean AP over the SAME seed set the SD is computed on (which INCLUDES SPLIT_SEED, the seed
    # the headline `average_precision` is fit at). Reported so a reader can read "AP ±seed_std"
    # as a band around a member of its own population, not around an unsampled point.
    average_precision_seed_mean: float = 0.0
    # Share of features that survived protection as usable numbers, which is the mechanism
    # behind any degradation.
    usable_feature_rate: float = 0.0
    # Operational metrics: what a fixed-capacity review queue actually sees.
    aml: dict[str, Any] = field(default_factory=dict)
    # Mean absolute SHAP per feature, compared across scopes, shows whether protection
    # changed what the model relies on rather than only how well it scores.
    shap_importance: dict[str, float] = field(default_factory=dict)
    # Recall per planted typology at the operating threshold. The aggregate AP hides that the
    # model barely sees some typologies, measured: funnel 91%, trade-based 25%, and a bank
    # tunes scenarios per typology, not per average.
    per_typology_recall: dict[str, dict] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    # The fitted bundle itself, so callers can log the model artifact (signature, registry)
    # without refitting. Deliberately absent from `to_dict`, it is an object, not a result.
    bundle: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope_name,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_positive": self.n_positive,
            "average_precision": round(self.average_precision, 4),
            "average_precision_seed_std": round(self.average_precision_seed_std, 4),
            "average_precision_seed_mean": round(self.average_precision_seed_mean, 4),
            "roc_auc": round(self.roc_auc, 4),
            "usable_feature_rate": round(self.usable_feature_rate, 4),
            "aml_metrics": self.aml,
            "per_typology_recall": {
                k: {"recall": round(v["recall"], 4), "n_caught": v["n_caught"],
                    "n_total": v["n_total"]}
                for k, v in sorted(self.per_typology_recall.items())
            },
            "shap_importance": {
                k: round(v, 4) for k, v in list(self.shap_importance.items())[:10]
            },
            "feature_importances": {
                k: round(v, 4) for k, v in sorted(
                    self.feature_importances.items(), key=lambda kv: -kv[1]
                )
            },
        }


def _as_number(value: Any) -> float | None:
    """Parse a ledger value, or None when protection has made it unparseable.

    This is the whole mechanism. A tokenized amount is still a *string that looks like* an
    amount, format-preserving tokenization guarantees it, so it parses fine and carries a
    plausible but meaningless magnitude. A tokenized date likewise. The classifier therefore
    does not see missing features; it sees **wrong** ones, which is a harder failure than
    absence and closer to what a real deployment would experience.
    """
    if value is None:
        return None
    try:
        return float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _parse_date(value: Any) -> float | None:
    """Days since epoch, or None if the value no longer parses as a date."""
    text = str(value or "")
    parts = text.split("-")
    if len(parts) != 3:
        return None
    try:
        year, month, day = (int(p) for p in parts)
        return year * 365.25 + month * 30.44 + day
    except (ValueError, TypeError):
        return None


FEATURE_NAMES = (
    "amount",
    "amount_below_threshold",
    "amount_log",
    "day_index",
    "is_deposit_channel",
    "is_wire_channel",
    "origin_out_degree",
    "beneficiary_in_degree",
    "counterparty_shared_count",
    "memo_has_invoice_ref",
)


def extract_features(
    transactions: list[dict], fit_on: list[dict] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Build a feature matrix from a ledger, protected or clear.

    `fit_on` supplies the rows the population aggregates (degrees, pair counts) are computed
    from; features are still emitted for every row in `transactions`. Pass the training fold
    to avoid leaking test-fold structure into training rows. Defaults to `transactions` for
    callers that score a whole ledger with a model already fitted elsewhere.

    Features deliberately span three groups so degradation can be attributed:

      * **Value features** (amount, threshold proximity), destroyed when AMOUNT is tokenized.
      * **Temporal features** (day index), destroyed when DATETIME is tokenized.
      * **Structural features** (degrees, shared counterparties, channel), survive every
        scope, because party ids and channels are never protected.

    If the classifier retains signal at wide scopes, structure is why.
    """
    # Aggregates are fit on `fit_on`, the training fold, and *applied* to every row.
    #
    # Accumulating them over all transactions computed each training row's degree from a
    # ledger that included the test fold, which is transductive leakage: the model saw
    # aggregate structure derived from rows it was about to be scored on. The comment block in
    # `train_scope` was emphatic about exactly this hazard for GuiltyWalker while these
    # counters, and the graph the walk features are built from, leaked it silently.
    #
    # Applying training-fold counts to test rows is the correct deployment analogue: a model
    # in production scores today's transaction against history it has already seen, never
    # against the future.
    fit_rows = transactions if fit_on is None else fit_on
    out_degree: dict[str, int] = {}
    in_degree: dict[str, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for txn in fit_rows:
        origin, beneficiary = txn["origin_party_id"], txn["beneficiary_party_id"]
        out_degree[origin] = out_degree.get(origin, 0) + 1
        in_degree[beneficiary] = in_degree.get(beneficiary, 0) + 1
        pair_counts[(origin, beneficiary)] = pair_counts.get((origin, beneficiary), 0) + 1

    rows: list[list[float]] = []
    usable = 0
    total = 0

    for txn in transactions:
        amount = _as_number(txn.get("amount"))
        day = _parse_date(txn.get("value_date"))
        channel = str(txn.get("channel", "")).lower()
        memo = str(txn.get("memo", ""))

        total += 2
        usable += (amount is not None) + (day is not None)

        # Unparseable values become sentinels rather than being dropped: a row with a
        # tokenized amount is still a row a deployed system must classify.
        rows.append([
            amount if amount is not None else -1.0,
            1.0 if (amount is not None and 0 < amount < 10_000) else 0.0,
            float(np.log1p(amount)) if amount is not None and amount > 0 else -1.0,
            day if day is not None else -1.0,
            1.0 if "deposit" in channel else 0.0,
            1.0 if channel in ("wire", "swift", "correspondent") else 0.0,
            float(out_degree.get(txn["origin_party_id"], 0)),
            float(in_degree.get(txn["beneficiary_party_id"], 0)),
            float(pair_counts.get(
                (txn["origin_party_id"], txn["beneficiary_party_id"]), 0
            )),
            1.0 if "INV-" in memo else 0.0,
        ])

    return np.asarray(rows, dtype=float), list(FEATURE_NAMES)


def _split_by_instance(
    transactions: list[dict],
    ground_truth: list[dict],
    clear_dates: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test split that keeps every typology instance wholly on one side.

    Held-out evaluation is supposed to answer "would this model catch a pattern it has never
    seen". A transaction-level split answers a much easier question, "would it catch the rest
    of a pattern it has already seen most of", because the planted instances that constitute
    the positive class straddle the fold boundary.

    The split is **temporal**: train on the earlier part of the ledger, test on the later.
    That is how every deployed AML model is validated, you have the past and you are asked
    about the future, and it is the only split that answers the question a bank asks.

    Two alternatives were measured and rejected on this corpus:

      * *By transaction* (the original). Instances straddle the boundary, so the train-fold
        illicit set was identical to the full illicit set: **100%** of test positives touched
        a party already labelled illicit, against 57% of benign. `b_gw_mean_length`, the top
        feature by SHAP, was the label one hop removed.
      * *By typology instance.* Better but insufficient, 73 of 196 illicit parties appear in
        more than one instance, leaving a 65-point gap (94.1% of positives vs 29.3% of benign).
      * *By connected component of shared parties.* This removes the bleed entirely, and is
        impossible here: 119 of 122 instances form a single component, so the split degenerates
        to 98% of positives in one fold. That the typology parties are this interconnected is
        itself a property of the corpus worth recording.

    A temporal split cuts across all three problems: a party active in both periods is
    legitimately known to the model, exactly as it would be in production, while the *pattern*
    being scored has not been seen. Instances are assigned wholly to the fold their activity
    predominantly falls in, so no instance straddles the boundary.
    """
    instance_of: dict[str, int] = {}
    for i, instance in enumerate(ground_truth):
        for txn_id in instance["transaction_ids"]:
            instance_of[txn_id] = i

    # Clear dates when supplied, so the split survives DATETIME being tokenized.
    date_of = clear_dates or {
        txn["transaction_id"]: txn["value_date"] for txn in transactions
    }
    dates = sorted(date_of[txn["transaction_id"]] for txn in transactions)
    cutoff = dates[int(len(dates) * (1 - TEST_FRACTION))]

    # An instance is assigned by its **earliest** transaction, and every row of it follows.
    # Assigning by majority let a pattern beginning before the cutoff be scored as test data
    # while its opening transactions sat in training, which both leaks and breaks the
    # temporal guarantee the split exists to provide.
    instance_side: dict[int, bool] = {}
    for i, instance in enumerate(ground_truth):
        instance_dates = [
            date_of[t] for t in instance["transaction_ids"] if t in date_of
        ]
        if instance_dates:
            instance_side[i] = min(instance_dates) >= cutoff

    train_idx: list[int] = []
    test_idx: list[int] = []
    for i, txn in enumerate(transactions):
        instance = instance_of.get(txn["transaction_id"])
        if instance is not None:
            in_test = instance_side.get(instance, False)
        else:
            in_test = date_of[txn["transaction_id"]] >= cutoff
        (test_idx if in_test else train_idx).append(i)

    return np.asarray(train_idx), np.asarray(test_idx)


@dataclass
class ClassifierBundle:
    """Everything a scored classifier run produces, from one construction path.

    This bundle exists because the alternative was measured and it failed: the hybrid script
    duplicated the split/feature/illicit-set logic inline, the copy went stale, and the
    published queue precision was computed on a model carrying every leak `train_scope` had
    eliminated, transaction-level split, whole-ledger feature fitting, no dual-membership
    exclusion. A four-piece surface callers must assemble correctly is a surface callers will
    assemble incorrectly; one function they cannot mis-assemble is the fix.
    """

    transactions: list[dict]
    model: Any
    features: "np.ndarray"
    feature_names: list[str]
    labels: "np.ndarray"
    train_idx: "np.ndarray"
    test_idx: "np.ndarray"
    test_scores: "np.ndarray"
    usable_rate: float
    # Deterministic identity of this classifier build, for the five-year reconstruction
    # obligation: same corpus + same code parameters = same hash.
    model_hash: str


def build_classifier(protected_dir: Path, corpus_dir: Path) -> ClassifierBundle:
    """The one construction path for the leak-guarded classifier.

    Encapsulates, in order: labels from clear ground truth; **temporal instance split on
    clear dates** (tokenized dates land the cutoff nowhere); training-fold illicit set with
    **dual-membership exclusion**; features and graph **fit on the training fold only**.
    Each of those clauses corresponds to a measured leak we caught and documented, callers
    that need a classifier take the bundle and cannot skip one.
    """
    import hashlib

    transactions = json.loads((protected_dir / "transactions.json").read_text())
    ground_truth = json.loads((corpus_dir / "ground_truth.json").read_text())
    flagged = {t for instance in ground_truth for t in instance["transaction_ids"]}

    clear_transactions = json.loads((corpus_dir / "transactions.json").read_text())
    clear_dates = {t["transaction_id"]: t["value_date"] for t in clear_transactions}

    labels = np.asarray(
        [1 if txn["transaction_id"] in flagged else 0 for txn in transactions]
    )
    train_idx, test_idx = _split_by_instance(transactions, ground_truth, clear_dates)

    train_rows = [transactions[i] for i in train_idx]
    train_ids = {txn["transaction_id"] for txn in train_rows}
    test_ids = {transactions[i]["transaction_id"] for i in test_idx}

    train_illicit_parties = {
        party
        for instance in ground_truth
        for party in instance["party_ids"]
        if any(t in train_ids for t in instance["transaction_ids"])
    }
    test_instance_parties = {
        party
        for instance in ground_truth
        for party in instance["party_ids"]
        if any(t in test_ids for t in instance["transaction_ids"])
    }
    train_illicit_parties -= test_instance_parties

    features, names = extract_features(transactions, fit_on=train_rows)
    graph = graph_features.extract(
        transactions, illicit_nodes=train_illicit_parties, fit_on=train_rows
    )
    features = np.hstack([features, graph.values])
    names = names + graph.names

    amount_usable = float(np.mean(features[:, 0] >= 0))
    day_usable = float(np.mean(features[:, 3] >= 0))

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=SPLIT_SEED,
        n_jobs=-1,
    )
    model.fit(features[train_idx], labels[train_idx])
    test_scores = model.predict_proba(features[test_idx])[:, 1]

    # The feature-matrix digest is what makes the hash scope-sensitive. Hashing only the
    # parameters and split produced the same identity for every protection scope, the exact
    # models the hash exists to tell apart, since the whole experiment is that their inputs
    # differ.
    model_hash = hashlib.sha256(
        (
            f"rf:n=200:depth=8:seed={SPLIT_SEED}:"
            f"features={','.join(names)}:"
            f"train={hashlib.sha256(''.join(sorted(train_ids)).encode()).hexdigest()[:16]}:"
            f"matrix={hashlib.sha256(features.tobytes()).hexdigest()[:16]}"
        ).encode()
    ).hexdigest()[:16]

    return ClassifierBundle(
        transactions=transactions,
        model=model,
        features=features,
        feature_names=names,
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        test_scores=test_scores,
        usable_rate=(amount_usable + day_usable) / 2,
        model_hash=model_hash,
    )


def train_scope(
    protected_dir: Path, corpus_dir: Path, scope_name: str
) -> TrainingResult:
    """Fit and evaluate a classifier on one scope's protected ledger.

    Labels come from the **clear** corpus's planted ground truth, matched by transaction id.
    Transaction ids are surrogate keys and are never protected, so this leaks nothing: the
    label says *which rows are suspicious*, which is what any supervised AML model is trained
    on, and the features come exclusively from the protected ledger.
    """
    bundle = build_classifier(protected_dir, corpus_dir)

    # The leak-guarded construction, temporal instance split, dual-membership exclusion,
    # training-fold-only feature fitting, lives entirely in `build_classifier`, shared with
    # the hybrid path. This function only measures what the bundle produced. SHAP is
    # best-effort: an explainability failure must not lose the measurement, but it is named
    # when it happens rather than silently reported as an empty importance table.
    y_test = bundle.labels[bundle.test_idx]
    # Select the operating threshold on the TRAINING fold and apply it to the test fold, so the
    # reported precision/recall/FPR and per-typology recall carry no threshold-overfitting bias
    # (selecting and scoring the operating point on the same test fold inflates every
    # thresholded metric). AP and ROC-AUC are threshold-free and unaffected either way.
    train_scores = bundle.model.predict_proba(bundle.features[bundle.train_idx])[:, 1]
    y_train = bundle.labels[bundle.train_idx]
    train_threshold = aml_metrics.select_f1_threshold(y_train, train_scores)
    operational = aml_metrics.compute(
        y_test, bundle.test_scores, operating_threshold=train_threshold
    )

    # Recall by typology at the operating threshold, because the blended number hides the
    # failure that matters: a scenario-tuning team asks "which typologies do we miss", and
    # trade-based laundering, whose defining indicator (invoice value mismatch) cannot exist
    # in a ledger-only corpus, is nearly invisible to this model.
    ground_truth = json.loads((corpus_dir / "ground_truth.json").read_text())
    typology_of = {
        txn_id: instance["typology"]
        for instance in ground_truth
        for txn_id in instance["transaction_ids"]
    }
    # Use the UNROUNDED threshold, the exact bytes metrics.compute applied for the aggregate
    # precision/recall/FPR. `to_dict()` rounds to 4dp, which can shift the cut for a score sitting
    # on the boundary, so the per-typology recall table and the aggregate recall would disagree by
    # a transaction on a small-denominator typology (moving its rate double digits).
    threshold = getattr(operational, "operating_threshold", None)
    if threshold is None:
        threshold = 0.5
    caught: dict[str, int] = {}
    totals: dict[str, int] = {}
    for index, score in zip(bundle.test_idx, bundle.test_scores):
        txn_id = bundle.transactions[index]["transaction_id"]
        typology = typology_of.get(txn_id)
        if typology is None:
            continue
        totals[typology] = totals.get(typology, 0) + 1
        if score >= threshold:
            caught[typology] = caught.get(typology, 0) + 1
    # Recall AND its denominator per typology: the recall alone hides that trade_based is
    # 2/8 and layering 7/16, small enough that a single transaction moves the rate double
    # digits. The table must carry "n" so it reads as the anecdote it is, not a stable rate.
    per_typology = {
        typology: {
            "recall": caught.get(typology, 0) / total,
            "n_caught": caught.get(typology, 0),
            "n_total": total,
        }
        for typology, total in totals.items()
    }

    # AP variance across RF seeds, so a reader knows the point estimate's noise floor. The
    # split, features and labels are all fixed; only the forest's randomness changes, which
    # is exactly the "is this scope difference real or seed noise" question.
    #
    # The reported headline AP is fit at random_state=SPLIT_SEED. The SD must characterize the
    # SAME population, so SPLIT_SEED is INCLUDED in the seed set (previously the SD sampled only
    # seeds 0..10, excluding the reported seed — so "AP ±seed_std" bracketed a point the SD never
    # sampled). We also report the seed-mean AP and the seed set, so a band is read as an interval
    # around a member of its own population, not around an unsampled outlier.
    from sklearn.metrics import average_precision_score

    seed_set = [SPLIT_SEED, *range(11)]
    seed_aps = []
    for seed in seed_set:
        m = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        m.fit(bundle.features[bundle.train_idx], bundle.labels[bundle.train_idx])
        seed_aps.append(average_precision_score(
            bundle.labels[bundle.test_idx],
            m.predict_proba(bundle.features[bundle.test_idx])[:, 1],
        ))
    ap_seed_std = float(np.std(seed_aps))
    ap_seed_mean = float(np.mean(seed_aps))

    try:
        shap_values = global_importance(
            bundle.model,
            bundle.features[bundle.test_idx],
            bundle.feature_names,
            background=bundle.features[bundle.train_idx],
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("SHAP failed for %s: %s: %s", scope_name, type(exc).__name__, str(exc)[:100])
        shap_values = {}

    return TrainingResult(
        scope_name=scope_name,
        n_train=len(bundle.train_idx),
        n_test=len(bundle.test_idx),
        n_positive=int(bundle.labels.sum()),
        average_precision=operational.average_precision,
        average_precision_seed_std=ap_seed_std,
        average_precision_seed_mean=ap_seed_mean,
        roc_auc=operational.roc_auc,
        aml=operational.to_dict(),
        shap_importance=shap_values,
        usable_feature_rate=bundle.usable_rate,
        per_typology_recall=per_typology,
        feature_importances=dict(
            zip(bundle.feature_names, bundle.model.feature_importances_.tolist())
        ),
        bundle=bundle,
    )
