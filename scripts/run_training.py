"""Train a classifier on each protected corpus and report the utility-vs-scope curve.

    python scripts/run_training.py                    # the committed CURVE_ORDER curve
    python scripts/run_training.py quasi-yearclear    # one opt-in scope (not in CURVE_ORDER)

No LLM calls, no API cost. This measures what protection costs a *trained model*, which is the
stage the primary challenge names and which no prompt-copying strategy can shortcut.

Passing scope names on the command line trains exactly those scopes instead of CURVE_ORDER, so
an opt-in scope like `quasi-yearclear` (year-in-clear dates) can be measured for $0
without re-billing the eight-scope curve or being wedged into the committed ordering.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from protometer.explain import basis_shift
from protometer.scopes import CURVE_ORDER, get_scope
from protometer.tracking import Tracker
from protometer.training import train_scope


def main() -> int:
    corpus = ROOT / "data" / "corpus"
    tracker = Tracker("protometer-training")
    from protometer.persist import RUN_ID
    from protometer.tracking import corpus_source_fingerprint

    corpus_fp = corpus_source_fingerprint(corpus)
    # Explicit scope names on argv override CURVE_ORDER, so an opt-in scope can be trained on its
    # own. Validate each name against the registry up front, an unknown scope should fail loudly,
    # not silently train nothing.
    requested = sys.argv[1:]
    if requested:
        for name in requested:
            get_scope(name)  # raises on an unknown scope name
        scopes = requested
    else:
        scopes = list(CURVE_ORDER)
    results = []
    for name in scopes:
        d = ROOT / "data" / "protected" / get_scope(name).slug
        if not (d / "transactions.json").exists():
            print(f"skip {name}: no protected corpus at {d.relative_to(ROOT)} "
                  f"(ingest this scope first)")
            continue
        r = train_scope(d, corpus, name)
        results.append(r)
        with tracker.run(f"training:{name}", {"scope": name, "stage": "training"}):
            tracker.log_metrics({
                "average_precision": r.average_precision,
                "roc_auc": r.roc_auc,
                "usable_feature_rate": r.usable_feature_rate,
            })
            tracker.log_nested("aml", r.aml)
            tracker.log_nested("shap", r.shap_importance)
            tracker.log_nested("recall_by_typology",
                               {k: v["recall"] for k, v in r.per_typology_recall.items()})
            tracker.log_artifact_json(f"{name}.json", r.to_dict())
            # The model as a first-class MLflow artifact: signature, input example, registry
            # entry. classifier_hash rides along as a tag so the homemade provenance and
            # MLflow's cross-check each other.
            if r.bundle is not None:
                slug = get_scope(name).slug
                # Capture the exact version this call registered, so governance blesses that
                # version rather than re-querying "latest" (which a concurrent run could move).
                registered_version = tracker.log_model(
                    r.bundle.model, slug,
                    r.bundle.features[r.bundle.test_idx][:5],
                )
                tracker.log_tags({"classifier_hash": r.bundle.model_hash})
                tracker.log_dataset(
                    r.bundle.features[r.bundle.train_idx],
                    source=str(d), name=f"{slug}-train",
                )
                # Plots make the logged numbers legible: the PR/ROC curves behind AP and
                # ROC-AUC, and the SHAP families behind the `shap.*` reliance metrics. Built
                # from the same bundle and the same interventional explainer, so a plot and a
                # metric can never disagree. Failures here never abort a training run.
                try:
                    from protometer.explain import explanation_for
                    from protometer.plotting import evaluation_figures, shap_figures

                    b = r.bundle
                    for fname, fig in evaluation_figures(
                        b.labels[b.test_idx], b.test_scores, name
                    ).items():
                        tracker.log_figure(fig, fname)

                    explanation = explanation_for(
                        b.model, b.features[b.test_idx], b.feature_names,
                        background=b.features[b.train_idx],
                    )
                    for fname, fig in shap_figures(explanation, name).items():
                        tracker.log_figure(fig, fname)
                except Exception as exc:  # noqa: BLE001, plots are illustrative, not the result
                    from protometer.log import get_logger

                    get_logger("run_training").warning(
                        "plotting failed for %s: %s: %s", name, type(exc).__name__, exc
                    )
                # Governance: tag the version this run registered with the facts that
                # identify it, and move the `champion` alias to it ONLY IF its AP is at least the
                # incumbent champion's (champion_if_best). "champion" means the best build for the
                # scope, not merely the newest, so a regressed retrain never silently demotes a
                # better model. `models:/protometer-<slug>@champion` resolves to the best version.
                # Use the version log_model returned; fall back to a "latest" lookup only if
                # the registry did not report one. The promote script archives superseded ones.
                ver = registered_version
                if not ver:
                    from protometer.tracking import _latest_model_version

                    ver = _latest_model_version(tracker, f"protometer-{slug}")
                if ver:
                    tracker.govern_model(f"protometer-{slug}", ver, {
                        "classifier_hash": r.bundle.model_hash,
                        "corpus_fingerprint": corpus_fp,
                        "average_precision": round(r.average_precision, 4),
                        "run_id": RUN_ID,
                    }, alias="champion", champion_if_best=True)

    if not results:
        sys.exit("No protected corpora found. Run scripts/ingest_all.py first.")

    base = results[0].average_precision
    print(f"{'scope':<28}{'avg precision':>14}{'ROC-AUC':>9}{'retained':>10}{'usable feat':>13}")
    print("-" * 76)
    for r in results:
        print(f"{r.scope_name:<28}{r.average_precision:>14.3f}{r.roc_auc:>9.3f}"
              f"{r.average_precision/base:>9.0%}{r.usable_feature_rate:>13.0%}")

    # Does protection change what the model *relies on*, not just how well it scores?
    if len(results) > 1 and results[0].shap_importance and results[-1].shap_importance:
        shift = basis_shift(results[0].shap_importance, results[-1].shap_importance)
        moved = [(k, v) for k, v in shift.items() if abs(v) >= 0.02][:4]
        if moved:
            print(f"\nreliance shift {results[0].scope_name} -> {results[-1].scope_name}:")
            for k, v in moved:
                print(f"  {k:<28}{v:+.3f}")

    print(f"\ntop features ({results[0].scope_name}):")
    for k, v in list(sorted(results[0].feature_importances.items(), key=lambda kv: -kv[1]))[:5]:
        print(f"  {k:<28}{v:.3f}")

    out = ROOT / "data" / "eval" / "training.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    from protometer.persist import RUN_ID, atomic_write_json

    # Stamp lineage per row: the classifier hash (ties each row to its MLflow training run
    # and to the hybrid artifact scored by the same model) and the run id. training.json was
    # the one artifact whose rows could only be tied to their inputs by recomputation.
    rows = []
    for r in results:
        row = r.to_dict()
        row["classifier_hash"] = r.bundle.model_hash if r.bundle else None
        row["run_id"] = RUN_ID
        # Persist the corpus fingerprint on the row so governance reads it from provenance
        # instead of recomputing or hardcoding it (the exact corpus that produced this model).
        row["corpus_fingerprint"] = corpus_fp
        rows.append(row)
    atomic_write_json(out, rows)
    print(f"\nwritten to {out.relative_to(ROOT)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
