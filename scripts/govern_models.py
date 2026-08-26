"""Reconcile model governance in the MLflow registry from the training artifact.

Idempotent and re-runnable: for each scope's registered model it tags the newest version
with the run's provenance (classifier hash, corpus fingerprint, AP), aliases it `@champion`,
and aliases every superseded version `@archived-vN`. Separate from `run_training.py` so
governance can be reconciled without retraining (e.g. after a registry restore), and so a
timing race during registration never leaves a model un-governed.

    python scripts/govern_models.py

MLflow 3 pattern: aliases + tags, not the deprecated None->Staging->Production stages.
`models:/amlguard-<scope>@champion` then resolves to the current best build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)


def main() -> int:
    import mlflow
    from mlflow import MlflowClient

    from amlguard.scopes import CURVE_ORDER, get_scope
    from amlguard.tracking import DEFAULT_SERVER_URI, _server_reachable

    if not _server_reachable(DEFAULT_SERVER_URI):
        sys.exit(f"MLflow server not reachable at {DEFAULT_SERVER_URI}; start it first.")
    mlflow.set_tracking_uri(DEFAULT_SERVER_URI)
    client = MlflowClient()

    from amlguard.tracking import corpus_source_fingerprint

    training = {r["scope"]: r for r in json.loads((ROOT / "data" / "eval" / "training.json").read_text())}
    # Compute the corpus fingerprint live rather than hardcoding it: a hardcoded hash silently
    # lies the moment the corpus changes, tagging every champion with a fingerprint that no
    # longer joins to the corpus the model was trained on. Prefer the value the training row
    # persisted (the exact corpus that produced this model); fall back to the current corpus.
    live_fingerprint = corpus_source_fingerprint(ROOT / "data" / "corpus")

    for name in CURVE_ORDER:
        slug = get_scope(name).slug
        model_name = f"amlguard-{slug}"
        row = training.get(name)
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except Exception as exc:  # noqa: BLE001
            print(f"  {model_name}: not registered ({type(exc).__name__})")
            continue
        if not versions:
            print(f"  {model_name}: no versions")
            continue

        # Champion = BEST average_precision, not newest version. A retrain that scores WORSE
        # (e.g. after a corpus regression) must NOT silently become champion — the alias means
        # "the best build for the scope", which every consumer of models:/...@champion trusts.
        # Each version carries an `average_precision` tag; pick the max, breaking ties toward the
        # newer version. A version with no AP tag sorts last (a build we cannot vouch for should
        # not out-rank a measured one). If NONE carry an AP, fall back to newest so the loop still
        # blesses something.
        def _ap(v) -> float:
            raw = (v.tags or {}).get("average_precision", "")
            try:
                return float(raw)
            except (TypeError, ValueError):
                return float("-inf")

        best = max(versions, key=lambda v: (_ap(v), int(v.version)))
        if _ap(best) == float("-inf"):
            best = max(versions, key=lambda v: int(v.version))
        # Tag the champion with the cross-tool join keys.
        if row:
            for k, v in {
                "classifier_hash": row.get("classifier_hash", ""),
                "corpus_fingerprint": row.get("corpus_fingerprint") or live_fingerprint,
                "average_precision": row.get("average_precision", ""),
                "run_id": row.get("run_id", ""),
            }.items():
                if v:
                    client.set_model_version_tag(model_name, best.version, k, str(v))
        client.set_registered_model_alias(model_name, "champion", best.version)

        # Archive every superseded version so the registry states which are retired.
        archived = 0
        for v in versions:
            if v.version != best.version:
                client.set_registered_model_alias(
                    model_name, f"archived-v{v.version}", v.version
                )
                archived += 1
        print(f"  {model_name}: champion=v{best.version} (AP {_ap(best)}), archived={archived}")

    print("\nGovernance reconciled. Resolve the best build with "
          "`models:/amlguard-<scope>@champion`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
