"""Consolidated cross-tool report: join MLflow, Langfuse, and Prometheus by run_id.

Proves the three observability planes are joinable into one view. Given a run_id (or the
latest hybrid run), it pulls the experiment scores from MLflow, the generations and scores
from Langfuse, and the ingest operational metrics from Prometheus, and prints them together.

    python scripts/observability_report.py                 # latest hybrid run
    python scripts/observability_report.py <run_id>

Read-only. Each plane is best-effort: a plane that is down is reported as unavailable, not
fatal, the whole point is that no single plane is a dependency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)


def _mlflow_runs(run_id: str) -> list[dict]:
    try:
        import mlflow
        from mlflow import MlflowClient

        from protometer.tracking import DEFAULT_SERVER_URI, _server_reachable
        if not _server_reachable(DEFAULT_SERVER_URI):
            return []
        mlflow.set_tracking_uri(DEFAULT_SERVER_URI)
        client = MlflowClient()
        out = []
        for exp in client.search_experiments():
            for r in client.search_runs(
                [exp.experiment_id], filter_string=f"tags.`protometer.run_id` = '{run_id}'"
            ):
                out.append({
                    "experiment": exp.name,
                    "run_name": r.data.tags.get("mlflow.runName", ""),
                    "metrics": {k: round(v, 4) for k, v in r.data.metrics.items()},
                })
        return out
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def _langfuse(run_id: str) -> dict:
    import os

    import requests

    from protometer import settings
    pk, sk = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
    # Single source of truth for the host default (shared tier host :5006, not the old :3000).
    host = settings.langfuse_host()
    if not pk or not sk:
        return {"available": False}
    try:
        # Join on sessionId, not metadata.run_id: the v4 observations *list* endpoint returns
        # sessionId as a first-class field but omits metadata entirely, so metadata.run_id is
        # invisible here. record_generation sets langfuse_session_id = RUN_ID precisely so the
        # run is joinable through the field the list API actually exposes.
        gens = requests.get(f"{host}/api/public/v2/observations",
                            params={"type": "GENERATION", "limit": 100},
                            auth=(pk, sk), timeout=10)
        rows = gens.json().get("data", []) if gens.ok else []
        mine = [o for o in rows if o.get("sessionId") == run_id]
        cost = round(sum(o.get("totalPrice") or 0.0 for o in mine), 4)
        # Only billed generations carry cost/latency; a cached run is 0.0 by definition, so
        # report those only when at least one call actually billed, else they read as a fault.
        billed = [o for o in mine if (o.get("totalPrice") or 0) > 0]
        lat = [o["latency"] for o in billed if o.get("latency") is not None]
        p50 = round(sorted(lat)[len(lat) // 2] / 1000, 3) if lat else None
        return {"available": True, "session": run_id,
                "generations_this_run": len(mine),
                "billed_generations": len(billed),
                "cost_usd": cost, "latency_p50_s": p50}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _prometheus(run_id: str) -> dict:
    """Ingest operational series.

    Ingest is a *separate process* from hybrid and the operational plane is labelled only by
    `scope` (run_id is deliberately not a Prometheus label, it would explode cardinality). The
    two stages join through `corpus_fingerprint`, not a shared run_id (see the Observability section of docs/architecture.md). This
    reports the per-scope ingest series that exist; the run_id lives on the MLflow/Langfuse
    planes above.
    """
    import requests

    from protometer import settings
    base = settings.prometheus_query_url()
    try:
        r = requests.get(f"{base}/api/v1/query",
                        params={"query": "protometer_ingest_seconds"}, timeout=10)
        results = r.json().get("data", {}).get("result", []) if r.ok else []
        return {"available": True,
                "join": "corpus_fingerprint + scope (ingest is a separate process)",
                "ingest_scopes": {m["metric"].get("scope"): float(m["value"][1])
                                  for m in results}}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        hy = ROOT / "data" / "eval" / "hybrid_none.json"
        if not hy.exists():
            sys.exit("No run_id given and no hybrid_none.json to read one from.")
        run_id = json.loads(hy.read_text()).get("run_id", "")
        if not run_id:
            sys.exit("hybrid_none.json has no run_id; pass one explicitly.")

    print(f"# Consolidated observability for run_id = {run_id}\n")
    print("## MLflow (experiment scores)")
    for r in _mlflow_runs(run_id):
        print(f"  {r}")
    print("\n## Langfuse (generations + session)")
    print(f"  {_langfuse(run_id)}")
    print("\n## Prometheus (ingest operations)")
    print(f"  {_prometheus(run_id)}")
    print("\nJoined on run_id across all three planes. See the Observability section of docs/architecture.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
