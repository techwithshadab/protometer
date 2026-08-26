"""Operational metrics to Prometheus, for the batch stages whose health is time-series.

MLflow answers "which run produced this score" (experiment comparison) and Langfuse answers
"what did this prompt cost and say" (per-generation). Neither is the right home for the
*operational* signal of a batch stage: ingest rate, discovery latency, per-scope duration,
no-op and failure counts over time. Those are exactly what Prometheus + Grafana exist for.

This module pushes ingest metrics to a Prometheus **Pushgateway** (the batch-job pattern:
a scraped `/metrics` endpoint would be empty between runs, so the job pushes on completion
and Prometheus scrapes the gateway). Grafana reads Prometheus.

Same posture as `tracking.Tracker` and `observability`: entirely optional. No gateway, no
`prometheus_client`, or `AMLGUARD_NO_METRICS=1` and every call is a no-op, the pipeline
unchanged. Telemetry must never be a dependency.
"""

from __future__ import annotations

import os

from amlguard.log import get_logger

_log = get_logger("metrics")

# Pushgateway address; grouping job name groups a run's series together.
from amlguard import settings as _settings

GATEWAY = _settings.pushgateway()
JOB = "amlguard_ingest"


def push_from_report(scope_slug: str, report: dict) -> None:
    """Push a scope's operational metrics from its ingestion-report dict.

    Works from the persisted `ingestion_report.json` shape, so the Prometheus plane is
    rebuildable from artifacts at $0 the same way the MLflow and Langfuse dashboards are
    repopulated from cache: re-running ingest over an already-protected corpus skips the
    paid protect calls yet still refreshes the operational series from the report on disk.

    Latency percentiles are only present in reports produced after that instrumentation
    landed; older reports omit them and this pushes only what exists (a 0.0 gauge would be a
    false reading, an absent series correctly says "not measured for this corpus").
    """
    stats = report.get("protection_stats") or {}
    seconds = report.get("seconds") or 0.0
    api_calls = stats.get("api_calls", 0)
    values = {
        "seconds": seconds,
        "seconds_in_discovery": report.get("seconds_in_discovery", 0.0),
        "entities_found": report.get("entities_found", 0),
        "entities_protected": report.get("entities_protected", 0),
        "api_calls": api_calls,
        "cache_hit_rate": stats.get("cache_hit_rate", 0.0),
        "retries": stats.get("retries", 0),
        "noops": sum((report.get("protection_noops") or {}).values()),
        "failures": sum((report.get("protection_failures") or {}).values()),
        "api_rate": (api_calls / seconds if seconds else 0),
    }
    for pct in ("p50", "p90", "p95", "p99"):
        if f"latency_{pct}" in stats:
            values[f"api_latency_{pct}"] = stats[f"latency_{pct}"]
    # The report stamps the corpus it was produced from as `source_fingerprint`. Carry it onto the
    # Prometheus series as the cross-plane join key (documented below), so the operational plane can
    # actually be joined to MLflow/Langfuse by (corpus_fingerprint, scope) rather than only by scope.
    push_ingest_metrics(scope_slug, values, corpus_fingerprint=report.get("source_fingerprint"))


def push_ingest_metrics(scope: str, values: dict[str, float],
                        corpus_fingerprint: str | None = None) -> None:
    """Push one scope's ingest operational metrics, labelled by scope (and corpus_fingerprint).

    Gauges, not counters: each is the measured value for this scope's run, and Grafana
    graphs them over time (the pushgateway keeps the last push per {job, scope}). A push
    failure is logged and swallowed, never raised.
    """
    if os.getenv("AMLGUARD_NO_METRICS") == "1":
        return
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


        # corpus_fingerprint is a BOUNDED label: one value per corpus (not per run), so it does not
        # cause the per-run cardinality blow-up that a run_id label would. Absent (older reports) ->
        # "unknown", still one bounded value. This is the label the documented cross-plane join needs.
        fp = corpus_fingerprint or "unknown"
        registry = CollectorRegistry()
        for name, value in values.items():
            if not isinstance(value, (int, float)):
                continue
            # Label by `scope` AND `corpus_fingerprint`, both bounded sets. `run_id` must NOT be a
            # series-defining label: it changes every process-run, so labelling by it would make
            # Prometheus persist a brand-new series per run (unbounded cardinality, the classic TSDB
            # memory blow-up) and fragment each metric into disjoint single-sample series that never
            # form one line in Grafana.
            gauge = Gauge(
                f"amlguard_ingest_{name}",
                f"AMLGuard ingest {name.replace('_', ' ')}",
                ["scope", "corpus_fingerprint"],
                registry=registry,
            )
            gauge.labels(scope=scope, corpus_fingerprint=fp).set(float(value))
        # run_id provenance is NOT put on Prometheus at all: it would be either a
        # cardinality-exploding label or a churning series. The operational plane joins to the
        # MLflow/Langfuse planes through `corpus_fingerprint` + `scope` (now both real labels on the
        # series; see the Observability section of docs/architecture.md and observability_report.py, which reads run_id from those
        # planes, not from here). Both go into the grouping_key so the pushgateway keeps one series
        # per (scope, corpus) rather than overwriting across corpora.
        push_to_gateway(GATEWAY, job=JOB,
                        grouping_key={"scope": scope, "corpus_fingerprint": fp}, registry=registry)
    except Exception as exc:  # noqa: BLE001, metrics must never break ingest
        _log.warning("pushgateway export failed: %s: %s", type(exc).__name__, str(exc)[:100])
