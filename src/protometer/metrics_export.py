"""Operational metrics to Prometheus, for the batch stages whose health is time-series.

MLflow answers "which run produced this score" (experiment comparison) and Langfuse answers
"what did this prompt cost and say" (per-generation). Neither is the right home for the
*operational* signal of a batch stage: ingest rate, discovery latency, per-scope duration,
no-op and failure counts over time. Those are exactly what Prometheus + Grafana exist for.

This module pushes ingest metrics to a Prometheus **Pushgateway** (the batch-job pattern:
a scraped `/metrics` endpoint would be empty between runs, so the job pushes on completion
and Prometheus scrapes the gateway). Grafana reads Prometheus.

Same posture as `tracking.Tracker` and `observability`: entirely optional. No gateway, no
`prometheus_client`, or `PROTOMETER_NO_METRICS=1` and every call is a no-op, the pipeline
unchanged. Telemetry must never be a dependency.
"""

from __future__ import annotations

import os

from protometer.log import get_logger

_log = get_logger("metrics")

# Pushgateway address; grouping job name groups a run's series together.
from protometer import settings as _settings

GATEWAY = _settings.pushgateway()
JOB = "protometer_ingest"


def push_from_report(scope_slug: str, report: dict, domain: str = "aml") -> None:
    """Push a scope's operational metrics from its ingestion-report dict.

    `domain` labels the series by use case (aml / healthcare / customer-support), so a future
    multi-domain ingest keeps each domain's operational metrics distinguishable in one Prometheus /
    Grafana (the idiomatic per-tenant approach: a bounded label, not a separate instance). Defaults
    to "aml" — the only domain whose ingest currently pushes metrics.

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
    push_ingest_metrics(scope_slug, values, corpus_fingerprint=report.get("source_fingerprint"),
                        domain=domain)


def push_ingest_metrics(scope: str, values: dict[str, float],
                        corpus_fingerprint: str | None = None, domain: str = "aml") -> None:
    """Push one scope's ingest operational metrics, labelled by domain, scope, corpus_fingerprint.

    Gauges, not counters: each is the measured value for this scope's run, and Grafana
    graphs them over time (the pushgateway keeps the last push per {job, domain, scope}). A push
    failure is logged and swallowed, never raised.
    """
    if os.getenv("PROTOMETER_NO_METRICS") == "1":
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
                f"protometer_ingest_{name}",
                f"Protometer ingest {name.replace('_', ' ')}",
                ["domain", "scope", "corpus_fingerprint"],
                registry=registry,
            )
            gauge.labels(domain=domain, scope=scope, corpus_fingerprint=fp).set(float(value))
        # run_id provenance is NOT put on Prometheus at all: it would be either a
        # cardinality-exploding label or a churning series. `domain` is a BOUNDED label (3 values),
        # so it segments cleanly per use case without cardinality risk. The operational plane joins to
        # the MLflow/Langfuse planes through `corpus_fingerprint` + `scope`; `domain` further isolates
        # each use case. All three go into the grouping_key so the pushgateway keeps one series per
        # (domain, scope, corpus) rather than overwriting across domains/corpora.
        push_to_gateway(GATEWAY, job=JOB,
                        grouping_key={"domain": domain, "scope": scope, "corpus_fingerprint": fp},
                        registry=registry)
    except Exception as exc:  # noqa: BLE001, metrics must never break ingest
        _log.warning("pushgateway export failed: %s: %s", type(exc).__name__, str(exc)[:100])
