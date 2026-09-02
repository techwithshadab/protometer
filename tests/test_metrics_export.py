"""The operational-metrics seam: report dict in, Prometheus gauge values out.

Both ingest paths (fresh protect and skip-with-persisted-report) cross `push_from_report`,
so it is the seam to test. The defects it guards against: (1) a skip-path rebuild silently
pushing nothing, and (2) an older report without latency percentiles pushing 0.0 gauges,
which read as "the API is instant" rather than "not measured for this corpus".
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard import metrics_export


def _capture(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        metrics_export, "push_ingest_metrics",
        lambda scope, values, corpus_fingerprint=None, domain="aml": captured.update(
            scope=scope, values=values, corpus_fingerprint=corpus_fingerprint, domain=domain),
    )
    return captured


def test_push_from_report_translates_the_dict(monkeypatch):
    captured = _capture(monkeypatch)
    report = {
        "seconds": 100.0,
        "seconds_in_discovery": 20.0,
        "entities_found": 500,
        "entities_protected": 480,
        "source_fingerprint": "abc123def456",
        "protection_stats": {"api_calls": 200, "cache_hit_rate": 0.5, "retries": 3},
        "protection_noops": {"PERSON": 2},
        "protection_failures": {"IBAN": 1},
    }
    metrics_export.push_from_report("direct", report)

    assert captured["scope"] == "direct"
    # the report's source_fingerprint is threaded through as the cross-plane join label
    assert captured["corpus_fingerprint"] == "abc123def456"
    v = captured["values"]
    assert v["seconds"] == 100.0
    assert v["api_calls"] == 200
    assert v["noops"] == 2
    assert v["failures"] == 1
    assert v["api_rate"] == 2.0  # 200 calls / 100 s


def test_absent_latency_percentiles_are_not_pushed(monkeypatch):
    """An older report has no latency_* keys; a 0.0 gauge would be a false reading."""
    captured = _capture(monkeypatch)
    metrics_export.push_from_report(
        "direct", {"seconds": 10.0, "protection_stats": {"api_calls": 5}}
    )
    v = captured["values"]
    assert not any(k.startswith("api_latency_") for k in v)


def test_present_latency_percentiles_are_pushed(monkeypatch):
    captured = _capture(monkeypatch)
    metrics_export.push_from_report("direct", {
        "seconds": 10.0,
        "protection_stats": {
            "api_calls": 5, "latency_p50": 0.1, "latency_p90": 0.3,
            "latency_p95": 0.4, "latency_p99": 0.9,
        },
    })
    v = captured["values"]
    assert v["api_latency_p50"] == 0.1
    assert v["api_latency_p99"] == 0.9


def test_zero_seconds_does_not_divide(monkeypatch):
    """The `none` scope protects nothing (seconds ~0); api_rate must not raise."""
    captured = _capture(monkeypatch)
    metrics_export.push_from_report("none", {"seconds": 0.0, "protection_stats": {}})
    assert captured["values"]["api_rate"] == 0


def test_compute_uses_supplied_threshold_not_test_fold_selection():
    """The operating threshold, when supplied, must be applied verbatim (no re-selection).

    Pins the fix for threshold-overfitting: training selects the F1 point on the TRAIN fold
    and passes it to compute() for the TEST fold. If compute() ignored the supplied value and
    re-selected on the test labels, the reported operating point would be optimistically
    biased. Here a deliberately non-F1 threshold must survive into the metrics.
    """
    import numpy as np

    from amlguard import metrics as m

    labels = np.array([0] * 90 + [1] * 10)
    scores = np.clip(labels * 0.5 + np.linspace(0, 0.4, 100), 0, 1)
    supplied = 0.123
    out = m.compute(labels, scores, operating_threshold=supplied)
    assert abs(out.operating_threshold - supplied) < 1e-9


def test_select_f1_threshold_is_degenerate_safe():
    import numpy as np

    from amlguard import metrics as m
    # single-class input -> 0.5 fallback, never a crash
    assert m.select_f1_threshold(np.zeros(20, dtype=int), np.random.default_rng(0).random(20)) == 0.5


def test_prometheus_gauges_are_labelled_by_bounded_keys_never_run_id(monkeypatch):
    """run_id must NOT be a Prometheus label: it changes per run and would explode TSDB cardinality
    and fragment each metric in Grafana. The gauges carry ONLY bounded labels — `domain` (3 use
    cases), `scope`, and `corpus_fingerprint` (one value per corpus, the documented cross-plane join
    key) — never run_id."""
    captured = {}

    class _FakeGauge:
        def __init__(self, name, doc, labelnames, registry=None):
            captured.setdefault("labelnames", []).append(tuple(labelnames))

        def labels(self, **kw):
            captured.setdefault("label_calls", []).append(kw)
            return self

        def set(self, v):
            pass

    import sys
    import types
    fake_pc = types.ModuleType("prometheus_client")
    fake_pc.CollectorRegistry = lambda: object()
    fake_pc.Gauge = _FakeGauge
    fake_pc.push_to_gateway = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "prometheus_client", fake_pc)
    monkeypatch.delenv("AMLGUARD_NO_METRICS", raising=False)

    metrics_export.push_ingest_metrics("all", {"seconds": 100.0, "api_calls": 5},
                                       corpus_fingerprint="fp0011223344", domain="healthcare")

    # every gauge is labelled by exactly ("domain", "scope", "corpus_fingerprint") — never run_id
    assert captured["labelnames"], "no gauges were created"
    for names in captured["labelnames"]:
        assert names == ("domain", "scope", "corpus_fingerprint"), \
            f"gauge labelled by {names}, must be ('domain', 'scope', 'corpus_fingerprint')"
    for call in captured["label_calls"]:
        assert set(call) == {"domain", "scope", "corpus_fingerprint"} and "run_id" not in call
        assert call["domain"] == "healthcare"   # the domain label is threaded through


def test_missing_fingerprint_falls_back_to_unknown_bounded_label(monkeypatch):
    """An older report without source_fingerprint must still push one bounded label value
    ('unknown'), not omit the label (which would make the series unjoinable) or crash."""
    captured = {}

    class _FakeGauge:
        def __init__(self, name, doc, labelnames, registry=None):
            captured.setdefault("labelnames", []).append(tuple(labelnames))

        def labels(self, **kw):
            captured.setdefault("label_calls", []).append(kw)
            return self

        def set(self, v):
            pass

    import sys
    import types
    fake_pc = types.ModuleType("prometheus_client")
    fake_pc.CollectorRegistry = lambda: object()
    fake_pc.Gauge = _FakeGauge
    fake_pc.push_to_gateway = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "prometheus_client", fake_pc)
    monkeypatch.delenv("AMLGUARD_NO_METRICS", raising=False)

    metrics_export.push_ingest_metrics("none", {"seconds": 0.0}, corpus_fingerprint=None)
    for call in captured["label_calls"]:
        assert call["corpus_fingerprint"] == "unknown"
