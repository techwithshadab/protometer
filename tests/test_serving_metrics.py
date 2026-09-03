"""The live-serving metrics seam: a TurnResult in, the right Prometheus counters out.

Pins the outcome taxonomy that a subtle bug got wrong: a TurnResult stamps
`error="egress-blocked: …"` on a turn the egress guard WITHHELD, but that is a GUARD ACTION, not a
failure. So an egress block must classify as outcome=egress_blocked (never "error") and must NOT
appear in the error-kind breakdown — else a normal guard action reads as a serving failure on the
dashboard. A genuine error (generation-failed, …) is the only thing that is outcome=error.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _fresh_module(monkeypatch):
    """Import serving_metrics with its own private prometheus registry so counts are isolated per
    test (the module builds metrics on the default registry once; we reset its cache and swap the
    registry so each test sees clean series)."""
    monkeypatch.delenv("PROTOMETER_NO_METRICS", raising=False)
    import prometheus_client
    from prometheus_client import CollectorRegistry
    reg = CollectorRegistry()
    # Point the default registry at a fresh one for the duration of the test.
    monkeypatch.setattr(prometheus_client, "REGISTRY", reg, raising=False)
    import protometer.serving_metrics as sm
    monkeypatch.setattr(sm, "_METRICS", None, raising=False)
    # Rebuild the metric objects against the fresh registry by monkeypatching Counter/Histogram to
    # register there. Simplest: rebuild via the real API but into `reg`.
    import prometheus_client as pc
    orig_counter, orig_hist = pc.Counter, pc.Histogram
    monkeypatch.setattr(pc, "Counter", lambda *a, **k: orig_counter(*a, registry=reg, **{kk: vv for kk, vv in k.items() if kk != "registry"}))
    monkeypatch.setattr(pc, "Histogram", lambda *a, **k: orig_hist(*a, registry=reg, **{kk: vv for kk, vv in k.items() if kk != "registry"}))
    return sm, reg


def _val(reg, name, labels):
    return reg.get_sample_value(name, labels) or 0.0


def test_egress_block_is_a_guard_action_not_an_error(monkeypatch):
    sm, reg = _fresh_module(monkeypatch)

    class Blocked:
        egress_blocked = True
        error = "egress-blocked: rejected"   # the TurnResult stamps this on a withheld reply
        entities_protected = 1
        revealed = 0
        canary_hits = 0
        out_of_scope = 0

    sm.record_turn("customer-support", "support_agent", Blocked())

    # Classified as egress_blocked, NOT error.
    assert _val(reg, "protometer_serving_turns_total",
                {"domain": "customer-support", "role": "support_agent", "outcome": "egress_blocked"}) == 1.0
    assert _val(reg, "protometer_serving_turns_total",
                {"domain": "customer-support", "role": "support_agent", "outcome": "error"}) == 0.0
    # Counted in the egress-block counter.
    assert _val(reg, "protometer_serving_egress_blocks_total", {"domain": "customer-support"}) == 1.0
    # NOT counted as an error kind (that would double-signal a normal guard action as a failure).
    assert _val(reg, "protometer_serving_errors_total",
                {"domain": "customer-support", "kind": "egress-blocked"}) == 0.0


def test_genuine_error_is_outcome_error_and_counted_by_kind(monkeypatch):
    sm, reg = _fresh_module(monkeypatch)

    class Errored:
        egress_blocked = False
        error = "generation-failed: TimeoutError"
        entities_protected = 0
        revealed = 0
        canary_hits = 0
        out_of_scope = 0

    sm.record_turn("aml", "investigator", Errored())
    assert _val(reg, "protometer_serving_turns_total",
                {"domain": "aml", "role": "investigator", "outcome": "error"}) == 1.0
    assert _val(reg, "protometer_serving_errors_total",
                {"domain": "aml", "kind": "generation-failed"}) == 1.0


def test_clean_turn_records_ok_and_protection_counters(monkeypatch):
    sm, reg = _fresh_module(monkeypatch)

    class Ok:
        egress_blocked = False
        error = ""
        entities_protected = 3
        revealed = 2
        canary_hits = 0
        out_of_scope = 0

    sm.record_turn("aml", "investigator", Ok(), model="llama3.2", input_tokens=100, output_tokens=40)
    assert _val(reg, "protometer_serving_turns_total",
                {"domain": "aml", "role": "investigator", "outcome": "ok"}) == 1.0
    assert _val(reg, "protometer_serving_entities_protected_total", {"domain": "aml"}) == 3.0
    assert _val(reg, "protometer_serving_revealed_total", {"domain": "aml", "role": "investigator"}) == 2.0
    assert _val(reg, "protometer_serving_llm_tokens_total",
                {"domain": "aml", "model": "llama3.2", "direction": "input"}) == 100.0


def test_metrics_off_is_a_noop(monkeypatch):
    monkeypatch.setenv("PROTOMETER_NO_METRICS", "1")
    import protometer.serving_metrics as sm
    monkeypatch.setattr(sm, "_METRICS", None, raising=False)
    # record_turn must not raise and must produce no metrics object.
    class Ok:
        egress_blocked = False; error = ""; entities_protected = 1
        revealed = 0; canary_hits = 0; out_of_scope = 0
    sm.record_turn("aml", "investigator", Ok())
    assert sm._metrics() is None
    assert sm.metrics_asgi_app() is None
