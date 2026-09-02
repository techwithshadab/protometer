"""Live-serving operational metrics for the AMLGuard chat API, exposed for Prometheus to SCRAPE.

Batch ingest PUSHES to a Pushgateway (a batch job's series would otherwise be empty between runs);
a long-running server is the opposite case — it is always up, so the idiomatic pattern is a scraped
`/metrics` endpoint whose counters/histograms ACCUMULATE across turns. Mixing the two is a classic
mistake: pushing a per-turn gauge overwrites the last turn and churns cardinality. So this module
owns COUNTERS + HISTOGRAMS on the default registry, and `ui/api/app.py` mounts `/metrics`.

Every metric is labelled by `domain` (aml / healthcare / customer-support) and, where it matters,
`role` and `outcome`, so one process serving all three domains keeps them distinguishable and each
domain gets its own Grafana view. All label sets are BOUNDED (fixed domains/roles/outcomes), never a
per-turn id, so the series count stays flat.

Same posture as `metrics_export` and `observability`: entirely optional. `AMLGUARD_NO_METRICS=1` or a
missing `prometheus_client` makes every call a no-op and the pipeline is unchanged. Telemetry must
never be a dependency of serving a turn.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

from amlguard.log import get_logger

_log = get_logger("serving-metrics")

# Latency buckets tuned for a protected LLM turn: protect + retrieval + generation + egress. A turn
# on a local model runs seconds; a hosted model can be longer. Buckets span 50ms .. 60s.
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60)


def _disabled() -> bool:
    return os.getenv("AMLGUARD_NO_METRICS") == "1"


# Metric objects are created ONCE (prometheus_client forbids re-registering a name on the default
# registry). Cached module-level; the first _metrics() call builds them, later calls reuse them.
_METRICS: dict | None = None


def _metrics() -> dict | None:
    """Lazily build (and cache) the metric objects. Returns None if metrics are unavailable/off, so
    every call site degrades to a no-op."""
    global _METRICS
    if _disabled():
        return None
    if _METRICS is not None:
        return _METRICS
    try:
        from prometheus_client import Counter, Histogram
    except Exception:  # noqa: BLE001, no prometheus_client -> telemetry off, serving unaffected
        return None
    m = {
        # Core: turn volume + outcome. outcome in {ok, egress_blocked, refused, error}. The success
        # rate is derived in Grafana as ok / sum(by domain).
        "turns": Counter(
            "amlguard_serving_turns_total", "Live chat turns served",
            ["domain", "role", "outcome"],
        ),
        # End-to-end turn latency (protect -> reason -> egress -> reveal).
        "latency": Histogram(
            "amlguard_serving_turn_latency_seconds", "End-to-end live turn latency",
            ["domain"], buckets=_LATENCY_BUCKETS,
        ),
        # PII / protection counters — the security signal, proving the boundary works live.
        "entities_protected": Counter(
            "amlguard_serving_entities_protected_total", "PII entities tokenized on inbound turns",
            ["domain"],
        ),
        "revealed": Counter(
            "amlguard_serving_revealed_total", "Identifiers re-identified for the entitled role",
            ["domain", "role"],
        ),
        "egress_blocks": Counter(
            "amlguard_serving_egress_blocks_total", "Replies withheld by the egress guard",
            ["domain"],
        ),
        "canary_hits": Counter(
            "amlguard_serving_canary_hits_total", "Reveals that tripped the canary tripwire",
            ["domain"],
        ),
        "out_of_scope": Counter(
            "amlguard_serving_out_of_scope_total", "Tokens withheld by scope-bound reveal",
            ["domain"],
        ),
        "errors": Counter(
            "amlguard_serving_errors_total", "Turns that ended in an error, by kind",
            ["domain", "kind"],
        ),
        # Model / token usage — cost + efficiency signal (Langfuse has per-generation detail; this is
        # the aggregate time-series view).
        "llm_calls": Counter(
            "amlguard_serving_llm_calls_total", "LLM generation calls on the serving path",
            ["domain", "model"],
        ),
        "tokens": Counter(
            "amlguard_serving_llm_tokens_total", "LLM tokens on the serving path, by direction",
            ["domain", "model", "direction"],
        ),
    }
    _METRICS = m
    return m


@contextmanager
def time_turn(domain: str):
    """Context manager measuring a turn's wall-clock and observing it into the latency histogram.
    A no-op timer when metrics are off. Use around the whole turn() call."""
    start = time.perf_counter()
    try:
        yield
    finally:
        m = _metrics()
        if m is not None:
            try:
                m["latency"].labels(domain=domain).observe(time.perf_counter() - start)
            except Exception as exc:  # noqa: BLE001
                _log.debug("latency observe failed: %s", exc)


def record_turn(
    domain: str,
    role: str,
    result,  # amlguard.serving.TurnResult (duck-typed to avoid an import cycle)
    *,
    model: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Record all per-turn counters from a completed TurnResult. Best-effort: a metrics failure is
    logged and swallowed, never raised into the turn. Call AFTER the turn completes (or after an
    exception, with a synthesized error result)."""
    m = _metrics()
    if m is None:
        return
    try:
        egress_blocked = bool(getattr(result, "egress_blocked", False))
        error = getattr(result, "error", "") or ""
        # Outcome taxonomy — an egress block is a GUARD ACTION, not a failure, even though the
        # TurnResult also stamps `error="egress-blocked: …"` on it. So classify a block as
        # "egress_blocked" FIRST; only a NON-block error counts as "error" (a real failure like
        # generation-failed / inbound-protection-failed / egress-unavailable). A clean turn is "ok".
        # ("refused" is reserved for a deliberate non-error, non-block withholding — none on the AML
        # path today, but the label stays stable for parity with botox's grounded refusals.)
        if egress_blocked:
            outcome = "egress_blocked"
        elif error:
            outcome = "error"
        else:
            outcome = "ok"
        m["turns"].labels(domain=domain, role=role, outcome=outcome).inc()

        n_prot = int(getattr(result, "entities_protected", 0) or 0)
        if n_prot:
            m["entities_protected"].labels(domain=domain).inc(n_prot)
        n_rev = int(getattr(result, "revealed", 0) or 0)
        if n_rev:
            m["revealed"].labels(domain=domain, role=role).inc(n_rev)
        if egress_blocked:
            m["egress_blocks"].labels(domain=domain).inc()
        n_canary = int(getattr(result, "canary_hits", 0) or 0)
        if n_canary:
            m["canary_hits"].labels(domain=domain).inc(n_canary)
        n_oos = int(getattr(result, "out_of_scope", 0) or 0)
        if n_oos:
            m["out_of_scope"].labels(domain=domain).inc(n_oos)
        # Count only GENUINE errors in the error-kind breakdown — an egress block already shows up
        # as outcome=egress_blocked + the egress_blocks counter, so re-counting it here as an
        # "error" kind would double-signal a normal guard action as a failure.
        if error and not egress_blocked:
            # kind = the class prefix before the first ':' so cardinality stays bounded
            # (e.g. "generation-failed", "inbound-protection-failed", "egress-unavailable").
            kind = error.split(":", 1)[0].strip()[:40] or "unknown"
            m["errors"].labels(domain=domain, kind=kind).inc()

        if model:
            m["llm_calls"].labels(domain=domain, model=model).inc()
            if input_tokens:
                m["tokens"].labels(domain=domain, model=model, direction="input").inc(int(input_tokens))
            if output_tokens:
                m["tokens"].labels(domain=domain, model=model, direction="output").inc(int(output_tokens))
    except Exception as exc:  # noqa: BLE001, metrics must never break a turn
        _log.warning("serving metric record failed: %s: %s", type(exc).__name__, str(exc)[:100])


def record_error_turn(domain: str, role: str, kind: str) -> None:
    """Record a turn that raised before a TurnResult existed (e.g. a 502 from an unexpected
    exception). Counts one turn with outcome=error plus the error kind."""
    m = _metrics()
    if m is None:
        return
    try:
        m["turns"].labels(domain=domain, role=role, outcome="error").inc()
        m["errors"].labels(domain=domain, kind=(kind or "unknown")[:40]).inc()
    except Exception as exc:  # noqa: BLE001
        _log.warning("serving error-metric failed: %s: %s", type(exc).__name__, str(exc)[:100])


def metrics_asgi_app():
    """The ASGI app that serves the Prometheus exposition format on /metrics. Returns None if
    prometheus_client is unavailable, so the caller can skip mounting it."""
    if _disabled():
        return None
    try:
        from prometheus_client import make_asgi_app
    except Exception:  # noqa: BLE001
        return None
    return make_asgi_app()
