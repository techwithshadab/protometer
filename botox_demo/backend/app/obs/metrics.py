"""Live-serving Prometheus metrics for the BOTOX chatbot, exposed for Prometheus to SCRAPE.

A long-running server is scraped (not pushed like a batch job), so this module owns COUNTERS +
HISTOGRAMS on the default registry and the API mounts `/metrics`. Every metric carries a fixed
`demo="botox"` label so it sits beside the Protometer serving metrics in one Prometheus without
colliding, and outcomes/labels are BOUNDED (fixed sets), never a per-turn id, so cardinality is flat.

Entirely optional, same as tracing: `BOTOX_NO_METRICS=1` (or missing prometheus_client) makes every
call a no-op and a turn is served unchanged. Telemetry is never a dependency of answering.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

_log = logging.getLogger("botox.metrics")

_DEMO = "botox"
# Latency buckets for a protected GraphRAG turn: emergency-check + protect + retrieve + graph-expand
# + generate + egress. Local Ollama runs seconds; span 50ms .. 60s.
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60)

_METRICS: dict | None = None


def _disabled() -> bool:
    return os.getenv("BOTOX_NO_METRICS") == "1"


def _metrics() -> dict | None:
    global _METRICS
    if _disabled():
        return None
    if _METRICS is not None:
        return _METRICS
    try:
        from prometheus_client import Counter, Histogram
    except Exception:  # noqa: BLE001, no prometheus_client -> metrics off, serving unaffected
        return None
    m = {
        # Turn volume + outcome. outcome in {ok, refused, blocked, error}:
        #   ok       - a grounded answer returned
        #   refused  - a deliberate refusal (off-topic / grounded-refusal / protection-down)
        #   blocked  - the egress guard withheld/redirected a reply
        #   error    - an unexpected failure
        "turns": Counter("botox_serving_turns_total", "Chatbot turns served",
                         ["demo", "outcome"]),
        "latency": Histogram("botox_serving_turn_latency_seconds", "End-to-end turn latency",
                             ["demo"], buckets=_LATENCY_BUCKETS),
        # PII / protection — the security signal for the public bot.
        "entities_protected": Counter("botox_serving_entities_protected_total",
                                      "Visitor PII entities tokenized at ingress", ["demo"]),
        "egress_blocks": Counter("botox_serving_egress_blocks_total",
                                "Replies withheld/redirected by the egress guard", ["demo"]),
        "protection_down": Counter("botox_serving_protection_down_total",
                                  "Turns refused because Protegrity was unavailable (fail-closed)",
                                  ["demo"]),
        "emergency_hits": Counter("botox_serving_emergency_hits_total",
                                 "Turns short-circuited by the emergency (urgent-symptom) check",
                                 ["demo"]),
        # Grounding quality: distribution of the grounding score (0..1) over answered turns.
        "grounding": Histogram("botox_serving_grounding_score", "Grounding score of answered turns",
                              ["demo"], buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
        # Model / token usage (estimated ~4 chars/token; Ollama does not report usage uniformly).
        "llm_calls": Counter("botox_serving_llm_calls_total", "LLM generation calls",
                            ["demo", "model"]),
        "tokens": Counter("botox_serving_llm_tokens_total", "Estimated LLM tokens, by direction",
                        ["demo", "model", "direction"]),
        "errors": Counter("botox_serving_errors_total", "Turns that ended in an error, by kind",
                        ["demo", "kind"]),
    }
    _METRICS = m
    return m


@contextmanager
def time_turn():
    """Measure a turn's wall-clock into the latency histogram. No-op when metrics are off."""
    start = time.perf_counter()
    try:
        yield
    finally:
        m = _metrics()
        if m is not None:
            try:
                m["latency"].labels(demo=_DEMO).observe(time.perf_counter() - start)
            except Exception as exc:  # noqa: BLE001
                _log.debug("latency observe failed: %s", exc)


def record_turn(answer, *, model: str = "", input_tokens: int = 0, output_tokens: int = 0,
                emergency: bool = False) -> None:
    """Record per-turn counters from a completed Answer (duck-typed to avoid an import cycle).
    Best-effort: a metrics failure is logged and swallowed, never raised into the turn."""
    m = _metrics()
    if m is None:
        return
    try:
        refused = bool(getattr(answer, "refused", False))
        blocked = bool(getattr(answer, "blocked", False))
        retryable = bool(getattr(answer, "retryable", False))
        # Outcome taxonomy (mutually exclusive, in priority order): a blocked reply is a guard action;
        # a retryable refusal means protection was down; any other refusal is a deliberate refusal;
        # otherwise ok.
        if blocked:
            outcome = "blocked"
        elif refused and retryable:
            outcome = "refused"
            m["protection_down"].labels(demo=_DEMO).inc()
        elif refused:
            outcome = "refused"
        else:
            outcome = "ok"
        m["turns"].labels(demo=_DEMO, outcome=outcome).inc()

        if emergency:
            m["emergency_hits"].labels(demo=_DEMO).inc()
        if blocked:
            m["egress_blocks"].labels(demo=_DEMO).inc()

        n_prot = int(getattr(answer, "entities_protected", 0) or 0)
        if n_prot:
            m["entities_protected"].labels(demo=_DEMO).inc(n_prot)

        # Grounding score only meaningful for an actual answered turn.
        if outcome == "ok":
            gs = float(getattr(answer, "grounding_score", 0.0) or 0.0)
            m["grounding"].labels(demo=_DEMO).observe(gs)

        if model:
            m["llm_calls"].labels(demo=_DEMO, model=model).inc()
            if input_tokens:
                m["tokens"].labels(demo=_DEMO, model=model, direction="input").inc(int(input_tokens))
            if output_tokens:
                m["tokens"].labels(demo=_DEMO, model=model, direction="output").inc(int(output_tokens))
    except Exception as exc:  # noqa: BLE001, metrics must never break a turn
        _log.warning("serving metric record failed: %s: %s", type(exc).__name__, str(exc)[:100])


def record_error(kind: str) -> None:
    """Record a turn that raised before an Answer existed."""
    m = _metrics()
    if m is None:
        return
    try:
        m["turns"].labels(demo=_DEMO, outcome="error").inc()
        m["errors"].labels(demo=_DEMO, kind=(kind or "unknown")[:40]).inc()
    except Exception as exc:  # noqa: BLE001
        _log.warning("serving error-metric failed: %s: %s", type(exc).__name__, str(exc)[:100])


def metrics_asgi_app():
    """ASGI app serving the Prometheus exposition format. None if unavailable (caller skips mount)."""
    if _disabled():
        return None
    try:
        from prometheus_client import make_asgi_app
    except Exception:  # noqa: BLE001
        return None
    return make_asgi_app()
