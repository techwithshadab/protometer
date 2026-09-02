"""Langfuse agent tracing with a safe no-op fallback (Langfuse SDK v4).

Design goals:
  - The pipeline calls the same tracing API whether or not Langfuse is configured. If the keys are
    absent or the SDK isn't installed, every call is a cheap no-op and nothing breaks.
  - Tracing NEVER changes the answer and NEVER raises into the request path: a tracing failure is
    logged and swallowed. Observability must not be able to take down the bot.
  - We record what an operator needs to debug a turn: retrieval (which chunks, scores), model +
    latency, the guard verdict and grounding, and outcome (answered/refused/blocked), but NOT the
    visitor's cleartext PII. The pipeline tokenizes PII before anything here sees it; we trace the
    protected text and the entity COUNT, never the raw values.

SDK v4 note: this targets Langfuse **v4** (the shared observability platform runs v4). v4 removed the
v2 `client.trace()/.generation()/.event()/.score()` surface in favour of `start_observation()` +
`.score_trace()`. The PUBLIC API of this module (trace_turn / _Turn.score/span/event/finish /
record_feedback / shutdown / client) is UNCHANGED, so the pipeline code that calls it is untouched;
only the Langfuse calls underneath moved to v4.

Configuration (env, read once at import):
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST   -> enables tracing when all present
  BOTOX_TRACING=off                                         -> force-disable even if keys are set
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

_log = logging.getLogger("botox.obs")

_client: Any | None = None
_enabled = False


def _init() -> None:
    """Initialise the Langfuse client once. Any failure disables tracing (no-op) rather than
    raising, a misconfigured observability stack must not break the bot."""
    global _client, _enabled
    if os.getenv("BOTOX_TRACING", "").lower() in ("off", "0", "false", "no"):
        _log.info("tracing disabled via BOTOX_TRACING")
        return
    pub, sec = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
    if not (pub and sec):
        _log.info("tracing off: LANGFUSE_PUBLIC_KEY/SECRET_KEY not set")
        return
    host = os.getenv("LANGFUSE_HOST", "http://obs-langfuse:3000")
    try:
        from langfuse import Langfuse
        _client = Langfuse(public_key=pub, secret_key=sec, host=host)
        _enabled = True
        _log.info("tracing on: Langfuse (v4) at %s", host)
    except Exception as exc:  # noqa: BLE001, never let observability break startup
        _log.warning("tracing init failed (%s); running without tracing", type(exc).__name__)
        _client = None
        _enabled = False


_init()


def enabled() -> bool:
    return _enabled and _client is not None


def client() -> Any | None:
    """The shared Langfuse client, or None when tracing/observability is disabled. Exposed so the
    prompt manager can reuse the one authenticated client (same keys, same host) instead of building
    a second one. Callers must treat None as 'Langfuse unavailable' and fall back."""
    return _client if _enabled else None


class _Turn:
    """One chat turn's trace. Wraps a Langfuse v4 ROOT observation (a span) when enabled; a no-op
    recorder otherwise. The pipeline uses `.event(name, **data)` for point-in-time facts and
    `.span(name)` for timed sub-steps, then `.finish(output=..., **meta)` at the end. Scores attach
    to the root's trace via `score_trace`."""

    def __init__(self, root: Any | None, trace_id: str | None = None) -> None:
        self._root = root                 # v4 root observation (LangfuseSpan), or None
        self._trace_id = trace_id
        self._t0 = time.perf_counter()

    @property
    def trace_id(self) -> str | None:
        """The Langfuse trace id for this turn (None when tracing is disabled). The API returns it
        so a later /api/feedback call can attach a user-feedback score to this exact trace."""
        return self._trace_id

    def score(self, name: str, value: Any, *, comment: str | None = None) -> None:
        """Emit a first-class Langfuse SCORE on this turn's trace (chartable/filterable KPI). `value`
        may be numeric (grounding, latency, counts) or a short string (categorical). v4: scores land
        via the root observation's `score_trace`. Never raises into the request path."""
        if self._root is None:
            return
        try:
            kwargs: dict[str, Any] = {"name": name}
            if isinstance(value, bool):
                kwargs["value"] = int(value)
                kwargs["data_type"] = "NUMERIC"
            elif isinstance(value, (int, float)):
                kwargs["value"] = value
                kwargs["data_type"] = "NUMERIC"
            else:
                kwargs["value"] = str(value)
                kwargs["data_type"] = "CATEGORICAL"
            if comment:
                kwargs["comment"] = comment
            self._root.score_trace(**kwargs)
        except Exception as exc:  # noqa: BLE001
            _log.debug("score failed (%s): %s", name, exc)

    @contextmanager
    def span(self, name: str, **inp: Any) -> Iterator["_SpanRec"]:
        rec = _SpanRec(self._root, name, inp) if self._root is not None else _SpanRec(None, name, inp)
        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            rec._latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            rec._close()

    @contextmanager
    def generation(self, name: str, *, model: str | None = None, prompt: Any | None = None,
                   **inp: Any) -> Iterator["_GenRec"]:
        """A model call, as a first-class Langfuse GENERATION (not a plain span): it carries the
        model name, input, token usage, and — when `prompt` is the Langfuse prompt object from
        Prompt Management — LINKS the generation to that prompt version (UI lineage). `.set(...)`
        attaches output/usage before close. Use for the LLM step so cost/usage/prompt tracking work."""
        rec = _GenRec(self._root, name, model, prompt, inp) if self._root is not None \
            else _GenRec(None, name, model, prompt, inp)
        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            rec._latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            rec._close()

    def event(self, name: str, **data: Any) -> None:
        """A point-in-time fact. v4 has no standalone event; model it as a zero-duration child span
        whose output carries the data (starts and ends immediately)."""
        if self._root is None:
            return
        try:
            ev = self._root.start_observation(name=name, as_type="span", metadata=_clean(data))
            ev.end()
        except Exception as exc:  # noqa: BLE001
            _log.debug("trace event failed: %s", exc)

    def finish(self, *, output: str = "", **meta: Any) -> None:
        if self._root is None:
            return
        try:
            meta = dict(meta)
            meta["total_ms"] = round((time.perf_counter() - self._t0) * 1000, 1)
            self._root.update(output=output, metadata=_clean(meta))
            self._root.end()
        except Exception as exc:  # noqa: BLE001
            _log.debug("trace finish failed: %s", exc)


class _SpanRec:
    """A single timed child span. `.set(**data)` attaches structured output; latency added on close."""

    def __init__(self, root: Any | None, name: str, inp: dict) -> None:
        self._name = name
        self._data: dict = {}
        self._latency_ms: float | None = None
        self._span = None
        if root is not None:
            try:
                self._span = root.start_observation(
                    name=name, as_type="span", input=_clean(inp))
            except Exception as exc:  # noqa: BLE001
                _log.debug("span start failed: %s", exc)

    def set(self, **data: Any) -> None:
        self._data.update(data)

    def _close(self) -> None:
        if self._span is None:
            return
        try:
            out = dict(self._data)
            if self._latency_ms is not None:
                out["latency_ms"] = self._latency_ms
            self._span.update(output=_clean(out))
            self._span.end()
        except Exception as exc:  # noqa: BLE001
            _log.debug("span end failed: %s", exc)


class _GenRec:
    """A model call as a Langfuse GENERATION. Carries model + input at start, links the prompt
    version (when a Langfuse prompt object is passed), and takes output + token usage via `.set()`
    / `.usage()` before close. `output=` is the completion; `usage_details` gives input/output
    tokens so cost/usage tracking works in the UI."""

    def __init__(self, root: Any | None, name: str, model: str | None,
                 prompt: Any | None, inp: dict) -> None:
        self._name = name
        self._data: dict = {}          # -> output/metadata on close
        self._usage: dict | None = None
        self._latency_ms: float | None = None
        self._gen = None
        if root is not None:
            try:
                kwargs: dict[str, Any] = {"name": name, "as_type": "generation",
                                          "input": _clean(inp)}
                if model:
                    kwargs["model"] = model
                # Link to the Prompt-Management version, but ONLY when `prompt` is a real Langfuse
                # prompt object (has .name/.version); a plain string would break the SDK call.
                if prompt is not None and hasattr(prompt, "name") and hasattr(prompt, "version"):
                    kwargs["prompt"] = prompt
                self._gen = root.start_observation(**kwargs)
            except Exception as exc:  # noqa: BLE001
                _log.debug("generation start failed: %s", exc)

    def set(self, **data: Any) -> None:
        """Structured fields to attach to the generation output on close."""
        self._data.update(data)

    def usage(self, *, input_tokens: int, output_tokens: int) -> None:
        """Record token usage so the UI can chart/aggregate cost & tokens for this generation."""
        self._usage = {"input": int(input_tokens), "output": int(output_tokens),
                       "total": int(input_tokens) + int(output_tokens)}

    def _close(self) -> None:
        if self._gen is None:
            return
        try:
            out = dict(self._data)
            output = out.pop("output", None)
            if self._latency_ms is not None:
                out["latency_ms"] = self._latency_ms
            upd: dict[str, Any] = {"metadata": _clean(out)}
            if output is not None:
                upd["output"] = output
            if self._usage is not None:
                upd["usage_details"] = self._usage
            self._gen.update(**upd)
            self._gen.end()
        except Exception as exc:  # noqa: BLE001
            _log.debug("generation end failed: %s", exc)


@contextmanager
def trace_turn(*, user_input: str, conversation_id: str, user_id: str | None = None,
               **meta: Any) -> Iterator[_Turn]:
    """Open a trace for one chat turn. `user_input` is the PROTECTED (tokenized) message, never
    pass cleartext PII here. `user_id` is the anonymous visitor id (opaque token, no PII) used to
    group a returning visitor's turns. Yields a `_Turn`; always safe, even when tracing is
    disabled.

    v4: the trace is defined by its ROOT observation. Session/user grouping and tags are attached
    to the trace via the OTel-attribute metadata convention (`langfuse_session_id` /
    `langfuse_user_id` / `langfuse_tags`), which v4 promotes to first-class trace fields."""
    root = None
    trace_id = None
    if enabled():
        try:
            md = _clean(meta)
            md["langfuse_session_id"] = conversation_id
            if user_id:
                md["langfuse_user_id"] = user_id
            md["langfuse_tags"] = ["botox", "graphrag"]
            root = _client.start_observation(  # type: ignore[union-attr]
                name="botox-chat-turn",
                as_type="span",
                input=user_input,
                metadata=md,
            )
            trace_id = getattr(root, "trace_id", None)
        except Exception as exc:  # noqa: BLE001
            _log.debug("trace open failed: %s", exc)
            root = None
    turn = _Turn(root, trace_id)
    # No per-turn flush: the SDK has a background flusher (and flushes at shutdown, see shutdown()),
    # so blocking the request thread on a synchronous flush every turn would tie a worker to
    # Langfuse's round-trip. The trace/spans/scores are enqueued above and sent async.
    yield turn


def shutdown() -> None:
    """Flush any queued events. Call once on app shutdown so in-flight traces aren't lost when the
    background flusher hasn't run yet. Never raises."""
    if not enabled() or _client is None:
        return
    try:
        _client.flush()
    except Exception as exc:  # noqa: BLE001
        _log.debug("shutdown flush failed: %s", exc)


def record_feedback(*, trace_id: str, rating: int, comment: str | None = None) -> bool:
    """Attach a user thumbs-up/down as a `user_feedback` SCORE on an existing trace. `rating` is
    +1 (up) or -1 (down). Returns True if it was recorded, False if tracing is off or it failed.
    Runs from the /api/feedback endpoint, outside any trace context; never raises.

    v4: a trace-scoped score with no live observation is created via `create_score(trace_id=...)`."""
    if not enabled() or _client is None or not trace_id:
        return False
    try:
        _client.create_score(trace_id=trace_id, name="user_feedback", value=rating,
                             data_type="NUMERIC", comment=comment or None)
        # No per-call flush: the SDK batches and flushes on its own timer / at shutdown.
        return True
    except Exception as exc:  # noqa: BLE001
        _log.debug("record_feedback failed: %s", exc)
        return False


def _clean(d: dict) -> dict:
    """Drop None values and keep the payload JSON-friendly. Defensive: metadata must never carry a
    surprise object that breaks serialization inside the request path."""
    out: dict = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool, list, dict)):
            out[k] = v
        else:
            out[k] = str(v)
    return out
