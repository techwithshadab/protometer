"""FastAPI app: the public BOTOX chatbot's protected GraphRAG endpoint.

    POST /api/chat          {message, conversation_id, visitor_id?}
                              -> {answer, safety, refused, blocked, retryable, conversation_id, trace_id}
    POST /api/chat/stream   same, as Server-Sent Events (token* then one final)
    POST /api/feedback      {trace_id, rating}   -> attach a user_feedback score to the turn's trace
    GET  /api/health                             -> readiness + which backends are active
    POST /api/support/reveal {protected_text}    -> ROLE-GATED re-identification (support token only)

The PUBLIC chat path is zero-reveal: the response never contains detected cleartext PII (tokenized
at ingress, guarded at egress). Re-identification happens ONLY on /api/support/reveal, gated by a
shared support token. Sources are not returned to the UI (they live in the trace). Loopback-friendly;
behind nginx (rate-limiting + security headers) in the compose stack.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import threading
import uuid

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("botox.api")

app = FastAPI(title="BOTOX Information Assistant", version="1.0")

# Prometheus SCRAPE endpoint for live-serving metrics (turn volume/outcome, latency, PII-protection
# counters, grounding, model/token usage — see app/obs/metrics.py). Served as an explicit GET route
# (not app.mount) so a bare "/metrics" with no trailing slash always resolves. No-op / empty body if
# prometheus_client is absent or BOTOX_NO_METRICS=1, so serving never depends on telemetry. Reachable
# on the backend loopback port only; nginx does not proxy it publicly (like the operator endpoints).
@app.get("/metrics")
def _prometheus_metrics():
    from starlette.responses import Response
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        from app.obs.metrics import _disabled
        if _disabled():
            return Response(status_code=404)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception:  # noqa: BLE001, telemetry must never break; report unavailable
        return Response(status_code=404)

# Feedback anti-forgery: we hand the client a SIGNED trace handle ("<trace_id>.<sig>") with each
# answer and only accept feedback whose signature verifies. This binds a rating to an answer the
# backend actually produced, so a caller can't spam/forge scores on arbitrary or guessed trace ids.
# The key defaults to BOTOX_TOKEN_KEY (already required to be set for any real deployment).
_FEEDBACK_KEY = (os.getenv("BOTOX_FEEDBACK_KEY") or os.getenv("BOTOX_TOKEN_KEY")
                 or "botox-demo-deterministic-key").encode()


def _sign_trace(trace_id: str) -> str:
    """Return '<trace_id>.<hex-sig>', the handle the client echoes back on /api/feedback."""
    sig = hmac.new(_FEEDBACK_KEY, trace_id.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{trace_id}.{sig}"


def _verify_trace(handle: str) -> str | None:
    """Verify a signed handle and return the bare trace_id, or None if the signature is invalid."""
    if not handle or "." not in handle:
        return None
    trace_id, _, sig = handle.rpartition(".")
    expected = hmac.new(_FEEDBACK_KEY, trace_id.encode(), hashlib.sha256).hexdigest()[:32]
    return trace_id if hmac.compare_digest(sig, expected) else None


# Support/internal ROLE token for the re-identification endpoint. Re-identifying a visitor's PII is a
# privileged operation, so it is gated behind a shared secret the operator holds, sent as
# `Authorization: Bearer <token>` and compared in constant time. There is NO default: if
# SUPPORT_API_TOKEN is unset, the reveal endpoint is DISABLED (returns 404-like unavailability),
# never open. This is the ONLY path that reverses tokenization; the public chat path never does.
_SUPPORT_TOKEN = os.getenv("SUPPORT_API_TOKEN") or ""


def _require_support_role(authorization: str | None) -> None:
    """Authorize a support-role caller, or raise. Fails closed: unconfigured => disabled; missing or
    wrong token => 401. Constant-time compare so the token can't be guessed by timing."""
    if not _SUPPORT_TOKEN:
        # No token configured: the reveal capability is off. Use FastAPI's default 404 detail casing
        # ("Not Found") so a disabled endpoint isn't fingerprintable against a genuinely-missing route.
        raise HTTPException(status_code=404, detail="Not Found")
    # Tolerant Bearer parsing: case-insensitive scheme, tolerate surrounding whitespace. (Fail-closed
    # either way; this just avoids rejecting a well-formed but differently-cased header.)
    header = (authorization or "").strip()
    scheme, _, presented = header.partition(" ")
    presented = presented.strip()
    if scheme.lower() != "bearer" or not presented \
            or not hmac.compare_digest(presented, _SUPPORT_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")

# CORS restricted to the site origins; override with BOTOX_ALLOWED_ORIGINS for a real deployment.
# Default matches the frontend's published host port (9001 in the shared port map; :5173 is the Vite
# dev server). The compose sets BOTOX_ALLOWED_ORIGINS explicitly, so this default only applies to a
# bare `uvicorn` run.
_origins = os.getenv("BOTOX_ALLOWED_ORIGINS",
                     "http://localhost:9001,http://127.0.0.1:9001,http://localhost:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in _origins],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

_ORCH = None
_ORCH_LOCK = threading.Lock()


def _orchestrator():
    """Build the orchestrator + GraphRAG retriever once (thread-safe). Kept lazy so /health works
    even if the index isn't built yet (it reports not-ready instead of crashing on import). The lock
    prevents a concurrent request burst from building two orchestrators (and, transitively, two
    Protegrity logins)."""
    global _ORCH
    if _ORCH is None:
        with _ORCH_LOCK:
            if _ORCH is None:               # re-check inside the lock
                from app.graph import vectorstore
                from app.pipeline.orchestrator import Orchestrator
                # Ensure the index loads (raises if never built, which /health reports as not-ready).
                vectorstore._ensure_loaded()
                _ORCH = Orchestrator(retriever=vectorstore)
    return _ORCH


@app.on_event("startup")
def _warmup() -> None:
    """Warm the retriever (index + embedding encoder) at startup so the FIRST user query is fast.
    The encoder's first encode costs a few seconds; paying it here, not interactively, is the
    single biggest latency win. Best-effort: if the index isn't built yet, /health reports
    not-ready and the first query warms it lazily instead."""
    try:
        from app.graph import vectorstore
        vectorstore.warmup()
        _orchestrator()  # build singletons (protector, model client) too
        _log.info("warmup complete: index + encoder ready")
    except Exception as exc:  # noqa: BLE001, startup must not crash if the index isn't built yet
        _log.warning("warmup skipped (%s): first query will warm lazily", type(exc).__name__)


@app.on_event("shutdown")
def _shutdown() -> None:
    """Flush any queued traces on shutdown (per-turn flushing was dropped to keep the request path
    off Langfuse's round-trip; this catches anything the background flusher hasn't sent yet)."""
    from app.obs import tracing
    tracing.shutdown()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    # Anonymous, opaque visitor id from the browser (localStorage). No PII, used only to group a
    # returning visitor's turns in tracing/analytics. Bounded length; never parsed for content.
    visitor_id: str | None = Field(default=None, max_length=128)


@app.get("/api/health")
def health() -> dict:
    """Readiness probe. NOT ready unless BOTH the GraphRAG index AND Protegrity protection are up:
    protection is mandatory (no mock), so an unreachable Protegrity means the bot cannot serve a
    turn and must report not-ready. Never 500s."""
    from app.pipeline.llm import resolve_model
    from app.protect.protector import ProtectionUnavailable, get_protector
    model, provider = resolve_model()

    # Protection must be constructible (Discovery + tokenizer reachable). This is the hard gate. We go
    # through the SHARED protector so a health probe never opens a new Protegrity login (opening a
    # session is a login, and /auth/login is separately rate-limited, a fresh login per 10s probe
    # would trip it). Once built, this reuses the one process-wide session.
    protection_ok, protection_detail = True, "ok"
    try:
        get_protector()
    except ProtectionUnavailable as exc:
        protection_ok, protection_detail = False, str(exc)
    except Exception as exc:  # noqa: BLE001, health must never 500
        protection_ok, protection_detail = False, f"protection error: {type(exc).__name__}"

    # Index must be loadable.
    index_ok, index_detail = True, "ok"
    try:
        _orchestrator()
    except Exception as exc:  # noqa: BLE001
        index_ok, index_detail = False, f"index not ready: {type(exc).__name__}"

    ready = protection_ok and index_ok
    detail = "ok" if ready else "; ".join(
        d for d in ((None if protection_ok else f"protection: {protection_detail}"),
                    (None if index_ok else index_detail)) if d)
    return {
        "ok": True,
        "ready": ready,
        "detail": detail,
        "protection": "up" if protection_ok else "down",
        "protection_backend": "protegrity",     # tokenizer (no mock)
        "pii_detector": "discovery",             # detection (no regex)
        "model": model,
        "provider": provider,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    cid = req.conversation_id or str(uuid.uuid4())
    # Keep the visitor id to a safe, opaque token: strip anything but url-safe chars, bound length.
    visitor_id = re.sub(r"[^A-Za-z0-9._-]", "", req.visitor_id or "")[:128] or None
    # Defense in depth: cap and sanitize control chars before anything sees the string.
    message = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", req.message).strip()[:2000]
    if not message:
        return {"answer": "Please type a question about BOTOX®.",
                "safety": False, "refused": True, "blocked": False, "retryable": False,
                "conversation_id": cid, "trace_id": None}
    from app.obs import metrics as _metrics
    try:
        # Measure end-to-end latency; record the per-turn counters after it returns. Both are
        # best-effort no-ops when metrics are off, so a turn is never blocked on telemetry.
        with _metrics.time_turn():
            result = _orchestrator().answer(message, conversation_id=cid, visitor_id=visitor_id)
        _metrics.record_turn(
            result, model=getattr(result, "model", ""),
            input_tokens=getattr(result, "input_tokens", 0),
            output_tokens=getattr(result, "output_tokens", 0),
            emergency=getattr(result, "emergency", False),
        )
    except Exception as exc:  # noqa: BLE001, never surface a stack trace to a public caller
        # Log only the exception TYPE, the message/args could contain a fragment of the user's
        # (possibly PII-bearing) input.
        _log.warning("chat turn failed: %s", type(exc).__name__)
        _metrics.record_error(type(exc).__name__)
        return {"answer": "Sorry, something went wrong on our side. Please try again.",
                "safety": False, "refused": True, "blocked": False, "retryable": False,
                "conversation_id": cid, "trace_id": None}
    # Sources are deliberately NOT returned to the public UI, the visitor sees a clean, grounded
    # answer; the per-chunk provenance lives in the Langfuse trace for operators. (result.citations
    # is still computed and traced.)
    return {
        "answer": result.answer,
        "safety": result.safety,
        "refused": result.refused,
        "blocked": result.blocked,
        "retryable": result.retryable,
        "conversation_id": cid,
        # SIGNED handle for this turn's trace; the UI echoes it back on /api/feedback, which only
        # accepts a valid signature, so feedback can't be forged onto arbitrary trace ids. Null
        # when tracing is off (no trace to rate).
        "trace_id": _sign_trace(result.trace_id) if result.trace_id else None,
    }


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """Streaming variant of /api/chat as Server-Sent Events. Emits `token` events as the draft
    generates, then one `final` event with the guard-approved result (see orchestrator.answer_stream
    for the optimistic-stream-then-guard-retract design). The `final` event's trace_id is SIGNED,
    same as /api/chat, so feedback can attach."""
    import json as _json

    cid = req.conversation_id or str(uuid.uuid4())
    visitor_id = re.sub(r"[^A-Za-z0-9._-]", "", req.visitor_id or "")[:128] or None
    message = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", req.message).strip()[:2000]

    def sse(event: dict) -> str:
        return f"data: {_json.dumps(event)}\n\n"

    def gen():
        if not message:
            yield sse({"type": "final", "answer": "Please type a question about BOTOX®.",
                       "safety": False, "refused": True, "blocked": False, "retryable": False,
                       "conversation_id": cid, "trace_id": None})
            return
        try:
            for event in _orchestrator().answer_stream(message, conversation_id=cid,
                                                        visitor_id=visitor_id):
                if event.get("type") == "final":
                    event["conversation_id"] = cid
                    tid = event.get("trace_id")
                    event["trace_id"] = _sign_trace(tid) if tid else None
                yield sse(event)
        except Exception as exc:  # noqa: BLE001, never surface a stack trace to a public caller
            _log.warning("stream turn failed: %s", type(exc).__name__)
            yield sse({"type": "final",
                       "answer": "Sorry, something went wrong on our side. Please try again.",
                       "safety": False, "refused": True, "blocked": False, "retryable": False,
                       "conversation_id": cid, "trace_id": None})

    # X-Accel-Buffering: no tells any intermediary nginx not to buffer the stream (so tokens flush
    # to the browser immediately); Cache-Control none so a proxy doesn't cache the event stream.
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=200)   # the SIGNED handle from /api/chat
    rating: int = Field(ge=-1, le=1)          # +1 thumbs up, -1 thumbs down (0 ignored)
    comment: str | None = Field(default=None, max_length=1000)


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """Record a visitor's thumbs-up/down on an answer as a `user_feedback` score on its trace. The
    trace handle must carry a valid signature issued by /api/chat, so scores can't be forged."""
    if req.rating == 0:
        return {"ok": False, "recorded": False}
    trace_id = _verify_trace(req.trace_id)
    if trace_id is None:
        # Bad/forged signature, refuse quietly (don't reveal why).
        return {"ok": False, "recorded": False}
    from app.obs import tracing
    comment = None
    if req.comment:
        comment = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", req.comment).strip()[:1000] or None
    recorded = tracing.record_feedback(trace_id=trace_id, rating=req.rating, comment=comment)
    return {"ok": True, "recorded": recorded}


class RevealRequest(BaseModel):
    # A protected transcript/message carrying [TYPE]token[/TYPE] wrappers, e.g. as stored in a
    # trace. The endpoint re-identifies the tokens via Protegrity unprotect. Bounded to keep the
    # request small and the number of unprotect round-trips sane.
    protected_text: str = Field(min_length=1, max_length=8000)


@app.post("/api/support/reveal")
def support_reveal(req: RevealRequest, authorization: str | None = Header(default=None)) -> dict:
    """ROLE-GATED re-identification. An entitled support/internal caller (holding SUPPORT_API_TOKEN)
    submits a PROTECTED transcript and gets the cleartext back, reversing the ingress tokenization
    via Protegrity `unprotect`. This is the ONLY endpoint that re-identifies PII, the public chat
    path is always zero-reveal. Fails closed: disabled unless SUPPORT_API_TOKEN is set; 401 on a bad
    token; a token that can't be reversed (a redaction / unprotect failure) becomes an explicit
    marker, never a guessed value. Loopback-bound behind nginx like the rest.
    """
    _require_support_role(authorization)   # raises 404 (disabled) or 401 (bad token)
    try:
        from app.protect.protector import ProtectionUnavailable, get_protector
        protector = get_protector()
    except Exception as exc:  # noqa: BLE001, protection unavailable -> can't reveal
        _log.warning("reveal unavailable: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Protection service unavailable") from exc
    try:
        cleartext = protector.reveal_text(req.protected_text)
    except Exception as exc:  # noqa: BLE001, never surface internals to the caller
        _log.warning("reveal failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Reveal failed") from exc
    return {"ok": True, "revealed": cleartext}
