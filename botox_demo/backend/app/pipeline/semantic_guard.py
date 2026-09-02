"""Protegrity Semantic Guardrail: an ML egress scan layered on top of the regex egress guard.

The regex guard (guardrail.scan_reply) is always-on and self-contained. This adds Protegrity's
Semantic Guardrail service as a second, stronger opinion on the OUTBOUND reply when it is wired
(PROTEGRITY_SEMANTIC_GUARD=on). It runs locally in Docker, no API key.

    POST {SEMANTIC_GUARD_URL}
    body: {"messages": [{"from": "ai", "to": "user", "content": "<reply>", "processors": ["pii"]}]}
    -> {"messages": [{"outcome": "approved|rejected|skipped", "score": 0..1,
                      "processors": [{"name","score","explanation"}]}], "batch": {...}}

For a health domain, Protegrity's own measurements show the `healthcare` domain model discriminates
well (unlike the AML case), and the `pii` processor flags a leaked identifier at ~0.99 with
character offsets. We use the `pii` processor on the reply: a high-scoring PII finding means the
model leaked an identifier, which is exactly what the protection boundary exists to prevent.

Design (fail-closed, best-effort-available):
  * When the service is UNREACHABLE, this returns `available=False` and the caller keeps the regex
    guard's verdict, observability degrades, safety does not (the regex leak check still ran).
  * When the service REJECTS the reply (PII/injection above its threshold), that is a BLOCK: the
    reply is retracted to the safe fallback, on top of whatever the regex guard decided.
  * The surrogate-token false positive Protegrity has on its own protection tokens is handled: a
    rejection whose only finding is one of OUR wrapper tokens (which the regex guard already blocks
    explicitly) is not double-counted, but since a leaked wrapper is blocked anyway, we keep the
    block. We never DOWNGRADE a rejection to a pass.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

_log = logging.getLogger("botox.semguard")

SEMANTIC_GUARD_URL = os.getenv(
    "PROTEGRITY_SEMANTIC_GUARD_URL",
    "http://vendor-de-guardrail:8581/pty/semantic-guardrail/v1.1/conversations/messages/scan",
)
# The processor to run on the outbound reply. `pii` scores identifiers in the response; it is the
# one measured to work well for leaked names/emails/IDs.
_RESPONSE_PROCESSOR = os.getenv("PROTEGRITY_SEMANTIC_PROCESSOR", "pii")


def enabled() -> bool:
    """True when the semantic guard is switched on (it is opt-in, off by default so the demo runs
    with no vendor services)."""
    return os.getenv("PROTEGRITY_SEMANTIC_GUARD", "").lower() in ("on", "1", "true", "yes")


@dataclass
class SemanticVerdict:
    available: bool                 # did we get a verdict from the service at all?
    rejected: bool = False          # service said reject (block the reply)
    score: float = 0.0              # 0..1, higher = riskier
    findings: list[str] = field(default_factory=list)   # e.g. ["EMAIL_ADDRESS : [9, 25]"]
    reason: str = ""


class _Client:
    """Lazy session-reusing client. Imports requests on construction only."""

    def __init__(self) -> None:
        import requests
        self._session = requests.Session()

    def scan_response(self, reply: str, *, timeout: float = 10.0) -> SemanticVerdict:
        import requests
        payload = {"messages": [{"from": "ai", "to": "user", "content": reply,
                                 "processors": [_RESPONSE_PROCESSOR]}]}
        try:
            r = self._session.post(SEMANTIC_GUARD_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            body = r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError, ValueError) as exc:
            # Unreachable/undecodable: no verdict. The caller keeps the regex guard's decision;
            # safety is not weakened (the regex leak check still ran), only this extra opinion is
            # missing. Logged at warning so an operator sees the degraded observability.
            _log.warning("semantic guard unavailable (%s); using regex guard only",
                         type(exc).__name__)
            return SemanticVerdict(available=False, reason=f"unavailable: {type(exc).__name__}")

        messages = body.get("messages") or [{}]
        msg = messages[0]
        outcome = str(msg.get("outcome", "unknown"))
        score = float(msg.get("score", 0.0))
        findings: list[str] = []
        for proc in msg.get("processors") or []:
            expl = str(proc.get("explanation", "")).strip()
            if expl:
                findings.append(expl)
        rejected = outcome == "rejected"
        return SemanticVerdict(
            available=True, rejected=rejected, score=score, findings=findings,
            reason=(f"semantic guard rejected (score {score:.2f})" if rejected else ""))


_CLIENT: _Client | None = None
_TRIED = False


def _client() -> _Client | None:
    """The process-wide client, built once. Returns None if requests/construction fails."""
    global _CLIENT, _TRIED
    if not _TRIED:
        _TRIED = True
        try:
            _CLIENT = _Client()
        except Exception as exc:  # noqa: BLE001
            _log.warning("semantic guard client init failed (%s)", type(exc).__name__)
            _CLIENT = None
    return _CLIENT


def scan_response(reply: str) -> SemanticVerdict:
    """Scan an outbound reply. Returns available=False when the guard is off or unreachable, so the
    caller can fall back to the regex guard's verdict without any safety regression."""
    if not enabled():
        return SemanticVerdict(available=False, reason="disabled")
    client = _client()
    if client is None:
        return SemanticVerdict(available=False, reason="client unavailable")
    return client.scan_response(reply)
