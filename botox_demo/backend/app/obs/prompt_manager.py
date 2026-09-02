"""Langfuse-managed system prompt, with the hardcoded prompt as a fail-safe fallback.

Design goals (mirror tracing.py):
  - The pipeline gets its system prompt from ONE place, `get_system_prompt()`, whether or not
    Langfuse is reachable. When Langfuse Prompt Management has a published version, we use it (so the
    prompt can be edited and versioned in the Langfuse UI without a redeploy). When Langfuse is
    absent, unreachable, or has no prompt yet, we fall back to the hardcoded `SYSTEM_PROMPT`.
  - Fetching NEVER raises into the request path and NEVER blocks it for long: results are cached with
    a short TTL, and any failure is logged and swallowed (returns the fallback).
  - The returned object carries the Langfuse prompt handle when there is one, so the caller can LINK
    the generation to that exact prompt version in the trace (lineage in the UI). No handle ->
    nothing to link, and the trace still records that the fallback was used.

Configuration (env):
  BOTOX_PROMPT_NAME       Langfuse prompt name to fetch (default "botox-system")
  BOTOX_PROMPT_LABEL      label/version to pull (default "production")
  BOTOX_PROMPT_TTL        cache seconds (default 60); 0 disables caching
  BOTOX_PROMPT_SOURCE=local  force the hardcoded prompt, skip Langfuse entirely
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from app.obs import tracing
from app.pipeline.llm import SYSTEM_PROMPT

_log = logging.getLogger("botox.prompt")

_NAME = os.getenv("BOTOX_PROMPT_NAME", "botox-system")
_LABEL = os.getenv("BOTOX_PROMPT_LABEL", "production")


def _ttl() -> float:
    try:
        return float(os.getenv("BOTOX_PROMPT_TTL", "60"))
    except ValueError:
        return 60.0


@dataclass(frozen=True)
class ResolvedPrompt:
    """The system prompt for a turn plus its provenance.

    text     the system-prompt string to send to the model.
    source   "langfuse" when it came from Prompt Management, "local" for the hardcoded fallback.
    version  the Langfuse prompt version (int) when source == "langfuse", else None.
    handle   the Langfuse prompt object when source == "langfuse" (used to link a generation to
             this prompt version in the trace), else None.
    """

    text: str
    source: str
    version: int | None = None
    handle: Any | None = None


_FALLBACK = ResolvedPrompt(text=SYSTEM_PROMPT, source="local", version=None, handle=None)

# module-level cache: (resolved, fetched_at)
_cache: tuple[ResolvedPrompt, float] | None = None


def get_system_prompt() -> ResolvedPrompt:
    """Return the current system prompt. Langfuse-managed when available, hardcoded otherwise.

    Cheap and safe to call once per turn: served from a short-TTL cache, and any failure returns the
    hardcoded fallback rather than raising.
    """
    global _cache

    if os.getenv("BOTOX_PROMPT_SOURCE", "").lower() == "local":
        return _FALLBACK

    ttl = _ttl()
    now = time.monotonic()
    if _cache is not None and ttl > 0 and (now - _cache[1]) < ttl:
        return _cache[0]

    resolved = _fetch() or _FALLBACK
    if ttl > 0:
        _cache = (resolved, now)
    return resolved


def _fetch() -> ResolvedPrompt | None:
    lf = tracing.client()
    if lf is None:
        return None  # tracing/Langfuse disabled -> fall back, quietly
    try:
        # get_prompt(name, label=..., cache_ttl_seconds=...) -> a prompt client (v4 API, same call shape).
        # The SDK has its own client-side cache too; our TTL bounds staleness at the app level.
        prompt = lf.get_prompt(_NAME, label=_LABEL, cache_ttl_seconds=int(_ttl()))
        text = prompt.prompt if isinstance(prompt.prompt, str) else str(prompt.prompt)
        if not text.strip():
            _log.warning("Langfuse prompt %r is empty; using local fallback", _NAME)
            return None
        version = getattr(prompt, "version", None)
        return ResolvedPrompt(text=text, source="langfuse", version=version, handle=prompt)
    except Exception as exc:  # noqa: BLE001, prompt fetch must never break a turn
        _log.info("prompt fetch fell back to local (%s: %s)", type(exc).__name__, str(exc)[:120])
        return None


def invalidate_cache() -> None:
    """Drop the cached prompt so the next call re-fetches. Used by the seed script and tests."""
    global _cache
    _cache = None
