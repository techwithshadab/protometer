"""Protection client, the only place that talks to Protegrity's hosted tokenization API.

Everything the rest of the pipeline needs from tokenization goes through here, because the
transport has properties the pipeline must not scatter across itself:

  * **Hosted only.** No local tokenization service exists; every protect call is an HTTPS
    round-trip to api.developer-edition.protegrity.com.
  * **Undocumented rate limits.** The SDK retries on HTTP 429, which establishes throttling
    exists while leaving its thresholds unstated. We design against an unknown ceiling.
  * **Batching is polymorphic.** `session.protect(list)` returns `(results, error_codes)` in
    one round-trip; `session.protect(scalar)` returns a scalar. Batching is the single
    largest lever on ingestion wall-clock, so structured data always takes the list path.

A process-local memo cache sits in front of the API. Tokenization is deterministic, so the same plaintext under the same data element always yields the same token
- which makes caching safe and, on a corpus where parties recur across thousands of
transactions, removes the large majority of calls.

The cache is deliberately keyed on the IV as well: the determinism ablation supplies an IV
precisely to break token stability, and a cache ignoring it would silently restore the
determinism the ablation exists to remove.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

# The SDK's own retry layer, configured explicitly rather than left at its defaults.
#
# `appython.service.request_handler` mounts a urllib3 `Retry` with
# `status_forcelist=(429, 500, 502, 503, 504)`, `backoff_factor=0.5` and
# `respect_retry_after_header=True`, driven by these two environment variables
# (`appython/service/config.py`). Nothing in this project set them, so the SDK ran three
# internal retries with 30-second timeouts *inside* each of this module's own six retries -
# compounding to the 210-second single-value protect call that was originally misread as a
# hard block on /protect.
#
# One retry layer is enough, and it should be the one that can see the HTTP status and the
# `Retry-After` header. The SDK's is kept for transport-level faults; this module's is
# narrowed to session and burst handling above it.
os.environ.setdefault("PTY_MAX_RETRIES", "2")
os.environ.setdefault("PTY_REQUEST_TIMEOUT", "30")

# Batch success codes, from the SDK's own `RETURN_CODE` constant
# (appython/utils/constants.py: {"protect": 6, "unprotect": 8, "reprotect": 50}, and
# documented in `LOG_RETURN_CODE_SUCCESS`). The SDK synthesizes a per-item code tuple on
# success and raises on failure, so these comparisons are belt-and-braces rather than a
# server behaviour we reverse-engineered. Kept named for readability; 50 (reprotect) is the
# one a casual reader is least likely to expect.
PROTECT_OK = 6
UNPROTECT_OK = 8
REPROTECT_OK = 50

# Conservative default. The API's real batch ceiling is undocumented; 200 keeps payloads
# well under the documented 1MB-per-call recommendation for typical field lengths.
DEFAULT_BATCH_SIZE = 200

# Where the login actually happens, verified against the installed SDK (appython 1.2.1):
# `Protector` has no `__init__` and never authenticates. `Protector.create_session(...)`
# constructs a `Session`, and `Session.__init__` -> `create_auth_provider(...).initialize()`
# is what POSTs to `/auth/login` (appython/service/auth_provider.py). So a *session open* is a
# login; a *protect call on an open session* is not.
#
# This project's original design opened a session per scope, per proactive renewal, and per
# retry of a failed open, which multiplied logins against an endpoint that is rate-limited
# independently of `/protect`. Measured consequence: after a bursty run the endpoint returned
# `{"message":"Limit Exceeded"}` / HTTP 429 to *every* login, including one sent with a
# deliberately wrong password, which is how the block was distinguished from a credential
# fault. Protect calls were unaffected; the account simply could not obtain a token.
#
# What this shared Protector actually buys is not "one login": each `create_session` still
# authenticates. The win is that ONE open session is reused for every protect call in a scope
# (hundreds of calls, one login), instead of the pathological earlier paths that re-opened
# per batch or per retry. Renewal is deliberately infrequent (below), so a full eight-scope
# ingest costs roughly one login per scope plus the occasional renewal, not one per call.
_SHARED_PROTECTOR: Any = None


def _shared_protector() -> Any:
    """The process-wide `appython.Protector`, constructed on first use."""
    global _SHARED_PROTECTOR
    if _SHARED_PROTECTOR is None:
        from appython import Protector as _Protector  # lazy so tests can stub

        _SHARED_PROTECTOR = _Protector()
    return _SHARED_PROTECTOR


def reset_shared_protector() -> None:
    """Drop the cached Protector, forcing a fresh login on next use.

    Only needed when credentials change mid-process, or in tests.
    """
    global _SHARED_PROTECTOR
    _SHARED_PROTECTOR = None


def _is_auth_limit(exc: Exception) -> bool:
    """True when an exception is the login-rate block rather than a credential fault.

    The SDK collapses both into `InitializationError: Could not authenticate user.`, so the
    text alone cannot separate them; the distinguishing evidence is that the endpoint returns
    429 for valid and invalid credentials alike while the block is active. Treated as
    non-retryable either way, retrying a rejected login only extends the block, and retrying
    genuinely bad credentials never succeeds.
    """
    message = str(exc).lower()
    return (
        "could not authenticate" in message
        or "limit exceeded" in message
        or "429" in message
    )


# Public alias: other modules (e.g. reidentify) need to distinguish an auth-throttle from a
# per-item failure so they can fail loud on the throttle rather than swallowing it.
def is_auth_limit(exc: Exception) -> bool:
    """Whether `exc` is the Protegrity login-rate block (see `_is_auth_limit`)."""
    return _is_auth_limit(exc)


class ProtectionError(RuntimeError):
    """Raised when the hosted API fails in a way retrying will not fix."""


@dataclass
class ProtectionStats:
    """Call accounting. Ingestion cost is dominated by network round-trips, so these
    numbers are what tell us whether batching and caching are actually working."""

    api_calls: int = 0
    values_protected: int = 0
    cache_hits: int = 0
    retries: int = 0
    seconds_in_api: float = 0.0
    # Per-call durations, so latency is reportable as a distribution (p50/p90/p95/p99), not
    # only as a total. A single "seconds_in_api" hid whether one slow call or many mediocre
    # ones dominated, the exact question an operator asks when the API feels slow.
    call_seconds: list = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.values_protected
        return self.cache_hits / total if total else 0.0

    def latency_percentile(self, pct: float) -> float:
        if not self.call_seconds:
            return 0.0
        s = sorted(self.call_seconds)
        return s[min(int(len(s) * pct / 100), len(s) - 1)]

    def summary(self) -> str:
        return (
            f"api_calls={self.api_calls} values={self.values_protected} "
            f"cache_hits={self.cache_hits} ({self.cache_hit_rate:.1%}) "
            f"retries={self.retries} api_time={self.seconds_in_api:.1f}s"
        )

    def to_dict(self) -> dict:
        """Structured form for reports. The formatted `summary()` string was what got
        persisted, which no dashboard or script can consume without a regex, a metric that
        must be parsed back out of its own presentation is not a metric."""
        return {
            "api_calls": self.api_calls,
            "values_protected": self.values_protected,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "retries": self.retries,
            "seconds_in_api": round(self.seconds_in_api, 2),
            "latency_p50": round(self.latency_percentile(50), 3),
            "latency_p90": round(self.latency_percentile(90), 3),
            "latency_p95": round(self.latency_percentile(95), 3),
            "latency_p99": round(self.latency_percentile(99), 3),
        }


@dataclass
class Protector:
    """Batching, caching, retrying wrapper over a Protegrity session.

    `external_iv` is set only by the determinism ablation. When present it is passed to
    every call and folded into the cache key, so tokens stop being stable across calls in
    exactly the way the ablation requires.
    """

    policy_user: str = "superuser"
    batch_size: int = DEFAULT_BATCH_SIZE
    external_iv: bytes | None = None
    max_retries: int = 6
    # Minimum seconds between calls. The published limits are 50 req/s with burst 100;
    # 0.02s enforces the sustained ceiling by mechanism rather than by the accident of
    # network latency (a colocated client would otherwise exceed it). The unbatched
    # ablation path raises its own floor further.
    min_call_interval: float = 0.02
    # First backoff after a throttle. Measured recovery is ~4 minutes, so retries must be
    # patient rather than fast.
    throttle_backoff_seconds: float = 30.0
    # Session lifetime in MINUTES (the SDK's unit). Sessions expire mid-run on long paths,
    # so this is renewed proactively rather than relied upon to outlast the work.
    session_minutes: int = 30
    # The determinism ablation. A *fixed* external_iv is not enough: tokenization stays
    # deterministic for that IV, so the same value still yields the same token and
    # cross-document entity resolution survives. Rotating the IV per call is what actually
    # destroys token stability, which is the property the ablation exists to remove.
    rotate_iv: bool = False
    stats: ProtectionStats = field(default_factory=ProtectionStats)

    _session: Any = field(default=None, repr=False)
    _cache: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)
    _iv_counter: int = field(default=0, repr=False)
    _last_call_at: float = field(default=0.0, repr=False)
    _session_opened_at: float = field(default=0.0, repr=False)

    def _open_session(self) -> None:
        """Create a fresh Protegrity session and record when it was opened.

        Sessions expire, `create_session(timeout=...)` is in **minutes**, defaulting to 15
, and expiry surfaces mid-run as "User session is invalid or timed out!!". The
        rotating ablation path runs for far longer than that, so the session is renewed on a
        timer rather than assumed to last the run.
        """
        # Session creation is retried in its own right. It is a network call like any other,
        # and an unretried one is a single point of failure for an unattended run: a transient
        # DNS or TLS blip during `create_session` aborted a completed-but-for-one-scope
        # ingestion after the API had already been reachable for thousands of calls.
        delay = 2.0
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self._session = _shared_protector().create_session(
                    self.policy_user, timeout=self.session_minutes
                )
                self._session_opened_at = time.monotonic()
                return
            except Exception as exc:  # noqa: BLE001, SDK wraps transport errors opaquely
                last = exc
                # Authentication failures are not transient and must not be retried.
                #
                # `appython.Protector()` performs a POST /auth/login in its constructor, and
                # that endpoint is rate-limited *separately from* /protect. Retrying a failed
                # login six times issues six more logins against an endpoint that is already
                # refusing them, which deepens the block instead of waiting it out, the SDK's
                # own auth path deliberately applies no retries for exactly this reason.
                #
                # Measured: once tripped, `POST /auth/login` returns
                # `{"message":"Limit Exceeded"}` with **HTTP 429** for every request,
                # including ones carrying deliberately wrong credentials. The endpoint is
                # rejecting before it authenticates, so "Could not authenticate user" from
                # the SDK in this state means *throttled*, not *bad password*.
                if _is_auth_limit(exc):
                    raise ProtectionError(
                        "Protegrity /auth/login is rate-limited (HTTP 429 'Limit Exceeded'). "
                        "This is a login-rate block, not a credential problem, the same "
                        "response is returned for invalid credentials while it is active. "
                        "Wait for it to clear rather than retrying, which extends the block."
                    ) from exc
                if attempt < self.max_retries - 1:
                    self.stats.retries += 1
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)

        raise ProtectionError(f"Could not open session after {self.max_retries} attempts: {last}")

    def _ensure_session(self) -> None:
        """Keep a valid session, refreshing before expiry rather than after a failure.

        Each open IS a login (see the module note), so both mechanisms are kept
        deliberately infrequent to spare the rate-limited auth endpoint:

          * *Proactive*, renew two minutes before the server's own expiry, so a long call
            cannot straddle the boundary and fail halfway through a batch. The SDK's session
            ttl is idle-based and refreshes on use, so a busy scope rarely triggers this at
            all; it exists for the gaps between scopes.
          * *Reactive*, `_call_with_retry` clears `_session` when the API reports
            "session is invalid" or "timed out", and the next attempt lands here.

        The reused open session, not a cached login, is what keeps the per-scope login count
        at roughly one; `create_session` carries the policy user and authenticates each time.
        """
        age_limit = max(60.0, (self.session_minutes - 2) * 60)
        if self._session is None or time.monotonic() - self._session_opened_at > age_limit:
            self._open_session()

    def __post_init__(self) -> None:
        for var in ("DEV_EDITION_EMAIL", "DEV_EDITION_PASSWORD", "DEV_EDITION_API_KEY"):
            if not os.getenv(var):
                raise ProtectionError(f"Missing credential: {var}")

        # Rotation batches by repeat index rather than one value per call, see `_chunks`.
        # The old design sent a single value per round-trip (34,874 for this corpus), which
        # exhausted the burst limit and left the ablation the one scope that could not finish.
        # Batching cuts that to ~556 calls with identical semantics, so the pacing floor can
        # be much smaller: it now guards the gap between batches, not between values.
        if self.rotate_iv and not self.min_call_interval:
            self.min_call_interval = 0.05

        self._open_session()

    # -- internals ---------------------------------------------------------------

    def _next_iv(self) -> bytes | None:
        """The IV for the next call. Rotates per call under the ablation."""
        if not self.rotate_iv:
            return self.external_iv
        self._iv_counter += 1
        return self._iv_counter.to_bytes(8, "big")

    def _kwargs(self) -> dict:
        iv = self._next_iv()
        return {"external_iv": iv} if iv else {}

    def _cache_key(self, value: str, data_element: str) -> tuple[str, str] | None:
        """Cache key, or None when results must not be cached.

        Under IV rotation every call is intended to produce a different token, so caching
        would defeat the ablation by serving the first token back for every later occurrence.
        """
        if self.rotate_iv:
            return None
        iv = self.external_iv.hex() if self.external_iv else ""
        return (f"{data_element}|{iv}", value)

    def _call_with_retry(self, operation: str, *args, **kwargs):
        """Retry with exponential backoff, pacing calls to stay under the burst limit.

        Developer Edition's published limits are 50 req/s, burst 100, and 10,000
        requests/user/day. The *behaviour* at the limit is still only measurable: a
        sustained run of unbatched calls returns **403 Forbidden**, not 429, and the
        account stays blocked for roughly four minutes, after which it recovers and
        returns the same deterministic tokens as before.

        Two mitigations follow. `min_call_interval` paces outgoing calls so the limit is not
        tripped, and the backoff below starts long enough to outlast a block that has
        already happened. Ingestion runs unattended across five scopes, so a
        throttle must cost minutes rather than aborting the run.
        """
        delay = self.throttle_backoff_seconds
        last: Exception | None = None

        for attempt in range(self.max_retries):
            # Pace: keep a floor between consecutive calls rather than discovering the
            # ceiling the hard way.
            if self.min_call_interval:
                elapsed = time.monotonic() - self._last_call_at
                if elapsed < self.min_call_interval:
                    time.sleep(self.min_call_interval - elapsed)
                self._last_call_at = time.monotonic()

            self._ensure_session()

            started = time.monotonic()
            try:
                # Resolved per attempt, not captured up front: a renewed session must
                # actually be the one used by the retry.
                result = getattr(self._session, operation)(*args, **kwargs)
                _elapsed = time.monotonic() - started
                self.stats.seconds_in_api += _elapsed
                self.stats.call_seconds.append(_elapsed)
                self.stats.api_calls += 1
                return result
            except Exception as exc:  # noqa: BLE001, SDK exception types are not public
                self.stats.seconds_in_api += time.monotonic() - started
                last = exc
                message = str(exc).lower()
                # Input the API will never accept, retrying only wastes the rate budget
                # and, worse, contributes to tripping the burst limit.
                if "not valid" in message or "invalid input" in message:
                    raise ProtectionError(f"Rejected by API: {exc}") from exc

                # An expired session is recoverable and cheap to fix: drop it so the next
                # attempt opens a fresh one, and retry immediately rather than backing off.
                if "session is invalid" in message or "timed out" in message:
                    self._session = None
                    self.stats.retries += 1
                    continue

                if attempt < self.max_retries - 1:
                    self.stats.retries += 1
                    # A 403 here means the burst limit, not a credential problem: the same
                    # key recovers on its own and returns identical tokens afterwards.
                    # Recovery was measured at ~4 minutes, so back off in minutes.
                    throttled = "forbidden" in message or "429" in message
                    time.sleep(delay if throttled else min(delay, 5.0))
                    delay = min(delay * 2, 300.0)

        raise ProtectionError(f"Failed after {self.max_retries} attempts: {last}") from last

    # -- public API --------------------------------------------------------------

    def protect_value(self, value: str, data_element: str) -> str:
        """Protect a single value. Prefer `protect_values`, this costs a full round-trip."""
        if not value:
            return value

        key = self._cache_key(value, data_element)
        if key is not None and key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]

        token = self._call_with_retry("protect", value, data_element, **self._kwargs())
        self.stats.values_protected += 1
        if key is not None:
            self._cache[key] = token
        return token

    @staticmethod
    def is_noop(plaintext: str, token: str) -> bool:
        """True when the API returned the input unchanged, i.e. did not protect it.

        Measured: `session.protect("seven hundred twelve thousand dollars",
        "number")` returns that exact string back, with a success code and no error. A caller
        that trusts the status believes the value is protected while the plaintext is still
        there.

        This is the most dangerous failure shape in the system, worse than an exception,
        which at least stops the run, so every protected value is checked against its input.
        """
        return bool(plaintext) and plaintext == token

    def protect_values(self, values: list[str], data_element: str) -> list[str]:
        """Protect many values under one data element, batching uncached ones.

        This is the path that makes ingestion affordable: one round-trip per batch rather
        than per value. Order is preserved so callers can zip results back onto records.
        """
        results: list[str | None] = [None] * len(values)
        pending: list[tuple[int, str]] = []

        for i, value in enumerate(values):
            if not value:
                results[i] = value
                continue
            key = self._cache_key(value, data_element)
            cached = self._cache.get(key) if key is not None else None
            if cached is not None:
                self.stats.cache_hits += 1
                results[i] = cached
            else:
                pending.append((i, value))

        # Deduplicate within the batch: a party recurring across transactions should cost
        # one slot, not one per occurrence. Under IV rotation each occurrence must get its
        # own token, so every position is kept distinct instead.
        unique: dict[str, list[int]] = {}
        if self.rotate_iv:
            for i, value in pending:
                unique[f"{i}\x00{value}"] = [i]
        else:
            for i, value in pending:
                unique.setdefault(value, []).append(i)

        uniques = list(unique)
        for chunk in self._chunks(uniques, data_element):
            # Under rotation the keys carry a position prefix to keep occurrences distinct;
            # the API must still receive the bare values.
            payload = [k.split("\x00", 1)[1] for k in chunk] if self.rotate_iv else chunk

            tokens, codes = self._call_with_retry("protect", payload, data_element, **self._kwargs())

            for key, value, token, code in zip(chunk, payload, tokens, codes):
                if code != PROTECT_OK:
                    raise ProtectionError(
                        f"protect failed for data_element={data_element!r} code={code}"
                    )
                cache_key = self._cache_key(value, data_element)
                if cache_key is not None:
                    self._cache[cache_key] = token
                self.stats.values_protected += 1
                for i in unique[key]:
                    results[i] = token

        assert all(r is not None for r in results), "protect_values left a gap"
        return results  # type: ignore[return-value]

    def _chunks(self, keys: list[str], data_element: str) -> "list[list[str]]":
        """Group keys into batches the API can serve in one call each.

        Without IV rotation this is a plain slice: values are already deduplicated, so any
        grouping is safe.

        **Under rotation it is not.** One `external_iv` applies to the whole call, so two
        copies of the same plaintext inside a single batch tokenize identically, verified
        against the live API: `protect(['Alice','Alice'], external_iv=3)` returns
        `['w6epe pyGz1', 'w6epe pyGz1']`, while the same values under IVs 1 and 2 differ. That
        collision is precisely what the ablation must avoid, since it exists to destroy token
        stability across occurrences.

        The previous design concluded from this that rotation cannot batch at all, and sent
        **one value per call**, 34,874 round-trips for this corpus, which is what exhausted
        the burst limit and made the ablation the only scope that could not finish. The
        conclusion was too strong: the constraint is not *no batching*, it is *no duplicate
        inside a batch*.

        So batches are formed by **repeat index** rather than by position: the first
        occurrence of every distinct value goes in round one, the second occurrence of every
        value that has one goes in round two, and so on. No batch can contain a duplicate by
        construction, each round still gets its own IV, and the number of calls falls to the
        count of rounds rather than the count of occurrences, 34,874 to roughly 556 here, a
        63x reduction with identical ablation semantics.
        """
        if not self.rotate_iv:
            return [
                keys[start : start + self.batch_size]
                for start in range(0, len(keys), self.batch_size)
            ]

        # Keys are "position\x00value" under rotation; group by the value part.
        rounds: dict[str, list[str]] = {}
        for key in keys:
            value = key.split("\x00", 1)[1]
            rounds.setdefault(value, []).append(key)

        by_round: list[list[str]] = []
        for depth in range(max((len(v) for v in rounds.values()), default=0)):
            layer = [occurrences[depth] for occurrences in rounds.values()
                     if len(occurrences) > depth]
            # Still honour the batch cap, and the 1 MB-per-call guidance behind it.
            by_round.extend(
                layer[start : start + DEFAULT_BATCH_SIZE]
                for start in range(0, len(layer), DEFAULT_BATCH_SIZE)
            )
        return by_round

    def reprotect_values(
        self, tokens: list[str], old_element: str, new_element: str
    ) -> list[str]:
        """Migrate tokens from one data element to another, plaintext never touching us.

        This is the API's key-rotation story: `reprotect` unprotects under the old element
        and re-protects under the new one **server-side**, so a re-keying migration never
        exposes plaintext to the application performing it. Verified live: a `string` token
        migrates to `name` and back, round-tripping to both the original plaintext and the
        original token, determinism holds in both directions.

        Both elements must be tokenization-type (SDK constraint). Not cached: migration is a
        one-way administrative operation, not a hot path, and caching a mapping between two
        element namespaces would double the cache's blast radius for no repeat-call benefit.
        """
        if not tokens:
            return []
        out: list[str] = []
        for start in range(0, len(tokens), self.batch_size):
            chunk = tokens[start : start + self.batch_size]
            result = self._call_with_retry("reprotect", chunk, old_element, new_element)
            # Polymorphic like protect: list in -> (values, codes) out.
            values, codes = result if isinstance(result, tuple) else (result, [])
            for code in codes:
                if code != REPROTECT_OK:
                    raise ProtectionError(
                        f"reprotect failed {old_element}->{new_element} code={code}"
                    )
            out.extend(values)
        return out

    def unprotect_value(self, token: str, data_element: str) -> str:
        """Re-identify a single token. Called only at the presentation boundary."""
        if not token:
            return token
        return self._call_with_retry("unprotect", token, data_element, **self._kwargs())

    def unprotect_values(self, tokens: list[str], data_element: str) -> list[str]:
        """Re-identify many tokens in one round-trip."""
        if not tokens:
            return []
        non_empty = [t for t in tokens if t]
        if not non_empty:
            return list(tokens)

        values, codes = self._call_with_retry("unprotect", non_empty, data_element, **self._kwargs())
        for code in codes:
            if code != UNPROTECT_OK:
                raise ProtectionError(f"unprotect failed code={code}")

        recovered = iter(values)
        return [next(recovered) if t else t for t in tokens]
