"""User-side PII protection for the public BOTOX chatbot: Protegrity only.

The bot is public: a visitor may type a name, email, phone, or "my doctor is Dr. Smith" into the
chat. That PII must be tokenized BEFORE it reaches retrieval, the model, or any log, and never
re-appear in a stored trace. This module is the single protection boundary, and it uses Protegrity
Developer Edition for both halves of the job:

* **Detection** via the Data **Discovery** service (`protect/discovery.py`): an ML/NER classifier
  that finds the PII spans in free text (names, emails, phones, addresses, DOB, member IDs).
* **Tokenization** via the Data **Protection** service through the `appython` SDK: each detected
  span becomes a deterministic surrogate token.

There is **no mock and no regex fallback**. Protegrity is mandatory: if Discovery or the tokenizer
is unavailable, protection cannot run, and the pipeline fails **closed** (a turn refuses safely, and
`/api/health` reports not-ready) rather than letting unprotected text through. `Protector()` raises
`ProtectionUnavailable` when it cannot reach Protegrity, so startup can surface it loudly.

Tokens are wrapped `[TYPE]token[/TYPE]` so the egress guard and re-identifier can find them. A no-op
"protection" (the API returns the value unchanged, a measured Protegrity failure shape) is treated as
a hard failure for that value, we never emit cleartext under a protection claim.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

_log = logging.getLogger("botox.protect")

# The PII wrapper types this protector emits (the targets of the Discovery->wrapper mapping in
# discovery.py). A precise wrapper regex built from THIS list (not a broad `[A-Z_]+`) is what the
# pipeline uses to strip/detect our tokens, so a legitimate bracketed acronym in an answer ("[FDA]")
# is never mistaken for a PII wrapper and deleted or blocked. Covers the identifiers a patient chat
# realistically surfaces; each maps to a Protegrity data element in _ELEMENT below.
PII_TYPES: tuple[str, ...] = (
    "EMAIL", "PHONE", "SSN", "DOCTOR", "NAME", "ZIP",
    "CREDITCARD", "DOB", "ADDRESS", "PASSPORT", "DL", "MEMBERID", "AGE",
)
_TYPE_ALT = "|".join(PII_TYPES)
# The [TYPE]...[/TYPE] framing we splice around each token. Used to detect a leaked wrapper (egress
# guard) and to strip the framing before the model sees the query.
TOKEN_RE = re.compile(rf"\[/?(?:{_TYPE_ALT})\]")
WRAPPER_RE = TOKEN_RE


class ProtectionUnavailable(RuntimeError):
    """Raised when Protegrity protection cannot run (Discovery or the tokenizer is unreachable). The
    pipeline fails closed on this: no unprotected text flows."""


class _NoopInBatch(Exception):
    """Internal signal: a batched protect returned an unchanged (no-op) value, so re-run that batch
    value-by-value to isolate and redact the offender instead of trusting the no-op."""


# Prefix marking a REDACTION token: a value the tokenizer could not protect, replaced with a one-way
# hash. Non-reversible, so reveal() returns a marker for these rather than attempting an unprotect.
_REDACTION_PREFIX = "RD"


def _redaction_token(etype: str, value: str) -> str:
    """A deterministic, non-reversible surrogate for a value the tokenizer could not protect. Same
    (type,value) -> same token within the process, so it stays stable in a conversation, but it is a
    one-way hash and reveals nothing. Shaped like a wrapperable token (no separators)."""
    import hashlib
    digest = hashlib.sha256(f"REDACT:{etype}:{value}".encode()).hexdigest()[:12]
    return f"{_REDACTION_PREFIX}{digest}"


# One shared Protector per process. Constructing a Protector opens a Protegrity SESSION, and opening
# a session is a LOGIN against `/auth/login`, which is rate-limited INDEPENDENTLY of `/protect`
# (measured; the SDK misreports its 429 as bad credentials). So a fresh Protector per request, or per
# health probe, would issue a login every few seconds and trip the limit, taking the whole bot down.
# The orchestrator and the health check both go through `get_protector()` so the process holds exactly
# one login. `get_protector()` re-raises the same ProtectionUnavailable on every call until Protegrity
# becomes reachable, so health keeps reporting not-ready without hammering the login endpoint.
_SHARED: "Protector | None" = None
# FastAPI runs sync endpoints in a threadpool, so two requests can hit get_protector() concurrently
# before the singleton exists. Without a lock, both would construct a Protector = two Protegrity
# LOGINS at once, the login storm the singleton exists to prevent. Double-checked locking builds
# exactly one.
_SHARED_LOCK = __import__("threading").Lock()


def get_protector() -> "Protector":
    """The process-wide Protector, built once (thread-safe). Raises ProtectionUnavailable while
    Protegrity is unreachable (retried on the NEXT call, never more than one construction attempt at
    a time, so a health probe or a concurrent request burst does not spawn a login storm)."""
    global _SHARED
    if _SHARED is None:
        with _SHARED_LOCK:
            if _SHARED is None:                 # re-check inside the lock
                _SHARED = Protector()           # raises ProtectionUnavailable if Protegrity is down
    return _SHARED


def reset_protector() -> None:
    """Drop the shared Protector (tests, or a credential change mid-process)."""
    global _SHARED
    with _SHARED_LOCK:
        _SHARED = None


@dataclass
class ProtectResult:
    protected: str                       # message with PII wrapped as [TYPE]token[/TYPE]
    entities: int = 0                    # count of PII spans tokenized
    types: dict[str, int] = field(default_factory=dict)
    mapping: dict[str, str] = field(default_factory=dict)  # token -> original (for role reveal)


class TokenRegistry:
    """A PERSISTED allowlist of the tokens THIS system has issued, mapping each token to the data
    ELEMENT it was protected under, and NOTHING else, no cleartext is ever stored.

    Its only job is to let reveal PROVE a token is one we minted (closing the unprotect-oracle): a
    token in the allowlist is unprotected on demand via Protegrity; a token that is not simply is not
    revealable. Persisting the allowlist (not the cleartext) is what makes reveal survive restarts
    and cross-process support workflows without turning the store into a PII datastore.

    Backed by SQLite (stdlib, no dependency), chosen deliberately over a JSON file:
      * INSERT OR IGNORE is O(1) with no full-file rewrite, so a busy deployment doesn't rewrite an
        ever-growing file inside every chat turn (avoids the O(n)-per-add / O(N^2) cliff).
      * A point-lookup SELECT means a reveal miss is a single indexed query, NOT a reload of the whole
        file, so a transcript of junk tokens can't amplify into N full-file reads.
      * SQLite's own locking (WAL mode) handles CONCURRENT writers across threads AND processes safely,
        no shared temp file to corrupt, no allowlist getting silently wiped to empty.
    The DB file is created 0600 (only tokens + element types, but still least-privilege). All DB
    errors are swallowed and degrade to an in-memory set, they never break protect().
    """

    def __init__(self, path=None) -> None:
        import threading
        from app.paths import DATA_DIR
        # Default to a .db path; a caller may pass any path (tests use a temp dir).
        self._path = path or (DATA_DIR / "reveal-registry.db")
        self._lock = threading.Lock()          # guards the single shared connection
        self._mem: dict[str, str] = {}         # small hot cache (token -> etype); NOT the source of truth
        self._conn = None
        self._init_db()

    def _init_db(self) -> None:
        try:
            import os as _os
            import sqlite3
            self._path.parent.mkdir(parents=True, exist_ok=True)
            existed = self._path.exists()
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=5.0)
            self._conn.execute("PRAGMA journal_mode=WAL")      # concurrent readers + one writer, cross-process
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS issued (token TEXT PRIMARY KEY, etype TEXT NOT NULL)")
            self._conn.commit()
            if not existed:
                try:
                    _os.chmod(self._path, 0o600)                # least-privilege; no cleartext, but tidy
                except OSError:
                    pass
        except Exception as exc:  # noqa: BLE001, DB unavailable -> in-memory only, never break protect()
            _log.warning("reveal registry DB init failed (%s); in-memory only", type(exc).__name__)
            self._conn = None

    def add(self, token: str, etype: str) -> None:
        """Record that `token` was issued under wrapper type `etype`. Idempotent and O(1); a repeated
        value (deterministic tokenization) is a cheap no-op INSERT OR IGNORE."""
        if not token or self._mem.get(token) == etype:
            return
        self._mem[token] = etype
        # Keep the hot cache bounded; the DB is the source of truth, so evicting is safe.
        if len(self._mem) > 10000:
            self._mem.clear()
            self._mem[token] = etype
        if self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute("INSERT OR IGNORE INTO issued(token, etype) VALUES (?, ?)",
                                   (token, etype))
                self._conn.commit()
        except Exception as exc:  # noqa: BLE001, persistence is best-effort; never break protect()
            _log.warning("reveal registry persist failed (%s)", type(exc).__name__)

    def etype_of(self, token: str) -> str | None:
        """The element type a token was issued under, or None if we never issued it. A single indexed
        point lookup, cheap on a miss, so it can't be amplified into large file reads."""
        cached = self._mem.get(token)
        if cached is not None:
            return cached
        if self._conn is None:
            return None
        try:
            with self._lock:
                row = self._conn.execute("SELECT etype FROM issued WHERE token = ?", (token,)).fetchone()
            if row:
                self._mem[token] = row[0]
                return row[0]
        except Exception as exc:  # noqa: BLE001
            _log.debug("reveal registry lookup failed (%s)", type(exc).__name__)
        return None


class Protector:
    """The user-side PII protector, Protegrity-backed. `protect()` detects (Discovery) and tokenizes
    (Data Protection) every PII span. Construction requires both services; it raises
    ProtectionUnavailable otherwise so the app fails closed rather than running unprotected."""

    def __init__(self) -> None:
        # Always Protegrity. Reported for /health; there is no other backend.
        self.backend = "protegrity"
        self.detector = "discovery"
        # Deterministic tokenization means the same (type, value) always yields the same token, so a
        # process-local cache is safe and removes repeat round-trips for values that recur across a
        # conversation (a name, a doctor). Keyed (etype, value) since the element depends on the type.
        self._cache: dict[tuple[str, str], str] = {}
        # ISSUED-TOKEN ALLOWLIST (persisted, NO cleartext): the set of tokens this system minted, each
        # mapped to its data element. reveal() unprotects a token ONLY if it is in this allowlist, so
        # a support caller cannot submit a GUESSED token to fish for another user's PII (the
        # unprotect-oracle stays closed), and because it is persisted, reveal survives restarts and
        # works from a separate support process. No original values are stored, cleartext is recovered
        # on demand via Protegrity unprotect.
        self._registry = TokenRegistry()
        try:
            from app.protect.discovery import DiscoveryClient
            self._discovery = DiscoveryClient()
        except Exception as exc:  # noqa: BLE001
            raise ProtectionUnavailable(
                f"Protegrity Discovery unavailable ({type(exc).__name__}); PII detection cannot run. "
                "Set PROTEGRITY_DISCOVERY_URL to the local Discovery service.") from exc
        try:
            self._pty = _ProtegrityClient()
        except Exception as exc:  # noqa: BLE001
            raise ProtectionUnavailable(
                f"Protegrity tokenization unavailable ({type(exc).__name__}); check DEV_EDITION_* "
                "credentials and the hosted Data Protection service.") from exc

    def protect(self, text: str) -> ProtectResult:
        """Detect (Discovery) and tokenize (Data Protection) every PII span in `text`, in place.
        Fails closed: a Discovery failure or an un-protectable value raises ProtectionUnavailable
        rather than letting cleartext through."""
        mapping: dict[str, str] = {}
        types: dict[str, int] = {}
        spans: list[tuple[int, int, str, str]] = []  # (start, end, type, token)
        claimed: list[tuple[int, int]] = []

        # 1) Collect the non-overlapping detected spans (position matters for the splice).
        detected: list[tuple[int, int, str, str]] = []   # (start, end, type, value)
        for s, e, etype, value in self._detect_spans(text):
            if any(not (e <= cs or s >= ce) for cs, ce in claimed):
                continue
            detected.append((s, e, etype, value))
            claimed.append((s, e))

        # 2) Tokenize by BATCH, one hosted round-trip per data element instead of one per span, and
        # dedup within the message (a repeated "Dr. Smith" costs one protect, not two). Tokenization
        # is deterministic, so the same value under the same element always maps to the same token.
        tokens = self._tokenize_spans(detected)   # {(etype, value): token}, or raises fail-closed

        # 3) Build the mapping/types and the token list for splicing.
        for s, e, etype, value in detected:
            token = tokens[(etype, value)]
            spans.append((s, e, etype, token))
            mapping[token] = value
            # Record token -> element in the PERSISTED allowlist (no cleartext) so reveal can later
            # prove we issued it and unprotect it, even after a restart. Redaction tokens are not
            # reversible, so they are not registered (reveal marks them explicitly).
            if not token.startswith(_REDACTION_PREFIX):
                self._registry.add(token, etype)
            types[etype] = types.get(etype, 0) + 1

        # Splice tokens back by descending offset so earlier indices stay valid.
        out = text
        for s, e, etype, token in sorted(spans, key=lambda x: x[0], reverse=True):
            out = out[:s] + f"[{etype}]{token}[/{etype}]" + out[e:]
        return ProtectResult(protected=out, entities=len(spans), types=types, mapping=mapping)

    def _detect_spans(self, text: str) -> list[tuple[int, int, str, str]]:
        """Detect PII spans as (start, end, type, value) via Protegrity Discovery. A Discovery
        failure is fatal (raised as ProtectionUnavailable): undetected PII is unprotected PII."""
        from app.protect.discovery import DiscoveryError
        try:
            found = self._discovery.discover(text)
        except DiscoveryError as exc:
            raise ProtectionUnavailable(f"Discovery failed: {exc}") from exc
        return [(d["start"], d["end"], d["type"], d["value"]) for d in found]

    def _tokenize_spans(self, detected: list[tuple[int, int, str, str]]) -> dict[tuple[str, str], str]:
        """Tokenize the distinct (type, value) pairs in `detected`, batched by data element and served
        from a deterministic process cache. Returns {(etype, value): token}.

        Robustness (all measured against the live API):
          * Values are normalised per element first (a `ccn` rejects separators; strip them).
          * The fast path is one batched protect() per element. If a batch RAISES (one bad value fails
            the whole batch, verified), we fall back to per-value protection so one problematic value
            can't refuse the entire turn.
          * Per value: try its mapped element; on failure retry under `string` (accepts any value);
            if THAT still fails, or the API returns the input unchanged (a no-op), REDACT the value
            (a fixed surrogate) rather than emit cleartext. We never leak, and one odd value never
            fails the whole message closed.
        """
        want: dict[tuple[str, str], None] = {}
        result: dict[tuple[str, str], str] = {}
        for _s, _e, etype, value in detected:
            key = (etype, value)
            if key in self._cache:
                result[key] = self._cache[key]
            else:
                want[key] = None

        by_element: dict[str, list[tuple[str, str]]] = {}
        for etype, value in want:
            by_element.setdefault(_ELEMENT.get(etype, "string"), []).append((etype, value))

        for element, pairs in by_element.items():
            norm = [_normalise_for_element(v, element) for (_t, v) in pairs]
            try:
                tokens = self._pty.protect_values(norm, element)
                ok = all(tok and tok != nv for tok, nv in zip(tokens, norm))
                if not ok:
                    raise _NoopInBatch()                      # force the per-value path to redact
                for (etype, value), token in zip(pairs, tokens):
                    self._cache[(etype, value)] = token
                    result[(etype, value)] = token
            except Exception:  # noqa: BLE001, batch failed -> recover value by value, never fail-all
                for (etype, value), nv in zip(pairs, norm):
                    token = self._protect_one(etype, element, nv, value)
                    self._cache[(etype, value)] = token
                    result[(etype, value)] = token
        return result

    def _protect_one(self, etype: str, element: str, norm_value: str, orig_value: str) -> str:
        """Protect a single value, degrading safely: mapped element -> `string` -> REDACT. Never
        returns the input unchanged (a no-op) and never raises, so one un-protectable value cannot
        fail the whole turn closed. Redaction loses utility but never emits cleartext."""
        for el in (element, "string") if element != "string" else ("string",):
            try:
                tok = self._pty.protect_values([norm_value], el)[0]
                if tok and tok != norm_value:
                    return tok
            except Exception:  # noqa: BLE001, try the next element, then redact
                continue
        # Could not protect this value under any element; redact deterministically (per type+value),
        # so it is stable within a conversation but reveals nothing.
        _log.warning("could not tokenize a %s value; redacting (no cleartext emitted)", etype)
        return _redaction_token(etype, orig_value)

    def reveal(self, token: str, etype: str | None = None) -> str:
        """Re-identify one token (role-gated caller only). Unprotects ONLY tokens that are in the
        persisted ISSUED allowlist, so a guessed token is never decrypted (the unprotect-oracle stays
        closed) and reveal survives restarts. A token we issued is unprotected on demand via Protegrity
        under the element it was issued under; a redaction token is marked; anything else is unknown.
        `etype` is ignored, the element comes from the allowlist we recorded at issue time."""
        if token.startswith(_REDACTION_PREFIX):
            return "[unrevealable: value was redacted at ingress]"
        issued_etype = self._registry.etype_of(token)
        if issued_etype is None:
            # Not a token we minted: refuse to unprotect caller-supplied input.
            return "[unknown token]"
        try:
            return self._pty.unprotect(issued_etype, token)
        except Exception as exc:  # noqa: BLE001, never surface internals; never guess
            _log.warning("reveal unprotect failed for a %s token: %s",
                         issued_etype, type(exc).__name__)
            return "[unrevealable]"

    def reveal_text(self, protected: str) -> str:
        """Re-identify EVERY [TYPE]token[/TYPE] span in a protected string. For a role-gated support
        caller ONLY, this restores a stored (tokenized) transcript we PRODUCED: each token is checked
        against the persisted issued-allowlist and unprotected on demand, it does NOT decrypt arbitrary
        input. A token we did not issue becomes "[unknown token]", never guessed. The wrappers are
        removed so the output reads naturally.
        """
        # Bound the token body: our tokens are short (format-preserving surrogates); a huge inner run
        # would only be junk, and this keeps the scan cheap and avoids pathological backtracking.
        wrapped = re.compile(rf"\[({_TYPE_ALT})\]([^\[\]]{{1,128}}?)\[/\1\]")

        # Cap how many token spans one request may re-identify. A real transcript has a handful; a
        # request packed with hundreds of (likely guessed) tokens is abuse, capping it bounds the work
        # per request (each miss is a cheap indexed lookup, but this stops amplification outright).
        seen = [0]
        _MAX = 256

        def _sub(m: "re.Match") -> str:
            seen[0] += 1
            if seen[0] > _MAX:
                return m.group(0)                   # leave excess wrappers untouched; don't process them
            etype, token = m.group(1), m.group(2)
            return self.reveal(token, etype)

        return wrapped.sub(_sub, protected)

    @staticmethod
    def strip_tags(text: str) -> str:
        """Bare tokens for the model: the [TYPE] wrappers are internal plumbing the LLM never needs.
        Strips only OUR known PII wrappers, so a legitimate bracketed acronym ("[FDA]") is left
        intact."""
        return WRAPPER_RE.sub("", text)


# Our PII wrapper TYPE -> Protegrity Data Protection element. Mirrors the SDK's DATA_ELEMENT_MAPPING
# and the Protometer integration, with the ONE deliberate divergence measured against the live API:
# EMAIL maps to `string`, NOT `email`. The `email` element tokenizes only the local part and returns
# the domain VERBATIM ("jane@clinic.com" -> "VUOQhl23@clinic.com"), leaking the domain (and, under
# determinism, a shared local part); `string` tokenizes the whole value and still returns an
# email-shaped result. The rest follow the vendor mapping, VERIFIED live: ccn/ssn/phone/datetime/
# number/passport/nin/address all round-trip. (DOCTOR is a clinician NAME -> string.)
_ELEMENT = {
    "EMAIL": "string", "PHONE": "phone", "SSN": "ssn", "DOCTOR": "string", "NAME": "string",
    "ZIP": "address", "ADDRESS": "address", "CREDITCARD": "ccn", "DOB": "datetime",
    "PASSPORT": "passport", "DL": "number", "MEMBERID": "number", "AGE": "number",
}

# Elements that REJECT values carrying their conventional separators (measured live: `ccn` accepts
# "4111111111111111" but rejects "4111 1111 1111 1111" / "4111-1111-1111-1111" with error 44). We
# strip these characters before protecting; the surrounding text keeps its original shape because we
# only tokenize the span's value, and the token is spliced back in place. Other elements (ssn, phone,
# datetime-ISO) accept their natural separators, so they are not listed.
_SEPARATOR_SENSITIVE = {"ccn": " -"}


def _normalise_for_element(value: str, element: str) -> str:
    """Strip separators an element rejects, leaving other elements' values untouched."""
    seps = _SEPARATOR_SENSITIVE.get(element)
    return "".join(ch for ch in value if ch not in seps) if seps else value


class _ProtegrityClient:
    """Adapter over Developer-Edition Data Protection via the `appython` SDK, the correct hosted
    tokenization service (NOT the anonymization/k-anon service an earlier HTTP stub targeted).

    Constructing this performs a login (the SDK's `Protector()` + `create_session`), so a missing SDK
    or bad credentials RAISES here. One session is opened and reused across calls; opening a session
    is a login, and `/auth/login` is rate-limited independently of `/protect`, so we never open one
    per value.
    """

    def __init__(self) -> None:
        # Keep the SDK's own retry layer explicit and modest; a second retry layer stacking on the
        # default 3x30s produced 210s calls in the Protometer measurements.
        os.environ.setdefault("PTY_MAX_RETRIES", "2")
        os.environ.setdefault("PTY_REQUEST_TIMEOUT", "30")
        from appython import Protector as _PtyProtector  # raises if the SDK is absent
        self._policy_user = os.getenv("PROTEGRITY_POLICY_USER", "superuser")
        self._session_minutes = int(os.getenv("PROTEGRITY_SESSION_MINUTES", "30"))
        self._protector = _PtyProtector()
        self._session = self._protector.create_session(self._policy_user,
                                                        timeout=self._session_minutes)

    def _ensure_session(self):
        if self._session is None:
            self._session = self._protector.create_session(self._policy_user,
                                                           timeout=self._session_minutes)
        return self._session

    def protect(self, etype: str, value: str) -> str:
        """Tokenize one value under the mapped data element. Returns the token; a no-op (input
        returned unchanged) is handled by the caller as a hard failure."""
        el = _ELEMENT.get(etype, "string")
        try:
            token = self._ensure_session().protect(value, el)
        except Exception as exc:  # noqa: BLE001, SDK exception types aren't public
            msg = str(exc).lower()
            # An expired session is recoverable: drop it and retry once with a fresh one.
            if "session is invalid" in msg or "timed out" in msg:
                self._session = None
                token = self._ensure_session().protect(value, el)
            else:
                raise
        return token

    # Batch success code for protect, from the SDK's RETURN_CODE constant.
    _PROTECT_OK = 6

    def protect_values(self, values: list[str], element: str) -> list[str]:
        """Tokenize many values under one data element in ONE round-trip. The SDK is polymorphic:
        a list in returns (tokens, codes). Order is preserved so the caller can zip results back.
        Retries once on an expired session; raises otherwise (the caller fails closed)."""
        if not values:
            return []

        def _call() -> list[str]:
            result = self._ensure_session().protect(values, element)
            # Polymorphic: list-in -> (tokens, codes); be tolerant if a build returns just a list.
            if isinstance(result, tuple):
                tokens, codes = result
                for code in codes or []:
                    if code != self._PROTECT_OK:
                        raise RuntimeError(f"protect failed for element={element!r} code={code}")
            else:
                tokens = result
            return list(tokens)

        try:
            return _call()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "session is invalid" in msg or "timed out" in msg:
                self._session = None
                return _call()
            raise

    def unprotect(self, etype: str, token: str) -> str:
        """Re-identify one token under the mapped data element (role-gated callers only)."""
        el = _ELEMENT.get(etype, "string")
        try:
            return self._ensure_session().unprotect(token, el)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "session is invalid" in msg or "timed out" in msg:
                self._session = None
                return self._ensure_session().unprotect(token, el)
            raise
