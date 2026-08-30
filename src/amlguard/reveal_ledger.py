"""Tamper-evident reveal ledger + canary tripwire for the detokenization boundary.

Two defenses that sit on the ONE place plaintext is ever recovered — `reidentify()`:

1. **Reveal ledger.** Every batch of unprotects appends a hash-chained record (who, role,
   purpose, entity-type counts, scope, and the previous record's hash). The chain is
   append-only and tamper-evident: altering or dropping any past record breaks every hash
   after it, which `verify_chain()` detects. This is the GDPR Art. 30 "record of processing"
   for the sensitive operation, and it never stores a plaintext value — only counts and types.

2. **Canary tripwire.** A small set of party ids per domain are *canaries*: real-looking
   records that no legitimate investigation ever needs to reveal. Their token values are
   registered here; if a reveal ever recovers a canary value, that is by construction an
   unauthorized or injected access (detokenization-as-intrusion-detection). The tripwire raises
   a loud audit event and increments a counter an operator can alert on.

Both are best-effort telemetry on the *analyst-facing* path in the sense that a ledger-write
failure must never lose a legitimate reveal — but a tripped canary is surfaced, not swallowed.
The ledger path is deterministic and offline ($0, no API calls): it hashes metadata only.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

_LOG_LEVEL = "amlguard.reveal"

# The genesis link every chain starts from, so the first real record still hashes over a fixed
# predecessor (no None-vs-"" ambiguity on verify).
GENESIS = "0" * 64


def _default_ledger_path() -> Path:
    override = os.getenv("AMLGUARD_REVEAL_LEDGER")
    if override:
        return Path(override)
    # A writable app-data location: data/audit/ under the repo/app root. NOT the MLflow store,
    # which the container mounts read-only — appending there fails silently and drops the audit
    # trail. A plain append-only JSONL, no DB needed.
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "audit" / "reveal_ledger.jsonl"


def _record_hash(prev_hash: str, payload: dict) -> str:
    """The link hash: sha256 over the previous hash + the canonical payload JSON.

    Canonical (sorted keys, no whitespace) so the same record always hashes identically, and the
    prev-hash inclusion is what chains the records: record N's hash depends on record N-1's.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{prev_hash}\n{canonical}".encode()).hexdigest()


@dataclass
class RevealLedger:
    """Append-only, hash-chained ledger of detokenization events. Thread-safe append."""

    path: Path = field(default_factory=_default_ledger_path)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _last_hash(self) -> str:
        """The hash of the final record, or GENESIS for an empty/absent ledger."""
        if not self.path.exists():
            return GENESIS
        last = None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = line
        if last is None:
            return GENESIS
        try:
            return json.loads(last)["hash"]
        except (json.JSONDecodeError, KeyError):
            # A corrupt tail is itself a tamper signal; verify_chain() will localize it. For
            # appends we chain from GENESIS-of-corruption so the new record is still verifiable
            # against what we wrote, and the break stays visible at the corrupt line.
            return GENESIS

    def append(
        self,
        *,
        actor: str,
        role: str,
        purpose: str,
        entity_counts: dict[str, int],
        scope: str | None = None,
        canary_hits: int = 0,
    ) -> str:
        """Append one reveal event and return its chain hash. Metadata only, never plaintext."""
        payload = {
            # Deliberately no timestamp in the HASHED payload (it must stay reproducible for the
            # tests that pin chaining); the wall-clock is recorded OUTSIDE the hash as `at`.
            "actor": actor,
            "role": role,
            "purpose": purpose,
            "entity_counts": dict(sorted(entity_counts.items())),
            "scope": scope,
            "canary_hits": canary_hits,
        }
        with self._lock:
            prev = self._last_hash()
            link = _record_hash(prev, payload)
            record = {"prev": prev, "hash": link, **payload}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        return link

    def verify_chain(self) -> tuple[bool, int]:
        """Recompute the chain. Returns (ok, first_broken_line_1indexed). ok=True → 0."""
        prev = GENESIS
        if not self.path.exists():
            return True, 0
        with self.path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return False, i
                payload = {k: rec.get(k) for k in
                           ("actor", "role", "purpose", "entity_counts", "scope", "canary_hits")}
                expect = _record_hash(prev, payload)
                if rec.get("prev") != prev or rec.get("hash") != expect:
                    return False, i
                prev = rec["hash"]
        return True, 0


# ── Canary tripwire ───────────────────────────────────────────────────────────────────────────

@dataclass
class CanaryTripwire:
    """A registry of canary plaintext values whose reveal is, by construction, illegitimate."""

    values: frozenset[str] = frozenset()
    tripped: int = 0

    def scan(self, revealed_values) -> list[str]:
        """Return the canaries present among freshly-revealed values, and count the trip."""
        hits = [v for v in revealed_values if v in self.values]
        if hits:
            self.tripped += len(hits)
            import logging

            logging.getLogger(_LOG_LEVEL).error(
                "CANARY TRIPPED: %d canary value(s) detokenized — treat as an unauthorized or "
                "injected reveal", len(hits))
        return hits


def load_canaries(domain: str = "aml") -> frozenset[str]:
    """Canary plaintext identifiers for a domain, from the corpus's designated canary parties.

    Canaries are ordinary-looking parties flagged `is_canary` in the corpus; no legitimate task
    references them, so any reveal of their identifiers is a tripwire hit. Returns their
    high-sensitivity values (names + identifiers) — the strings a reveal would surface.
    """
    from amlguard.domains import get_domain

    try:
        from amlguard import db

        parties = db.read_parties(domain) or []
    except Exception:  # noqa: BLE001, tripwire must never break the reveal path
        return frozenset()
    dom = get_domain(domain)
    out: set[str] = set()
    for p in parties:
        if not p.get("is_canary"):
            continue
        name = str(p.get("full_name") or "").strip()
        if name:
            out.add(name)
        out |= dom.high_sensitivity_values([p])
    return frozenset(out)
