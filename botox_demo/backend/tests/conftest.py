"""Test fixtures: run the Protegrity-only protector offline.

Protection is now mandatory Protegrity (no mock, no regex). The real `Protector()` reaches out to
Discovery + the appython SDK on construction, which no test should do. This fixture replaces those
two collaborators with deterministic in-memory stubs, so the protection BOUNDARY (detect -> tokenize
-> splice -> strip, fail-closed) is exercised exactly as in production, without any vendor service.

The stubs are intentionally faithful to the contracts the code depends on:
  * Discovery returns non-overlapping (start, end, type, value) spans for a small set of test PII.
  * The tokenizer is deterministic (same value -> same token) and never returns the input unchanged,
    so the no-op guard is not tripped by the stub.
Tests that want a Discovery failure or a no-op can override the stubs on the instance.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.protect import protector as _protector_mod  # noqa: E402


# Minimal, deterministic detection for tests: the same entity types the real Discovery maps to.
_TEST_PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("DOCTOR", re.compile(r"\b(?:Dr\.?|Doctor)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")),
    ("NAME", re.compile(r"(?i:\bmy name is|\bi am|\bi'm|\bthis is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")),
]


class _StubDiscovery:
    """Deterministic in-memory Discovery: returns non-overlapping PII spans for the test patterns."""

    def discover(self, text: str) -> list[dict]:
        spans: list[dict] = []
        for etype, pat in _TEST_PATTERNS:
            for m in pat.finditer(text):
                gi = 1 if (m.re.groups and m.group(1) is not None) else 0
                s, e = (m.start(gi), m.end(gi)) if gi else (m.start(), m.end())
                spans.append({"type": etype, "start": s, "end": e, "value": text[s:e], "score": 0.99})
        spans.sort(key=lambda d: (d["start"], -(d["score"] or 0)))
        out: list[dict] = []
        for d in spans:
            if out and d["start"] < out[-1]["end"]:
                continue
            out.append(d)
        return out


class _StubTokenizer:
    """Deterministic, REVERSIBLE tokenizer stub matching the _ProtegrityClient interface: batched
    protect_values (same value -> same token), and unprotect that reverses it (so reveal round-trips
    like the real deterministic service)."""

    def __init__(self) -> None:
        self._back: dict[str, str] = {}    # token -> original, so unprotect can reverse

    def _token(self, value: str) -> str:
        t = "TK" + hashlib.sha256(value.encode()).hexdigest()[:10]
        self._back[t] = value
        return t

    def protect_values(self, values: list[str], element: str) -> list[str]:
        return [self._token(v) for v in values]

    def unprotect(self, etype: str, token: str) -> str:
        if token not in self._back:
            raise RuntimeError("stub: token not issued")   # mimic an unprotect failure/oracle miss
        return self._back[token]


@pytest.fixture(autouse=True)
def stub_protegrity(monkeypatch, tmp_path):
    """Make every `Protector()` in the suite construct with the in-memory Protegrity stubs, so the
    protection boundary runs offline and fail-closed semantics are testable. The reveal allowlist is
    persisted to a per-test temp file (never the real data dir)."""
    from app.protect.protector import TokenRegistry

    def _init(self):
        self.backend = "protegrity"
        self.detector = "discovery"
        self._cache = {}
        self._registry = TokenRegistry(path=tmp_path / "reveal-registry.json")
        self._discovery = _StubDiscovery()
        self._pty = _StubTokenizer()

    monkeypatch.setattr(_protector_mod.Protector, "__init__", _init)
    # The shared process-wide Protector is cached in a module global; reset it so each test gets a
    # fresh stubbed instance rather than one leaking across tests.
    _protector_mod.reset_protector()
    yield
    _protector_mod.reset_protector()
