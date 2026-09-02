"""Trace-redaction seam: clear PII must not reach Langfuse's at-rest store.

At scope none/partial the prompt and completion carry real identifiers; Langfuse persists
bodies at rest, so a run installs the corpus's forbidden values and record_generation scrubs
them before export. These tests drive record_generation with a fake client and assert the
bodies that would leave are scrubbed, and that redaction is off by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer import observability as obs


class _FakeObs:
    def __init__(self, sink):
        self._sink = sink

    def update(self, **kwargs):
        self._sink.update(kwargs)

    def end(self):
        pass


class _FakeClient:
    def __init__(self):
        self.captured = {}

    def start_observation(self, **kwargs):
        self.captured.update(kwargs)
        return _FakeObs(self.captured)


def _record(monkeypatch, **over):
    fake = _FakeClient()
    monkeypatch.setattr(obs, "_get_client", lambda project=None: fake)
    obs.record_generation(
        component="eval", model="m",
        system="grade this", prompt="Leila Rahman moved 712000.00",
        completion="Reviewed Leila Rahman", input_tokens=1, output_tokens=1,
        cost_usd=0.0, latency_s=0.0, cached=False, **over,
    )
    return fake.captured


def test_installed_values_are_scrubbed_from_bodies(monkeypatch):
    obs.set_trace_redaction({"Leila Rahman", "712000.00"})
    try:
        cap = _record(monkeypatch)
        blob = str(cap.get("input", "")) + str(cap.get("output", ""))
        assert "Leila Rahman" not in blob
        assert "712000.00" not in blob
        assert "[REDACTED-PII]" in blob
    finally:
        obs.reset_trace_redaction()


def test_redaction_catches_case_and_homoglyph_variants(monkeypatch):
    """The trace redactor must be at least as strong as the egress leak-check, which normalizes
    (NFKC, casefold, ignorable-stripped). A case variant / fullwidth homoglyph of a forbidden
    value must not slip into Langfuse's at-rest store just because it differs from the stored
    casing."""
    obs.set_trace_redaction({"Leila Rahman"})
    try:
        # lowercase variant
        assert "leila rahman" not in obs._redact("reviewed leila rahman").lower() \
            or "[REDACTED-PII]" in obs._redact("reviewed leila rahman")
        assert "[REDACTED-PII]" in obs._redact("reviewed leila rahman")
        # NFKC-foldable fullwidth variant of the same name
        fullwidth = "Ｌｅｉｌａ　Ｒａｈｍａｎ"
        assert "[REDACTED-PII]" in obs._redact(f"note about {fullwidth} here")
        # a clean body is returned unchanged (not needlessly canonicalized)
        assert obs._redact("nothing sensitive here") == "nothing sensitive here"
    finally:
        obs.reset_trace_redaction()


def test_redaction_is_off_by_default(monkeypatch):
    obs.reset_trace_redaction()
    cap = _record(monkeypatch)
    assert "Leila Rahman" in str(cap.get("input", ""))
