"""The prompt store seam: file is the durable source, Langfuse the editing surface.

Guards the invariants that make "prompts out of code" safe: the pipeline resolves prompts
with all telemetry off (file is the floor), a missing prompt file is a loud error not a
silent empty string, path traversal is rejected, and save_prompt only writes on a real change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard import prompts


def test_all_three_pipeline_prompts_ship_on_disk():
    for name in ("amlguard-investigation-system", "amlguard-rationale-system",
                 "amlguard-judge-system"):
        text = prompts.load_prompt(name)
        assert text and "money-laundering" in text.lower()


def test_missing_prompt_is_a_loud_error():
    with pytest.raises(FileNotFoundError):
        prompts.load_prompt("amlguard-does-not-exist")


def test_path_traversal_is_rejected():
    for bad in ("../secrets", "a/b", "..\\x", ".hidden"):
        with pytest.raises(ValueError):
            prompts.load_prompt(bad)


def test_save_prompt_writes_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "PROMPT_DIR", tmp_path)
    assert prompts.save_prompt("x-test", "hello world") is True   # first write
    assert prompts.load_prompt("x-test") == "hello world"
    assert prompts.save_prompt("x-test", "hello world") is False  # no-op on identical
    assert prompts.save_prompt("x-test", "changed") is True       # write on change
    assert prompts.load_prompt("x-test") == "changed"


def test_managed_prompt_falls_back_to_file_when_tracing_off(monkeypatch):
    monkeypatch.setenv("AMLGUARD_NO_TRACING", "1")
    # force a fresh client decision
    from amlguard import observability
    monkeypatch.setattr(observability, "_initialised", False)
    monkeypatch.setattr(observability, "_client", None)
    text = observability.managed_prompt("amlguard-judge-system")
    assert text == prompts.load_prompt("amlguard-judge-system")
