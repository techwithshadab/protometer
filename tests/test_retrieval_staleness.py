"""The eval READ path must reject a stale narrative index, not query it silently.

The build path (`index_narratives`) already raises StaleIndexError on a corpus mismatch. The
eval read path constructs a NarrativeIndex and calls `search()` directly, which only guards
against an EMPTY index — so a non-empty index left over from a superseded corpus would be
queried with no error, contaminating retrieved notes. `validate_fingerprint()` closes that gap;
these tests pin it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("chromadb")

from amlguard.retrieval import (  # noqa: E402
    NarrativeIndex,
    StaleIndexError,
    corpus_fingerprint,
)


def test_fingerprint_changes_on_same_length_content_edit():
    """A same-length text swap (what tokenization does) MUST change the fingerprint — a
    length-only digest collided and let a stale index pass the guard."""
    a = [{"document_id": "n1", "text": "Allison Hill"}]
    b = [{"document_id": "n1", "text": "Zxq7be Yk925"}]  # same length, different content
    assert len(a[0]["text"]) == len(b[0]["text"])
    assert corpus_fingerprint(a) != corpus_fingerprint(b)


def test_eval_narrative_fingerprint_matches_index_guard(tmp_path):
    """The fingerprint the eval runner logs to MLflow / the artifact (`index_fingerprint`) MUST
    equal the value the index staleness guard stores, or the tracked number could not be used to
    detect drift. This pins that `_narrative_fingerprint` and the index guard agree."""
    from amlguard.eval.runner import _narrative_fingerprint

    # lay out data/protected/<scope>/narratives.json under a temp protected_root
    scope = "none"
    (tmp_path / scope).mkdir(parents=True)
    import json
    (tmp_path / scope / "narratives.json").write_text(json.dumps(_NARRATIVES))

    logged = _narrative_fingerprint(scope, tmp_path)
    guard = corpus_fingerprint(_NARRATIVES)
    assert logged == guard and logged != ""

    # A missing scope yields "" (not a crash), matching the empty-fingerprint no-op path.
    assert _narrative_fingerprint("absent", tmp_path) == ""

_NARRATIVES = [
    {"document_id": "n1", "text": "cash structuring below the reporting threshold"},
    {"document_id": "n2", "text": "wire to a shell company overseas"},
]


def test_validate_fingerprint_raises_on_corpus_mismatch(tmp_path):
    persist = tmp_path / "idx"
    # Build with fingerprint "AAAA".
    built = NarrativeIndex("none", persist, "AAAA")
    built.index_narratives(_NARRATIVES)
    assert len(built) == 2

    # Re-open the SAME on-disk index but expecting a DIFFERENT corpus fingerprint (as the eval
    # read path would after the corpus was regenerated without rebuilding the index).
    reopened = NarrativeIndex("none", persist, "BBBB")
    with pytest.raises(StaleIndexError):
        reopened.validate_fingerprint()


def test_validate_fingerprint_passes_when_fingerprint_matches(tmp_path):
    persist = tmp_path / "idx"
    NarrativeIndex("none", persist, "AAAA").index_narratives(_NARRATIVES)
    # Same fingerprint -> no raise, and search works.
    idx = NarrativeIndex("none", persist, "AAAA")
    idx.validate_fingerprint()
    hits = idx.search("structuring cash", top_k=1)
    assert hits


def test_validate_fingerprint_is_noop_without_expected_fingerprint(tmp_path):
    persist = tmp_path / "idx"
    NarrativeIndex("none", persist, "AAAA").index_narratives(_NARRATIVES)
    # No expected fingerprint supplied -> cannot validate, must not raise (back-compat).
    NarrativeIndex("none", persist).validate_fingerprint()
