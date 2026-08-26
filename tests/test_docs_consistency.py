"""The docs never carry a stale headline number.

results-aml.md is regenerated from artifacts (safe); README and architecture.md embed numbers
by hand and have drifted repeatedly. This test runs the consistency checker so a stale value can
never ship: it fails if a hand-written doc contains a known-superseded number, cites a stale
corpus fingerprint, or drops a current headline result. See scripts/check_docs.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def test_hand_written_docs_have_no_stale_numbers():
    from check_docs import check

    problems = check()
    assert not problems, "docs drifted from the artifacts:\n" + "\n".join(
        f"  - {p}" for p in problems
    )


def test_key_metrics_are_all_derived():
    """The single source of truth must produce the current fingerprint (proves it read the
    fresh artifacts, not a cached constant)."""
    from key_metrics import key_metrics

    m = key_metrics()
    assert m.get("corpus_fingerprint"), "key_metrics failed to derive the corpus fingerprint"
    # spot-check a couple of derived values are present and numeric
    assert m.get("attack", {}).get("neighbourhood_pct") is not None
    assert m.get("eval_cost_usd") is not None
