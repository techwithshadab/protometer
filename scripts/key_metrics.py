"""Single source of truth for every headline number the docs cite.

The generated results doc (`results-aml.md`) is safe because it is regenerated from artifacts.
The hand-written docs (README, architecture) embed numbers by hand and drift. This module
extracts the canonical headline metrics from the committed artifacts into one dict, so a
consistency checker (`scripts/check_docs.py`) can verify the hand-written docs against it, and
so there is exactly one place the "current numbers" are defined.

    python scripts/key_metrics.py            # prints the metrics as JSON
    python scripts/key_metrics.py --write    # writes data/eval/key_metrics.json

Every value is derived, never typed. Absent artifacts yield absent keys (never a stale default).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

EVAL = ROOT / "data" / "eval"


def _load(rel: str):
    p = EVAL / rel
    return json.loads(p.read_text()) if p.exists() else None


def key_metrics() -> dict:
    """Every headline number, derived from the committed artifacts."""
    m: dict = {}

    # corpus fingerprint (ties the whole set to one run)
    try:
        from protometer.tracking import corpus_source_fingerprint
        m["corpus_fingerprint"] = corpus_source_fingerprint(ROOT / "data" / "corpus")
    except Exception:  # noqa: BLE001
        pass

    training = _load("training.json")
    if training:
        by = {r["scope"]: r for r in training}
        base = by.get("none", {}).get("average_precision")
        m["training"] = {
            # `is not None`, not truthiness: an AP of exactly 0.0 is a real (if degenerate)
            # measurement, not a missing value, and must not be silently dropped to None.
            "none_ap": round(base, 3) if base is not None else None,
            "monetary_ap": round(by["direct-plus-monetary"]["average_precision"], 3)
            if "direct-plus-monetary" in by else None,
            # The division keeps a truthiness guard on purpose: `base` here is also the
            # denominator, so 0.0 must fall through to None (not divide by zero).
            "monetary_retained_pct": round(
                by["direct-plus-monetary"]["average_precision"] / base * 100)
            if base and "direct-plus-monetary" in by else None,
            "all_ap": round(by["all"]["average_precision"], 3) if "all" in by else None,
        }

    for scope in ("none", "quasi"):
        h = _load(f"hybrid_{scope}.json")
        if h:
            m.setdefault("hybrid", {})[scope] = {
                "p_at_50": h.get("precision_at_capacity"),
                "queue": h.get("queue_length"),
                "distinct": h.get("distinct_subjects_in_head"),
                "cost_usd": round(h.get("llm_cost_usd", 0), 2),
            }

    attacks = _load("attacks.json")
    if attacks:
        nl = next((at for sc, v in attacks.items() if isinstance(v, list)
                   for at in v if "neighbour" in at.get("attack", "").lower()), None)
        if nl:
            m["attack"] = {
                "neighbourhood_pct": round(nl["accuracy"] * 100, 1),
                "control_pct": round((nl.get("control_accuracy") or 0) * 100, 2),
            }

    erasure = _load("semantic_erasure.json")
    if erasure and "none" in erasure and "direct" in erasure:
        m["erasure"] = {
            "identity_none": erasure["none"].get("identity_found"),
            "identity_direct": erasure["direct"].get("identity_found"),
            "behavioural_none": erasure["none"].get("behavioural_found"),
            "behavioural_direct": erasure["direct"].get("behavioural_found"),
            "fisher_p": erasure["direct"].get("identity_fisher_p_vs_baseline"),
        }

    # eval curve cost (sum of billed scope costs)
    curve_dir = EVAL / "bedrock-sonnet-5"
    if curve_dir.is_dir():
        cost = 0.0
        for f in curve_dir.glob("*.json"):
            if f.name in ("tasks.json", "comparison.json"):
                continue
            j = json.loads(f.read_text())
            cost += j.get("llm_stats", {}).get("total_cost_usd", 0) or 0
        if cost:
            m["eval_cost_usd"] = round(cost, 2)

    frontier = _load("protection_methods.json")
    if frontier:
        _tstr = frontier.get("synthetic_ledger", {}).get("tstr", {})
        m["frontier"] = {
            "marketer_risk": frontier.get("metadata_risk", {}).get("marketer_risk"),
            "kanon_ap": frontier.get("generalized_amounts", {}).get("average_precision"),
            "tstr_retained": _tstr.get("utility_retained"),
            # Whether the retention ratio's REFERENCE task cleared chance. None when the artifact
            # predates the guard (older run) -> callers must treat as "not above chance" so the
            # caveat is never dropped. See compare_protection_methods._tstr.
            "tstr_reference_above_chance": _tstr.get("reference_above_chance"),
            "tstr_trtr": _tstr.get("train_real_test_real"),
            "tstr_chance": _tstr.get("chance_macro_f1"),
            "dp_available": frontier.get("differential_privacy", {}).get("available"),
        }

    return m


def main() -> int:
    m = key_metrics()
    if "--write" in sys.argv:
        from protometer.persist import atomic_write_json

        dest = EVAL / "key_metrics.json"
        atomic_write_json(dest, m)
        print(f"written to {dest.relative_to(ROOT)}")
    else:
        print(json.dumps(m, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
