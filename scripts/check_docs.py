"""Fail if a hand-written doc cites a headline number that disagrees with the artifacts.

`results-aml.md` is regenerated from artifacts and is safe by construction. README and
architecture.md embed numbers by hand and drift (this has happened repeatedly). This checker is
the guardrail: it takes the canonical `key_metrics` (the single source of truth, derived from
artifacts) and verifies that where a hand-written doc mentions one of those metrics, it agrees,
and that the doc does NOT contain a KNOWN-STALE value the metric has since moved past.

It is intentionally conservative: it does not try to parse every number (that would false-alarm
on p-values, citations, and dates). It checks two things per metric:
  1. the CURRENT value appears in the doc (presence), and
  2. no value from a curated STALE list appears (absence),
so a number that silently reverts to an old figure is caught.

    python scripts/check_docs.py          # exits non-zero on any drift, prints the diffs

Run in CI / as a test (tests/test_docs_consistency.py) so stale data can never ship.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

# Numbers that were TRUE on a previous corpus and must not reappear as current claims. Add to
# this list whenever a headline number changes, it is the "never regress to this" set.
STALE_VALUES = [
    "0.534", "0.483", "0.492",          # old training APs
    "49.7%", "49.7 ", "49.7%",          # old neighbourhood linkage
    "5.4 × 10", "5.4e-10",              # old Fisher p
    "30/40", "3/40 (7.5%)",             # old erasure identity recall (as prose, not table)
    "25/40", "(63%)",                   # identity recall drifted to 25/40 once; truth is 26/40 (65%)
    "96 invariant", "96 tests",         # old test-count prose (suite grew past it; now 163)
    "$2.77", "$2.72",                   # old eval costs (superseded by fresh-run values)
    "P@50 0.28", "0.28 vs 0.24",        # old hybrid precision
    "0.34 vs 0.36", "34 vs 34",         # older hybrid-queue prose (pre fresh-run 0.48/0.48, 18/19)
    "34/50",                            # old distinct-subjects denominator (queue was 50, now 25)
    "0.776", "0.706-0.741",             # oldest LLM curve means
    "0.785-0.838",                      # LLM band before `all` fell to 0.750 (now 0.750-0.838)
    "AP 0.451", "AP 0.485",             # old frontier APs (now measured 0.4505 / 0.4855)
    "channel L1 0.032",                 # old synthetic L1 (now 0.0354)
    "≈ 80% utility retained",           # old TSTR headline w/o the near-chance caveat
    "80% utility retained",
]

# The hand-written docs the checker guards. (results-aml.md is generated, excluded.)
GUARDED_DOCS = ["README.md", "docs/architecture.md"]


def check() -> list[str]:
    """Return a list of human-readable problems; empty means consistent."""
    from key_metrics import key_metrics  # noqa: E402

    m = key_metrics()
    problems: list[str] = []

    docs = {d: (ROOT / d).read_text() for d in GUARDED_DOCS if (ROOT / d).exists()}

    # 1. No known-stale value may appear anywhere in a guarded doc. Match on NUMBER boundaries,
    #    not raw substring: a stale "0.534" must not be reported inside a current "10.534" or
    #    "0.5341". The boundary is "not preceded/followed by a digit or dot", so the stale token
    #    stands as its own number, not a fragment of a larger, possibly-current one.
    import re as _re
    for name, text in docs.items():
        for stale in STALE_VALUES:
            pattern = r"(?<![\d.])" + _re.escape(stale) + r"(?![\d.])"
            if _re.search(pattern, text):
                problems.append(f"{name}: contains STALE value {stale!r} (a number the "
                                f"current run has moved past)")

    # 2. If a doc stamps a fingerprint-shaped LITERAL (12 hex chars), it must be the current
    #    one. Prose that merely uses the word "fingerprint" (the mechanism) is not flagged, only
    #    an actual stale hash value is.
    import re

    fp = m.get("corpus_fingerprint")
    if fp:
        for name, text in docs.items():
            for lit in set(re.findall(r"\b[0-9a-f]{12}\b", text)):
                if lit != fp:
                    problems.append(f"{name}: contains fingerprint literal {lit!r} != current "
                                    f"{fp!r} (stale run reference)")

    # 3. Key current values should be PRESENT in the README (the doc that must carry results).
    #    Presence-only (formatting varies), so this catches a number silently dropped, not
    #    every phrasing. Percentages are checked as integers to tolerate '%'.
    readme = docs.get("README.md", "")
    want_present = []
    if m.get("attack", {}).get("neighbourhood_pct"):
        want_present.append(("neighbourhood linkage", f"{m['attack']['neighbourhood_pct']}"))
    if m.get("hybrid", {}).get("none", {}).get("p_at_50"):
        want_present.append(("hybrid none P@50", f"{m['hybrid']['none']['p_at_50']}"))
    if m.get("eval_cost_usd"):
        want_present.append(("eval cost", f"{m['eval_cost_usd']}"))
    for label, val in want_present:
        if val not in readme:
            problems.append(f"README.md: current {label} ({val}) is not present, "
                            f"the README must carry the current result")

    # 4. A committed healthcare artifact must carry the HONESTY-GATE fields the current code always
    #    writes. Their absence means the artifact predates the gate: it may internally over-claim
    #    a HIPAA standard its own numbers do not meet. Flag it so a stale, pre-gate artifact can
    #    never ship silently, re-run scripts/healthcare_deidentify.py to refresh it.
    hc_path = ROOT / "data" / "eval" / "healthcare" / "deidentify.json"
    if hc_path.exists():
        import json as _json
        hc = _json.loads(hc_path.read_text())
        ed = hc.get("expert_determination", {})
        sh = hc.get("safe_harbor", {})
        if "expert_determination_met" not in ed:
            problems.append(
                "data/eval/healthcare/deidentify.json: missing 'expert_determination_met' "
                "(artifact predates the Expert-Determination honesty gate; re-run "
                "scripts/healthcare_deidentify.py)")
        if "noop_names_redacted" not in sh:
            problems.append(
                "data/eval/healthcare/deidentify.json: missing 'noop_names_redacted' "
                "(artifact predates the Safe-Harbor no-op guard; re-run "
                "scripts/healthcare_deidentify.py)")

    return problems


def main() -> int:
    problems = check()
    if not problems:
        print("docs consistency: OK (README/architecture agree with the artifacts)")
        return 0
    print("docs consistency: FAILED\n")
    for p in problems:
        print(f"  - {p}")
    print("\nFix the doc, or if the number legitimately changed, update the current value and "
          "move the old one into STALE_VALUES in scripts/check_docs.py.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
