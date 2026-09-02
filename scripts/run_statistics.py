"""Confidence intervals, paired tests, and power for the curve.

    python scripts/run_statistics.py                     # default model directory
    python scripts/run_statistics.py qwen2.5-14b         # any other
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from protometer.eval.statistics import (
    adjust_family,
    bootstrap_mean,
    compare_scopes,
    required_sample_size,
)
from protometer.scopes import CURVE_ORDER


def main(argv: list[str]) -> int:
    # `--help` is the first thing anyone types. Treating it as a scope name produced
    # `KeyError: "Unknown scope '--help'"`, which reads as a broken script.
    if any(a in ("-h", "--help") for a in argv[1:]):
        print(__doc__)
        return 0

    # Model directory is an argument, not a constant. Hardcoding `bedrock-sonnet-5` meant
    # anyone evaluating a local model got three empty section headers and then
    # `KeyError: 'none'`, and since this is the only script that computes confidence
    # intervals and McNemar tests, its output never reached `docs/results-aml.md` at all.
    model = argv[1] if len(argv) > 1 else "bedrock-sonnet-5"
    d = ROOT/"data"/"eval"/model
    if not d.is_dir():
        available = sorted(p.name for p in (ROOT/"data"/"eval").iterdir() if p.is_dir())
        sys.exit(f"No results for model {model!r}. Available: {', '.join(available) or 'none'}")

    res = {}
    for f in d.glob("*.json"):
        if f.name == "tasks.json": continue
        j = json.loads(f.read_text())
        if j.get("llm_stats",{}).get("billed_calls",0) or j.get("llm_stats",{}).get("cache_hits",0):
            res[j["scope"]] = j
    # An empty result set is a missing run, not a null finding. Falling through printed
    # section headers with no rows before dying on a KeyError several lines later.
    if "none" not in res:
        sys.exit(
            f"No usable results in {d} (found {len(res)} scope(s), baseline 'none' absent).\n"
            f"Run: python scripts/run_eval.py --model {model}"
        )
    order = [s for s in CURVE_ORDER if s in res]

    print("MEAN SCORE WITH 95% CI (bootstrap over tasks, respecting clustering)\n")
    print(f"{'scope':<28}{'mean [95% CI]':<28}")
    for s in order:
        print(f"{s:<28}{str(bootstrap_mean(res[s])):<28}")

    # p is a paired bootstrap test on the SAME score-delta the CI is built from (so CI and p
    # agree); McNemar on binary pass/fail is reported beside it as context, not as the verdict.
    # The family of six comparisons is Holm-Bonferroni corrected, and the verdict is taken on
    # the adjusted p (`p*`), so a "different" claim survives the multiplicity of the table.
    comparisons = [
        compare_scopes(res["none"], res[s], f"none vs {s}")
        for s in order if s != "none"
    ]
    comparisons = adjust_family(comparisons)
    print("\n\nPAIRED COMPARISONS vs BASELINE (bootstrap on score deltas; "
          "Holm-corrected p*; McNemar for context)\n")
    print(f"{'comparison':<34}{'diff [95% CI]':<26}{'p':>8}{'p*':>8}{'McN p':>8}"
          f"{'MDE':>8}  verdict")
    for c in comparisons:
        adj = c.adjusted_p_value if c.adjusted_p_value is not None else c.p_value
        print(f"{c.label:<34}{str(c.difference):<26}{c.p_value:>8.3f}{adj:>8.3f}"
              f"{c.mcnemar_p_value:>8.3f}{c.detectable_effect:>8.3f}  {c.verdict}")

    print("\n\nPOWER: checkpoints needed to detect an effect at 80% power")
    for e in (0.03, 0.05, 0.10, 0.20, 0.30):
        print(f"  {e:.0%} difference -> {required_sample_size(e):>5} checkpoints")
    n = sum(len(t['checkpoints']) for t in res['none']['tasks'])
    print(f"\n  this evaluation has {n} checkpoints per scope")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
