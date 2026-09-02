"""Consolidate evaluation results across models and protection scopes.

Reads every `data/eval/<model>/<scope>.json` produced by `run_eval.py` and renders the
utility-vs-scope curve per model, plus cost and latency.

Running the same scopes on more than one model is what separates two claims a sceptical
reviewer will not let you conflate:

    "protection degrades reasoning"        (the finding)
    "this model degrades on tokenized text" (an artefact)

A consistent shape across independent model families supports the former. A shape that
appears on only one model supports the latter.

    python scripts/compare_models.py
    python scripts/compare_models.py --format markdown > docs/results-aml.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.scopes import CURVE_ORDER  # noqa: E402

EVAL_ROOT = ROOT / "data" / "eval"


def load_results() -> dict[str, dict[str, dict]]:
    """Return {model: {scope: result}} for every evaluation on disk."""
    results: dict[str, dict[str, dict]] = {}
    skipped: list[str] = []

    for scope_file in sorted(EVAL_ROOT.rglob("*.json")):
        if scope_file.name in ("tasks.json", "comparison.json"):
            continue
        if "llm_cache" in scope_file.parts or "judge_cache" in scope_file.parts:
            continue
        try:
            payload = json.loads(scope_file.read_text())
        except json.JSONDecodeError:
            continue
        if "scope" not in payload or "mean_checkpoint_score" not in payload:
            continue

        # Reject runs where the model never actually answered.
        #
        # A scope whose LLM calls all failed still writes a result file, with every checkpoint
        # scored zero, indistinguishable in the table from protection destroying utility, and
        # far more damaging, because it reads as a dramatic finding. Measured case: loading a
        # 19GB model while an ingestion job held memory pushed load time past the request
        # timeout, so four scopes recorded 0.000 with `billed_calls == 0`.
        #
        # Zero billed calls means no measurement took place. Such runs are skipped and
        # reported, never silently dropped and never plotted.
        stats = payload.get("llm_stats") or {}
        if stats.get("billed_calls", 0) == 0 and stats.get("cache_hits", 0) == 0:
            skipped.append(f"{scope_file.parent.name}/{payload['scope']}: no LLM calls completed")
            continue

        # The model recorded inside the result is authoritative; the directory name is only
        # a convenience and can disagree after a fallback.
        model = payload.get("model") or scope_file.parent.name
        results.setdefault(model, {})[payload["scope"]] = payload

    # Surfaced rather than swallowed: a reader must know which scopes are missing from the
    # table and why, otherwise an incomplete curve looks like a complete one.
    if skipped:
        print("Excluded runs (no measurement took place):", file=sys.stderr)
        for note in skipped:
            print(f"  {note}", file=sys.stderr)
        print(file=sys.stderr)

    return results


def format_table(results: dict[str, dict[str, dict]], markdown: bool = False) -> str:
    lines: list[str] = []

    for model, scopes in sorted(results.items()):
        lines.append(f"\n### {model}\n" if markdown else f"\n=== {model} ===")

        strata = sorted({s for r in scopes.values() for s in r.get("stratum_scores", {})})
        header = ["scope", "mean", "verifiable", "completion", *strata, "cost$", "p50s"]

        if markdown:
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join("---" for _ in header) + "|")
        else:
            lines.append(
                f"{'scope':<26}{'mean':>7}{'verif':>8}{'compl':>8}"
                + "".join(f"{s[:9]:>10}" for s in strata)
                + f"{'cost$':>9}{'p50s':>7}"
            )

        for scope in CURVE_ORDER:
            result = scopes.get(scope)
            if not result:
                continue
            stats = result.get("llm_stats", {})
            cells = [
                scope,
                f"{result['mean_checkpoint_score']:.3f}",
                f"{result.get('verifiable_score', 0):.3f}",
                f"{result['task_completion_rate']:.0%}",
                *[f"{result.get('stratum_scores', {}).get(s, 0):.3f}" for s in strata],
                f"{stats.get('total_cost_usd', 0):.4f}",
                f"{stats.get('latency_p50', 0):.1f}",
            ]
            if markdown:
                lines.append("| " + " | ".join(cells) + " |")
            else:
                lines.append(
                    f"{cells[0]:<26}{cells[1]:>7}{cells[2]:>8}{cells[3]:>8}"
                    + "".join(f"{c:>10}" for c in cells[4 : 4 + len(strata)])
                    + f"{cells[-2]:>9}{cells[-1]:>7}"
                )

        # The delta that is the whole point: what protection costs relative to clear text.
        baseline = scopes.get("none")
        if baseline:
            base = baseline["mean_checkpoint_score"]
            lines.append("" if markdown else "")
            for scope in CURVE_ORDER[1:]:
                result = scopes.get(scope)
                if not result or base == 0:
                    continue
                retained = result["mean_checkpoint_score"] / base
                note = f"{scope} retains {retained:.1%} of baseline utility"
                lines.append(f"- {note}" if markdown else f"  {note}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "markdown"), default="text")
    args = parser.parse_args()

    results = load_results()
    if not results:
        sys.exit(f"No evaluation results under {EVAL_ROOT}. Run scripts/run_eval.py first.")

    print(format_table(results, markdown=args.format == "markdown"))

    (EVAL_ROOT / "comparison.json").write_text(
        json.dumps(
            {
                model: {
                    scope: {
                        "mean": r["mean_checkpoint_score"],
                        "verifiable": r.get("verifiable_score"),
                        "completion": r["task_completion_rate"],
                        "strata": r.get("stratum_scores", {}),
                        "llm_stats": r.get("llm_stats", {}),
                    }
                    for scope, r in scopes.items()
                }
                for model, scopes in results.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
