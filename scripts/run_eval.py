"""Run the utility-vs-scope evaluation.

    python scripts/run_eval.py                          # every scope, default model
    python scripts/run_eval.py --scopes none direct     # named scopes
    python scripts/run_eval.py --model claude-sonnet-5  # any model in config/models.yaml
    python scripts/run_eval.py --tasks 2 --no-resume    # quick smoke test

Results are written per scope to data/eval/, incrementally, so an interrupted run resumes
rather than repaying the cost of completed scopes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# README step 2 is `cp .env.example .env`; make that instruction true.
from protometer.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

from protometer.eval.cost import estimate_run  # noqa: E402
from protometer.eval.runner import run_evaluation  # noqa: E402
from protometer.scopes import CURVE_ORDER  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scopes", nargs="*", default=None, choices=list(CURVE_ORDER))
    parser.add_argument("--model", default=None, help="model name from config/models.yaml")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="model for narrative grading (defaults to the main model)",
    )
    parser.add_argument("--tasks", type=int, default=8, help="number of typology tasks")
    # Concurrency was hardcoded at 5 in the runner with no way to change it. Hosted
    # providers throttle differently by account and time of day, and the only lever when
    # that happens should not be editing source.
    parser.add_argument("--workers", type=int, default=5,
                        help="concurrent tasks per scope; lower if the provider throttles")
    parser.add_argument(
        "--no-resume", action="store_true", help="re-run scopes even if results exist"
    )
    parser.add_argument("--output", default=None, help="output directory")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="bypass the response cache entirely, every call hits the model",
    )
    parser.add_argument(
        "--detection-on-protected", action="store_true",
        help=(
            "run the deterministic detectors on each scope's PROTECTED ledger (end-to-end "
            "deployment behaviour). Default runs them on clear text so the curve isolates "
            "the model rather than the detectors."
        ),
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the cost confirmation prompt",
    )
    args = parser.parse_args()

    from protometer.persist import acquire_run_lock

    try:
        _lock = acquire_run_lock(ROOT / "data")  # held for process lifetime  # noqa: F841
    except RuntimeError as exc:
        sys.exit(str(exc))

    # A model-specific output directory keeps runs from overwriting each other, so results
    # across models can be compared rather than clobbered.
    default_output = ROOT / "data" / "eval"
    if args.model and not args.output:
        default_output = ROOT / "data" / "eval" / args.model.replace("/", "_")

    # The resolved provider must answer before anything is estimated or spent. A recorded incident: a
    # silently-dead provider once routed an entire "Sonnet 5" curve to a local 14B model.
    from protometer.llm import preflight

    resolved = preflight(args.model)
    print(f"preflight OK, {resolved} answers")

    # Cost is estimated and confirmed BEFORE any call is made. A hosted run should never be
    # a surprise on someone's bill.
    scope_count = len(args.scopes) if args.scopes else len(CURVE_ORDER)
    estimate = estimate_run(args.model, scope_count, args.tasks)
    print(
        f"\nPlanned: {scope_count} scopes x ~{estimate['tasks_per_scope']} tasks on "
        f"{estimate['model']} ({estimate['provider']})\n"
        f"Estimated: {estimate['calls']} calls, {estimate['input_tokens']:,} input tokens, "
        f"{estimate['cost_label']}\n"
    )
    if estimate["cost_usd"] > 0 and not args.yes:
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted before any call was made.")
            return 0

    results = run_evaluation(
        corpus_dir=ROOT / "data" / "corpus",
        protected_root=ROOT / "data" / "protected",
        index_root=ROOT / "data" / "index",
        output_dir=Path(args.output) if args.output else default_output,
        scopes=args.scopes,
        model=args.model,
        judge_model=args.judge_model,
        max_typologies=args.tasks,
        resume=not args.no_resume,
        no_cache=args.no_cache,
        max_workers=args.workers,
        detection_dir=(
            None if not args.detection_on_protected else Path("__per_scope__")
        ),
    )

    if results:
        print(f"{'scope':<28} {'mean':>7} {'verifiable':>11} {'completion':>11}")
        for name, result in results.items():
            print(
                f"{name:<28} {result.mean_checkpoint_score:>7.3f} "
                f"{result.verifiable_score():>11.3f} {result.task_completion_rate:>10.1%}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
