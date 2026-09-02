"""Estimate the cost of an evaluation run before spending anything.

Token counts are **measured**, not guessed: 51 billed calls across completed local scopes
averaged 4,029 input and 75 output tokens. The prompt is dominated by the transaction network
and retrieved case notes, whose size is set by the corpus rather than the model, so those
averages carry across providers.

    python scripts/estimate_cost.py                        # every hosted model
    python scripts/estimate_cost.py --model bedrock-opus-5
    python scripts/estimate_cost.py --scopes 8 --tasks 9

Output is an *upper bound*: it assumes no cache hits. Re-runs over unchanged prompts cost
nothing, so a second pass on the same corpus is effectively free.
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

from protometer.llm import ModelRegistry  # noqa: E402

# Measured averages. See module docstring.
AVG_INPUT_TOKENS = 4_029
AVG_OUTPUT_TOKENS = 75

# One extra call per narrative checkpoint for LLM-as-judge grading.
JUDGE_CALLS_PER_TASK = 1
JUDGE_INPUT_TOKENS = 400
JUDGE_OUTPUT_TOKENS = 40


def estimate(spec, scopes: int, tasks: int, judge: bool = True) -> dict:
    task_calls = scopes * tasks
    judge_calls = task_calls * JUDGE_CALLS_PER_TASK if judge else 0

    input_tokens = task_calls * AVG_INPUT_TOKENS + judge_calls * JUDGE_INPUT_TOKENS
    output_tokens = task_calls * AVG_OUTPUT_TOKENS + judge_calls * JUDGE_OUTPUT_TOKENS

    return {
        "model": spec.name,
        "provider": spec.provider,
        "calls": task_calls + judge_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": spec.cost_usd(input_tokens, output_tokens),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="estimate one model instead of all")
    parser.add_argument("--scopes", type=int, default=8, help="protection scopes to run")
    parser.add_argument("--tasks", type=int, default=9, help="tasks per scope")
    parser.add_argument("--no-judge", action="store_true", help="exclude judge grading calls")
    args = parser.parse_args()

    registry = ModelRegistry.load()
    specs = (
        [registry.get(args.model)]
        if args.model
        else sorted(registry.specs.values(), key=lambda s: s.cost_per_1m_input)
    )

    print(
        f"Estimate for {args.scopes} scopes x {args.tasks} tasks"
        f"{'' if args.no_judge else ' (+ judge calls)'}\n"
        f"Measured averages: {AVG_INPUT_TOKENS:,} input / {AVG_OUTPUT_TOKENS} output per call\n"
    )
    print(f"{'model':<22}{'provider':<12}{'calls':>7}{'input tok':>12}{'cost USD':>11}")
    print("-" * 64)

    for spec in specs:
        row = estimate(spec, args.scopes, args.tasks, judge=not args.no_judge)
        cost = "free (local)" if spec.is_local else f"${row['cost_usd']:.2f}"
        print(
            f"{row['model']:<22}{row['provider']:<12}{row['calls']:>7}"
            f"{row['input_tokens']:>12,}{cost:>11}"
        )

    print(
        "\nUpper bound: assumes zero cache hits. Repeat runs over unchanged prompts are served"
        "\nfrom cache and cost nothing. Prompt caching, where the provider supports it, removes"
        "\nthe shared system prompt from repeat input cost as well."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
