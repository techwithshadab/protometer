"""Pre-run cost estimation, so a hosted evaluation is never a surprise on a bill.

Token counts are **measured**, not guessed: 51 billed calls across completed scopes averaged
4,029 input and 75 output tokens. The prompt is dominated by the transaction network and
retrieved case notes, whose size is set by the corpus rather than the model, so those averages
carry across providers.

The estimate is an **upper bound**: it assumes no cache hits and charges every judge call.
"""

from __future__ import annotations

from protometer.llm import ModelRegistry

# Measured averages. See module docstring.
AVG_INPUT_TOKENS = 4_029
AVG_OUTPUT_TOKENS = 75

# Narrative checkpoints are graded by a second call.
JUDGE_INPUT_TOKENS = 400
JUDGE_OUTPUT_TOKENS = 40

# Typology tasks plus the entity-resolution, triage and cross-document tasks the suite adds.
EXTRA_TASKS = 4


def estimate_run(
    model: str | None, scopes: int, typology_tasks: int, judge: bool = True
) -> dict:
    """Estimate calls, tokens and USD for a planned run, before anything is invoked."""
    registry = ModelRegistry.load()
    spec = registry.get(model or registry.default)

    tasks_per_scope = typology_tasks + EXTRA_TASKS
    task_calls = scopes * tasks_per_scope
    judge_calls = task_calls if judge else 0

    input_tokens = task_calls * AVG_INPUT_TOKENS + judge_calls * JUDGE_INPUT_TOKENS
    output_tokens = task_calls * AVG_OUTPUT_TOKENS + judge_calls * JUDGE_OUTPUT_TOKENS
    cost = spec.cost_usd(input_tokens, output_tokens)

    return {
        "model": spec.name,
        "provider": spec.provider,
        "tasks_per_scope": tasks_per_scope,
        "calls": task_calls + judge_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "cost_label": (
            "free (local model, no per-token cost)"
            if spec.is_local
            else f"${cost:.2f} upper bound (assumes zero cache hits)"
        ),
    }
