"""Seed / update the botox system prompt in Langfuse Prompt Management.

Pushes the hardcoded `SYSTEM_PROMPT` (the fail-safe fallback and current source of truth) to Langfuse
as a versioned, labeled prompt, so it becomes editable and version-tracked in the Langfuse UI. The
running app then prefers the Langfuse copy at request time (see prompt_manager.get_system_prompt),
and still falls back to the hardcoded text if Langfuse is unreachable.

Idempotent: creating a prompt with the same name adds a new VERSION (Langfuse never overwrites), and
re-running when the text is unchanged is harmless. Run once after the stack is up:

    docker compose exec backend python -m app.obs.seed_prompt

Env (same as prompt_manager): BOTOX_PROMPT_NAME (default "botox-system"),
BOTOX_PROMPT_LABEL (default "production").
"""

from __future__ import annotations

import os
import sys

from app.obs import prompt_manager, tracing
from app.pipeline.llm import SYSTEM_PROMPT


def main() -> int:
    lf = tracing.client()
    if lf is None:
        print("Langfuse is not configured (LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST). "
              "The app will use the hardcoded prompt. Nothing to seed.", file=sys.stderr)
        return 1

    name = os.getenv("BOTOX_PROMPT_NAME", "botox-system")
    label = os.getenv("BOTOX_PROMPT_LABEL", "production")

    # If a version with identical text already carries this label, don't create a duplicate.
    try:
        existing = lf.get_prompt(name, label=label, cache_ttl_seconds=0)
        if isinstance(existing.prompt, str) and existing.prompt.strip() == SYSTEM_PROMPT.strip():
            print(f"'{name}' (label '{label}', v{getattr(existing, 'version', '?')}) already matches "
                  f"the hardcoded prompt; nothing to do.")
            return 0
        print(f"'{name}' exists but differs from the hardcoded prompt; creating a new version...")
    except Exception:  # noqa: BLE001, not found / unreachable -> attempt create below
        print(f"'{name}' not found in Langfuse; creating the first version...")

    try:
        created = lf.create_prompt(
            name=name,
            prompt=SYSTEM_PROMPT,
            labels=[label],
            config={"temperature": 0.1, "model_hint": "grounded, non-advisory, safety-forward"},
        )
        version = getattr(created, "version", "?")
        print(f"Pushed '{name}' v{version} labeled '{label}'. "
              f"Edit it in the Langfuse UI; the app picks up new '{label}' versions within the TTL.")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to create prompt: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    prompt_manager.invalidate_cache()
    # Verify the app-side resolver now sees Langfuse (not the fallback).
    resolved = prompt_manager.get_system_prompt()
    print(f"Resolver now returns source={resolved.source} version={resolved.version}.")
    lf.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
