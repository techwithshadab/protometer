#!/usr/bin/env python3
"""Seed every Protometer domain prompt into ITS OWN Langfuse project, versioned.

Protometer has three domains (aml / healthcare / customer-support), each with its own
investigation / rationale / judge prompt (config/prompts/<name>.txt) and its own Langfuse project.
`managed_prompt` seeds a prompt lazily on first use, but that only registers the ones a run happens
to exercise (e.g. only the investigation prompt of a domain you chat with). This script registers
ALL of them up front, each in the right project, so every use case's prompts are versioned in
Langfuse from the start.

Idempotent: `managed_prompt` creates v1 only when the prompt doesn't already exist in that project,
and returns the existing text otherwise. Safe to re-run.

Requires the Langfuse keys for each domain's project in the environment (see .env):
  aml         -> LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (the default/AML project)
  healthcare  -> LANGFUSE_PUBLIC_KEY_HEALTHCARE / _SECRET_KEY_HEALTHCARE
  customer-support -> LANGFUSE_PUBLIC_KEY_SUPPORT / _SECRET_KEY_SUPPORT
A domain whose keys are unset falls back to the AML project (still seeded, just not isolated).

Usage:  python scripts/seed_langfuse_prompts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protometer.domains import domain_names, get_domain  # noqa: E402
from protometer.observability import _get_client, domain_project, managed_prompt  # noqa: E402


def main() -> int:
    if _get_client() is None:
        print("Langfuse is not configured (LANGFUSE_PUBLIC_KEY / SECRET_KEY unset), or tracing is "
              "off (PROTOMETER_NO_TRACING=1). Nothing to seed.", file=sys.stderr)
        return 1

    total = 0
    for dname in domain_names():
        domain = get_domain(dname)
        project = domain_project(dname)
        proj_label = project or "aml(default)"
        # Each domain's three prompts: investigation (live + batch), rationale (hybrid), judge (eval).
        for prompt_name in (domain.investigation_prompt, domain.rationale_prompt, domain.judge_prompt):
            text = managed_prompt(prompt_name, project=project)
            ok = bool(text and text.strip())
            print(f"  [{proj_label:16}] {prompt_name:35} -> {'seeded/present' if ok else 'FAILED'}")
            total += 1

    from protometer.observability import flush
    flush()
    print(f"\nDone. {total} prompt(s) ensured across the domain projects. "
          f"Edit any of them in the Langfuse UI; the app picks up new versions within the TTL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
