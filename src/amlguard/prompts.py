"""File-backed prompt store: the code-side source of truth for system prompts.

Prompts are *content*, not code, and hardcoding them meant a wording change was a code
change. They now live in `config/prompts/<name>.txt`, editable without touching Python, and
this module is the single seam that reads them. It has no dependency on Langfuse or any
telemetry: even with all observability off, the pipeline resolves its prompts from disk.

The relationship with the Langfuse registry (see `observability.managed_prompt`) is
deliberate: Langfuse is the *editing surface* (versioned, UI-editable at runtime), and this
file is the *durable fallback*. When `managed_prompt` fetches a newer version from Langfuse,
it writes it back here via `save_prompt`, so the on-disk fallback is always the latest
version the code has seen. If Langfuse is then unreachable, the pipeline uses that
last-synced latest text, not a stale constant frozen at deploy time.
"""

from __future__ import annotations

from pathlib import Path

from amlguard.log import get_logger

_log = get_logger("prompts")

# config/prompts/ relative to the repo root (this file is src/amlguard/prompts.py).
PROMPT_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"


def _path(name: str) -> Path:
    # Names are our own registry keys (amlguard-*-system); reject anything that could escape
    # the prompt directory rather than trusting the caller.
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"invalid prompt name: {name!r}")
    return PROMPT_DIR / f"{name}.txt"


def load_prompt(name: str) -> str:
    """The prompt text from disk. Raises if the file is missing, that is a packaging bug,
    not a runtime condition to paper over: a prompt the pipeline needs must ship with it."""
    path = _path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"prompt {name!r} not found at {path}. Prompts live in config/prompts/; "
            f"a required prompt file is missing from the checkout."
        )
    return path.read_text().strip()


def save_prompt(name: str, text: str) -> bool:
    """Persist `text` as the on-disk prompt, only when it differs, atomically.

    Returns True if the file was updated. Called by `managed_prompt` to keep the durable
    fallback in step with the latest Langfuse version. A write failure is logged and
    swallowed: syncing the fallback must never break a run.
    """
    path = _path(name)
    new = text.strip() + "\n"
    try:
        if path.exists() and path.read_text() == new:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".txt.tmp")
        tmp.write_text(new)
        tmp.replace(path)
        _log.info("prompt %s synced from registry to %s", name, path.name)
        return True
    except OSError as exc:
        _log.warning("could not sync prompt %s to disk: %s", name, exc)
        return False
