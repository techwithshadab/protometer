"""One-time setup for the open-source local model, so the live app runs without cloud credentials.

The app's live chat prefers a hosted model (Bedrock) when AWS credentials are present, and otherwise
falls back to a local open-source model served by Ollama (default `llama3.2`, ~2GB). This script
performs the one-time download of that model. It is the explicit alternative to the
`PROTOMETER_AUTO_PULL_MODEL=true` flag, which pulls on first use instead.

Prerequisite: Ollama installed and running (https://ollama.com/download). This script does NOT
install Ollama; it only pulls a model into an already-running server.

    python scripts/setup_local_model.py                 # pull the configured local model
    python scripts/setup_local_model.py --model qwen2.5:7b
    PROTOMETER_LOCAL_MODEL=qwen2.5:7b python scripts/setup_local_model.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

from protometer import llm, settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None,
                    help="Ollama model id to pull (default: the configured PROTOMETER_LOCAL_MODEL "
                         "resolved through config/models.yaml).")
    args = ap.parse_args()

    if args.model:
        model_id = args.model
        label = args.model
    else:
        label = settings.local_model()
        try:
            model_id = llm.ModelRegistry.load().get(label).model_id
        except llm.LLMConfigError:
            # Not a config key — treat it as a raw Ollama tag.
            model_id = label

    print(f"Local model: {label} (Ollama id: {model_id})")
    print(f"Ollama endpoint: {settings.ollama_url()}")

    if not llm.ollama_reachable():
        print("\nERROR: Ollama is not reachable.", file=sys.stderr)
        print("  Install it from https://ollama.com/download, start it, then re-run this.",
              file=sys.stderr)
        print("  (If Ollama runs on another host or port, set OLLAMA_URL accordingly.)",
              file=sys.stderr)
        return 1

    if llm.ollama_has_model(model_id):
        print(f"\n'{model_id}' is already pulled. Nothing to do — live chat will use it.")
        return 0

    try:
        llm.ollama_pull(model_id)
    except Exception as exc:  # noqa: BLE001, a pull can fail mid-stream (dropped connection, timeout,
        # partial chunk) as well as up front; any of those should print a clean message, not a
        # traceback, since this is a user-facing setup command.
        print(f"\nERROR: could not pull '{model_id}': {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  Check that Ollama is running and reachable, then re-run.", file=sys.stderr)
        return 1

    print(f"\nDone. Live chat will now run locally on '{label}' when no cloud credentials are set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
