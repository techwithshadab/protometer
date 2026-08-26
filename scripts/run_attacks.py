"""Run the adversarial evaluation against a protected corpus.

    python scripts/run_attacks.py            # every ingested scope
    python scripts/run_attacks.py direct
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# README step 2 is `cp .env.example .env`; make that instruction true.
from amlguard.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)
from amlguard.attacks import run_all
from amlguard.scopes import CURVE_ORDER, get_scope


def main(argv: list[str]) -> int:
    # `--help` is the first thing anyone types. Treating it as a scope name produced
    # `KeyError: "Unknown scope '--help'"`, which reads as a broken script.
    if any(a in ("-h", "--help") for a in argv[1:]):
        print(__doc__)
        return 0

    names = argv[1:] or [s for s in CURVE_ORDER
                         if (ROOT/"data"/"protected"/get_scope(s).slug/"narratives.json").exists()]
    # An empty scope list means nothing is ingested, not that nothing leaked. Writing `{}` and
    # exiting 0 made "no attack ran" indistinguishable from "no attack succeeded", the worst
    # possible confusion for a *security* evaluation, and it silently overwrote real results.
    if not names:
        sys.exit(
            "No ingested scopes found, nothing was attacked, and no result was written.\n"
            "Run: python scripts/ingest_all.py"
        )
    out = {}
    for name in names:
        scope = get_scope(name)
        if not scope.entities:
            continue  # baseline protects nothing; attacking it is meaningless
        results = run_all(ROOT/"data"/"corpus", ROOT/"data"/"protected"/scope.slug)
        out[name] = [r.to_dict() for r in results]
        print(f"\n=== {name} ===")
        print(f"{'attack':<34}{'acc':>8}{'chance':>9}{'lift':>7}  notes")
        for r in results:
            print(f"{r.name:<34}{r.accuracy:>8.1%}{r.baseline_accuracy:>9.2%}{r.lift:>7.1f}x  {r.notes[:44]}")
    if not out:
        sys.exit("Every scope was skipped, no results to write.")
    dest = ROOT/"data"/"eval"/"attacks.json"
    dest.parent.mkdir(parents=True, exist_ok=True)  # absent on a fresh clone
    from amlguard.persist import atomic_write_json

    atomic_write_json(dest, out)
    print(f"\nwrote {dest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
