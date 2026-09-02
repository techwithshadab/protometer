"""Load the local corpus JSON into Postgres, per domain schema. Idempotent, $0, no API calls.

The file corpus (data/corpus/*.json) stays the source of truth; this mirrors it into Postgres so
the app can query tables instead of parsing JSON, and so relational joins are possible. Re-run any
time the corpus is regenerated - it rebuilds the tables from the current files and stamps the
corpus fingerprint on the mirror.

    python scripts/load_corpus_db.py                 # load AML (default)
    python scripts/load_corpus_db.py --domain aml    # explicit
    python scripts/load_corpus_db.py --all           # every domain that has a corpus dir

Fails soft: if Postgres is not running it prints how to start it and exits non-zero, without a
traceback (the app still works from files regardless).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

from protometer import db  # noqa: E402
from protometer.tracking import corpus_source_fingerprint  # noqa: E402


def _load_one(domain: str, corpus_dir: Path) -> bool:
    fp = corpus_source_fingerprint(corpus_dir)
    counts = db.load_domain_corpus(domain, corpus_dir, fingerprint=fp)
    if not counts:
        return False
    # AML canaries live in a separate file so they never touch the MEASUREMENT corpus's
    # fingerprint (the eval reproduces from data/corpus/*.json, not these). They are appended to
    # the parties table with is_canary=True; the reveal tripwire flags any detokenization of them.
    # Support/healthcare canaries are generated inline by build_domain_corpus.py, so only AML
    # needs this side file.
    n_canary = 0
    canary_file = corpus_dir / "canaries.json"
    if domain == "aml" and canary_file.exists():
        n_canary = db.append_parties(domain, json.loads(canary_file.read_text()))
    total = sum(counts.values())
    extra = f" + {n_canary} canaries" if n_canary else ""
    print(f"  {domain}: {total} rows across {len(counts)} tables "
          f"({', '.join(f'{k}={v}' for k, v in counts.items())}){extra}; fingerprint {fp}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="aml", help="domain to load (default: aml)")
    parser.add_argument("--all", action="store_true", help="load every domain with a corpus dir")
    args = parser.parse_args()

    if not db.available():
        sys.exit(
            "Postgres is not reachable. Start it with:\n"
            "  cd docker/app/postgres && docker compose up -d\n"
            f"(expected at {__import__('protometer.settings', fromlist=['postgres_url']).postgres_url()})"
        )

    # The AML corpus lives in data/corpus; the non-AML live-chat domains carry a party master under
    # data/corpus/<domain>/ (built by scripts/build_domain_corpus.py). --all loads whatever exists.
    domain_dirs = {
        "aml": ROOT / "data" / "corpus",
        "customer-support": ROOT / "data" / "corpus" / "customer-support",
        "healthcare": ROOT / "data" / "corpus" / "healthcare",
    }
    if args.all:
        targets = domain_dirs.items()
    else:
        d = domain_dirs.get(args.domain)
        if d is None:
            sys.exit(f"no corpus dir configured for domain {args.domain!r}; known: {sorted(domain_dirs)}")
        targets = [(args.domain, d)]

    print("Loading corpus into Postgres:")
    loaded_any = False
    for domain, cdir in targets:
        if not (cdir / "parties.json").exists():
            print(f"  {domain}: skipped (no corpus at {cdir.relative_to(ROOT)})")
            continue
        loaded_any = _load_one(domain, cdir) or loaded_any
    if not loaded_any:
        sys.exit("nothing loaded (no corpus found, or Postgres write failed)")
    print("Done. The app reads its corpus from Postgres (source of truth; no file fallback). "
          "Re-run this loader whenever the file corpus changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
