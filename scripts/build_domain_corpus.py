"""Generate synthetic party corpora for the non-AML live-chat domains and write them to disk.

AML has its own richer generator (`build_corpus.py`, transactions + alerts + narratives). Support
and healthcare need only a *party master* so a live chatbot turn has a roster to protect against. This writes one `parties.json` per domain under `data/corpus/<domain>/`, the layout the
DB loader reads (`scripts/load_corpus_db.py --domain <domain>`).

    python scripts/build_domain_corpus.py            # both domains, default sizes
    python scripts/build_domain_corpus.py --domain healthcare --count 200

Deterministic ($0, no API calls): a fixed per-domain seed, so a rebuild is byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlguard.corpus.domain_parties import DOMAIN_PARTY_GENERATORS, generate_domain_parties

# Fixed seeds, one per domain, so each corpus is reproducible and independent of the others.
DOMAIN_SEEDS = {"customer-support": 4187, "healthcare": 9302}
DEFAULT_COUNT = 150


def _build_one(domain: str, count: int, root: Path) -> Path:
    parties = generate_domain_parties(domain, count, seed=DOMAIN_SEEDS[domain])
    out_dir = root / "data" / "corpus" / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "parties.json"
    path.write_text(json.dumps(parties, indent=1, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=sorted(DOMAIN_PARTY_GENERATORS),
                        help="one domain to build (default: all)")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help=f"parties per domain (default: {DEFAULT_COUNT})")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    domains = [args.domain] if args.domain else sorted(DOMAIN_PARTY_GENERATORS)
    for domain in domains:
        path = _build_one(domain, args.count, root)
        rel = path.relative_to(root)
        print(f"  {domain:<18} {args.count:>4} parties -> {rel}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
