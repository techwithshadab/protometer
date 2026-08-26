"""Generate the synthetic AML corpus and write it to data/corpus/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlguard.corpus.generate import generate_corpus, write_corpus


def main() -> int:
    # Scaled for statistical power. At 28 typologies the evaluation reaches only ~110
    # checkpoints and a minimum detectable effect of 0.119, larger than the entire spread
    # observed across protection scopes (0.135), so no ordering could be resolved. 120
    # typologies brings MDE to ~0.057, which can resolve the differences actually seen.
    #
    # `n_parties` is 4,000 rather than 600 to bring the **illicit population density** into a
    # realistic range. 122 typology instances need ~513 party-slots; drawn from 257
    # organizations they forced heavy reuse, leaving 196 of 600 parties (32.7%) illicit and
    # one party in 54 separate typologies. Real AML populations are well under 1%, and at
    # 32.7% the graph features cannot be separated from the label by *any* train/test split -
    # 43.5% of benign transactions touched a known-illicit party, so "walks to a known-illicit
    # node" was genuinely predictive of benign activity too.
    #
    # Parties are cheap to protect and narratives are not: the structured path batches
    # (600 parties cost 14 API calls), while the unstructured path is one round-trip per
    # entity per line. Growing the party pool while holding narratives at 500 buys
    # a realistic base rate at almost no additional tokenization cost.
    corpus = generate_corpus(
        n_parties=4000,
        n_benign_transactions=6000,
        n_structuring=34,
        n_layering=26,
        n_round_tripping=22,
        n_funnel_account=22,
        n_trade_based=18,
        n_benign_narratives=500,
    )
    out = Path(__file__).resolve().parents[1] / "data" / "corpus"
    paths = write_corpus(corpus, out)

    print(f"parties      {len(corpus.parties):>5}")
    print(f"transactions {len(corpus.transactions):>5}")
    print(f"narratives   {len(corpus.narratives):>5}")
    print(f"typologies   {len(corpus.typologies):>5}")
    print(f"alerts       {len(corpus.alerts):>5}")
    print()
    for name, path in paths.items():
        print(f"  {name:<13} {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
