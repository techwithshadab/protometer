"""Measure Semantic Erasure, the project's headline finding, reproducibly.

The claim is that tokenization does not degrade retrieval uniformly. It leaves **behavioural**
queries substantially intact, because the surrounding unprotected prose still carries the
signal, and **destroys identity** queries, because a token has no embedding relationship to the
name it replaced.

Published work establishes that blanket redaction costs retrieval quality, TRIP-RAG reports
Recall@1 falling from 1.000 to 0.430, but no published work found decomposes that loss by query
type. This script is the evidence for that decomposition, and it exists so a reviewer can
re-derive the numbers in the README rather than take them on trust.

    python scripts/measure_semantic_erasure.py
    python scripts/measure_semantic_erasure.py --scopes none direct quasi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.retrieval import NarrativeIndex  # noqa: E402
from protometer.scopes import CURVE_ORDER, get_scope  # noqa: E402

# Queries about *behaviour*, one per typology, each paired with the documents that genuinely
# describe it. Their signal lives in prose that protection never touches, so they should
# survive.
#
# Both arms are now scored by **rank of a known-correct target**, which they were not before.
# The behavioural arm previously reported the *mean distance of the top hit*, a metric that
# cannot detect failure, because in a several-hundred-document index something is always
# nearby, and it never checks whether the retrieved document is the right one. Reporting
# "behavioural retrieval survives" on a measure incapable of showing it failing is not a
# finding, and it made the two arms incommensurable: one measured proximity, the other
# correctness.
BEHAVIOURAL_QUERIES: tuple[tuple[str, str], ...] = (
    ("cash deposits just below the reporting threshold", "structuring"),
    ("funds moved through a chain of intermediaries, each hop shedding a commission",
     "layering"),
    ("value leaving an account and returning to it through a circuit of companies",
     "round_tripping"),
    ("many depositors paying cash into one collection account, swept out quickly",
     "funnel_account"),
    ("repeated round-value payments against a single invoice reference", "trade_based"),
)


def _identity_queries(narratives: list[dict], parties: dict[str, dict], limit: int = 5):
    """Queries naming a specific party, the signal protection removes.

    Built from narratives that actually name someone, so each query has a known-correct target
    document and the measurement is a rank, not an impression.
    """
    queries = []
    for narrative in narratives:
        subject = parties.get(narrative.get("subject_party_id", ""))
        if not subject or not subject.get("full_name"):
            continue
        queries.append(
            (
                f"investigation concerning {subject['full_name']}",
                narrative["document_id"],
            )
        )
        if len(queries) >= limit:
            break
    return queries


def _behavioural_targets(narratives: list[dict], ground_truth: list[dict]) -> list:
    """Pair each behavioural query with every narrative describing that typology.

    Gives the behavioural arm known-correct targets, so it can be scored by the same rank
    metric as the identity arm, and so it can *fail*, which the previous top-hit-distance
    measure made impossible.
    """
    typology_of = {
        instance["typology_id"]: instance["typology"] for instance in ground_truth
    }
    by_typology: dict[str, set[str]] = {}
    for narrative in narratives:
        typology = typology_of.get(narrative.get("typology_id") or "")
        if typology:
            by_typology.setdefault(typology, set()).add(narrative["document_id"])

    return [
        (query, by_typology.get(typology, set()))
        for query, typology in BEHAVIOURAL_QUERIES
        if by_typology.get(typology)
    ]


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p for the 2x2 table [[a, b], [c, d]].

    Exact hypergeometric enumeration with math.comb, no scipy: the docs quote this
    p-value, so the repository must be able to re-derive it. A quoted number whose
    computation lives nowhere is a claim, not a measurement.
    """
    import math

    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2

    def prob(x: int) -> float:
        return (
            math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)
        )

    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(col1, row1)
    return min(1.0, sum(
        prob(x) for x in range(lo, hi + 1) if prob(x) <= observed + 1e-12
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scopes", nargs="*", default=None, choices=list(CURVE_ORDER))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=40,
                        help="identity queries to run (n=5 was underpowered)")
    args = parser.parse_args()

    corpus = ROOT / "data" / "corpus"
    narratives = json.loads((corpus / "narratives.json").read_text())
    parties = {p["party_id"]: p for p in json.loads((corpus / "parties.json").read_text())}
    ground_truth = json.loads((corpus / "ground_truth.json").read_text())
    # 40 identity queries rather than 5. At n=5 the reported "3/5 found -> 0/5 found" gives
    # Fisher's exact p = 0.167, not significant, for the project's flagship claim, while the
    # LLM curve next to it carries bootstrap CIs and exact McNemar tests. The asymmetry in
    # rigour was the first thing a reviewer objected to.
    identity_queries = _identity_queries(narratives, parties, limit=args.queries)
    behavioural_targets = _behavioural_targets(narratives, ground_truth)

    scope_names = args.scopes or [
        s for s in CURVE_ORDER if (ROOT / "data" / "index" / get_scope(s).slug).exists()
    ]
    if not scope_names:
        sys.exit("No indexes found. Run scripts/ingest_all.py and build indexes first.")

    results: dict[str, dict] = {}

    print(
        f"{'scope':<26}{'behav found':>13}{'b.rank':>10}{'ident found':>15}{'i.rank':>10}"
    )
    print("-" * 74)

    for name in scope_names:
        scope = get_scope(name)
        index = NarrativeIndex(scope.slug, ROOT / "data" / "index" / scope.slug)

        # Behavioural: does a document genuinely describing this typology appear in the top-k,
        # and where? Same metric as the identity arm below.
        behavioural_hits = 0
        behavioural_ranks: list[int] = []
        for query, targets in behavioural_targets:
            ids = [hit.document_id for hit in index.search(query, top_k=args.top_k)]
            rank = next((i + 1 for i, doc in enumerate(ids) if doc in targets), None)
            if rank is not None:
                behavioural_hits += 1
                behavioural_ranks.append(rank)
        behavioural_mean_rank = (
            sum(behavioural_ranks) / len(behavioural_ranks) if behavioural_ranks else None
        )

        # Identity: where does the correct document rank, if it appears at all?
        ranks: list[int | None] = []
        for query, expected_doc in identity_queries:
            ids = [hit.document_id for hit in index.search(query, top_k=args.top_k)]
            ranks.append(ids.index(expected_doc) + 1 if expected_doc in ids else None)

        found = [r for r in ranks if r is not None]
        mean_rank = sum(found) / len(found) if found else 0.0

        results[name] = {
            "behavioural_found": f"{behavioural_hits}/{len(behavioural_targets)}",
            "behavioural_mean_rank": (
                round(behavioural_mean_rank, 2) if behavioural_mean_rank else None
            ),
            "identity_mean_rank": round(mean_rank, 2) if found else None,
            "identity_found": f"{len(found)}/{len(ranks)}",
            "identity_ranks": ranks,
        }

        rank_label = f"{mean_rank:.1f}" if found else "not found"
        beh_label = (
            f"{behavioural_mean_rank:.1f}" if behavioural_mean_rank else "not found"
        )
        print(
            f"{name:<26}{behavioural_hits:>6}/{len(behavioural_targets):<7}{beh_label:>10}"
            f"{len(found):>8}/{len(ranks):<7}{rank_label:>10}"
        )

    baseline = results.get(scope_names[0], {})
    # The identity-arm significance test, computed here rather than quoted: baseline
    # found-vs-missed against each protected scope's found-vs-missed.
    def found_count(row: dict) -> tuple[int, int]:
        # identity_found is stored presentation-shaped ("30/40"); parse both halves.
        raw = str(row.get("identity_found") or "0/0")
        found_s, _, total_s = raw.partition("/")
        return int(found_s or 0), int(total_s or 0)

    base_found, n_queries = found_count(baseline)
    n_queries = n_queries or args.queries
    for name in scope_names[1:]:
        found, _ = found_count(results[name])
        results[name]["identity_fisher_p_vs_baseline"] = fisher_exact_two_sided(
            base_found, n_queries - base_found, found, n_queries - found
        )

    print(
        f"\nBoth arms scored the same way: recall of a known-correct document in the top-k.\n"
        f"Baseline behavioural {baseline.get('behavioural_found')}, "
        f"identity {baseline.get('identity_found')}."
    )
    for name in scope_names[1:]:
        p = results[name].get("identity_fisher_p_vs_baseline")
        if p is not None:
            print(f"  identity arm, {name} vs baseline: Fisher exact p = {p:.2e}")

    out = ROOT / "data" / "eval" / "semantic_erasure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    from protometer.persist import atomic_write_json

    atomic_write_json(out, results)
    print(f"written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
