"""Graph features, following the two published production systems for AML.

Two sources, both with measured lift on real deployments rather than benchmarks:

**Feedzai** (AAAI-22 workshop, ~500k alerted transfers at a real bank, LightGBM, no GNN):

  * *Neighbour-aggregated degree*, in/out degree of the party **plus** mean/min/max in/out
    degree of its one-hop successors and predecessors. Reported **+11.6 percentage points** in
    recall@20%FPR over a tuned tabular baseline. Amount-weighted variants did **not** beat
    unweighted, so weighting is deliberately omitted.
  * *GuiltyWalker*, random walks from the party that terminate at known-illicit nodes;
    features are walk-length statistics, hit rate, and the count of distinct illicit nodes
    reached. Reported **+13.4 percentage points**, strongest at low false-positive rates.
    Combined with degree features: **+15.5** (overlapping but additive).

**IBM Graph Feature Preprocessor** (shipped in Snap ML): scatter-gather patterns, temporal and
simple cycles up to length 10, and vertex statistics, extracted from a streaming in-memory graph
and appended to each transaction row for a gradient-boosted model.

Two properties make these the right choice for this project specifically:

  * They are **invariant under protection**. Party ids are surrogate keys and are
    never tokenized, so the graph is byte-identical at every protection scope. Whatever these
    features contribute, protection cannot take away.
  * They are **exact and explainable**. A k-core number or a cycle membership is a fact about
    the ledger, not a learned embedding, so it survives the reconstructability requirement an
    examiner would apply.

**Leakage discipline.** GuiltyWalker uses known-illicit labels, so it can only ever see labels
from the *training* period. A walk that reaches a node labelled illicit by the very split being
predicted would be leakage of the worst kind, a feature that encodes the answer. The illicit
set is therefore passed in explicitly rather than derived inside, and callers must supply
training-fold labels only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx
import numpy as np

# Walks per node. Feedzai used 50; the return is flat well before that on a graph this size,
# and the cost is linear in this parameter.
WALKS_PER_NODE = 50
MAX_WALK_LENGTH = 10

# The "no walk signal" encoding, in GUILTY_WALK_NAMES order: hit_rate 0, mean/min length at
# the never-reached sentinel, distinct-illicit 0. One definition, because the two ad-hoc
# copies of this list managed to invert the column order twice.
_NO_WALK_SIGNAL = [0.0, float(MAX_WALK_LENGTH + 1), float(MAX_WALK_LENGTH + 1), 0.0]
WALK_SEED = 20260813

# Named because they are cache-key inputs, not only tuning knobs. Every parameter that
# changes the computed features must reach `_cache_path`, or a tuning change silently
# serves stale features until someone remembers a hand-bumped version string.
CYCLE_LENGTH_BOUND = 6
BETWEENNESS_SAMPLE = 100


@dataclass(frozen=True)
class GraphFeatureSet:
    """Feature matrix plus names, kept together so columns cannot drift from labels."""

    values: np.ndarray
    names: list[str]


def build_graph(transactions: list[dict]) -> nx.DiGraph:
    """Directed multigraph collapsed to a simple digraph with edge counts.

    Direction matters, layering is a *directed* path and round-tripping a *directed* cycle -
    so an undirected projection would erase the signal these typologies are made of.
    """
    graph = nx.DiGraph()
    for txn in transactions:
        origin, beneficiary = txn["origin_party_id"], txn["beneficiary_party_id"]
        if graph.has_edge(origin, beneficiary):
            graph[origin][beneficiary]["weight"] += 1
        else:
            graph.add_edge(origin, beneficiary, weight=1)
    return graph


def _neighbour_degree_stats(graph: nx.DiGraph) -> dict[str, list[float]]:
    """Feedzai's neighbour-aggregated degree features, per node.

    The insight is that a party's own degree says less than the *shape of the company it
    keeps*. A mule account and a payroll provider can both have high in-degree; what separates
    them is whether their counterparties are themselves hubs or leaves.
    """
    out_degree = dict(graph.out_degree())
    in_degree = dict(graph.in_degree())
    stats: dict[str, list[float]] = {}

    for node in graph.nodes():
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))

        def summarise(nodes: list[str], degrees: dict[str, int]) -> list[float]:
            values = [float(degrees.get(n, 0)) for n in nodes]
            if not values:
                return [0.0, 0.0, 0.0]
            return [float(np.mean(values)), float(min(values)), float(max(values))]

        stats[node] = [
            float(out_degree.get(node, 0)),
            float(in_degree.get(node, 0)),
            *summarise(successors, out_degree),
            *summarise(successors, in_degree),
            *summarise(predecessors, out_degree),
            *summarise(predecessors, in_degree),
        ]
    return stats


NEIGHBOUR_DEGREE_NAMES = (
    "out_degree", "in_degree",
    "succ_out_mean", "succ_out_min", "succ_out_max",
    "succ_in_mean", "succ_in_min", "succ_in_max",
    "pred_out_mean", "pred_out_min", "pred_out_max",
    "pred_in_mean", "pred_in_min", "pred_in_max",
)


def _guilty_walk_stats(
    graph: nx.DiGraph, illicit_nodes: set[str], rng: random.Random
) -> dict[str, list[float]]:
    """GuiltyWalker: how close is this party to known-bad activity, by random walk?

    Distance to a known-illicit node is a different signal from local structure. A party two
    hops from a confirmed launderer through several distinct paths is suspicious in a way no
    degree statistic captures, and it is exactly how an investigator reasons, follow the money
    and see whose company you end up in.

    Walks traverse the graph as undirected, because proximity to bad activity is meaningful in
    either direction: receiving from a launderer and paying one are both informative.
    """
    undirected = graph.to_undirected()
    stats: dict[str, list[float]] = {}

    for node in graph.nodes():
        lengths: list[int] = []
        reached: set[str] = set()

        for _ in range(WALKS_PER_NODE):
            current = node
            for step in range(1, MAX_WALK_LENGTH + 1):
                neighbours = list(undirected.neighbors(current))
                if not neighbours:
                    break
                current = rng.choice(neighbours)
                # The origin being illicit is not informative about itself.
                if current in illicit_nodes and current != node:
                    lengths.append(step)
                    reached.add(current)
                    break

        hit_rate = len(lengths) / WALKS_PER_NODE
        stats[node] = [
            hit_rate,
            float(np.mean(lengths)) if lengths else float(MAX_WALK_LENGTH + 1),
            float(min(lengths)) if lengths else float(MAX_WALK_LENGTH + 1),
            float(len(reached)),
        ]
    return stats


GUILTY_WALK_NAMES = (
    "gw_hit_rate", "gw_mean_length", "gw_min_length", "gw_distinct_illicit",
)


def _motif_membership(graph: nx.DiGraph) -> dict[str, list[float]]:
    """Explicit motif participation, the IBM Graph Feature Preprocessor approach.

    The typologies this corpus plants *are* these motifs, so counting them directly is both the
    most informative feature available and fully explainable. Cycle enumeration is
    output-sensitive and exponential in the worst case, which is why every production system
    imposes a hop limit. Measured at the current corpus (2,479 nodes / 5,438 edges):
    9.47M bounded cycles in ~134s, which is why `extract` is disk-memoized, and why
    another order of magnitude needs a streaming approach (IBM GFP-style), not
    enumeration.
    """
    in_cycle: dict[str, int] = {}
    cycle_lengths: dict[str, list[int]] = {}
    for cycle in nx.simple_cycles(graph, length_bound=CYCLE_LENGTH_BOUND):
        for node in cycle:
            in_cycle[node] = in_cycle.get(node, 0) + 1
            cycle_lengths.setdefault(node, []).append(len(cycle))

    # Scatter-gather: value fanning out from one party and reconverging on another. The
    # defining shape of layering through intermediaries.
    scatter: dict[str, int] = {}
    gather: dict[str, int] = {}
    for node in graph.nodes():
        successors = set(graph.successors(node))
        predecessors = set(graph.predecessors(node))
        scatter[node] = len(successors)
        gather[node] = len(predecessors)

    return {
        node: [
            float(in_cycle.get(node, 0)),
            float(min(cycle_lengths.get(node, [0])) if node in cycle_lengths else 0),
            float(scatter.get(node, 0)),
            float(gather.get(node, 0)),
        ]
        for node in graph.nodes()
    }


MOTIF_NAMES = ("cycle_count", "min_cycle_length", "scatter_width", "gather_width")


# Disk memo for the full feature extraction. Justified by measurement: at the current corpus
# (2,479 nodes / 5,438 edges) `nx.simple_cycles(length_bound=6)` alone enumerates 9.47M cycles
# in ~134 seconds, and `extract` runs inside `build_classifier`, i.e. on every training scope
# AND every hybrid invocation, recomputing an identical result each time. The ledger is
# immutable between corpus regenerations, so this is a pure function of its inputs.
#
# The key hashes the ORDERED endpoint sequence of both ledgers, not a set of ids: the walk
# features are input-order-dependent (rng.choice over insertion-ordered neighbour lists, a
# known, documented property), so two orderings of the same rows are genuinely different
# inputs and must not share a cache entry.
def _cache_path(transactions, illicit_nodes, fit_on):
    import hashlib
    import os
    from pathlib import Path

    root = os.getenv(
        "PROTOMETER_GRAPH_CACHE",
        str(Path(__file__).resolve().parents[2] / "data" / "cache" / "graph_features"),
    )
    if root in ("", "off", "0"):
        return None
    digest = hashlib.sha256()
    for txn in transactions:
        digest.update(f"{txn['origin_party_id']}>{txn['beneficiary_party_id']};".encode())
    digest.update(b"|fit|")
    for txn in (fit_on if fit_on is not None else transactions):
        digest.update(f"{txn['origin_party_id']}>{txn['beneficiary_party_id']};".encode())
    digest.update(b"|illicit|")
    for node in sorted(illicit_nodes or ()):
        digest.update(node.encode())
    # Every feature-shaping parameter, spelled out. The previous key carried only the seed
    # plus a hand-bumped "v2", so changing walks-per-node or the cycle bound served stale
    # features until someone remembered to bump it, a cache key you have to remember to
    # maintain is a cache key that will eventually lie.
    digest.update(
        f"|seed={WALK_SEED}|wpn={WALKS_PER_NODE}|mwl={MAX_WALK_LENGTH}"
        f"|cyc={CYCLE_LENGTH_BOUND}|btw={BETWEENNESS_SAMPLE}".encode()
    )
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest.hexdigest()[:24]}.npz"


def extract(
    transactions: list[dict],
    illicit_nodes: set[str] | None = None,
    fit_on: list[dict] | None = None,
) -> GraphFeatureSet:
    """Disk-memoized front door over `_extract_uncached`, same signature, same result."""
    import os

    path = _cache_path(transactions, illicit_nodes, fit_on)
    if path is not None and path.exists():
        # A corrupt entry recomputes; it must never crash. Party ids are identical across
        # protection scopes, so concurrent scope runs share ONE cache path, a half-written
        # zip is a live possibility, not a theoretical one, and an unguarded load would turn
        # a single collision into a permanent crash on every future run.
        try:
            loaded = np.load(path, allow_pickle=False)
            return GraphFeatureSet(
                values=loaded["values"], names=list(loaded["names"].tolist())
            )
        except Exception:  # noqa: BLE001, any unreadable entry means "recompute"
            try:
                path.unlink()
            except OSError:
                pass
    result = _extract_uncached(transactions, illicit_nodes, fit_on)
    if path is not None:
        # Write-then-rename so a concurrent reader can only ever see a complete file.
        tmp = path.parent / f".{path.stem}.{os.getpid()}.tmp.npz"
        try:
            np.savez_compressed(tmp, values=result.values, names=np.array(result.names))
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)  # cache write failure must not fail the extraction
    return result


def _extract_uncached(
    transactions: list[dict],
    illicit_nodes: set[str] | None = None,
    fit_on: list[dict] | None = None,
) -> GraphFeatureSet:
    """Per-transaction graph features for both endpoints.

    `illicit_nodes` must contain **training-fold labels only**. Passing the full label set would
    let a walk reach a node labelled by the split being predicted, which is leakage of the
    worst kind, a feature encoding the answer.

    `fit_on` is the ledger the *graph itself* is built from; features are still emitted for
    every transaction in `transactions`. Building it from all transactions gave each training
    row a degree, k-core, PageRank and cycle count computed over a graph containing the test
    fold, the same leakage as above, arriving through the topology rather than the labels,
    and affecting roughly 30 of the 38 features rather than 4.
    """
    graph = build_graph(transactions if fit_on is None else fit_on)
    rng = random.Random(WALK_SEED)

    neighbour = _neighbour_degree_stats(graph)
    motifs = _motif_membership(graph)
    walks = (
        _guilty_walk_stats(graph, illicit_nodes, rng)
        if illicit_nodes
        # Column order is GUILTY_WALK_NAMES: (hit_rate, mean_length, min_length,
        # distinct_illicit). "No labels" therefore means hit_rate 0.0 and the LENGTH columns
        # at the never-reached sentinel (MAX_WALK_LENGTH + 1), a 0.0 length reads as
        # "reached an illicit node in zero hops", maximal guilt.
        #
        # Both prior versions of this line inverted it: the original filled all zeros, and
        # the "fix" put the sentinel in the hit_rate slot ([11.0, 0.0, 0.0, 0.0]), an
        # impossible rate and zero-hop lengths, the exact bug the fix described. The order is
        # now pinned by a regression test against GUILTY_WALK_NAMES rather than trusted to a
        # comment, which is what let it invert twice.
        else {
            node: _NO_WALK_SIGNAL.copy()
            for node in graph.nodes()
        }
    )

    # Global centralities. Betweenness is O(VE) exactly, so it is sampled, the ranking is
    # stable well before the full node set is used.
    pagerank = nx.pagerank(graph)
    kcore = nx.core_number(graph.to_undirected())
    clustering = nx.clustering(graph.to_undirected())
    betweenness = nx.betweenness_centrality(
        graph, k=min(BETWEENNESS_SAMPLE, graph.number_of_nodes()), seed=WALK_SEED
    )

    empty_neighbour = [0.0] * len(NEIGHBOUR_DEGREE_NAMES)
    # Nodes absent from the (training-fold) graph get the same "no signal" encoding as the
    # no-labels case. All-zeros here was the live-path version of the sentinel inversion:
    # every test-fold-only party scored gw_mean_length = 0.0, "reached illicit in zero
    # hops", a phantom-guilt bias applied precisely to the rows being evaluated. Measured
    # before the fix: 28 affected test rows with mean score 0.329 against 0.086 for the rest.
    empty_walk = _NO_WALK_SIGNAL.copy()
    empty_motif = [0.0] * len(MOTIF_NAMES)

    rows: list[list[float]] = []
    for txn in transactions:
        origin, beneficiary = txn["origin_party_id"], txn["beneficiary_party_id"]
        row: list[float] = []
        for node in (origin, beneficiary):
            row.extend(neighbour.get(node, empty_neighbour))
            row.extend(walks.get(node, empty_walk))
            row.extend(motifs.get(node, empty_motif))
            row.extend([
                float(pagerank.get(node, 0.0)),
                float(kcore.get(node, 0)),
                float(clustering.get(node, 0.0)),
                float(betweenness.get(node, 0.0)),
            ])
        rows.append(row)

    centrality_names = ("pagerank", "kcore", "clustering", "betweenness")
    names = [
        f"{side}_{name}"
        for side in ("o", "b")
        for name in (
            *NEIGHBOUR_DEGREE_NAMES, *GUILTY_WALK_NAMES, *MOTIF_NAMES, *centrality_names
        )
    ]
    return GraphFeatureSet(values=np.asarray(rows, dtype=float), names=names)
