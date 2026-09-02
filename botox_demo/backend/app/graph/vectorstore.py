"""Embedded vector store over chunks + graph-aware expansion, the retrieval half of GraphRAG.

Design choice, embedded, not a service: the corpus is ~100 chunks. A dedicated vector DB (Qdrant,
etc.) would add a container, a network hop, and operational surface for a dataset that fits in a
few hundred KB of numpy. So we embed each chunk with sentence-transformers all-MiniLM-L6-v2 (small,
CPU-friendly, offline after first download) and persist the matrix + metadata to data/index/. Cosine
similarity over a 100x384 matrix is sub-millisecond; there is nothing to gain from a server here and
one fewer moving part in the one-command stack. (Neo4j still runs as a service, the GRAPH half
genuinely benefits from a real graph engine; vectors do not.)

GraphRAG in two moves:
  1. `search(query, k)`  , vector similarity finds the seed chunks most relevant to the question.
  2. `graph_expand(ids)` , pull chunks that share an entity with the seeds (1-hop over the graph),
     so an answer about "chronic migraine" also sees the co-mentioned safety and cost context even
     if those passages didn't rank on raw vector similarity. Expansion prefers Neo4j (:Chunk
     -[:MENTIONS]-> :Entity <-[:MENTIONS]- :Chunk) and falls back to the local graph.json
     chunk_mentions map when Neo4j is unreachable, so retrieval works with or without the service.

    python -m app.graph.vectorstore --build     # build + persist the index
    python -m app.graph.vectorstore --query "what are the side effects?"   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.paths import INDEX, PROCESSED

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_MODEL = None  # lazily loaded; heavy import deferred so `--help` etc. stay instant


def _load_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # noqa: TRY003
            raise RuntimeError(
                "sentence-transformers is not installed. `pip install sentence-transformers`. "
                "It is required to build/query the vector index."
            ) from exc
        _MODEL = SentenceTransformer(EMBED_MODEL)
    return _MODEL


def _embed(texts: list[str]) -> np.ndarray:
    model = _load_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


# ── Build ─────────────────────────────────────────────────────────────────────────────────────
def build_index(chunks_path: Path | None = None) -> int:
    chunks_path = chunks_path or (PROCESSED / "chunks.jsonl")
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found, run `python -m app.ingest.chunk` first.")
    chunks = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not chunks:
        raise ValueError("no chunks to index")

    embeddings = _embed([c["text"] for c in chunks])  # (N, 384), already L2-normalised
    INDEX.mkdir(parents=True, exist_ok=True)
    np.save(INDEX / "embeddings.npy", embeddings)
    meta = [{"id": c["id"], "url": c["url"], "title": c["title"],
             "text": c["text"], "safety": c["safety"]} for c in chunks]
    (INDEX / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    print(f"indexed {len(chunks)} chunks ({embeddings.shape}) -> {INDEX}")
    return len(chunks)


# ── Load (process-cached) ─────────────────────────────────────────────────────────────────────
_EMB: np.ndarray | None = None
_META: list[dict] | None = None
_ID2ROW: dict[str, int] | None = None


def _ensure_loaded() -> None:
    global _EMB, _META, _ID2ROW
    if _EMB is not None:
        return
    emb_path, meta_path = INDEX / "embeddings.npy", INDEX / "meta.json"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"vector index not found in {INDEX}, run `python -m app.graph.vectorstore --build`."
        )
    _EMB = np.load(emb_path)
    _META = json.loads(meta_path.read_text())
    _ID2ROW = {m["id"]: i for i, m in enumerate(_META)}


def warmup() -> None:
    """Load the index, eagerly initialise the sentence-transformer encoder with a throwaway encode,
    and prime the Neo4j driver. The encoder's first `encode()` and the driver's first connect each
    cost time; doing them at startup means the first real user query is fast, not slow."""
    _ensure_loaded()
    _embed(["warmup"])          # forces _load_model() + a real encode pass
    try:
        graph_expand([])        # primes the Neo4j driver (or the local fallback) off the hot path
    except Exception:           # noqa: BLE001, warmup must never fail startup
        pass


# Minimum cosine similarity for a chunk to count as a real retrieval hit. Below this, the query is
# off-corpus and returning the "least-irrelevant" chunks would fabricate grounding: an off-topic
# question ("capital of France") tops out near 0.05-0.09 on this corpus, while genuine BOTOX
# questions score 0.67+, so a 0.30 floor separates them with wide margin. When every candidate is
# below the floor, search() returns [] and the orchestrator takes its grounded-refusal path. Tune
# via the `min_score` arg or the BOTOX_MIN_SIM env var without touching callers.
DEFAULT_MIN_SIM = float(os.getenv("BOTOX_MIN_SIM", "0.30"))


def search(query: str, k: int = 5, min_score: float | None = None) -> list[dict[str, Any]]:
    """Top-k chunks by cosine similarity to the query, keeping only those at or above `min_score`
    (default DEFAULT_MIN_SIM). Each result: meta + `score`. Returns [] when nothing clears the floor
    so an off-corpus query yields a grounded refusal instead of borrowed context."""
    _ensure_loaded()
    assert _EMB is not None and _META is not None
    floor = DEFAULT_MIN_SIM if min_score is None else min_score
    qv = _embed([query])[0]  # (384,), normalised
    sims = _EMB @ qv  # cosine, since both are L2-normalised
    k = min(k, len(_META))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [{**_META[i], "score": float(sims[i])} for i in top if float(sims[i]) >= floor]


# ── Graph expansion ───────────────────────────────────────────────────────────────────────────
def _local_mentions() -> dict[str, list[str]]:
    """chunk_id -> [entity_id] from graph.json, the offline fallback for expansion."""
    gp = PROCESSED / "graph.json"
    if not gp.exists():
        return {}
    return json.loads(gp.read_text()).get("chunk_mentions", {})


_ENTITY_IDF: dict[str, float] | None = None


def _entity_idf() -> dict[str, float]:
    """Per-entity inverse-document-frequency weight: log(N / df), where df is the number of chunks
    that mention the entity. A ubiquitous hub like `botox` (in nearly every chunk) gets a weight near
    0; a rare, discriminating entity (a specific condition) gets a high weight. This is what makes
    graph expansion rank on TOPICAL relatedness rather than on co-mentioning the hub. Computed once
    from graph.json and cached."""
    global _ENTITY_IDF
    if _ENTITY_IDF is None:
        import math
        mentions = _local_mentions()
        n = max(1, len(mentions))
        df: dict[str, int] = {}
        for ents in mentions.values():
            for e in set(ents):
                df[e] = df.get(e, 0) + 1
        # +1 smoothing so an entity in every chunk still gets a small positive weight, not exactly 0.
        _ENTITY_IDF = {e: math.log((n + 1) / (d + 1)) + 1e-6 for e, d in df.items()}
    return _ENTITY_IDF


def _shared_weight(shared: set[str]) -> float:
    """Rank score for a candidate chunk: the IDF-weighted sum of the entities it shares with the
    seeds, so a rare shared entity outweighs many mentions of the ubiquitous hub."""
    idf = _entity_idf()
    return sum(idf.get(e, 1.0) for e in shared)


def graph_expand(seed_chunk_ids: list[str], max_expand: int = 4) -> list[dict[str, Any]]:
    """Chunks that share an entity with the seeds (1-hop over the knowledge graph).

    Tries Neo4j first (:Chunk-[:MENTIONS]->:Entity<-[:MENTIONS]-:Chunk), falls back to the local
    graph.json chunk_mentions map. Returns up to `max_expand` NEW chunks (excluding the seeds),
    ranked by how many entities they share with the seed set. Each result carries `via` = the
    shared entity ids, so the answer layer can explain the graph hop.
    """
    _ensure_loaded()
    assert _META is not None and _ID2ROW is not None
    seeds = set(seed_chunk_ids)

    neighbours = _expand_neo4j(seeds) if _neo4j_available() else None
    if neighbours is None:
        neighbours = _expand_local(seeds)

    # Rank candidates by IDF-WEIGHTED shared-entity score (not raw count): a chunk sharing a rare,
    # topical entity ranks above one that merely co-mentions the ubiquitous `botox` hub. Ties break
    # on raw count. Keep the strongest new chunks.
    ranked = sorted(neighbours.items(), key=lambda kv: (-_shared_weight(kv[1]), -len(kv[1])))
    out: list[dict[str, Any]] = []
    for cid, shared in ranked:
        if cid in seeds or cid not in _ID2ROW:
            continue
        m = _META[_ID2ROW[cid]]
        out.append({**m, "via": sorted(shared), "expanded": True})
        if len(out) >= max_expand:
            break
    return out


def _expand_local(seeds: set[str]) -> dict[str, set[str]]:
    mentions = _local_mentions()
    seed_entities: set[str] = set()
    for s in seeds:
        seed_entities.update(mentions.get(s, []))
    neighbours: dict[str, set[str]] = {}
    for cid, ents in mentions.items():
        if cid in seeds:
            continue
        shared = seed_entities.intersection(ents)
        if shared:
            neighbours[cid] = shared
    return neighbours


def _neo4j_available() -> bool:
    # NEO4J_URL is what compose/docs set; NEO4J_URI/NEO4J_HOST are accepted too. If none is set we
    # skip Neo4j entirely and use the local graph.json expansion (works standalone).
    configured = any(os.getenv(v) for v in ("NEO4J_URL", "NEO4J_URI", "NEO4J_HOST"))
    return configured and _driver() is not None


_DRIVER = None
# When the LAST connection attempt failed, don't re-probe until this monotonic time. A SUCCESS is
# cached forever; a FAILURE is retried after a cooldown, so a startup race (the backend calls
# graph_expand before Neo4j finishes booting) does not pin the process to the local fallback for its
# whole lifetime, it retries once the cooldown passes and picks Neo4j up when it comes online.
_DRIVER_RETRY_AFTER = 0.0
_DRIVER_RETRY_COOLDOWN = 30.0


def _driver():
    global _DRIVER, _DRIVER_RETRY_AFTER
    if _DRIVER is not None:
        return _DRIVER
    if time.monotonic() < _DRIVER_RETRY_AFTER:
        return None                       # in cooldown after a recent failure
    uri = os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD", "botoxdemo123")
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(uri, auth=(user, pw))
        drv.verify_connectivity()
        _DRIVER = drv                     # cache the success for the process lifetime
    except Exception:  # noqa: BLE001, any failure -> local fallback, retry after the cooldown
        _DRIVER = None
        _DRIVER_RETRY_AFTER = time.monotonic() + _DRIVER_RETRY_COOLDOWN
    return _DRIVER


def _expand_neo4j(seeds: set[str]) -> dict[str, set[str]] | None:
    drv = _driver()
    if drv is None:
        return None
    try:
        with drv.session() as session:
            rows = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(n:Chunk)
                WHERE c.id IN $seeds AND NOT n.id IN $seeds
                RETURN n.id AS cid, collect(DISTINCT e.id) AS shared
                """,
                seeds=list(seeds),
            )
            return {r["cid"]: set(r["shared"]) for r in rows}
    except Exception:  # noqa: BLE001
        return None


def retrieve(query: str, k: int = 5, expand: int = 3) -> dict[str, Any]:
    """Full GraphRAG retrieval: vector seeds + graph expansion, merged and de-duped.

    Returns {"seeds": [...], "expanded": [...], "chunks": [...merged...]}. `chunks` is what the
    answer layer grounds on; seeds are ranked by vector score, expanded chunks follow.
    """
    seeds = search(query, k=k)
    expanded = graph_expand([s["id"] for s in seeds], max_expand=expand)
    merged = list(seeds)
    seen = {s["id"] for s in seeds}
    for e in expanded:
        if e["id"] not in seen:
            merged.append(e)
            seen.add(e["id"])
    return {"seeds": seeds, "expanded": expanded, "chunks": merged}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    if args.build:
        build_index()
    if args.query:
        res = retrieve(args.query, k=args.k)
        print(f"\nQ: {args.query}\n")
        for r in res["chunks"]:
            tag = "EXPANDED" if r.get("expanded") else f"score={r['score']:.3f}"
            safety = " [SAFETY]" if r.get("safety") else ""
            print(f"[{tag}]{safety} {r['title']}\n  {r['text'][:160]}...\n")
    if not args.build and not args.query:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
