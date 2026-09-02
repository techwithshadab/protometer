"""Load the extracted knowledge graph into Neo4j, idempotently, fail-soft.

Why Neo4j (vs. keeping the graph in-process): the vector half of GraphRAG is fine in numpy, but
the GRAPH half, "find chunks that share an entity with my seeds, 1-hop out", is exactly what a
graph engine does well, and it scales past a JSON map if the corpus grows. Neo4j runs as a service
in the compose stack; this loader mirrors graph.json into it.

Schema:
  (:Entity {id, label, type})                      the BOTOX knowledge-graph nodes
  (:Chunk  {id, text, url, title, safety})         the retrievable passages
  (:Entity)-[:REL {type}]->(:Entity)               typed relationships (TREATS, WARNS_ABOUT, ...)
  (:Chunk)-[:MENTIONS]->(:Entity)                  which passages mention which entity

Everything is MERGE, so re-running is idempotent (no duplicate nodes/edges). If Neo4j is
unreachable the loader prints a clear message and returns 0 rather than raising, the rest of the
pipeline (chunks, vectors, graph.json) is already built and the app degrades to local graph
expansion.

    python -m app.graph.build_neo4j        # loads data/processed/graph.json + chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.paths import PROCESSED


def _driver():
    # NEO4J_URL is what compose and the docs set; accept NEO4J_URI too (the older name) so either
    # works, then fall back to the compose-network default.
    uri = os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD", "botoxdemo123")
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(uri, auth=(user, pw))
    drv.verify_connectivity()
    return drv


def load(graph_path: Path | None = None, chunks_path: Path | None = None) -> int:
    graph_path = graph_path or (PROCESSED / "graph.json")
    chunks_path = chunks_path or (PROCESSED / "chunks.jsonl")
    if not graph_path.exists():
        raise FileNotFoundError(f"{graph_path} not found, run `python -m app.graph.extract` first.")

    try:
        drv = _driver()
    except Exception as exc:  # noqa: BLE001
        print(f"  Neo4j not reachable ({type(exc).__name__}: {str(exc)[:80]}). "
              f"Skipping graph load; the app will use local graph expansion until Neo4j is up.")
        return 0

    graph = json.loads(graph_path.read_text())
    chunks = {}
    if chunks_path.exists():
        chunks = {c["id"]: c for c in
                  (json.loads(l) for l in chunks_path.read_text().splitlines() if l.strip())}

    with drv.session() as session:
        # Constraints make MERGE fast and enforce id uniqueness (idempotency).
        session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")

        # A clean reload: wipe prior demo data so a re-run reflects the current extraction exactly.
        session.run("MATCH (n) WHERE n:Entity OR n:Chunk DETACH DELETE n")

        # Entities
        session.run(
            "UNWIND $nodes AS n MERGE (e:Entity {id:n.id}) SET e.label=n.label, e.type=n.type",
            nodes=graph["nodes"],
        )
        # Chunks (text lives here so graph expansion can return passages directly)
        chunk_rows = [
            {"id": cid, "text": c["text"], "url": c["url"], "title": c["title"],
             "safety": bool(c["safety"])}
            for cid, c in chunks.items()
        ]
        if chunk_rows:
            session.run(
                "UNWIND $rows AS r MERGE (c:Chunk {id:r.id}) "
                "SET c.text=r.text, c.url=r.url, c.title=r.title, c.safety=r.safety",
                rows=chunk_rows,
            )
        # Typed entity-entity relationships (one relationship type REL with a `type` property keeps
        # the schema simple while preserving the semantic label for queries/explanations).
        session.run(
            "UNWIND $edges AS e "
            "MATCH (a:Entity {id:e.src}), (b:Entity {id:e.dst}) "
            "MERGE (a)-[r:REL {type:e.rel}]->(b) SET r.sources=e.sources",
            edges=graph["edges"],
        )
        # Chunk -> Entity mentions (the join the vector store's graph_expand traverses)
        mention_rows = [
            {"cid": cid, "eid": eid}
            for cid, eids in graph.get("chunk_mentions", {}).items()
            for eid in eids
        ]
        if mention_rows:
            session.run(
                "UNWIND $rows AS r "
                "MATCH (c:Chunk {id:r.cid}), (e:Entity {id:r.eid}) MERGE (c)-[:MENTIONS]->(e)",
                rows=mention_rows,
            )

        counts = session.run(
            "MATCH (e:Entity) WITH count(e) AS ents "
            "MATCH (c:Chunk) WITH ents, count(c) AS chks "
            "MATCH ()-[r:REL]->() WITH ents, chks, count(r) AS rels "
            "MATCH ()-[m:MENTIONS]->() RETURN ents, chks, rels, count(m) AS mentions"
        ).single()

    drv.close()
    print(f"  Neo4j loaded: {counts['ents']} entities, {counts['chks']} chunks, "
          f"{counts['rels']} relationships, {counts['mentions']} mentions.")
    return counts["ents"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path, default=None)
    ap.add_argument("--chunks", type=Path, default=None)
    ap.parse_args()
    load()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
