"""One-command ingest: chunk -> extract graph -> load Neo4j -> build vector index.

Runs the full offline ($0, no LLM) indexing pipeline in dependency order. Neo4j is OPTIONAL: if it
is unreachable the pipeline still builds chunks, the entity graph (graph.json), and the vector
index, so the vector-retrieval path works standalone; only cross-chunk graph EXPANSION needs Neo4j
(and even that falls back to the local graph.json map). This is what the container entrypoint runs
after the crawl, and what a developer runs by hand to (re)build the index.

    python -m app.ingest.build_all          # assumes data/raw/pages.jsonl already exists
    python -m app.ingest.build_all --crawl  # crawl first, then build everything

Env: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD (Neo4j), EMBED_MODEL (embeddings).
"""

from __future__ import annotations

import argparse
import sys


def run(do_crawl: bool = False, crawl_limit: int = 40) -> int:
    if do_crawl:
        print("[1/6] crawling botox.com (robots-respecting)...")
        from app.ingest.crawl import crawl
        crawl(limit=crawl_limit)

    print("[2/6] chunking pages...")
    from app.ingest.chunk import build_chunks
    n_chunks = build_chunks()

    # No chunks means the crawl produced nothing, typically an OFFLINE run (every fetch failed) or a
    # first run with no `--crawl`. Stop here with a clear message rather than letting build_index
    # raise ValueError("no chunks to index") with a bare traceback.
    if n_chunks == 0:
        print("  no chunks were produced, nothing to index.")
        if do_crawl:
            print("  the crawl fetched no pages (offline, or the site was unreachable). Re-run with "
                  "a working network connection.")
        else:
            print("  run with --crawl first to fetch pages, or ensure data/raw/pages.jsonl exists.")
        return 1

    print("[3/6] extracting knowledge graph (entities + relationships)...")
    from app.graph.extract import extract
    graph = extract()

    print("[4/6] loading graph into Neo4j (optional)...")
    from app.graph.build_neo4j import load
    try:
        load()
    except FileNotFoundError as exc:
        print(f"  {exc}")
    except Exception as exc:  # noqa: BLE001, a Neo4j failure must not fail the whole build
        print(f"  Neo4j load skipped ({type(exc).__name__}: {str(exc)[:80]}); "
              f"local graph expansion will be used.")

    print("[5/6] building vector index (embeddings)...")
    from app.graph.vectorstore import build_index
    n_indexed = build_index()

    print("[6/6] seeding the system prompt into Langfuse (optional)...")
    # Push the hardcoded prompt to Langfuse Prompt Management so it's versioned and editable there;
    # the app prefers the Langfuse copy at runtime and falls back to the hardcoded text otherwise.
    # Best-effort: a Langfuse outage must not fail the build (same policy as the Neo4j step).
    try:
        from app.obs.seed_prompt import main as seed_prompt
        seed_prompt()
    except Exception as exc:  # noqa: BLE001
        print(f"  prompt seed skipped ({type(exc).__name__}: {str(exc)[:80]}); "
              f"the app will use the hardcoded prompt.")

    print(
        f"\nDone. {n_chunks} chunks, {len(graph['nodes'])} entities, "
        f"{len(graph['edges'])} relationships, {n_indexed} vectors."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crawl", action="store_true", help="crawl botox.com before building")
    ap.add_argument("--limit", type=int, default=40, help="crawl page limit")
    args = ap.parse_args()
    try:
        return run(do_crawl=args.crawl, crawl_limit=args.limit)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
