#!/usr/bin/env python3
"""Verify the hard-coded counts in the diagram HTML match the actual built artifacts.

The diagrams state concrete numbers ("102 chunks", "47 entities · 233 relationships"). Those are
hand-typed into the HTML, so they drift the moment the corpus is rebuilt. This check reads the real
artifacts (data/index/meta.json, data/processed/graph.json) and fails if any diagram's numbers are
stale, enforcing the project rule: docs claim only what an artifact backs. When it fails, update the
numbers in the named diagrams (and re-render), or re-run ingest so the artifacts and prose agree.

    python docs/diagrams/src/check_counts.py     # exits non-zero on any mismatch

It reports the artifact counts so you know exactly what to write.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]        # botox_demo/
SRC = Path(__file__).resolve().parent             # docs/diagrams/src/
META = ROOT / "data" / "index" / "meta.json"
GRAPH = ROOT / "data" / "processed" / "graph.json"


def artifact_counts() -> dict[str, int] | None:
    """(chunks, entities, relationships) from the built artifacts, or None if not built yet."""
    if not META.exists() or not GRAPH.exists():
        return None
    meta = json.loads(META.read_text())
    graph = json.loads(GRAPH.read_text())
    return {
        "chunks": len(meta),
        "entities": len(graph.get("nodes", [])),
        "relationships": len(graph.get("edges", [])),
    }


def main() -> int:
    counts = artifact_counts()
    if counts is None:
        print(f"SKIP: artifacts not built ({META} / {GRAPH} missing). Run ingest first.")
        return 0
    ch, en, rel = counts["chunks"], counts["entities"], counts["relationships"]
    print(f"artifacts: {ch} chunks, {en} entities, {rel} relationships")

    problems: list[str] = []

    # architecture-overview.html: "<N> chunks embedded" and "<E> entities · <R> relationships"
    ao = (SRC / "architecture-overview.html").read_text()
    for label, pat, want in (
        ("chunks", r"(\d+) chunks embedded", ch),
        ("entities", r"(\d+) entities", en),
        ("relationships", r"entities · (\d+) relationships", rel),
    ):
        m = re.search(pat, ao)
        if not m or int(m.group(1)) != want:
            got = m.group(1) if m else "MISSING"
            problems.append(f"architecture-overview.html {label}: says {got}, artifact has {want}")

    # ingest-pipeline.html: "chunks.jsonl · <N> chunks" and "graph.json · <E> · <R>"
    ip = (SRC / "ingest-pipeline.html").read_text()
    for label, pat, want in (
        ("chunks", r"chunks\.jsonl · (\d+) chunks", ch),
        ("entities", r"graph\.json · (\d+) · \d+", en),
        ("relationships", r"graph\.json · \d+ · (\d+)", rel),
    ):
        m = re.search(pat, ip)
        if not m or int(m.group(1)) != want:
            got = m.group(1) if m else "MISSING"
            problems.append(f"ingest-pipeline.html {label}: says {got}, artifact has {want}")

    # KPI-score count: the diagram claims "<N> KPI scores"; the truth is the distinct set of
    # turn.score(...) names in the orchestrator plus `user_feedback` from the feedback path. Counting
    # them from source keeps the diagram honest as scores are added/removed.
    orch = (ROOT / "backend" / "app" / "pipeline" / "orchestrator.py").read_text()
    score_names = set(re.findall(r'turn\.score\("([a-z_]+)"', orch))
    trace = (ROOT / "backend" / "app" / "obs" / "tracing.py").read_text()
    if 'name="user_feedback"' in trace:
        score_names.add("user_feedback")
    kpi = len(score_names)
    print(f"code: {kpi} distinct KPI scores")
    m = re.search(r"(\d+) KPI scores", ao)
    if not m or int(m.group(1)) != kpi:
        got = m.group(1) if m else "MISSING"
        problems.append(f"architecture-overview.html KPI scores: says {got}, code emits {kpi}")

    if problems:
        print("\nSTALE diagram/doc counts (fix the HTML and re-render):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK: all diagram counts match the artifacts and code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
