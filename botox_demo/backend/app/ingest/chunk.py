"""Chunk crawled pages into overlapping, sentence-aware passages for retrieval.

Why sentence-aware overlap: the botox.com pages are long (30k-80k chars) and mix marketing prose
with dense clinical safety text. A naive fixed-width split would cut mid-sentence and orphan a
safety clause from its subject ("...may cause serious side effects that" | "can be life
threatening"). We split on sentence boundaries and keep a ~100-char overlap so a passage that
starts mid-topic still carries the tail of the previous one, which keeps retrieval coherent and
prevents a safety warning from being severed from what it warns about.

Why per-chunk safety re-flagging: `safety_flagged` on the page is page-level (every botox.com page
carries the ISI footer, so it is almost always True and therefore useless for ranking). We re-flag
each CHUNK against a safety regex so the query pipeline can (a) always surface a real safety chunk
when the topic is safety, and (b) attach the Important Safety Information callout only when the
retrieved passage is genuinely safety-bearing.

    python -m app.ingest.chunk           # data/raw/pages.jsonl -> data/processed/chunks.jsonl

Output: one JSON object per chunk: {id, url, title, text, safety}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from app.paths import PROCESSED, RAW

# Target chunk sizing (characters). Sentence-aware, so these are soft targets: a chunk grows to the
# next sentence boundary and stops once it exceeds MAX; OVERLAP chars of the tail seed the next.
TARGET = 650
MAX = 800
OVERLAP = 120
MIN_CHUNK = 120  # drop slivers smaller than this (nav crumbs, stray labels)

# Safety-relevant content. Kept deliberately broad on the recall side: for a pharma assistant it is
# far worse to miss a warning than to over-flag. These map to the real ISI language on the site
# (boxed warning, distant spread of toxin, swallowing/breathing problems, contraindications).
SAFETY_RE = re.compile(
    r"\b("
    r"boxed warning|important safety information|serious side effect|life[- ]threatening|"
    r"side effect|adverse|do not (?:use|take|receive)|should not (?:use|take|receive)|"
    r"warning|contraindicat|spread of toxin|distant spread|"
    r"trouble (?:swallowing|breathing|speaking)|difficulty (?:swallowing|breathing)|"
    r"allergic reaction|tell your doctor|medication guide|prescribing information"
    r")\b",
    re.IGNORECASE,
)

# Residual boilerplate the crawler's text extraction leaves behind: AEM component markers, repeated
# nav labels, and the cookie/UI scaffolding that carries no answerable information.
_BOILERPLATE = re.compile(
    r"(Container \d+ ClassName: [a-z0-9-]+|Condition Type:|Insurance Type:|"
    r"Skip to Main content|ClassName: c\d+-i\d+)",
    re.IGNORECASE,
)

# Sentence boundary: end punctuation followed by whitespace and a capital/number/quote. We keep it
# conservative so decimals ("$182.46") and abbreviations don't over-split; the size cap is the real
# backstop against runaway chunks.
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"'“])")


def _clean(text: str) -> str:
    text = _BOILERPLATE.sub(" ", text)
    # Normalise the ® spacing the site uses inconsistently ("BOTOX ®" -> "BOTOX®") so entity
    # matching downstream sees a stable surface form.
    text = re.sub(r"\bBOTOX\s+®", "BOTOX®", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_id(url: str, idx: int, text: str) -> str:
    """Stable, content-addressed id so re-running the pipeline yields the same ids (idempotent
    downstream graph MERGE / vector upsert). Combines url + position + a short content hash."""
    h = hashlib.sha1(f"{url} {idx} {text[:80]}".encode()).hexdigest()[:10]
    return f"c_{h}"


def chunk_text(text: str) -> list[str]:
    """Split cleaned text into overlapping, sentence-aware chunks."""
    sentences = _sentences(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sentences:
        # A single monster "sentence" (rare, e.g. an un-split list) is hard-wrapped at MAX so it
        # can't blow past the embedding model's comfortable window.
        if len(sent) > MAX:
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            for i in range(0, len(sent), MAX - OVERLAP):
                chunks.append(sent[i : i + MAX])
            continue
        if cur_len + len(sent) + 1 > MAX and cur:
            chunks.append(" ".join(cur))
            # Seed the next chunk with an overlap tail from the end of the current one.
            tail, tlen = [], 0
            for s in reversed(cur):
                if tlen + len(s) > OVERLAP:
                    break
                tail.insert(0, s)
                tlen += len(s) + 1
            cur, cur_len = list(tail), tlen
        cur.append(sent)
        cur_len += len(sent) + 1
        if cur_len >= TARGET:
            chunks.append(" ".join(cur))
            tail, tlen = [], 0
            for s in reversed(cur):
                if tlen + len(s) > OVERLAP:
                    break
                tail.insert(0, s)
                tlen += len(s) + 1
            cur, cur_len = list(tail), tlen
    if cur:
        chunks.append(" ".join(cur))
    # De-dupe: the overlap machinery plus near-identical pages (/, /main) can yield exact repeats.
    seen: set[str] = set()
    out: list[str] = []
    for c in chunks:
        c = c.strip()
        if len(c) < MIN_CHUNK:
            continue
        key = c[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def build_chunks(pages_path: Path | None = None, out_path: Path | None = None) -> int:
    pages_path = pages_path or (RAW / "pages.jsonl")
    out_path = out_path or (PROCESSED / "chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pages_path.exists():
        raise FileNotFoundError(
            f"{pages_path} not found, run `python -m app.ingest.crawl` first."
        )

    n_pages = 0
    n_chunks = 0
    n_safety = 0
    global_seen: set[str] = set()  # cross-page de-dupe (/, /main are identical)
    with out_path.open("w", encoding="utf-8") as out:
        for line in pages_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            page = json.loads(line)
            n_pages += 1
            cleaned = _clean(page["text"])
            for idx, ch in enumerate(chunk_text(cleaned)):
                key = ch[:200]
                if key in global_seen:
                    continue
                global_seen.add(key)
                # The page-level flag is useless for ranking (every botox.com page carries the ISI
                # footer, so it is almost always True). We flag purely on the chunk's own content.
                safety = bool(SAFETY_RE.search(ch))
                rec = {
                    "id": _chunk_id(page["url"], idx, ch),
                    "url": page["url"],
                    "title": page["title"],
                    "text": ch,
                    "safety": safety,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1
                n_safety += int(safety)

    print(
        f"chunked {n_pages} pages -> {n_chunks} chunks "
        f"({n_safety} safety-flagged) -> {out_path}"
    )
    return n_chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.parse_args()
    build_chunks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
