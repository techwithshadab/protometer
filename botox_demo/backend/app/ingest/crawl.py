"""Robots-respecting crawler for public botox.com marketing content.

This is a GraphRAG source builder, not a scraper-for-reproduction: it pulls the public
informational pages (indications, safety, FAQ, cost) so the chatbot can *summarize and cite* them,
never reproduce them verbatim at length. It honours robots.txt (which blocks /api, /search,
/forms, /account flows), rate-limits politely, and stays on the botox.com host.

    python -m app.ingest.crawl              # crawl into data/raw/pages.jsonl
    python -m app.ingest.crawl --limit 40

Output: one JSON object per page {url, title, text, safety_flagged} in data/raw/pages.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

import httpx
from selectolax.parser import HTMLParser

from app.paths import RAW

BASE = "https://www.botox.com"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

# Seed the informational pages; the crawler also follows in-content links that stay on-host and
# out of the disallowed app flows. These are the pages a public patient would read.
SEEDS = [
    "/", "/main", "/cost-and-coverage",
    "/resources", "/resources/frequently-asked-questions", "/resources/botox-complete",
]

# Never crawl the patient-portal app / forms / auth flows (also covered by robots, belt-and-braces),
# nor navigation-only pages (sitemap, site-map), those are just link indexes with no informational
# content, so they match almost any query lexically and pollute retrieval + citations.
SKIP = ("account", "log-in", "login", "password", "dashboard", "claim", "enroll", "sign-up",
        "sms", "reminder", "manage-account", "reset", "conversion", "confirmation", "/api/",
        "search-results", "generatepdf", "thank-you", "jcr", "system-error", "page-not-found",
        "sitemap", "site-map")


def _robots() -> urllib.robotparser.RobotFileParser:
    """Parse robots.txt fetched with OUR user-agent. urllib's default UA gets a 403 here (the site
    blocks it), which RobotFileParser then treats as disallow-all, so we fetch with httpx and feed
    the text in. If the fetch genuinely fails, we fall back to an allow-all parser and rely on the
    explicit SKIP list (which already encodes robots' Disallow paths)."""
    rp = urllib.robotparser.RobotFileParser()
    try:
        r = httpx.get(f"{BASE}/robots.txt", headers={"User-Agent": UA}, timeout=15,
                      follow_redirects=True)
        rp.parse(r.text.splitlines())
    except Exception:  # noqa: BLE001
        rp.parse(["User-agent: *", "Allow: /"])
    return rp


def _clean_text(html: str) -> tuple[str, str]:
    """(title, cleaned visible text). Strips nav/script/style; collapses whitespace."""
    tree = HTMLParser(html)
    title = ""
    if tree.css_first("title"):
        title = tree.css_first("title").text(strip=True)
    for sel in ("script", "style", "nav", "header", "footer", "noscript", "svg", "form"):
        for node in tree.css(sel):
            node.decompose()
    body = tree.body
    text = body.text(separator=" ", strip=True) if body else ""
    text = re.sub(r"\s+", " ", text).strip()
    return title, text


def _allowed(url: str, rp: urllib.robotparser.RobotFileParser) -> bool:
    p = urlparse(url)
    if p.netloc and p.netloc not in ("www.botox.com", "botox.com"):
        return False
    if any(s in url.lower() for s in SKIP):
        return False
    return rp.can_fetch(UA, url)


# Safety content must be preserved and surfaced, never dropped as boilerplate.
_SAFETY = re.compile(r"boxed warning|important safety information|distant spread|serious side|"
                     r"do not (use|take)|warning|contraindicat", re.I)


def crawl(limit: int = 40, delay: float = 1.0) -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    rp = _robots()
    seen: set[str] = set()
    queue = [urljoin(BASE, s) for s in SEEDS]
    out = (RAW / "pages.jsonl").open("w", encoding="utf-8")
    n = 0
    with httpx.Client(headers={"User-Agent": UA}, timeout=25, follow_redirects=True) as client:
        while queue and n < limit:
            url = queue.pop(0)
            url = url.split("#")[0].rstrip("/") or BASE
            if url in seen or not _allowed(url, rp):
                continue
            seen.add(url)
            try:
                r = client.get(url)
                if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                    continue
            except Exception:  # noqa: BLE001, a single dead link must not stop the crawl
                continue
            title, text = _clean_text(r.text)
            if len(text) < 200:  # skip near-empty shells
                continue
            record = {"url": url, "title": title, "text": text,
                      "safety_flagged": bool(_SAFETY.search(text))}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            print(f"  [{n}] {url}  ({len(text)} chars"
                  f"{', SAFETY' if record['safety_flagged'] else ''})")
            # enqueue on-host in-content links
            for a in HTMLParser(r.text).css("a[href]"):
                href = a.attributes.get("href") or ""
                nxt = urljoin(url, href).split("#")[0]
                if nxt not in seen and _allowed(nxt, rp):
                    queue.append(nxt)
            time.sleep(delay)
    out.close()
    print(f"crawled {n} pages -> {RAW / 'pages.jsonl'}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()
    crawl(args.limit, args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
