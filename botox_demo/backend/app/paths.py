"""One place to resolve where crawled pages and the built index live.

The data dir must resolve identically whether the code runs from the host checkout
(`backend/app/...`, so `data/` is at the repo root) or inside the container (`WORKDIR /app`
with `app/` copied in, so `data/` is at `/app/data`). Hard-coding `parents[N]` breaks across
those two layouts because the module sits at a different depth. Instead we:

  1. honor an explicit BOTOX_DATA_DIR override (what a deployment sets), else
  2. walk up from this file until we find a directory that contains `data/` (or an `app/`
     sibling, marking the backend root), else
  3. fall back to CWD/data.

Every ingest/graph module imports DATA_DIR from here so there is a single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_data_dir() -> Path:
    override = os.getenv("BOTOX_DATA_DIR")
    if override:
        return Path(override).resolve()

    here = Path(__file__).resolve()
    # Walk up: the backend root is the dir that holds this `app/` package; `data/` sits beside it
    # (host: backend/data via repo root; container: /app/data). Prefer an existing `data/`, then
    # the package parent, so a first-time build still lands in the right place.
    for parent in here.parents:
        if (parent / "data").is_dir():
            return (parent / "data").resolve()
    # No data/ yet (first run before crawl): put it beside the `app/` package.
    pkg_parent = here.parent.parent  # .../app/paths.py -> .../app -> backend-root
    return (pkg_parent / "data").resolve()


DATA_DIR: Path = _resolve_data_dir()
RAW: Path = DATA_DIR / "raw"
PROCESSED: Path = DATA_DIR / "processed"
INDEX: Path = DATA_DIR / "index"
