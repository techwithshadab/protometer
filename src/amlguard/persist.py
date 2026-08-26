"""Durable artifact writing, and the identity that ties a run's records together.

Two small primitives with one shared motive: the repo's artifacts are its evidence, and
evidence must survive the machine being unkind.

`atomic_write_json` exists because every artifact writer except the graph-feature cache
used truncate-then-write: a disk-full or a kill mid-write left torn JSON where a committed
measurement used to be, and in one spot (the LLM response cache) a torn file crashed every
later run that touched the same prompt. Write-to-temp then `os.replace` makes a reader's
view all-or-nothing.

`RUN_ID` exists because the telemetry lived in three stores, MLflow runs, Langfuse traces,
and artifact JSON, with nothing joining them but wall-clock time. One id per process,
stamped into all three, turns "which prompts produced this number" from archaeology into a
query.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

# One id per process: every MLflow run, Langfuse record, and artifact written by this
# invocation carries it.
RUN_ID = uuid.uuid4().hex[:12]


def atomic_write_json(path: "Path | str", payload: Any, indent: int = 2) -> None:
    """Serialize then rename, so no reader ever sees a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps(payload, indent=indent, default=str))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def acquire_run_lock(root: "Path | str", name: str = "amlguard") -> Any:
    """One writer at a time across processes, or a clear refusal.

    Two concurrent ingests raced the manifest read-merge-write, double-paid discovery, and
    doubled logins against the separately rate-limited auth endpoint. An exclusive
    non-blocking flock turns that into an immediate, explainable exit. Returns the open
    file object, whose lifetime is the lock; raises RuntimeError when another run holds it.
    """
    import fcntl

    lock_path = Path(root) / f".{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(
            f"another run holds {lock_path}; wait for it to finish or remove the lock "
            f"if it is stale"
        ) from None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle
