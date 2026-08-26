"""Library logging: timestamped, stderr, configured once.

Degradation notices used to be bare prints to stdout: untimestamped (a hung 650-second
discovery loop could not be dated from its own output), interleaved with result tables,
and invisible to anything that captured only stderr. Progress output that a human reads
as the product of a script (tables, per-task lines) stays on stdout via print; anything
that describes the system's own health goes through here.
"""

from __future__ import annotations

import logging
import os
import sys

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s",
                              datefmt="%Y-%m-%dT%H:%M:%S")
        )
        root = logging.getLogger("amlguard")
        root.addHandler(handler)
        root.setLevel(os.getenv("AMLGUARD_LOG_LEVEL", "INFO").upper())
        root.propagate = False
        _configured = True
    return logging.getLogger(f"amlguard.{name}")
