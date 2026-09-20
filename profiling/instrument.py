"""Opt-in timeline of where a profiling run spends its wall clock.

A clean tp4/ep4 fill on four B200s takes 1564 s without the energy window. Its
CUPTI trace accounts for 2644 of the 6256 GPU-seconds; sampling `nvidia-smi`
during the run put mean occupancy at 56% with all four cards idle together 20%
of the time. So more than half the run is neither measurement nor per-spec
setup, and subtracting one estimate from another has already produced three
wrong answers about what it is. This records the spans directly instead.

Off unless ``VIBESIM_PROFILE_TIMELINE`` names a file. When it does, every span
appends one JSON line:

    {"t0": 1758..., "t1": 1758..., "pid": 123, "phase": "chunk.run", ...}

Timestamps are ``time.time()``, not ``perf_counter()``: the spans come from the
parent, from worker subprocesses, and from inside containers, and only a wall
clock is comparable across all three. Lines are written with a single
``os.write`` of under ``PIPE_BUF`` to an ``O_APPEND`` descriptor, which Linux
guarantees to be atomic, so concurrent chunk workers cannot interleave.

The path is handed to containers explicitly, like the other measurement-policy
variables, and its directory is mounted -- see ``exec/local.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator
from typing import Any

TIMELINE_ENV = "VIBESIM_PROFILE_TIMELINE"

_MAX_LINE = 4096  # PIPE_BUF: the size Linux promises to append atomically.


def timeline_path() -> str | None:
    path = os.environ.get(TIMELINE_ENV)
    return path if path else None


def emit(phase: str, t0: float, t1: float, **fields: Any) -> None:
    """Append one span. Never raises: a diagnostic must not fail a fill."""

    path = timeline_path()
    if path is None:
        return
    record = {"t0": round(t0, 6), "t1": round(t1, 6), "pid": os.getpid(), "phase": phase}
    record.update(fields)
    try:
        line = (json.dumps(record, default=str) + "\n").encode()
        if len(line) > _MAX_LINE:
            # Drop the payload rather than risk a torn line from a second write.
            line = (
                json.dumps(
                    {"t0": record["t0"], "t1": record["t1"], "pid": record["pid"], "phase": phase}
                )
                + "\n"
            ).encode()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)
    except OSError:
        return


@contextlib.contextmanager
def span(phase: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block and append it. The yielded dict accepts late fields, so a
    span can record something it only learns by running (a spec count, whether
    the row came from cache)."""

    extra: dict[str, Any] = {}
    started = time.time()
    try:
        yield extra
    finally:
        emit(phase, started, time.time(), **{**fields, **extra})
