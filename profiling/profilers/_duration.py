"""Shared helpers for duration-targeted profiling loops (Timer + Energy).

Both the timer and energy paths can size their measurement loop from a target
wall-clock budget (``min_duration_ms``) instead of a fixed iteration count:
estimate the per-iteration cost, then run enough iterations to cover the budget.

Agent note: because each process sizes that loop from its own local timing
estimate, the resulting iteration count is non-deterministic across processes.
Multi-GPU / collective runs where ranks must execute in lockstep MUST use a
fixed ``rep`` instead -- see ``warn_if_multi_gpu_duration_mode``.
"""

from __future__ import annotations

import importlib
import math
import warnings
from typing import Any

# Time-centric defaults shared by Timer and Energy: run for at least
# DEFAULT_MIN_DURATION_MS of wall clock, but never fewer than DEFAULT_MIN_REP
# iterations (so an ultra-slow kernel still gets a few reps for the median).
DEFAULT_MIN_DURATION_MS = 500
DEFAULT_MIN_REP = 3


def iters_for_duration(min_duration_ms: int, per_iter_time_ms: float) -> int:
    """Iterations needed to cover ``min_duration_ms`` at the measured rate."""

    if not math.isfinite(per_iter_time_ms) or per_iter_time_ms <= 0:
        raise ValueError("per_iter_time_ms must be finite and positive")
    if min_duration_ms <= 0:
        return 1
    return max(math.ceil(float(min_duration_ms) / per_iter_time_ms), 1)


def warn_if_multi_gpu_duration_mode(context: str) -> None:
    """Warn that duration-targeted iteration counts desync across ranks.

    ``min_duration_ms`` sizes the loop from a per-process timing estimate, so two
    ranks can pick different iteration counts. For collective/multi-GPU work that
    desyncs ranks (deadlock on collectives, or incomparable rows), so those runs
    must pass a fixed ``rep`` instead. We key off an initialized process group
    with ``world_size > 1`` rather than the visible device count, because a
    single-GPU job on a multi-GPU node is perfectly safe.
    """

    world_size = _distributed_world_size()
    if world_size <= 1:
        return
    warnings.warn(
        f"{context}: min_duration_ms sizes iterations from a per-process timing "
        f"estimate and is non-deterministic across the {world_size} ranks in this "
        "process group; multi-GPU/collective runs must pass a fixed rep instead",
        RuntimeWarning,
        stacklevel=3,
    )


def _distributed_world_size() -> int:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return 0
    dist: Any = getattr(torch, "distributed", None)
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return 0
    try:
        return int(dist.get_world_size())
    except Exception:
        return 0
