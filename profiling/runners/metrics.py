"""L1a runner return types.

Runners return these dataclasses and nothing DB-shaped. L1b owns persistence
and query policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeMetrics:
    time_ms: float
    tflops: float
    memory_bandwidth_gbps: float
    energy_j: float = 0.0


@dataclass(frozen=True)
class CommMetrics:
    # message_size is an args/sweep axis (the cache key), NOT a measured result,
    # so it is not carried here. The simulator derives a collective's moved bytes
    # from `busbw_gbps × time` (the real per-GPU fabric traffic), so the result
    # only needs the measured rates + time.
    time_ms: float
    algbw_gbps: float
    busbw_gbps: float
    energy_j: float = 0.0


Metrics = ComputeMetrics | CommMetrics


@dataclass(frozen=True)
class RunnerResult:
    """One item of a list-runner's return value.

    The runner contract is ``run(kwargs_list) -> list[RunnerResult]``, 1:1 and
    in-order with the input specs. A success carries ``metrics`` (and ``error is
    None``); a per-item failure carries ``error`` (and ``metrics is None``). This
    keeps per-item ok/error resilience inside the runner layer (the ``batched``
    adapter for compute, or the native comm parent) instead of the worker loop.
    Distinct from L1b's ``ChunkResult``, which additionally carries ``gpu_name``.
    """

    metrics: Metrics | None = None
    error: str | None = None
