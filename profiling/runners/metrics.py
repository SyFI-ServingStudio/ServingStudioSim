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
    time_ms: float
    algbw_gbps: float
    busbw_gbps: float
    message_size_bytes: int
    energy_j: float = 0.0


Metrics = ComputeMetrics | CommMetrics
