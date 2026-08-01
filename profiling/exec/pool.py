"""Execution backend contracts for L1 profiling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from profiling.db.kind import KernelKind
from profiling.runners.metrics import Metrics


@dataclass(frozen=True)
class ChunkResult:
    metrics: Metrics | None
    # Physical GPU the executing worker actually ran on, when observable.
    # Every successful execution must provide this provenance. It becomes the DB
    # key only when the controller did not request an explicit key.
    observed_gpu_name: str | None = None
    error: str | None = None


class GpuChunk(ABC):
    """Short-lived reservation for one profiling chunk."""

    @abstractmethod
    def run(self, kernel_kind: KernelKind, specs: list[dict]) -> list[ChunkResult]:
        raise NotImplementedError


class GpuPool(ABC):
    """Long-lived owner of local GPU slots or remote cluster quota."""

    @abstractmethod
    def acquire_chunks(self, k: int, max_concurrent: int) -> Iterator[GpuChunk]:
        raise NotImplementedError
