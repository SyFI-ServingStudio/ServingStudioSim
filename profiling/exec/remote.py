"""Remote execution backend boundary for L1 profiling."""

from __future__ import annotations

from collections.abc import Iterator

from profiling.exec.pool import GpuChunk, GpuPool


class RemoteGpuPool(GpuPool):
    def __init__(self, target: str):
        self.target = target

    def acquire_chunks(self, k: int, max_concurrent: int = 1) -> Iterator[GpuChunk]:
        del k, max_concurrent
        raise NotImplementedError(
            "RemoteGpuPool protocol is intentionally left for the ops implementation"
        )
