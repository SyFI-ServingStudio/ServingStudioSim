"""KernelKind — dispatch alphabet for the (kernel_kind, backend) registry.

Per L1 design.md §2.1: every runner is registered under a ``(KernelKind,
backend)`` pair; the value of each variant is the snake_case wire string used
in the ``profile.db`` ``kind`` column and in PyO3 marshaling to the Rust
bridge.

The current implemented variant is ``GEMM_SINGLE = "gemm_single"``. Future
variants land with their runner and ``KernelProfilerSpec`` entry.
"""

from __future__ import annotations

from enum import StrEnum


class KernelKind(StrEnum):
    """str-valued enum. Variants are filled in by L1 runners as they land."""

    GEMM_SINGLE = "gemm_single"

    def __str__(self) -> str:
        # Return the raw wire string ("gemm_single") rather than the default
        # "KernelKind.GEMM_SINGLE" repr. Keeps DB writes, PyO3 marshal across
        # Py↔Rust, and log lines using the canonical value.
        return self.value
