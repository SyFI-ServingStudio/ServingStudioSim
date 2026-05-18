"""KernelKind — dispatch alphabet for the (kernel_kind, backend) registry.

Per L1 design.md §2.1: every runner is registered under a ``(KernelKind,
backend)`` pair; the value of each variant is the snake_case wire string used
in the ``profile.db`` ``kind`` column and in PyO3 marshaling to the Rust
bridge.

Phase 0 ships only the **class declaration** — no variants. Each variant is a
payload that lands together with the runner it dispatches to, by adding a line
to the class body in this file (Phase 1+). Keeping the class here reserves the
import path so downstream modules can already write
``from profiling.db.kind import KernelKind`` without a stub error.
"""

from __future__ import annotations

from enum import Enum


class KernelKind(str, Enum):
    """str-valued enum. Variants are filled in by L1 runners as they land."""

    def __str__(self) -> str:
        # Return the raw wire string ("gemm_single") rather than the default
        # "KernelKind.GEMM_SINGLE" repr. Keeps DB writes, PyO3 marshal across
        # Py↔Rust, and log lines using the canonical value.
        return self.value
