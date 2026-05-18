"""KernelArgs — abstract base for per-kind args dataclasses.

Per L1 design.md §2.1: each ``(kernel_kind, backend)`` registry entry pairs a
runner function with a ``KernelArgs`` subclass. Field names == runner kwargs ==
DB key columns (strict three-way equality, enforced via framework reflection
in Phase 1).

This module ships only the **abstract contract** (the frozen-dataclass base
class + field-name conventions). Concrete per-kind subclasses
(``SingleGemmArgs`` / ``AttnPrefillArgs`` / ``AllReduceArgs`` / ...) are
**payloads**, not infra, and land alongside their respective L1 runners — see
``profiling/runners/{gemm,attention,comm,norm,...}/`` and the corresponding
registry entries in Phase 1+. The ``KernelKind`` dispatch enum is likewise a
Phase 1 concern: it gets authored with the variants the first runner needs
and grown as more runners arrive.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelArgs:
    """Base class for per-kernel-kind argument records.

    Subclasses only declare fields — no methods. Field name conventions
    (per L1 design.md §2.2):

    - identical to runner kwargs and to DB key column names;
    - declaration order = DB column order;
    - all fields frozen (the dataclass is hashable so it can key the cache).
    """
