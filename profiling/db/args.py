"""Typed argument records for L1 profile entries.

Per L1 design.md §2.1 / §2.2, each ``KernelProfilerSpec`` points at one
``KernelArgs`` subclass. Its field names are the shared contract between public
spec dicts, runner keyword arguments, and DB key columns.

Agent note: add concrete ``KernelArgs`` subclasses here, not in runners or the
registry. These classes are schema records only; sweep grids, cache policy,
table names, and backend behavior belong to Rust kernels or L1b registry/table
code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DType(StrEnum):
    """Canonical dtype strings used by profile DB rows and runner kwargs."""

    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"

    @classmethod
    def from_value(cls, value: Any) -> DType:
        if isinstance(value, cls):
            return value
        normalized = str(value).lower()
        aliases = {
            "float16": cls.FP16,
            "torch.float16": cls.FP16,
            "half": cls.FP16,
            "bfloat16": cls.BF16,
            "torch.bfloat16": cls.BF16,
            "float32": cls.FP32,
            "torch.float32": cls.FP32,
        }
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)

    def size_bytes(self) -> float:
        return {
            DType.FP16: 2,
            DType.BF16: 2,
            DType.FP32: 4,
            DType.FP8_E4M3: 1,
            DType.FP8_E5M2: 1,
            DType.INT8: 1,
            DType.INT4: 0.5,
        }[self]

    # Keep framework conversions lazy so importing DB schemas does not import
    # heavyweight runner libraries.
    def torch(self):
        import torch

        mapping = {
            DType.FP16: torch.float16,
            DType.BF16: torch.bfloat16,
            DType.FP32: torch.float32,
        }
        try:
            return mapping[self]
        except KeyError as exc:
            raise ValueError(f"{self.value} is not supported by the torch runner") from exc


@dataclass(frozen=True)
class KernelArgs:
    """Base class for per-kernel-kind argument records.

    Subclasses only declare fields — no methods. Field name conventions
    (per L1 design.md §2.2):

    - identical to runner kwargs and to DB key column names;
    - declaration order = DB column order;
    - all fields frozen so args can be reused as immutable DB/query identities.
    """


# Concrete args begin below. New classes should stay field-only and mirror the
# runner kwargs exactly; registry rows only reference them.
@dataclass(frozen=True)
class SingleGemmArgs(KernelArgs):
    m: int
    n: int
    k: int
    dtype: DType
