"""Indexed row copy kernel kind: eager PyTorch advanced indexing over whole rows.

Recurrent-state models keep one large state row per request in a slot pool and
move rows with plain tensor indexing around prefill: ``rows = pool[indices]``
(gather), ``rows[~mask, ...] = 0`` (masked zero of fresh requests), and
``pool[indices] = rows`` (scatter). Eager PyTorch runs these on its indexing
kernels, whose cost at a few rows is not a byte rate: a single-row gather is
slower than a two-row one, and a masked zero adds a mask inversion and a
nonzero pass. A byte-keyed ``elementwise`` proxy missed them by 24-77%.

Wire string: ``"indexed_row_copy"`` — matches the Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/indexed_row_copy.rs``.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "indexed_row_copy"

OPERATIONS: tuple[str, ...] = ("gather", "scatter", "masked_zero")


@dataclass(frozen=True)
class IndexedRowCopyArgs(KernelArgs):
    num_rows: int = arg(unit="rows", doc="Rows read, written or zeroed.")
    row_elements: int = arg(unit="elements", doc="Elements in each row.")
    dtype: DType = arg(doc="Element type of the rows.")
    operation: str = arg(
        doc=(
            "gather reads indexed pool rows into a new tensor; scatter writes "
            "rows into indexed pool slots; masked_zero zeroes the rows a "
            "boolean mask leaves unset."
        )
    )


DOC = KernelDoc(
    title="Indexed row copy",
    summary="Gather, scatter or zero whole state rows by index with eager PyTorch.",
    description=(
        "The slot-pool bookkeeping around a recurrent-state prefill. vLLM "
        "gathers each prefilling request's state row from the pool by slot "
        "index, zeroes the rows of requests that start without a prior state, "
        "and scatters the final rows back to their slots, all as eager tensor "
        "indexing. Indices are int32 slots spread across a larger pool."
    ),
    category="Other",
    formula=(
        "bytes(gather, scatter) = 2·num_rows·row_elements·bytes(dtype)",
        "bytes(masked_zero) = num_rows·row_elements·bytes(dtype)",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Every launch of the indexing statement is counted, including the mask "
        "inversion and nonzero pass of masked_zero. Pools, rows and indices are "
        "allocated before the capture."
    ),
    caveats=(
        "masked_zero zeroes all num_rows rows; a step whose mask keeps some rows "
        "is priced at the number it zeroes.",
        "Pool rows are contiguous; a pool whose slot stride pads the row is not modeled.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.FP32, DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.indexed_row_copy_torch",
            function_name="profile_indexed_row_copy",
        ),
        table_name=KIND,
        args_schema=IndexedRowCopyArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "Eager PyTorch advanced indexing: index for gather, index_put_ "
                "for scatter and for the masked zero."
            ),
        ),
    )
)
