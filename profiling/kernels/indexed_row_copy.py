"""Indexed row copy kernel kind: eager PyTorch indexed copies of whole rows.

Recurrent-state models keep one large state row per request in a slot pool and
move rows with plain tensor indexing around prefill: ``rows = pool[indices]``
(gather), ``rows[~mask, ...] = 0`` (masked zero of fresh requests), and
``pool[indices] = rows`` (scatter). Eager PyTorch runs these on its indexing
kernels, whose cost at a few rows is not a byte rate: a single-row gather is
slower than a two-row one, and a masked zero adds a mask inversion and a
nonzero pass. A byte-keyed ``elementwise`` proxy missed them by 24-77%.

The same statements move token rows when a step mixes speculative-decode rows
with plain rows: ``x.index_select(0, indices)`` splits the projection rows and
``out.index_copy_(0, indices, rows)`` merges the core outputs back. PyTorch runs
``index_select`` on ``indexSelectSmallIndex`` up to 16 indices (time grows with
each index) and on ``vectorized_gather_kernel`` above; ``index_copy_`` runs
``index_copy_kernel_impl``. On B200 the bitwise_not byte pass missed them by
0.17-1.7x and the strided-copy pass by 0.85-3.7x.

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

OPERATIONS: tuple[str, ...] = ("gather", "scatter", "masked_zero", "index_select", "index_copy")


@dataclass(frozen=True)
class IndexedRowCopyArgs(KernelArgs):
    num_rows: int = arg(unit="rows", doc="Rows read, written or zeroed.")
    row_elements: int = arg(unit="elements", doc="Elements in each row.")
    dtype: DType = arg(doc="Element type of the rows.")
    operation: str = arg(
        doc=(
            "gather reads indexed pool rows into a new tensor; scatter writes "
            "rows into indexed pool slots; masked_zero zeroes the rows a "
            "boolean mask leaves unset; index_select reads sorted token rows "
            "of a tensor; index_copy writes rows to sorted token rows."
        )
    )


DOC = KernelDoc(
    title="Indexed row copy",
    summary="Gather, scatter or zero whole rows by index with eager PyTorch.",
    description=(
        "The slot-pool bookkeeping around a recurrent-state prefill. vLLM "
        "gathers each prefilling request's state row from the pool by slot "
        "index, zeroes the rows of requests that start without a prior state, "
        "and scatters the final rows back to their slots, all as eager tensor "
        "indexing. Indices are int32 slots spread across a larger pool. The "
        "index_select and index_copy operations are the token-row split and "
        "merge of a step mixing speculative and plain rows, with sorted int64 "
        "token indices."
    ),
    category="Other",
    formula=(
        "bytes(gather, scatter, index_select, index_copy) = 2·num_rows·row_elements·bytes(dtype)",
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


# `torch` runs in the main environment; `torch_vllm` is the same runner in
# vLLM's pinned environment, for a model whose rows must come from the
# serving build.
for _backend, _env in (("torch", None), ("torch_vllm", "vllm_env")):
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            supports=BackendSupport(compute=frozenset({DType.FP32, DType.BF16})),
            runner_ref=RunnerRef(
                module_name="profiling.runners.elementwise.indexed_row_copy_torch",
                function_name="profile_indexed_row_copy",
            ),
            table_name=KIND,
            args_schema=IndexedRowCopyArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env=_env,
            doc=BackendDoc(
                summary=(
                    "Eager PyTorch advanced indexing: index for gather, index_put_ "
                    "for scatter and for the masked zero, index_select and "
                    "index_copy_ for the token-row operations."
                ),
            ),
        )
    )
