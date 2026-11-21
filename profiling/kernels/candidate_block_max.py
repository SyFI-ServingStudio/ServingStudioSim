"""DeepInfer's candidate block scores (``candidate_block_max``).

Wire string: ``"candidate_block_max"`` -- matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/candidate_block_max.rs``.

On the candidate source layer DeepInfer (``index_attn.py``
``_topk_candidate_block``) reduces each row's dense bf16 logits to one score
per ``block_size`` keys (``tilelang_ops/candidate_blocks.py``
``candidate_block_max``: a task-list launch, then the block-max launch; the
last valid block scores +inf). DeepSelect then keeps the top blocks
(``deep_select_topk``).

Why a new kind: a strided block max over a ragged logits matrix, with its
task-list launch, matches no existing reduction kind.

Key: one request's causal-tail chunk (as ``mqa_logits_prefill_chunk``) and
the block size.
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

KIND = "candidate_block_max"


@dataclass(frozen=True)
class CandidateBlockMaxArgs(KernelArgs):
    num_queries: int = arg(unit="tokens", doc="Logits rows of the chunk.")
    context_len: int = arg(
        unit="tokens", doc="Context of the request, the rows included; the rows are its last tokens."
    )
    compress_ratio: int = arg(unit="tokens", doc="Context tokens per compressed index key.")
    block_size: int = arg(unit="tokens", doc="Keys per scored block.")
    dtype: DType = arg(doc="Element type of the logits and block scores (bf16).")


DOC = KernelDoc(
    title="Indexer candidate block scores",
    summary="Reduce each row's indexer logits to per-block maxima for candidate selection.",
    description=(
        "Row i's valid logits are its key window; every block of block_size keys "
        "scores its maximum (capped at the largest finite bf16), the last valid block "
        "scores +inf so selection always keeps it."
    ),
    category="Attention",
    subcategory="DSA",
    formula=("bytes = 2 · P + 2 · P / block_size + 8 · num_queries, P = valid logits",),
    default_metric="time_ms",
    method=CUPTI_METHOD + " Both launches count. Checked against DeepInfer's candidate_block_max_ref first.",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tilelang",
        # TileLang JIT-compiles for the device; the kernel waits on PDL
        # (T.pdl_sync), which needs SM90 or newer.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(9, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.candidate_block_max_tilelang",
            function_name="profile_candidate_block_max_tilelang",
        ),
        table_name=KIND,
        args_schema=CandidateBlockMaxArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(summary="DeepInfer's TileLang candidate_block_max (two launches)."),
    )
)
