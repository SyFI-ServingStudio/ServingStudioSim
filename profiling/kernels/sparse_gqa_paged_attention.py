"""Sparse grouped-query attention over a paged K/V cache, with a sigmoid output gate.

An indexer has already chosen, for every query row, which logical tokens it
attends: ``token_topk / compress_ratio`` complete compressed groups, each
expanded to ``compress_ratio`` consecutive tokens, plus the causal tail of the
group the query sits in. The kernel gathers those tokens' keys and values from
the paged cache, runs an online softmax per query head (``num_q_heads /
num_kv_heads`` heads share one K/V head), and multiplies the result by
``sigmoid(gate)``. Split-K partials and their merge belong to the same public
call.
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

KIND = "sparse_gqa_paged_attention"


@dataclass(frozen=True)
class SparseGqaPagedAttentionArgs(KernelArgs):
    num_rows: int = arg(unit="tokens", doc="Query rows attended in the call.")
    context_len: int = arg(
        unit="tokens", doc="Tokens visible to every row, the row's own token included."
    )
    num_q_heads: int = arg(unit="heads", doc="Query heads, each with a gate head.")
    num_kv_heads: int = arg(unit="heads", doc="K/V heads, each shared by a query group.")
    head_dim: int = arg(unit="elements", doc="Width of each query, key and value head.")
    token_topk: int = arg(unit="tokens", doc="Tokens selected from complete groups per row.")
    compress_ratio: int = arg(unit="tokens", doc="Tokens per compressed group.")
    q_dtype: DType = arg(doc="Element type of the queries, gate and output.")
    kv_dtype: DType = arg(doc="Element type of the paged K/V cache.")
    prefill_config: bool = arg(
        doc=(
            "Whether the step holds a prefill row (max query length above the "
            "decode/verify length); it picks the launch configuration for large calls."
        )
    )


DOC = KernelDoc(
    title="Sparse GQA paged attention",
    summary=(
        "Attend each query row to its indexer-selected tokens in a paged K/V cache "
        "and apply the sigmoid output gate."
    ),
    description=(
        "Each row selects min(⌊context_len/ratio⌋, token_topk/ratio) complete "
        "groups of ratio tokens, scattered at random over its visible context, "
        "plus the causal tail of its own open group. The call reads only those "
        "tokens' keys and values through the block table. It splits the selection "
        "into split-K partials when there are few rows, then merges them; the "
        "configuration is the production table for this GPU. Under the uniform "
        "(decode) table every row is its own request with pages drawn at random "
        "from a pool larger than L2; under the prefill table the rows are one "
        "prefill chunk, sharing one request's consecutive pages."
    ),
    category="Attention",
    subcategory="Sparse GQA",
    formula=(
        "S = min(⌊c/r⌋, k/r)·r + (c mod r): tokens attended per row",
        "out = BF16(softmax(q·Kₛᵀ/√d)·Vₛ) · sigmoid(gate)",
        "TFLOPS = 4 · num_rows · num_q_heads · head_dim · S / time",
        "GB/s = num_rows · S · num_kv_heads · 2 · head_dim · bytes(kv) / time",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} Every launch of the call counts: the partial kernel and, "
        "when it splits, the merge. The selection is built with the production "
        "expansion kernel and checked against a PyTorch reference before timing."
    ),
    caveats=(
        "The output zero-fill the framework issues before the call is not included.",
        "Selected groups are uniform random over the visible context, not a real "
        "indexer's choice; contiguous selections (context ≤ token_topk) read faster.",
        "FP8 caches use per-tensor scales of 1.",
        "GB/s counts selected K/V bytes once; queries, gates, indices and the "
        "split-K workspace are excluded.",
    ),
    reference="profiling.runners.attention.sparse_gqa_paged_attention_reference",
)


# Triton bf16/fp8 tl.dot (fork models/qwen4_exp/nvidia/ops/qsa.py:31-289);
# Triton lowers fp8 e4m3 dots on SM89+, and the kernel is portable Triton.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16, DType.FP8_E4M3}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.sparse_gqa_paged_attention_vllm_triton",
            function_name="profile_sparse_gqa_paged_attention_vllm_triton",
        ),
        table_name=KIND,
        args_schema=SparseGqaPagedAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "The alignment vLLM fork's qsa_sparse_paged_attention: split-K "
                "Triton partials plus merge, gate fused."
            ),
            url="https://github.com/serendipity-zk/vllm/blob/892da0822f7e866cec8e519727f014c45a5ad792/vllm/models/qwen4_exp/nvidia/ops/qsa.py",
        ),
    )
)

__all__ = ["KIND", "SparseGqaPagedAttentionArgs"]
