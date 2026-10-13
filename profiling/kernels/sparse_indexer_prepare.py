"""Prepare a sparse-attention indexer step: normalize and rotate queries, compress keys.

One launch does both halves. Query heads get Gemma RMSNorm (weight + 1) and
NeoX RoPE on their first half and are written in the indexer cache dtype. Each
completed group of ``compress_ratio`` raw keys is mean-pooled, normalized,
rotated at the group's first position and stored in the compressed key cache;
the last ``ring_size`` raw keys of each request are kept in a ring for groups
that straddle steps.
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

KIND = "sparse_indexer_prepare"


@dataclass(frozen=True)
class SparseIndexerPrepareArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens in the step.")
    num_heads: int = arg(unit="heads", doc="Indexer query heads.")
    head_dim: int = arg(unit="elements", doc="Width of each query head and key; RoPE rotates half.")
    compress_ratio: int = arg(unit="tokens", doc="Raw keys pooled into one compressed key.")
    ring_size: int = arg(unit="tokens", doc="Raw keys kept per request between steps.")
    input_dtype: DType = arg(doc="Element type of the projected queries and raw keys.")
    cache_dtype: DType = arg(doc="Element type of the prepared queries and compressed keys.")


DOC = KernelDoc(
    title="Sparse indexer prepare",
    summary="Normalize and rotate indexer queries and pool completed key groups into the cache.",
    description=(
        "The fused Triton kernel runs query tiles and one program per completed "
        "key group (plus one per request that commits the raw-key ring) in a single "
        "grid. The measured step is one sequence of num_tokens new tokens starting "
        "at position 0: decode batches with many requests run within a few percent "
        "of a single sequence of the same token count."
    ),
    category="Attention",
    subcategory="Sparse GQA",
    formula=(
        "q' = RoPE(GemmaRMSNorm(q)); k_c = RoPE(GemmaRMSNorm(mean_ratio(k)))",
        "GB/s = num_tokens · (num_heads·head_dim·(bytes(in) + bytes(cache)) "
        "+ head_dim·bytes(in)) / time",
    ),
    default_metric="time_ms",
    method=f"{CUPTI_METHOD} The one launch of the public call is counted.",
    caveats=(
        "Text positions only (one position row); multimodal RoPE positions are not measured.",
        "GB/s excludes the compressed-key and ring writes and the metadata.",
    ),
    reference=None,
)


# Portable Triton (fork models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py:93-505).
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
            module_name="profiling.runners.attention.sparse_indexer_prepare_vllm_triton",
            function_name="profile_sparse_indexer_prepare_vllm_triton",
        ),
        table_name=KIND,
        args_schema=SparseIndexerPrepareArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="The alignment vLLM fork's fused qsa_pre_indexer Triton kernel.",
            url="https://github.com/serendipity-zk/vllm/blob/892da0822f7e866cec8e519727f014c45a5ad792/vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py",
        ),
    )
)

__all__ = ["KIND", "SparseIndexerPrepareArgs"]
