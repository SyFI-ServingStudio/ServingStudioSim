"""Prefill scores of a weight-free sparse-attention indexer over compressed paged keys.

One sequence's chunk of ``num_queries`` new tokens, after ``prefix_len`` cached
tokens, scores each row against the compressed keys it can see: row ``i`` sees
``(prefix_len + i + 1) // compress_ratio`` keys. The call splits the rows into
pieces whose FP32 score matrix stays under the framework's logits budget and
launches the scoring kernel once per piece; all of them count here. The top-k
on each piece is the separate ``dsa_persistent_topk_decode`` kind.
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

KIND = "sparse_indexer_logits_prefill"


@dataclass(frozen=True)
class SparseIndexerLogitsPrefillArgs(KernelArgs):
    num_queries: int = arg(unit="tokens", doc="New tokens of the sequence in this step.")
    prefix_len: int = arg(unit="tokens", doc="Tokens of the sequence already cached.")
    num_heads: int = arg(unit="heads", doc="Indexer query heads.")
    head_dim: int = arg(unit="elements", doc="Width of each query head and key.")
    compress_ratio: int = arg(unit="tokens", doc="Tokens per compressed key.")
    dtype: DType = arg(doc="Element type of the queries and the cached keys (they match).")


DOC = KernelDoc(
    title="Sparse indexer prefill logits",
    summary="Score a prefill chunk's rows against their causally visible compressed keys.",
    description=(
        "The score matrix is ⌈⌈(prefix_len + num_queries)/ratio⌉/64⌉·64 columns "
        "wide. The framework caps it at VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (512 MiB "
        "by default) and launches _qsa_mqa_paged_prefill_kernel once per row piece; "
        "every launch counts. Pages of keys are scattered at random."
    ),
    category="Attention",
    subcategory="Sparse GQA",
    formula=(
        "logit[i, c] = Σₕ ReLU(q[i, h] · key[c]) for c < (prefix_len + i + 1) // ratio",
        "K = Σᵢ (prefix_len + i + 1) // ratio: scored keys",
        "TFLOPS = 2 · num_heads · head_dim · K / time",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} The public prefill selection call runs whole with the "
        "default 512 MiB budget; only its _qsa_mqa_paged_prefill_kernel launches "
        "count. A PyTorch reference checks sampled rows first."
    ),
    caveats=(
        "One sequence per call; a step with several prefill sequences is several calls.",
        "GB/s counts each visible key once per row tile of 32 rows and the scores written.",
    ),
    reference="profiling.runners.attention.sparse_indexer_logits_reference",
)


# Portable Triton (fork models/qwen4_exp/nvidia/ops/qsa_indexer.py:117-219,
# 380-435); fp8 tl.dot lowers on SM89+.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP8_E4M3}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.sparse_indexer_logits_vllm_triton",
            function_name="profile_sparse_indexer_logits_prefill_vllm_triton",
        ),
        table_name=KIND,
        args_schema=SparseIndexerLogitsPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="The alignment vLLM fork's _qsa_mqa_paged_prefill_kernel (Triton).",
            url="https://github.com/serendipity-zk/vllm/blob/892da0822f7e866cec8e519727f014c45a5ad792/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py",
        ),
    )
)

__all__ = ["KIND", "SparseIndexerLogitsPrefillArgs"]
