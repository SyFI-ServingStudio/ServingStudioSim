"""DeepSeek V4 C4 indexer prefill MQA-logits operation."""

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

KIND = "deepseek_v4_indexer_mqa_logits_prefill"


@dataclass(frozen=True)
class DeepseekV4IndexerMqaLogitsPrefillArgs(KernelArgs):
    query_context_pairs: tuple[tuple[int, int], ...] = arg(
        doc="Per-request query count and context length, in tokens."
    )
    max_model_len: int = arg(unit="tokens", doc="Maximum supported context length per request.")
    max_num_batched_tokens: int = arg(unit="tokens", doc="Maximum query rows in the prefill batch.")
    max_logits_bytes: int = arg(
        unit="bytes", doc="Maximum logits buffer size used when splitting calls."
    )
    compress_ratio: int = arg(
        unit="tokens", doc="Original context tokens represented by one compressed key."
    )
    num_heads: int = arg(unit="heads", doc="Indexer query heads per token.")
    head_dim: int = arg(unit="elements", doc="Elements in each indexer query and key head.")
    q_dtype: DType = arg(doc="Element type of indexer queries.")
    k_dtype: DType = arg(doc="Element type of compressed keys.")
    k_scale_dtype: DType = arg(doc="Element type of per-key scales.")
    weight_dtype: DType = arg(doc="Element type of indexer head weights.")
    output_dtype: DType = arg(doc="Element type of the logits output.")
    clean_logits: bool = arg(
        doc="Whether unwritten output logits are filled with negative infinity."
    )


DOC = KernelDoc(
    title="DeepSeek V4 prefill indexer logits",
    summary="Score prefill queries against compressed keys for sparse token selection.",
    description=(
        "In prefill, DeepSeek V4's sparse-attention indexer scores each query "
        "against the compressed keys visible at its position, one key per "
        "compress_ratio context tokens: the ReLU of each head's query-key dot "
        "product, weighted and summed over the 64 heads, times the key's scale."
        " query_context_pairs sets each request's query count and context. When"
        " the gathered keys or the 512 MiB logits buffer would overflow, the "
        "batch is split into several calls, as vLLM does."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "logit(i, j) = Σ_heads weight(i, h) · ReLU(q(i, h) · (k(j) · key_scale(j)))",
        "valid_key_pairs = Σ_i (row_end(i) − row_start(i)), over all chunks",
        "TFLOPS = 2 · valid_key_pairs · 64 · 128 / time",
        "GB/s = (total_queries · 64 · 128 + valid_key_pairs · (128 + 4 + 4) + "
        "total_queries · (64 · 4 + 8)) / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "Sampled rows are checked against a PyTorch computation before timing; "
        "every DeepGEMM call the batch needs is counted together."
    ),
    caveats=(
        "Queries and keys are deterministic FP8 values; projection, "
        "quantization and key gathering are not included.",
        "GB/s counts logical bytes for the valid query-key pairs, not physical "
        "memory transactions.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_deepgemm_fp8",
        runner_ref=RunnerRef(
            module_name=(
                "profiling.runners.attention.deepseek_v4_indexer_mqa_logits_prefill_deepgemm"
            ),
            function_name="profile_deepseek_v4_indexer_mqa_logits_prefill_deepgemm",
        ),
        table_name=KIND,
        args_schema=DeepseekV4IndexerMqaLogitsPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "DeepGEMM fp8_fp4_mqa_logits through vLLM's deep_gemm wrapper, on "
                "unpaged compressed keys."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)

__all__ = ["DeepseekV4IndexerMqaLogitsPrefillArgs", "KIND"]
