"""DeepSeek V4 indexer decode MQA-logits operation."""

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

KIND = "deepseek_v4_indexer_mqa_logits_decode"


@dataclass(frozen=True)
class DeepseekV4IndexerMqaLogitsDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Decode requests scored together.")
    context_len: int = arg(unit="tokens", doc="Maximum compressed-key context length in the batch.")
    next_n: int = arg(unit="tokens", doc="New indexer queries per request.")
    max_model_len: int = arg(unit="tokens", doc="Width of each logits output row.")
    num_heads: int = arg(unit="heads", doc="Indexer query heads per request.")
    head_dim: int = arg(unit="elements", doc="Elements in each indexer query and key head.")
    block_size: int = arg(unit="tokens", doc="Compressed keys in each cache page.")
    q_dtype: DType = arg(doc="Element type of indexer queries.")
    cache_dtype: DType = arg(doc="Element type of compressed keys in the cache.")
    scale_dtype: DType = arg(doc="Element type of per-key cache scales.")
    weight_dtype: DType = arg(doc="Element type of indexer head weights.")
    output_dtype: DType = arg(doc="Element type of the logits output.")
    context_mode: str = arg(doc="Rule for deriving request lengths from context_len.")
    page_mapping: str = arg(doc="Assignment of physical cache pages to requests.")
    cache_format: str = arg(doc="Storage layout and scale format of the paged key cache.")
    clean_logits: bool = arg(
        doc="Whether unwritten output logits are filled with negative infinity."
    )


DOC = KernelDoc(
    title="DeepSeek V4 decode indexer logits",
    summary="Score decode queries against paged compressed keys for sparse token selection.",
    description=(
        "In decode, DeepSeek V4's sparse-attention indexer scores each "
        "request's compressed keys: for each key, the ReLU of each head's "
        "query-key dot product, weighted by the head weight and summed over the"
        " 64 heads, times the key's scale. The top-k step then picks the keys "
        "attention reads. Request b has max(1, context_len − b) keys, stored in"
        " contiguous cache pages; the scheduler metadata is built before "
        "timing."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "logit(b, j) = Σ_heads weight(b, h) · ReLU(q(b, h) · k(b, j)) · key_scale(b, j)",
        "length(b) = max(1, context_len − b); "
        "scheduled_tokens = Σ_b block_size · ⌈length(b) / block_size⌉",
        "TFLOPS = 2 · scheduled_tokens · num_heads · head_dim / time",
        "GB/s = (batch_size · next_n · num_heads · head_dim + "
        "batch_size · next_n · num_heads · 4 + scheduled_tokens · 132 + "
        "batch_size · ⌈context_len / block_size⌉ · 4 + batch_size · 4 + "
        "scheduled_tokens · 4) / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "Scheduler metadata and the output check, against a PyTorch "
        "computation, run before timing; every launch of the DeepGEMM call is "
        "counted."
    ),
    caveats=(
        "TFLOPS and GB/s count whole cache pages, including each request's "
        "padded tail; GB/s is logical, not physical, traffic.",
        "clean_logits is false, so unwritten logits are left as they are.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_deepgemm_fp8",
        runner_ref=RunnerRef(
            module_name=(
                "profiling.runners.attention."
                "deepseek_v4_indexer_mqa_logits_decode_deepgemm"
            ),
            function_name="profile_deepseek_v4_indexer_mqa_logits_decode_deepgemm",
        ),
        table_name=KIND,
        args_schema=DeepseekV4IndexerMqaLogitsDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(summary=(
                "DeepGEMM fp8_fp4_paged_mqa_logits through vLLM's deep_gemm wrapper, "
                "reading a paged FP8 key cache."
            ), url="https://github.com/deepseek-ai/DeepGEMM"),
    )
)

__all__ = ["DeepseekV4IndexerMqaLogitsDecodeArgs", "KIND"]
