"""DeepSeek V4 BF16 sparse-MLA prefill over one request batch."""

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

KIND = "deepseek_v4_sparse_mla_prefill"


@dataclass(frozen=True)
class DeepseekV4SparseMlaPrefillArgs(KernelArgs):
    query_context_pairs: tuple[tuple[int, int], ...] = arg(
        doc="Per-request (query tokens, context tokens) in the prefill batch."
    )
    max_model_len: int = arg(unit="tokens", doc="Maximum context length allowed by the model.")
    max_num_batched_tokens: int = arg(
        unit="tokens", doc="Maximum query tokens allowed in the batch."
    )
    prefill_chunk_size: int = arg(unit="requests", doc="Requests in each FlashMLA call.")
    compress_ratio: int = arg(
        unit="tokens", doc="Source tokens represented by one compressed cache position."
    )
    window_size: int = arg(unit="tokens", doc="Sliding-window keys available per query.")
    selected_k: int = arg(unit="tokens", doc="Maximum selected compressed keys per query.")
    selected_index_pattern: str = arg(doc="How selected and sliding-window indices are combined.")
    num_heads: int = arg(unit="heads", doc="Query heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads in the gathered cache.")
    head_dim: int = arg(unit="elements", doc="Elements in each query and key head.")
    value_dim: int = arg(unit="elements", doc="Elements in each output head.")
    softmax_scale: float = arg(unit="multiplier", doc="Multiplier applied to attention logits.")
    q_dtype: DType = arg(doc="Query element type.")
    cache_dtype: DType = arg(doc="Gathered cache element type.")
    index_dtype: str = arg(doc="Selected-index element type.")
    output_dtype: DType = arg(doc="Output element type.")
    cache_layout: str = arg(doc="Arrangement of gathered cache rows.")


DOC = KernelDoc(
    title="DeepSeek V4 sparse MLA prefill",
    summary="Attend from prefill queries to sliding-window and optional compressed keys.",
    description=(
        "DeepSeek V4's sparse MLA in prefill: each new query token attends to a"
        " 128-token sliding window and, in compressed layers, to its selected "
        "compressed keys, all from a BF16 workspace gathered beforehand. The "
        "measurement builds request-local indices with the compressed keys "
        "spread evenly across the visible context, and runs FlashMLA on groups "
        "of up to four requests, as vLLM does."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "O = softmax(q · K_selectedᵀ · softmax_scale) · V_selected, per query head",
        "Q = total query tokens; V = sum of valid index lengths; P = total padded index entries",
        "TFLOPS = 2·64·V·(512 + 512) / time",
        "GB/s = [2·Q·64·512 + 4·P + 4·Q + 2·V·512 + 2·Q·64·512 + 8·Q·64] / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "One FlashMLA launch per group of up to four requests is counted. The "
        "output checks and cache construction are excluded."
    ),
    caveats=(
        "Selected keys are spread evenly rather than chosen by an indexer.",
        "Gathering the cache and combining the indices are separate kinds and are not included.",
    ),
    # The PyTorch correctness calculation is local to the measured runner.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_flashmla_bf16",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepseek_v4_sparse_mla_prefill_flashmla",
            function_name="profile_deepseek_v4_sparse_mla_prefill_flashmla",
        ),
        table_name=KIND,
        args_schema=DeepseekV4SparseMlaPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's flash_mla_sparse_fwd calls BF16 FlashMLA on combined request-local indices."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/flashmla.py",
        ),
    )
)

__all__ = ["DeepseekV4SparseMlaPrefillArgs", "KIND"]
