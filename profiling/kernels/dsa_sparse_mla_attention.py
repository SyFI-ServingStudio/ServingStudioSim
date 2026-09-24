"""GLM-5.2 / GLM-5.3-Flash selected sparse MLA attention kernel kind."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "dsa_sparse_mla_attention"


@dataclass(frozen=True)
class DsaSparseMlaAttentionArgs(KernelArgs):
    num_queries: int
    num_cache_tokens: int
    num_heads: int
    num_kv_heads: int
    selected_k: int
    latent_dim: int
    rope_dim: int
    value_dim: int
    softmax_scale: float
    q_dtype: DType
    cache_dtype: DType
    index_dtype: str
    output_dtype: DType
    valid_counts: str
    index_distribution: str
    cache_layout: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_attention",
            function_name="profile_dsa_sparse_mla_attention_torch",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_fp8",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_attention",
            function_name="profile_dsa_sparse_mla_attention_flashinfer_trtllm_fp8",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)


# GLM-5.3-Flash (qk_rope_head_dim=0): FlashInfer's native no-rope sparse MLA,
# which exists only in the vLLM fork's FlashInfer 0.6.18. Accepts rope_dim=0,
# cache_layout "hnd_paged_mqa_fp8_latent", and selected_k (the page-table width
# passed as sparse_mla_top_k) of 2048 or the kpool buffer's 2176.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_fp8_vllm_fork",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_attention",
            function_name="profile_dsa_sparse_mla_attention_flashinfer_trtllm_fp8_vllm_fork",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_fork_env",
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_flashmla_bf16",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_attention",
            function_name="profile_dsa_sparse_mla_attention_vllm_flashmla_bf16",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
