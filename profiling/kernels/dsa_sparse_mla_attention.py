"""GLM-5.2 BF16 selected sparse MLA attention kernel kind."""

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
