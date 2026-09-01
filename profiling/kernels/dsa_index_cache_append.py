"""DSA index-key quantization and page-planar cache append kernel kind."""

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

KIND: str = "dsa_index_cache_append"


@dataclass(frozen=True)
class DsaIndexCacheAppendArgs(KernelArgs):
    num_tokens: int
    index_dim: int
    block_size: int
    quant_block_size: int
    input_dtype: DType
    cache_dtype: DType
    scale_format: str
    cache_format: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_index_cache_append",
            function_name="profile_dsa_index_cache_append_torch",
        ),
        table_name=KIND,
        args_schema=DsaIndexCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_index_cache_append",
            function_name="profile_dsa_index_cache_append_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=DsaIndexCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_fused_norm_rope_store",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_index_cache_append",
            function_name="profile_dsa_index_cache_append_sglang_fused_norm_rope_store",
        ),
        table_name=KIND,
        args_schema=DsaIndexCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
    )
)
