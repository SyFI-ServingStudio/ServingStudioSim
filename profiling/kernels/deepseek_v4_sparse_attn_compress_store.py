"""DeepSeek V4 sparse-attention compress, normalize, RoPE, and cache store."""

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

KIND = "deepseek_v4_sparse_attn_compress_store"


@dataclass(frozen=True)
class DeepseekV4SparseAttnCompressStoreArgs(KernelArgs):
    """Exact token topology plus the production model/cache identity."""

    row_positions: tuple[int, ...]
    row_request_ids: tuple[int, ...]
    state_block_table_width: int
    compress_ratio: int
    num_kv_heads: int
    head_dim: int
    rope_head_dim: int
    logical_block_size: int
    rms_eps: float
    state_dtype: DType
    norm_dtype: DType
    cache_dtype: str
    cache_layout: str
    scale_format: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_deepseek_v4_cutedsl",
        runner_ref=RunnerRef(
            module_name=(
                "profiling.runners.attention.deepseek_v4_sparse_attn_compress_store_cutedsl"
            ),
            function_name="profile_deepseek_v4_sparse_attn_compress_store_cutedsl",
        ),
        table_name=KIND,
        args_schema=DeepseekV4SparseAttnCompressStoreArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP32}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_deepseek_v4_triton",
        runner_ref=RunnerRef(
            module_name=(
                "profiling.runners.attention.deepseek_v4_sparse_attn_compress_store_triton"
            ),
            function_name="profile_deepseek_v4_sparse_attn_compress_store_triton",
        ),
        table_name=KIND,
        args_schema=DeepseekV4SparseAttnCompressStoreArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP32}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
    )
)

__all__ = ["DeepseekV4SparseAttnCompressStoreArgs", "KIND"]
