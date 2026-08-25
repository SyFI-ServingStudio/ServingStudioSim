"""DeepSeek V4 packed FP8/BF16 cache dequantize-and-gather."""

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

KIND = "deepseek_v4_packed_cache_gather"


@dataclass(frozen=True)
class DeepseekV4PackedCacheGatherArgs(KernelArgs):
    seq_lens: tuple[int, ...]
    gather_lens: tuple[int, ...]
    workspace_rows: int
    block_table_width: int
    block_size: int
    offset: int
    num_kv_heads: int
    head_dim: int
    fp8_dim: int
    quant_group_size: int
    cache_dtype: str
    output_dtype: DType
    cache_layout: str
    scale_format: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_deepseek_v4_cutedsl",
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.deepseek_v4_packed_cache_gather_cutedsl"),
            function_name="profile_deepseek_v4_packed_cache_gather_cutedsl",
        ),
        table_name=KIND,
        args_schema=DeepseekV4PackedCacheGatherArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
    )
)

__all__ = ["DeepseekV4PackedCacheGatherArgs", "KIND"]
