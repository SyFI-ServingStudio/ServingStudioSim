"""DeepSeek V4 FP8 sparse-MLA decode CUDA-graph replay."""

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

KIND = "deepseek_v4_sparse_mla_decode"


@dataclass(frozen=True)
class DeepseekV4SparseMlaDecodeArgs(KernelArgs):
    """Workload and model identity for one sparse decode graph replay.

    Counts remain per flattened query row because FlashMLA's planner consumes
    every row. Production also flattens speculative decode tokens this way and
    keeps ``s_q=1``. ``extra_index_capacity`` distinguishes runtime context
    configurations, notably C128 width 512 at 65K versus 8192 at 1M.
    """

    swa_valid_counts: tuple[int, ...]
    extra_valid_counts: tuple[int, ...]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    value_dim: int
    swa_window: int
    extra_index_capacity: int
    compress_ratio: int
    q_dtype: DType
    cache_dtype: DType
    output_dtype: DType
    planner_mode: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_flashmla_fp8_cudagraph",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepseek_v4_sparse_mla_decode_flashmla",
            function_name="profile_deepseek_v4_sparse_mla_decode_flashmla",
        ),
        table_name=KIND,
        args_schema=DeepseekV4SparseMlaDecodeArgs,
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

__all__ = ["DeepseekV4SparseMlaDecodeArgs", "KIND"]
