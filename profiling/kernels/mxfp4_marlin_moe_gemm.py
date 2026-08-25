"""One production packed-MXFP4 Marlin MoE GEMM launch."""

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

KIND = "mxfp4_marlin_moe_gemm"


@dataclass(frozen=True)
class Mxfp4MarlinMoeGemmArgs(KernelArgs):
    m: int
    n: int
    k: int
    dtype: DType
    input_top_k: int
    block_size_m: int
    mul_topk_weights: bool
    per_group_batches: tuple[int, ...]


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_marlin",
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.mxfp4_marlin_moe_gemm_vllm_marlin",
            function_name="profile_mxfp4_marlin_moe_gemm_vllm_marlin",
        ),
        table_name=KIND,
        args_schema=Mxfp4MarlinMoeGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
    )
)

__all__ = ["KIND"]
