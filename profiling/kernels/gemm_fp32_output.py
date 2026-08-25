"""BF16 matrix multiplication with FP32 output."""

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

KIND = "gemm_fp32_output"


@dataclass(frozen=True)
class GemmFp32OutputArgs(KernelArgs):
    m: int
    n: int
    k: int
    input_dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_cublas",
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.gemm_fp32_output_torch_cublas",
            function_name="profile_gemm_fp32_output_torch_cublas",
        ),
        table_name=KIND,
        args_schema=GemmFp32OutputArgs,
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
