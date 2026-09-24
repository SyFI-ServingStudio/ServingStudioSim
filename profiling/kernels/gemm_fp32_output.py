"""Matrix multiplication with FP32 output (BF16 or FP32 inputs).

``input_dtype`` is the dtype of both GEMM operands; the output is always FP32.
"""

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
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        subprocess_env="vllm_env",
    )
)

# GLM-5.3 serving stack. Its cuBLAS selects the production split-K SGEMM for
# the FP32 indexer head-weights form, which the container cuBLAS does not.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_cublas_vllm_fork",
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.gemm_fp32_output_torch_cublas",
            function_name="profile_gemm_fp32_output_torch_cublas_vllm_fork",
        ),
        table_name=KIND,
        args_schema=GemmFp32OutputArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP32}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        subprocess_env="vllm_fork_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_router_auto",
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.gemm_fp32_output_sglang_router",
            function_name="profile_gemm_fp32_output_sglang_router",
        ),
        table_name=KIND,
        args_schema=GemmFp32OutputArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        subprocess_env="sglang_env",
    )
)

__all__ = ["KIND"]
