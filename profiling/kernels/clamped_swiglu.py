"""DeepSeek routed-expert clamped SwiGLU."""

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

KIND = "clamped_swiglu"


@dataclass(frozen=True)
class ClampedSwigluArgs(KernelArgs):
    num_rows: int
    hidden_dim: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_inductor",
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.clamped_swiglu_vllm_inductor",
            function_name="profile_clamped_swiglu_vllm_inductor",
        ),
        table_name=KIND,
        args_schema=ClampedSwigluArgs,
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
