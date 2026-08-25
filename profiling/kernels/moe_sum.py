"""DeepSeek routed-expert BF16 top-k reduction."""

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

KIND = "moe_sum"


@dataclass(frozen=True)
class MoeSumArgs(KernelArgs):
    num_tokens: int
    top_k: int
    hidden_dim: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_sum_vllm_cuda",
            function_name="profile_moe_sum_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=MoeSumArgs,
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
