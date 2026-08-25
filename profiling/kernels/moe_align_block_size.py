"""vLLM MoE token-to-expert block alignment."""

from dataclasses import dataclass

from profiling.db.args import KernelArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "moe_align_block_size"


@dataclass(frozen=True)
class MoeAlignBlockSizeArgs(KernelArgs):
    num_tokens: int
    num_experts: int
    top_k: int
    block_size: int


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_align_block_size_vllm_cuda",
            function_name="profile_moe_align_block_size_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=MoeAlignBlockSizeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(compute=None, gpus=frozenset({"NVIDIA H200"})),
        subprocess_env="vllm_env",
    )
)

__all__ = ["KIND"]
