"""Fused MoE softmax/top-k router-selection kernel kind.

The Torch backend is a multi-launch semantic baseline. Production simulation
uses vLLM's single-launch CUDA backend.
"""

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

KIND: str = "moe_fused_topk"


@dataclass(frozen=True)
class MoeFusedTopkArgs(KernelArgs):
    num_tokens: int
    num_experts: int
    top_k: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_fused_topk_torch",
            function_name="profile_moe_fused_topk",
        ),
        table_name=KIND,
        args_schema=MoeFusedTopkArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="default_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_fused_topk_vllm_cuda",
            function_name="profile_moe_fused_topk_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=MoeFusedTopkArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
