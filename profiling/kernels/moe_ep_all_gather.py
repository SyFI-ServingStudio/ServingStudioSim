"""Grouped hidden-state and router-logit all-gatherv used by naive DP/EP MoE."""

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

KIND = "moe_ep_all_gather"


@dataclass(frozen=True)
class MoeEpAllGatherArgs(KernelArgs):
    num_gpus: int
    per_rank_tokens: tuple[int, ...]
    hidden_size: int
    num_experts: int
    hidden_dtype: DType
    router_dtype: DType
    fabric: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_pynccl",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.moe_ep_collectives_vllm_pynccl",
            function_name="profile_moe_ep_all_gather_batch",
        ),
        table_name=KIND,
        args_schema=MoeEpAllGatherArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
    )
)
