"""Kimi-K3 MXFP4 TRT-LLM fused MoE profiling contract."""

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

KIND = "mxfp4_fused_moe"


@dataclass(frozen=True)
class Mxfp4FusedMoeArgs(KernelArgs):
    num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    num_local_experts: int
    top_k: int
    input_dtype: DType
    weight_format: str
    group_size: int
    routing_method: str
    activation: str
    n_group: int
    topk_group: int
    routed_scaling_factor: float
    gemm1_alpha: float
    gemm1_clamp_limit: float
    per_expert_batches: tuple[int, ...]


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_trtllm_mxfp4",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.mxfp4_fused_moe",
            function_name="profile_mxfp4_fused_moe",
        ),
        table_name=KIND,
        args_schema=Mxfp4FusedMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_k3_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_trtllm_mxfp4_prefill",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.mxfp4_fused_moe",
            function_name="profile_mxfp4_fused_moe_prefill",
        ),
        table_name=KIND,
        args_schema=Mxfp4FusedMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_k3_env",
    )
)
