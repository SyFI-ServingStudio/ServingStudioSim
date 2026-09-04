"""Whole FlashInfer TRT-LLM BF16 MoE callable used by vLLM on SM100."""

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

KIND = "bf16_fused_moe"


@dataclass(frozen=True)
class Bf16FusedMoeArgs(KernelArgs):
    num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    num_local_experts: int
    top_k: int
    dtype: DType
    routing_method: str
    n_group: int
    topk_group: int
    routed_scaling_numerator: int
    routed_scaling_denominator: int
    per_expert_batches: tuple[int, ...]


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_sm100",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.bf16_fused_moe",
            function_name="profile_bf16_fused_moe_sm100",
        ),
        table_name=KIND,
        args_schema=Bf16FusedMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
