"""FlashInfer TRT-LLM monolithic NVFP4 MoE selected by vLLM on B200."""

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

KIND = "nvfp4_moe"


@dataclass(frozen=True)
class Nvfp4MoeArgs(KernelArgs):
    num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    num_local_experts: int
    local_expert_offset: int
    top_k: int
    input_dtype: DType
    weight_format: str
    group_size: int
    routing_method: str
    n_group: int
    topk_group: int
    routed_scaling_numerator: int
    routed_scaling_denominator: int


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.nvfp4_moe",
            function_name="profile_nvfp4_moe_flashinfer_trtllm",
        ),
        table_name=KIND,
        args_schema=Nvfp4MoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
