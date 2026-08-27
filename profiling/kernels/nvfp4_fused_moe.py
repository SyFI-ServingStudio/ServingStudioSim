"""Whole FlashInfer TRT-LLM NVFP4 MoE callable used on B200.

The operation starts with already-quantized activations and contains routing,
both expert GEMMs, SwiGLU, and finalize routing.  It is intentionally one L1
kind: SM100 uses PDL between several physical launches, so summing independently
profiled stages would double-count overlap and could select different tactics.
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

KIND = "nvfp4_fused_moe"


@dataclass(frozen=True)
class Nvfp4FusedMoeArgs(KernelArgs):
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
            module_name="profiling.runners.moe.nvfp4_fused_moe",
            function_name="profile_nvfp4_fused_moe_sm100",
        ),
        table_name=KIND,
        args_schema=Nvfp4FusedMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
