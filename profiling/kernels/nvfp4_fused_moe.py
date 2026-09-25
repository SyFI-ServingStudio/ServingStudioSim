"""Whole FlashInfer TRT-LLM block-quantized MoE callable used on B200.

The operation starts with already-quantized activations and contains routing,
both expert GEMMs, SwiGLU, and finalize routing.  It is intentionally one L1
kind: SM100 uses PDL between several physical launches, so summing independently
profiled stages would double-count overlap and could select different tactics.

Despite the kind name, `weight_format` selects the expert weight recipe and the
backend fixes the activation recipe: the NVFP4 backends take `nvfp4_e2m1` /
group 16 with NVFP4 activations; the MXFP4 backend
(`trtllm_fp4_block_scale_routed_moe`, DeepSeek-V4.1-Flash) takes `mxfp4_ue8m0` /
group 32 with MXFP8 activations and `routing_method="precomputed_dsv4"`: finished
top-k ids and weights come in, so it only permutes (n_group = topk_group = 1,
routed scale 1/1; the router already applied the scale). `input_dtype` is the
unquantized activation and output precision in every backend.
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

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_sm100_deferred_finalize",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.nvfp4_fused_moe",
            function_name="profile_nvfp4_fused_moe_deferred_finalize_sm100",
        ),
        table_name=KIND,
        args_schema=Nvfp4FusedMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_sm100_mxfp4",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.mxfp4_fused_moe",
            function_name="profile_mxfp4_fused_moe_sm100",
        ),
        table_name=KIND,
        args_schema=Nvfp4FusedMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_fork_env",
    )
)
