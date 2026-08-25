"""DeepSeek V4 fused inverse-RoPE and grouped FP8 quantization."""

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

KIND = "deepseek_v4_fused_inv_rope_fp8_quant"


@dataclass(frozen=True)
class DeepseekV4FusedInvRopeFp8QuantArgs(KernelArgs):
    num_tokens: int


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name=(
                "profiling.runners.attention.deepseek_v4_fused_inv_rope_fp8_quant_vllm_triton"
            ),
            function_name="profile_deepseek_v4_fused_inv_rope_fp8_quant_vllm_triton",
        ),
        table_name=KIND,
        args_schema=DeepseekV4FusedInvRopeFp8QuantArgs,
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
