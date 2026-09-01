"""FlashInfer fused MLA RoPE, FP8 quantization, and query-concat kernel kind."""

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

KIND: str = "mla_rope_quantize_fp8"


@dataclass(frozen=True)
class MlaRopeQuantizeFp8Args(KernelArgs):
    num_tokens: int
    num_heads: int
    kv_lora_rank: int
    rope_dim: int
    max_position: int
    is_neox_style: bool
    input_dtype: DType
    quant_dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.mla_rope_quantize_fp8",
            function_name="profile_mla_rope_quantize_fp8_flashinfer",
        ),
        table_name=KIND,
        args_schema=MlaRopeQuantizeFp8Args,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
    )
)

__all__ = ["KIND", "MlaRopeQuantizeFp8Args"]
