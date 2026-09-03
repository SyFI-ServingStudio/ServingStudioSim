"""SGLang fused DSA indexer-query RoPE and FP8 quantization kernel kind."""

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

KIND: str = "dsa_indexer_q_rope_quant"


@dataclass(frozen=True)
class DsaIndexerQRopeQuantArgs(KernelArgs):
    num_tokens: int
    num_heads: int
    head_dim: int
    rope_dim: int
    # Both fields select compile-time CUDA instantiations.
    rope_layout: str
    hadamard: bool
    input_dtype: DType
    q_output_dtype: DType
    weight_output_dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_indexer_q_rope_quant",
            function_name="profile_dsa_indexer_q_rope_quant_sglang_cuda",
        ),
        table_name=KIND,
        args_schema=DsaIndexerQRopeQuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
    )
)

__all__ = ["KIND", "DsaIndexerQRopeQuantArgs"]
