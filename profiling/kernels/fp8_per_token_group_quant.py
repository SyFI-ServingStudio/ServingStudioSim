"""Dense-linear BF16-to-FP8 per-token-group input quantization kind.

The vLLM DeepGEMM dense path quantizes each activation row in independent
``group_size`` chunks and writes UE8M0 FP32 scales in column-major layout before
QKV and output-projection GEMMs.  Group size and scale layout remain explicit
cache axes because both select kernel semantics and launch/storage behavior,
even though the first exact backend supports only the production pair below.
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

KIND: str = "fp8_per_token_group_quant"


@dataclass(frozen=True)
class Fp8PerTokenGroupQuantArgs(KernelArgs):
    num_tokens: int
    hidden_size: int
    group_size: int
    input_dtype: DType
    scale_format: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H100", "NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.fp8_per_token_group_quant",
            function_name="profile_fp8_per_token_group_quant_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=Fp8PerTokenGroupQuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
