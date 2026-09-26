"""Dense-linear BF16-to-FP8 per-token-group input quantization kind.

The vLLM FP8 block-scaled paths quantize each activation row in independent
``group_size`` chunks and write UE8M0 scales. Group size and scale layout remain
explicit cache axes because both select kernel semantics and launch/storage
behavior. ``scale_format`` values of the ``vllm_cuda`` backend:

- ``ue8m0_column_major`` (Hopper): FP32 scales, column-major (DeepGEMM SM90).
- ``ue8m0_row_major`` (B200): FP32 scales, row-major (TRT-LLM FP8 block MoE input).
- ``ue8m0_packed_int32`` (B200): four UE8M0 bytes per int32, MN-major,
  TMA-aligned (DeepGEMM SM100 dense inputs).
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
            # The runner pairs each scale_format with its verified GPUs.
            gpus=frozenset({"NVIDIA H100", "NVIDIA H200", "NVIDIA B200"}),
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

# GLM-5.3 serving stack: its ``_C`` build launches the Blackwell layouts on a
# different grid than the profiler container's vLLM.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_fork_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.fp8_per_token_group_quant",
            function_name="profile_fp8_per_token_group_quant_vllm_fork_cuda",
        ),
        table_name=KIND,
        args_schema=Fp8PerTokenGroupQuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_fork_env",
    )
)
