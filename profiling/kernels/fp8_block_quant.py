"""BF16-to-FP8 1x128 dynamic block quantization kernel kind.

The upper layer has already resolved routing and EP ownership before it reaches
L1. Consequently ``num_tokens`` is the final local-rank row count. The static
``num_problems`` (local expert count) is retained because it changes the grouped
kernel's launch and scale-storage layout; the full per-expert batch vector is
intentionally not part of this wire schema. The runner synthesizes uniform
boundaries with the same problem count while preserving the final row count.

The output contract is fixed for the first backend: FP8 E4M3 values plus FP32
inverse/dequant scales, with BF16 input and a 128-element block size.  Those
fixed choices do not become cache-key fields; ``input_dtype`` remains explicit
because dtype is a capability and future backend-selection axis.
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

KIND: str = "fp8_block_quant"


@dataclass(frozen=True)
class Fp8BlockQuantArgs(KernelArgs):
    num_tokens: int
    hidden_size: int
    num_problems: int
    input_dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H100", "NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.fp8_block_quant",
            function_name="profile_fp8_block_quant_flashinfer_trtllm",
        ),
        table_name=KIND,
        args_schema=Fp8BlockQuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="flashinfer_pip_env",
    )
)
