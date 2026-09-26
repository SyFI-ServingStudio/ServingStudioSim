"""RMSNorm kernel kind.

All Python-side per-kernel knowledge for ``rms_norm`` lives here: the wire
string ``KIND``, the ``RmsNormArgs`` schema, and the ``register(...)`` call that
wires this kernel into ``profiling.db.registry``.

Wire string: ``"rms_norm"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/rms_norm.rs`` and the Python facade stem used by
``profiling.facade`` to generate ``get_rms_norm_times`` /
``count_missing_rms_norm``.

Importing this module has a side effect: it appends a ``KernelProfilerSpec``
row to the registry. The runner module ``profiling.runners.norm.flashinfer`` is
referenced lazily via ``RunnerRef`` so the main process never eager-imports
torch/cuda/flashinfer.

Shape split (see L1 design §8.1): static config is ``(hidden, dtype)``; the
token count ``m`` is the runtime sweep axis (``Cache1DLinear``).
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

KIND: str = "rms_norm"


@dataclass(frozen=True)
class RmsNormArgs(KernelArgs):
    m: int
    hidden: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer",
        # Norms stay 16-bit even in an fp8 run (activation precision).
        supports=BackendSupport(compute=frozenset({DType.BF16, DType.FP16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.norm.flashinfer",
            function_name="profile_rms_norm",
        ),
        table_name=KIND,
        args_schema=RmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

# vLLM's own CUDA op (``RMSNorm.forward_cuda`` without residual), run in the
# pinned vLLM image; ``csrc/layernorm_kernels.cu`` is identical in the fork.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.norm.rms_norm_vllm_cuda",
            function_name="profile_rms_norm_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=RmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
