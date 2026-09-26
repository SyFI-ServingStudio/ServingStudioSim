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
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
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
    m: int = arg(unit="tokens", doc="Tokens in the batch.")
    hidden: int = arg(unit="elements", doc="Features normalized per token.")
    dtype: DType = arg(doc="Element type of the input and weight.")


DOC = KernelDoc(
    title="RMSNorm",
    summary=(
        "Normalize each token's hidden vector by its root mean square and scale it by a weight."
    ),
    description=(
        "RMSNorm on its own, without the residual add that residual_rms_norm "
        "fuses in. Models use it wherever a tensor is normalized alone, for "
        "example before the attention projections. The input has m token rows "
        "of hidden features; input and weight are random normal and ε is 1e-6."
    ),
    category="Normalization",
    formula=(
        "y = x / √(mean(x²) + 1e-6) · weight, per token",
        "TFLOPS = 5·m·hidden / time",
        "GB/s = 2·m·hidden·bytes per element / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=f"{CUPTI_METHOD} Only launches named RMSNormKernel are counted.",
    caveats=(
        "GB/s counts one read of x and one write of y; the weight read is left out.",
        "TFLOPS assumes 5 operations per element.",
    ),
    # No separate PyTorch reference implementation exists for this kind.
    reference=None,
)


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
        doc=BackendDoc(
            summary="FlashInfer's norm.rmsnorm.",
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/norm/__init__.py",
        ),
    )
)
