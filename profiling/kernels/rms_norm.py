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
        "of hidden features; input and weight are random normal. ε is 1e-6 "
        "for flashinfer and 1e-5 for vllm_cuda and neuron_torch_rms."
    ),
    category="Normalization",
    subcategory="RMSNorm",
    formula=(
        "y = x / √(mean(x²) + ε) · weight, per token",
        "flashinfer: TFLOPS = 5·m·hidden / time; GB/s = 2·m·hidden·bytes per element / time",
        "vllm_cuda and neuron_torch_rms: TFLOPS = (4·m·hidden + 2·m) / time; "
        "GB/s = (2·m·hidden + hidden)·bytes per element / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Only the norm kernel is counted: RMSNormKernel for "
        "flashinfer, rms_norm_kernel for vllm_cuda. vllm_cuda checks one output "
        "against a PyTorch RMSNorm before timing. Neuron times the compiled "
        "CustomRMSNorm expression, including its FP32 promotion and BF16 return, "
        "using the median native device interval union on one LNC2 unit."
    ),
    caveats=(
        "The backends count work differently: flashinfer's GB/s leaves out the "
        "weight read and its TFLOPS assumes 5 operations per element; vllm_cuda "
        "counts the weight read and 4 operations per element plus 2 per row.",
        "Neuron uses the same logical work/operand formulas as vllm_cuda; "
        "internal cast traffic is excluded. Energy sampling is unavailable and energy_j is zero.",
    ),
    # No separate PyTorch reference implementation exists for this kind.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="neuron_torch_rms",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            device_family="neuron",
            architectures=frozenset({"Trainium2"}),
        ),
        runner_ref=RunnerRef("profiling.runners.neuron.rms_norm", "profile_rms_norm"),
        table_name=KIND,
        args_schema=RmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="neuron_trace_env",
        doc=BackendDoc(
            summary=(
                "NxDI CustomRMSNorm's public RmsNorm.apply, compiled for LNC2; "
                "FP32 input promotion, epsilon1e-5, BF16 output, native device interval union."
            ),
            url="https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/modules/custom_calls.py",
        ),
    )
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

# vLLM's own CUDA op (``RMSNorm.forward_cuda`` without residual), run in the
# pinned vLLM image; ``csrc/layernorm_kernels.cu`` is identical in the fork.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        # No capability rule: a generic vLLM CUDA kernel built for every arch vLLM ships.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
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
        doc=BackendDoc(
            summary=(
                "vLLM's rms_norm CUDA op, as RMSNorm.forward_cuda launches it without a residual."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/layernorm_kernels.cu",
        ),
    )
)
