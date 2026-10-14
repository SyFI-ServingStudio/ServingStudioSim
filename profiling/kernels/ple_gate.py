"""PLE similarity gate (one fused Triton launch).

Source: alignment fork, ``vllm/models/qwen4_exp/nvidia/ops/ple.py``
``ple_gate`` -> ``torch.ops.vllm.qwen4_exp_ple_gate`` -> ``_ple_gate_kernel``,
grid ``(num_tokens, hc_mult)`` (``ple_layer.py`` ``Qwen4ExpPLELayer.forward``).
Per token and residual stream ``s``: grouped Gemma-style RMSNorm of the key
slice (from the ``kv_proj`` output) and of the hidden stream, scaled dot
product ``d``, gate ``g = sigmoid(sign(d) * sqrt(max(|d|, 1e-6)))``, gated
value ``g * v`` (one value row shared by every stream) and its RMSNorm (the
short-conv input).

Args:

- ``num_tokens``: rows of the hidden state.
- ``hidden_size``: elements per residual stream (the norm group, ``H``).
- ``hc_mult``: residual streams (hyper-connection copies); key and hidden are
  ``hc_mult * hidden_size`` wide, value is ``hidden_size`` wide.
- ``dtype``: activation dtype (bf16).
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

KIND = "ple_gate"


@dataclass(frozen=True)
class PleGateArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows of the hidden state.")
    hidden_size: int = arg(unit="elements", doc="Elements per residual stream (norm group).")
    hc_mult: int = arg(unit="streams", doc="Residual streams sharing one value row.")
    dtype: DType = arg(doc="Activation element type.")


DOC = KernelDoc(
    title="PLE similarity gate",
    summary=(
        "Gate a looked-up n-gram value by its normalized key's similarity to each "
        "residual stream, and RMS-normalize the result."
    ),
    description=(
        "For each token and residual stream, the key slice and the hidden "
        "stream are RMS-normalized (weight 1 + w), their dot product scaled by "
        "1/sqrt(hidden_size) becomes a signed-square-root sigmoid gate, the gate "
        "multiplies the token's value row, and the gated row is RMS-normalized "
        "again for the following short convolution. One launch writes both the "
        "gated and the normalized rows."
    ),
    category="Other",
    formula=(
        "d = <RMSNorm(k_s), RMSNorm(x_s)> / sqrt(H); g = sigmoid(sign(d)·sqrt(max(|d|, 1e-6)))",
        "gated_s = g·v; normed_s = RMSNorm(gated_s)",
        "bytes = num_tokens · (5·hc_mult·H + H) · bytes(dtype) + 3·hc_mult·H·bytes(dtype)",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Only _ple_gate_kernel is counted. Key and value are "
        "strided views of one [tokens, (hc_mult + 1)·H] projection output, as in "
        "production. Outputs are checked against a PyTorch reference first."
    ),
    caveats=("TFLOPS is not computed.",),
    reference="profiling.runners.ple.ple_gate",
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.ple.ple_gate",
            function_name="profile_ple_gate_vllm_triton",
        ),
        table_name=KIND,
        args_schema=PleGateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # The wrapper accepts BF16/FP16 activations, but only BF16 (the
        # production dtype) is profiled and declared. The kernel uses only
        # generic Triton ops (PDL wait/launch is gated on the device inside).
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        subprocess_env="vllm_env",
        doc=BackendDoc(summary="The fork's ple_gate custom op (_ple_gate_kernel)."),
    )
)

__all__ = ["KIND", "PleGateArgs"]
