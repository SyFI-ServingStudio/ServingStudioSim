"""PLE dilated depthwise short convolution with state update.

Source: alignment fork, ``vllm/models/qwen4_exp/nvidia/ops/ple.py`` ``ple_conv``
(called from ``ple_layer.py`` ``_short_conv`` -> ``_short_conv_dilated_dispatch``,
an eager break under piecewise graphs). One call is ``_ple_conv_kernel``,
grid ``(tokens, ceil(channels / 512))``, plus, in ``spec`` and ``prefill``
modes, ``_ple_conv_writeback_kernel``, grid ``(requests, ceil(channels / 512))``.
For every token and channel: ``y = silu(sum_k w[c, k] * x[t - (K - 1 - k) * D])``
over the request's history (the persistent state holds the last
``(K - 1) * D`` inputs), then ``residual = outer_residual + (residual + y)``
in place, and the state advances.

Modes (a constexpr of the kernel, chosen by the batch's request types):
``decode`` (one token per request, state shifted in the same launch),
``spec`` (``query_len`` speculative verify rows per request, rolled back to the
last accepted token) and ``prefill`` (``query_len`` new tokens per request).

Args:

- ``batch_size``: requests in the call.
- ``query_len``: tokens per request (1 in ``decode``).
- ``channels``: conv channels (``hc_mult * hidden_size``).
- ``kernel_size``, ``dilation``: filter taps and spacing.
- ``mode``: ``decode``, ``spec`` or ``prefill``.
- ``dtype``: activation and state dtype (bf16).
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

KIND = "ple_short_conv"


@dataclass(frozen=True)
class PleShortConvArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests in the call.")
    query_len: int = arg(unit="tokens", doc="Tokens per request (1 in decode mode).")
    channels: int = arg(unit="channels", doc="Depthwise conv channels.")
    kernel_size: int = arg(unit="taps", doc="Filter taps per channel.")
    dilation: int = arg(unit="tokens", doc="Spacing between taps.")
    mode: str = arg(doc="decode, spec (verify rows with rollback) or prefill.")
    dtype: DType = arg(doc="Activation and conv-state element type.")


DOC = KernelDoc(
    title="PLE dilated short convolution",
    summary=(
        "Depthwise dilated causal convolution with SiLU over the PLE stream, added "
        "into the residual, with the per-request conv state carried across steps."
    ),
    description=(
        "Each channel has its own kernel_size-tap filter with taps dilation tokens "
        "apart. Taps before the step's first token come from the request's "
        "persistent state, which holds the last (kernel_size − 1)·dilation inputs. "
        "The SiLU output is added to the gated PLE output and then to the outer "
        "residual in place. Decode shifts the state in the same launch; spec "
        "(speculative verify, rolled back to the last accepted token) and prefill "
        "rewrite it in a second launch."
    ),
    category="Other",
    formula=(
        "y[t, c] = SiLU(Σₖ w[c, k] · x[t − (K − 1 − k)·D, c]); res[t, c] = outer[t, c] + (res[t, c] + y[t, c])",
        "FLOPs = batch_size · query_len · channels · (2·kernel_size + 6)",
        "bytes ≈ (4·tokens·channels + 2·batch_size·channels·(K − 1)·D) · bytes(dtype)",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Every launch of one ple_conv call is counted (one in "
        "decode, two in spec and prefill). Requests are equal length with their "
        "own state slots in an SD-layout pool viewed as [slot, channel, window], "
        "as production canonicalizes it. Outputs and the updated state are checked "
        "against a PyTorch reference first."
    ),
    caveats=(
        "Requests have equal query_len; mixed batches pass a token map that the "
        "runner does not exercise.",
        "Every request has prior state (continuing prefill chunks).",
    ),
    reference="profiling.runners.ple.ple_short_conv",
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.ple.ple_short_conv",
            function_name="profile_ple_short_conv_vllm_triton",
        ),
        table_name=KIND,
        args_schema=PleShortConvArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # Generic Triton loads/stores; PDL is gated on the device inside.
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        subprocess_env="vllm_env",
        doc=BackendDoc(summary="The fork's ple_conv (_ple_conv_kernel + writeback)."),
    )
)

__all__ = ["KIND", "PleShortConvArgs"]
