"""TileKernels' Engram gate (``tile_kernels.engram.engram_gate_fwd``).

Wire string: ``"engram_gate"`` -- matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/engram_gate.rs``.

One ``engram_gate_fwd_kernel_cuda_kernel`` launch per Engram layer and GPU
merges the projected Engram embedding into the hyper-connection streams: for
each of the ``hc_mult`` streams it RMS-normalizes the hidden state and the
stream's key, forms a clamped signed-sqrt gate from their weighted dot product,
and adds the gated value row to the stream in place. DeepInfer calls it from
``v4_modules/engram.py`` (``Engram.forward``) after the ``fused_kv_projection``
linear.

Why a new kind: no existing kind reads ``hc_mult`` key rows plus one value row
per token and rewrites the streams; ``elementwise`` would only be a byte-rate
stand-in.

Key: the per-GPU token count, ``hc_mult`` and ``hidden_size``. Inference mode:
in place, no saved intermediates, no image-token mask.
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

KIND = "engram_gate"


@dataclass(frozen=True)
class EngramGateArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens on this GPU.")
    hc_mult: int = arg(unit="streams", doc="Hyper-connection streams per token.")
    hidden_size: int = arg(unit="elements", doc="Width of each stream.")
    dtype: DType = arg(doc="Element type of the streams and the projected key/value rows (bf16).")


DOC = KernelDoc(
    title="Engram gate",
    summary=("Gates the projected Engram value into every hyper-connection stream, in place."),
    description=(
        "For each token and stream the kernel computes the RMS norms of the stream "
        "and of its projected key, a weighted dot product of the two, a clamped "
        "signed-sqrt gate from it, and adds gate times the shared value row to the "
        "stream. Inputs: streams [num_tokens, hc_mult, hidden_size] and keys plus "
        "value [num_tokens, hc_mult + 1, hidden_size], both bf16; the streams are "
        "updated in place."
    ),
    category="Other",
    formula=("bytes = 2 · num_tokens · hidden_size · (2 · hc_mult + hc_mult + 1)",),
    default_metric="time_ms",
    method=CUPTI_METHOD,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tile_kernels",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.engram.engram_gate_tile_kernels",
            function_name="profile_engram_gate_tile_kernels",
        ),
        table_name=KIND,
        args_schema=EngramGateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "tile_kernels.engram.engram_gate_fwd as DeepInfer calls it: in place, "
                "save_for_backward=False, no image-token mask."
            ),
            url="https://github.com/deepseek-ai/TileKernels",
        ),
    )
)
