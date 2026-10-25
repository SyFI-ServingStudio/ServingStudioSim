"""PLE post on the Flash-Next Trainium engine.

The PLE block after its all-reduce: project, gate, conv and residual in one region.

Wire string ``'ple_post'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-flashnext/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``, ``<impl>`` the
op module's ``IMPLS`` key; the region is built by ``flashnext.neuron.sim_l1.ple``
(inputs as in the project's ``bench/bench_ple.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import neuron_spec

KIND = "ple_post"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class PlePostArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows on this rank.")
    embed_dim: int = arg(unit="elements", doc="PLE embedding width (heads x head_dim).")
    hidden_size: int = arg(
        unit="elements", doc="Model hidden size (features of one residual stream)."
    )
    hc_mult: int = arg(unit="streams", doc="Residual streams per token.")
    conv_kernel_size: int = arg(unit="taps", doc="PLE short-conv filter taps.")
    dilation: int = arg(unit="tokens", doc="Spacing between short-conv taps.")
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="PLE post (Neuron)",
    summary=("The PLE block after its all-reduce: project, gate, conv and residual in one region."),
    description=(
        "The post-collective PLE region: key/value projections, per-stream similarity "
        "gate, dilated depthwise short conv with zero state, and the residual add. Real"
        " layer-1 PLE weights; num_tokens <= 256 is a decode step (num_tokens sequences"
        " x 1 token), larger num_tokens one prefill chunk."
    ),
    category="Other",
    formula=("S' = S + gate(proj(emb), S) + conv(...)",),
    default_metric="time_ms",
    method=METHOD,
    caveats=(),
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch",
        PlePostArgs,
        "Project, gate, short conv and residual fused in one plain-torch region.",
    )
)

__all__ = ["KIND", "PlePostArgs"]
