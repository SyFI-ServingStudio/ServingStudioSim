"""HC mix on the Flash-Next Trainium engine.

Hyper-connection pre-block mix: grouped RMSNorm, low-rank gate and injection logits.

Wire string ``'hc_mix'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-flashnext/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``, ``<impl>`` the
op module's ``IMPLS`` key; the region is built by ``flashnext.neuron.sim_l1.hc``
(inputs as in the project's ``bench/bench_hc.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import neuron_spec

KIND = "hc_mix"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class HcMixArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows on this rank.")
    hidden_size: int = arg(
        unit="elements", doc="Model hidden size (features of one residual stream)."
    )
    hc_mult: int = arg(unit="streams", doc="Residual streams per token.")
    lowrank: int = arg(unit="elements", doc="Rank of the HC low-rank gate projection.")
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="HC mix (Neuron)",
    summary=(
        "Hyper-connection pre-block mix: grouped RMSNorm, low-rank gate and injection logits."
    ),
    description=(
        "The hyper-connection module before a block: Gemma RMSNorm per stream, a "
        "rank-lowrank gate (down, SiLU, up, sigmoid) over the hc_mult·hidden_size "
        "streams, the stream mean as the block input, and the injection logits for the "
        "next combine. HC weights are replicated, so the per-rank cost is the op at the"
        " rank's tokens. Seeded random weights and streams, as bench_hc.py."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "x = mean_s(σ(up(silu(down(GN(S))))) ⊙ GN(S))",
        "FLOPs = 2·num_tokens·hc_mult·hidden_size·(2·lowrank + hc_mult)",
    ),
    default_metric="time_ms",
    method=METHOD,
    caveats=(),
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND, "torch", HcMixArgs, "Grouped norm, low-rank gate and injection logits in plain torch."
    )
)
register(
    neuron_spec(
        KIND,
        "torch_cat",
        HcMixArgs,
        "As torch with the down and inject projections stacked into one GEMM.",
    )
)

__all__ = ["KIND", "HcMixArgs"]
