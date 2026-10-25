"""HC final mix on the Flash-Next Trainium engine.

The final hyper-connection mixer before the LM head (no injection).

Wire string ``'hc_mix_final'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "hc_mix_final"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class HcMixFinalArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows on this rank.")
    hidden_size: int = arg(
        unit="elements", doc="Model hidden size (features of one residual stream)."
    )
    hc_mult: int = arg(unit="streams", doc="Residual streams per token.")
    lowrank: int = arg(unit="elements", doc="Rank of the HC low-rank gate projection.")
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="HC final mix (Neuron)",
    summary="The final hyper-connection mixer before the LM head (no injection).",
    description=(
        "hc_mix without the injection projection: produces the stream that the LM head "
        "reads. Seeded random weights and streams, as bench_hc.py."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "x = mean_s(σ(up(silu(down(GN(S))))) ⊙ GN(S))",
        "FLOPs = 4·num_tokens·hc_mult·hidden_size·lowrank",
    ),
    default_metric="time_ms",
    method=METHOD,
    caveats=(),
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND, "torch", HcMixFinalArgs, "The final mixer (no injection logits) in plain torch."
    )
)

__all__ = ["KIND", "HcMixFinalArgs"]
