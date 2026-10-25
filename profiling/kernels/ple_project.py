"""PLE project on the Flash-Next Trainium engine.

PLE key and value projections of the gathered n-gram embedding.

Wire string ``'ple_project'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "ple_project"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class PleProjectArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows on this rank.")
    embed_dim: int = arg(unit="elements", doc="PLE embedding width (heads x head_dim).")
    hidden_size: int = arg(
        unit="elements", doc="Model hidden size (features of one residual stream)."
    )
    hc_mult: int = arg(unit="streams", doc="Residual streams per token.")
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="PLE project (Neuron)",
    summary="PLE key and value projections of the gathered n-gram embedding.",
    description=(
        "emb [T, embed_dim] to key [T, hc_mult·hidden_size] and value [T, hidden_size] "
        "with the replicated real layer-1 PLE weights. Seeded random embedding rows, as"
        " bench_ple.py."
    ),
    category="Other",
    formula=(
        "key = emb·Kᵀ, value = emb·Vᵀ",
        "FLOPs = 2·num_tokens·embed_dim·(hc_mult + 1)·hidden_size",
    ),
    default_metric="time_ms",
    method=METHOD,
    caveats=(),
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(KIND, "torch", PleProjectArgs, "Key and value projections, plain torch GEMMs.")
)

__all__ = ["KIND", "PleProjectArgs"]
