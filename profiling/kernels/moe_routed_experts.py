"""MoE routed experts on the Flash-Next Trainium engine.

The rank's local experts' SwiGLU MLPs over the routed tokens.

Wire string ``'moe_routed_experts'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-flashnext/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``, ``<impl>`` the
op module's ``IMPLS`` key; the region is built by ``flashnext.neuron.sim_l1.moe``
(inputs as in the project's ``bench/bench_moe.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import neuron_spec

KIND = "moe_routed_experts"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class MoeRoutedExpertsArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows on this rank.")
    hidden_size: int = arg(
        unit="elements", doc="Model hidden size (features of one residual stream)."
    )
    num_experts: int = arg(unit="experts", doc="Experts in the model's router (global).")
    top_k: int = arg(unit="experts", doc="Experts each token is routed to.")
    local_experts: int = arg(unit="experts", doc="Experts this rank holds (num_experts / ep).")
    intermediate_size: int = arg(
        unit="elements", doc="Expert intermediate columns on this rank (640 / moe tp)."
    )
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="MoE routed experts (Neuron)",
    summary="The rank's local experts' SwiGLU MLPs over the routed tokens.",
    description=(
        "Routed expert compute of one EP/TP rank: tokens replicated on the rank, "
        "routing given, each local expert (intermediate columns sharded by moe tp) runs"
        " gate/up, SiLU·mul and down, weighted by the routing weight. The partial "
        "output needs the caller's all-reduce. Weights and routing are the real layer-0"
        ' MoE (bench_moe.py `routing == "real"`). The timed rank is the critical-path '
        "EP rank (most local assignments)."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=(
        "y = Σ_{e∈local} w_e · down_e(silu(x·G_e) ⊙ x·U_e)",
        "FLOPs (useful) = 2·num_tokens·top_k·3·hidden_size·intermediate_size",
    ),
    default_metric="time_ms",
    method=METHOD,
    caveats=(
        ("The cost depends on the routing; rows use real layer-0 routing and the busiest EP rank."),
    ),
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "auto",
        MoeRoutedExpertsArgs,
        (
            "Fastest exact impl per shape (torch_gather_w at T=1, nki_tkg, nki_tkg_all, "
            "then nki_cte)."
        ),
    )
)
register(
    neuron_spec(
        KIND,
        "nki",
        MoeRoutedExpertsArgs,
        "nkilib moe_tkg (selective) for T <= 128, moe_cte otherwise.",
    )
)
register(
    neuron_spec(
        KIND, "nki_tkg", MoeRoutedExpertsArgs, "nkilib moe_tkg, selective-expert mode (T <= 128)."
    )
)
register(
    neuron_spec(
        KIND, "nki_tkg_all", MoeRoutedExpertsArgs, "nkilib moe_tkg, all-expert mode (T <= 512)."
    )
)
register(
    neuron_spec(
        KIND, "nki_cte", MoeRoutedExpertsArgs, "nkilib moe_cte, blockwise prefill (T >= 128)."
    )
)
register(
    neuron_spec(
        KIND,
        "nki_cte_cf2",
        MoeRoutedExpertsArgs,
        ("nkilib moe_cte with a rank-level capacity of 2x the uniform share (T >= 128; may drop)."),
    )
)
register(
    neuron_spec(
        KIND,
        "torch_gather_w",
        MoeRoutedExpertsArgs,
        "Gather the selected experts' weights per (token, slot), plain torch (T <= 16).",
    )
)
register(
    neuron_spec(
        KIND,
        "torch_dense",
        MoeRoutedExpertsArgs,
        ("Every local expert on every token weighted by dense affinities, plain torch (T <= 256)."),
    )
)
register(
    neuron_spec(
        KIND,
        "torch_gather",
        MoeRoutedExpertsArgs,
        "Static per-expert capacity dispatch (capacity = exact max load), plain torch.",
    )
)
register(
    neuron_spec(
        KIND,
        "torch_onehot",
        MoeRoutedExpertsArgs,
        "Static capacity with one-hot dispatch and combine, plain torch (T >= 128).",
    )
)

__all__ = ["KIND", "MoeRoutedExpertsArgs"]
