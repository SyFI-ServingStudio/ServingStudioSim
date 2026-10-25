"""MoE block on the Flash-Next Trainium engine.

Router, routed experts, shared expert and combine of one rank in one region.

Wire string ``'moe_block'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "moe_block"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class MoeBlockArgs(KernelArgs):
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
    shared_intermediate_size: int = arg(
        unit="elements", doc="Shared-expert intermediate columns on this rank (640 / split)."
    )
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="MoE block (Neuron)",
    summary="Router, routed experts, shared expert and combine of one rank in one region.",
    description=(
        "The whole MoE block of one EP/TP rank as one compiled region (bench impl "
        "torch_topk_first/<routed>/torch): the device router, the routed experts and "
        "the shared expert split over all ep·tp ranks. The partial output needs the "
        "caller's all-reduce. Weights and routing are the real layer-0 MoE "
        '(bench_moe.py `routing == "real"`). The timed rank is the critical-path EP '
        "rank."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=("y = routed(x, router(x)) + shared(x)",),
    default_metric="time_ms",
    method=METHOD,
    caveats=(),
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "auto",
        MoeBlockArgs,
        "Router torch_topk_first + routed auto + shared torch, one region.",
    )
)
register(
    neuron_spec(
        KIND,
        "nki",
        MoeBlockArgs,
        "Router torch_topk_first + routed nki + shared torch, one region.",
    )
)

__all__ = ["KIND", "MoeBlockArgs"]
