"""MoE router on the Flash-Next Trainium engine.

Router logits, top-k expert selection and renormalized weights.

Wire string ``'moe_router'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "moe_router"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class MoeRouterArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows on this rank.")
    hidden_size: int = arg(
        unit="elements", doc="Model hidden size (features of one residual stream)."
    )
    num_experts: int = arg(unit="experts", doc="Experts in the model's router (global).")
    top_k: int = arg(unit="experts", doc="Experts each token is routed to.")
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="MoE router (Neuron)",
    summary="Router logits, top-k expert selection and renormalized weights.",
    description=(
        "x [T, H] times the replicated router weight [E, H] gives fp32 logits; each "
        "token keeps its top_k experts with renormalized softmax weights. The real "
        "layer-0 router weight and MoE input."
    ),
    category="MoE",
    subcategory="Routing and combine",
    formula=(
        "w, ids = topk(softmax(x·Wᵍᵀ), top_k)",
        "FLOPs = 2·num_tokens·hidden_size·num_experts",
    ),
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
        MoeRouterArgs,
        ("fp32 logits, softmax over all experts, then top-k (the reference order), plain torch."),
    )
)
register(
    neuron_spec(
        KIND,
        "torch_topk_first",
        MoeRouterArgs,
        "fp32 logits, top-k on the logits, softmax over the k values, plain torch.",
    )
)
register(
    neuron_spec(
        KIND, "torch_bf16", MoeRouterArgs, "Like torch_topk_first with a BF16 router matmul."
    )
)
register(
    neuron_spec(
        KIND,
        "nki_topk",
        MoeRouterArgs,
        "fp32 logits (torch) and the nkilib rotational top-k kernel.",
    )
)

__all__ = ["KIND", "MoeRouterArgs"]
