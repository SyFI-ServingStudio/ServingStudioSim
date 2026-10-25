"""MoE shared expert on the Flash-Next Trainium engine.

The sigmoid-gated shared-expert SwiGLU MLP, split over ranks.

Wire string ``'moe_shared_expert'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "moe_shared_expert"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class MoeSharedExpertArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows on this rank.")
    hidden_size: int = arg(
        unit="elements", doc="Model hidden size (features of one residual stream)."
    )
    intermediate_size: int = arg(
        unit="elements", doc="Expert intermediate columns on this rank (640 / moe tp)."
    )
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="MoE shared expert (Neuron)",
    summary="The sigmoid-gated shared-expert SwiGLU MLP, split over ranks.",
    description=(
        "sigmoid(x·g) · down(silu(x·G) ⊙ x·U) with the rank's slice of the real layer-0"
        " shared expert (intermediate 640 / split). The partial output needs the "
        "caller's all-reduce."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=(
        "y = σ(x·gᵀ) · down(silu(x·G) ⊙ x·U)",
        "FLOPs = 2·num_tokens·3·hidden_size·intermediate_size",
    ),
    default_metric="time_ms",
    method=METHOD,
    caveats=(),
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(neuron_spec(KIND, "torch", MoeSharedExpertArgs, "Gated SwiGLU MLP in plain torch."))
register(
    neuron_spec(KIND, "nki", MoeSharedExpertArgs, "nkilib core mlp kernel, gate scaling in torch.")
)

__all__ = ["KIND", "MoeSharedExpertArgs"]
