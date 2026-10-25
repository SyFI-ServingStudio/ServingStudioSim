"""PLE residual on the Flash-Next Trainium engine.

Add the gated PLE value and its short-conv output into the streams.

Wire string ``'ple_residual'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "ple_residual"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class PleResidualArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows on this rank.")
    hidden_size: int = arg(
        unit="elements", doc="Model hidden size (features of one residual stream)."
    )
    hc_mult: int = arg(unit="streams", doc="Residual streams per token.")
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="PLE residual (Neuron)",
    summary="Add the gated PLE value and its short-conv output into the streams.",
    description=(
        "streams + bf16(gated + conv) over hc_mult·hidden_size features. Seeded random "
        "rows, as bench_ple.py."
    ),
    category="Other",
    formula=("S' = S + bf16(gated + conv)", "bytes ≈ 8·num_tokens·hc_mult·hidden_size"),
    default_metric="time_ms",
    method=METHOD,
    caveats=(),
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(neuron_spec(KIND, "torch", PleResidualArgs, "streams + bf16(gated + conv), plain torch."))

__all__ = ["KIND", "PleResidualArgs"]
