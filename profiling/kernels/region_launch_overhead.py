"""Region launch overhead on the Flash-Next Trainium engine.

Host wall time per step of back-to-back trivial compiled regions.

Wire string ``'region_launch_overhead'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-flashnext/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``, ``<impl>`` the
op module's ``IMPLS`` key; the region is built by ``flashnext.neuron.sim_l1.launch``
(inputs as in the project's ``scripts/probe_overhead.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import neuron_spec

KIND = "region_launch_overhead"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class RegionLaunchOverheadArgs(KernelArgs):
    num_regions: int = arg(
        unit="regions", doc="Distinct compiled regions launched back to back in one step."
    )
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="Region launch overhead (Neuron)",
    summary="Host wall time per step of back-to-back trivial compiled regions.",
    description=(
        "One step launches num_regions distinct one-NEFF regions back to back on a [1, "
        "2560] bf16 input, each consuming the previous output, then waits. The time is "
        "WALL clock (pipelined, per step), not device time: host launch, executor "
        "queueing and the inter-region gap, plus the trivial regions' own device time."
    ),
    category="Other",
    formula=("time = wall per step / 1",),
    default_metric="time_ms",
    method=(
        "Wall clock, not NRT device time: runtime.bench with one step in flight, mean "
        "per step over 50 steps after 5 warm-up steps, the median of 3 such runs, on "
        "one Trainium2 LNC2 logical core (scripts/probe_overhead.py chain_k_regions)."
    ),
    caveats=("Wall time depends on host load; measure on a quiet host.",),
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch",
        RegionLaunchOverheadArgs,
        "num_regions distinct trivial compiled regions (x + c) chained in one step.",
    )
)

__all__ = ["KIND", "RegionLaunchOverheadArgs"]
