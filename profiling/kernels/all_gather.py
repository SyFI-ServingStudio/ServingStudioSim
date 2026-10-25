"""All-gather across Trainium2 logical cores (in-graph, Neuron).

Gather every rank's shard so each holds the whole buffer.

Wire string ``'all_gather'``; args ``num_gpus, message_size_bytes, dtype, fabric``
like ``all_reduce`` (``doc/pr/trn2-flashnext/L1_CONTRACT.md``). Backend
``neuron_cc``: the list-native runner ``profiling.runners.neuron.collectives``
measures one-chip groups live; multi-chip rows come from the standalone sweep
``flashnext/neuron/sim_l1/collectives.py`` through ``tools/neuron-bench-import``.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import MetricFamily, register
from profiling.kernels.neuron_common import neuron_spec

KIND = "all_gather"


@dataclass(frozen=True)
class AllGatherArgs(KernelArgs):
    num_gpus: int = arg(unit="ranks", doc="Ranks in the group (one Neuron logical core each).")
    message_size_bytes: int = arg(
        unit="bytes", doc="Per-rank input shard (bytes); the output is num_gpus times larger."
    )
    dtype: DType = arg(doc="Element type of the buffer (bf16).")
    fabric: str = arg(
        doc="Rank placement: neuron_c1 (1 rank per chip), neuron_c2 (2) or neuron_c4 (4)."
    )


DOC = KernelDoc(
    title="All-gather (Neuron)",
    summary="Gather every rank's shard so each holds the whole buffer.",
    description=(
        "message_size_bytes is the per-rank input shard; each rank ends with num_gpus "
        "shards. Ranks are Neuron logical cores; fabric names how they sit on chips."
    ),
    category="Communication",
    formula=("algbw = num_gpus·message_size_bytes / time", "busbw = algbw · (N − 1) / N"),
    default_metric="busbw_gbps",
    method=(
        "In-graph collectives as vllm-neuron runs them: one process per Trainium2 LNC2 "
        "logical core, a gloo world group, and torch functional collectives traced into"
        " one compiled region. Two regions chain 1 and 33 collectives; the time is the "
        "slope of rank 0's NRT device time, (t33 − t1) / 32, with a CPU barrier before "
        "every call, the median of 3 repeats of 20 calls each "
        "(flashnext.neuron.sim_l1.collectives). Live runs cover one-chip groups "
        "(neuron_c2 with 2 ranks, neuron_c4 with 4); multi-chip groups come from the "
        "standalone sweep through tools/neuron-bench-import."
    ),
    caveats=(
        "The chained form adds one elementwise op per collective (reduce_scatter also re-tiles its "
        "shard back to the full size), which the slope includes.",
    ),
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "cc",
        AllGatherArgs,
        "torch functional all_gather traced into one region (gloo world, a process per core).",
        compute=None,
        metric_family=MetricFamily.COMM,
        module="profiling.runners.neuron.collectives",
        function="profile_all_gather_batch",
        list_native=True,
        logical_cores=4,
    )
)

__all__ = ["KIND", "AllGatherArgs"]
