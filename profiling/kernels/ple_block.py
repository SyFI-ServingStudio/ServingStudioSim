"""PLE block on the Flash-Next Trainium engine.

The whole PLE block of one table rank in one region, all-reduce omitted.

Wire string ``'ple_block'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "ple_block"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class PleBlockArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows on this rank.")
    num_heads: int = arg(unit="heads", doc="PLE table ids per token.")
    head_dim: int = arg(unit="elements", doc="Table elements in one row.")
    table_rows: int = arg(
        unit="rows", doc="Rows of this rank's PLE table slice (320001536 / shards)."
    )
    hidden_size: int = arg(
        unit="elements", doc="Model hidden size (features of one residual stream)."
    )
    hc_mult: int = arg(unit="streams", doc="Residual streams per token.")
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="PLE block (Neuron)",
    summary="The whole PLE block of one table rank in one region, all-reduce omitted.",
    description=(
        "N-gram hash ids, the masked gather from this rank's real table slice (rank 0 "
        "of 320001536 / table_rows shards), and the post region. The all-reduce of the "
        "gathered rows is left out. num_tokens <= 256 is a decode step, larger "
        "num_tokens one prefill chunk."
    ),
    category="Other",
    formula=("S' = post(gather(ngram_ids(tokens)), S)",),
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
        PleBlockArgs,
        "Hash ids, local table gather, project, gate, conv and residual in one region.",
    )
)

__all__ = ["KIND", "PleBlockArgs"]
