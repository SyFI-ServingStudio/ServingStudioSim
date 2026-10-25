"""Vocab argmax on the Flash-Next Trainium engine.

Greedy sampling's local step: max and index over this rank's vocab rows.

Wire string ``'vocab_argmax'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-flashnext/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``, ``<impl>`` the
op module's ``IMPLS`` key; the region is built by ``flashnext.neuron.sim_l1.head``
(inputs as in the project's ``bench/bench_head.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import neuron_spec

KIND = "vocab_argmax"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class VocabArgmaxArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows on this rank.")
    vocab_size: int = arg(unit="rows", doc="Vocab rows on this rank (248320 / tp).")
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="Vocab argmax (Neuron)",
    summary="Greedy sampling's local step: max and index over this rank's vocab rows.",
    description=(
        "fp32 logits [num_tokens, vocab_size] to (max value, global index) per token; "
        "the caller all-gathers the pairs over the vocab ranks. Seeded random logits, "
        "as bench_head.py."
    ),
    category="Other",
    formula=("(v, i) = max(logits, dim=-1)", "bytes = 4·num_tokens·vocab_size"),
    default_metric="time_ms",
    method=METHOD,
    caveats=(),
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(neuron_spec(KIND, "torch", VocabArgmaxArgs, "torch.max over the local fp32 logits."))
register(
    neuron_spec(
        KIND,
        "nki",
        VocabArgmaxArgs,
        "nkilib cascaded_max per 16384-column chunk (num_tokens <= 256).",
    )
)

__all__ = ["KIND", "VocabArgmaxArgs"]
