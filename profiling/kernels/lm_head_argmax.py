"""LM head + argmax on the Flash-Next Trainium engine.

This rank's LM-head logits and their local argmax in one region.

Wire string ``'lm_head_argmax'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "lm_head_argmax"
METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the median "
    "over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction and "
    "warm-up stay outside the trace. Host launch overhead is the separate kind "
    "region_launch_overhead."
)


@dataclass(frozen=True)
class LmHeadArgmaxArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows on this rank.")
    hidden_size: int = arg(
        unit="elements", doc="Model hidden size (features of one residual stream)."
    )
    vocab_size: int = arg(unit="rows", doc="Vocab rows on this rank (248320 / tp).")
    dtype: DType = arg(doc="Activation and weight element type.")


DOC = KernelDoc(
    title="LM head + argmax (Neuron)",
    summary="This rank's LM-head logits and their local argmax in one region.",
    description=(
        "x [num_tokens, hidden_size] times the rank's lm_head rows [vocab_size, "
        "hidden_size], then the local max and global index; the caller all-gathers the "
        "pairs. Seeded random weights, as bench_head.py."
    ),
    category="GEMM",
    formula=("(v, i) = max(x·Wᵀ, dim=-1)", "FLOPs = 2·num_tokens·hidden_size·vocab_size"),
    default_metric="tflops",
    method=METHOD,
    caveats=(),
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(neuron_spec(KIND, "torch", LmHeadArgmaxArgs, "LM-head GEMM and torch.max in one region."))
register(
    neuron_spec(
        KIND,
        "nki",
        LmHeadArgmaxArgs,
        "LM-head GEMM and nkilib cascaded_max in one region (num_tokens <= 256).",
    )
)

__all__ = ["KIND", "LmHeadArgmaxArgs"]
