"""LM head (Neuron).

Final stream mean + norm, the vocab-parallel lm_head GEMM over the rank's vocab shard and the local
greedy argmax.

Wire string ``'lm_head'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported
from the glm5-trn2
bench (bench_mhc_head.py kind lm_head) by ``tools/neuron-bench-import`` and are never profiled live
here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "lm_head"


@dataclass(frozen=True)
class LmHeadArgs(KernelArgs):
    num_tokens: int = arg(
        unit="tokens", doc="Rows sampled (decode rows, or one per prefill chunk)."
    )
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    vocab_size: int = arg(unit="elements", doc="Vocabulary rows on this rank (vocab / attn_tp).")
    hc_mult: int = arg(unit="elements", doc="Residual streams.")
    dtype: DType = arg(doc="Activation dtype.")


DOC = KernelDoc(
    title="LM head (Neuron)",
    summary=(
        "Final stream mean + norm, the vocab-parallel lm_head GEMM over the rank's vocab "
        "shard and the local greedy argmax."
    ),
    description=(
        "One compiled region of the glm5-trn2 Neuron engine on one Trainium2 LNC2 logical core, "
        "at per-rank shapes (heads / experts / vocab of this rank). Rows are imported from the "
        "engine's own benchmark (L1_CONTRACT.md); the region's TP / world collectives are"
        " separate kinds."
    ),
    category="GEMM",
    formula=("(max, argmax)(W_vocab_shard rmsnorm(mean_i s_i))",),
    default_metric="time_ms",
    method=(
        "NRT system-trace device time of one region per call (glmflash runtime.device_bench, "
        "median over the bench's iterations), imported by tools/neuron-bench-import; cells the "
        "bench did not measure are derived rows (analytic x measured ratio, "
        "backend_version derived:*)."
    ),
    caveats=("Imported, not profiled live: a missing row fails the build under --no-gpu.",),
    reference=None,
)


for _impl in GLM_IMPLS[KIND]:
    register(neuron_spec(KIND, _impl, LmHeadArgs, "LM head (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "LmHeadArgs"]
