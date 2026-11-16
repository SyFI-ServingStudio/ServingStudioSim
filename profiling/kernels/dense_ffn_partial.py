"""Dense FFN partial (Neuron).

One rank's 1/ep column shard of a dense FFN layer (layers 0-2) over every gathered row.

Wire string ``'dense_ffn_partial'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported from the glm5-trn2
bench (bench_moe.py kind dense_ffn_partial) by ``tools/neuron-bench-import`` and are never profiled live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "dense_ffn_partial"


@dataclass(frozen=True)
class DenseFfnPartialArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows gathered over the world.")
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    intermediate_size: int = arg(unit="elements", doc="Intermediate columns on this rank (12288 / ep).")
    dtype: DType = arg(doc="Activation dtype.")
    weight_dtype: DType = arg(doc="Weight storage dtype (FP8, upcast in-graph).")


DOC = KernelDoc(
    title="Dense FFN partial (Neuron)",
    summary="One rank's 1/ep column shard of a dense FFN layer (layers 0-2) over every gathered row.",
    description=(
        "One compiled region of the glm5-trn2 Neuron engine on one Trainium2 LNC2 logical core, "
        "at per-rank shapes (heads / experts / vocab of this rank). Rows are imported from the "
        "engine's own benchmark (L1_CONTRACT.md); the region's TP / world collectives are separate kinds."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=('y = W_down[:, shard] silu(W_gate[shard] x) * (W_up[shard] x)',),
    default_metric="time_ms",
    method=(
        "NRT system-trace device time of one region per call (glmflash runtime.device_bench, "
        "median over the bench's iterations), imported by tools/neuron-bench-import; cells the "
        "bench did not measure are derived rows (analytic x measured ratio, backend_version derived:*)."
    ),
    caveats=("Imported, not profiled live: a missing row fails the build under --no-gpu.",),
    reference=None,
)


for _impl in GLM_IMPLS[KIND]:
    register(neuron_spec(KIND, _impl, DenseFfnPartialArgs, "Dense FFN partial (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "DenseFfnPartialArgs"]
