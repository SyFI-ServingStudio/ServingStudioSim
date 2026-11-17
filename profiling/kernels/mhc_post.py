"""mHC post (Neuron).

Hyper-connection post: write the module output back into the residual streams (post and comb mix).

Wire string ``'mhc_post'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported
from the glm5-trn2
bench (bench_mhc_head.py kind mhc_post) by ``tools/neuron-bench-import`` and are never profiled
live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "mhc_post"


@dataclass(frozen=True)
class MhcPostArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows on this rank (its sequence-parallel slice).")
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    hc_mult: int = arg(unit="elements", doc="Residual streams.")
    dtype: DType = arg(doc="Activation dtype.")


DOC = KernelDoc(
    title="mHC post (Neuron)",
    summary=(
        "Hyper-connection post: write the module output back into the residual streams (post "
        "and comb mix)."
    ),
    description=(
        "One compiled region of the glm5-trn2 Neuron engine on one Trainium2 LNC2 logical core, "
        "at per-rank shapes (heads / experts / vocab of this rank). Rows are imported from the "
        "engine's own benchmark (L1_CONTRACT.md); the region's TP / world collectives are"
        " separate kinds."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=("s_i' = post_i y + sum_j comb_ji s_j",),
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
    register(neuron_spec(KIND, _impl, MhcPostArgs, "mHC post (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "MhcPostArgs"]
