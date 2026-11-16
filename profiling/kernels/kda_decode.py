"""KDA decode (Neuron).

The whole KDA block of one rank over a decode batch (one token per request, recurrent state update).

Wire string ``'kda_decode'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported from the glm5-trn2
bench (bench_kda.py kind kda_decode) by ``tools/neuron-bench-import`` and are never profiled live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "kda_decode"


@dataclass(frozen=True)
class KdaDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="tokens", doc="Decode rows (requests, bucket-padded).")
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    num_heads: int = arg(unit="elements", doc="KDA heads on this rank (64 / attn_tp).")
    head_dim: int = arg(unit="elements", doc="KDA head dimension.")
    conv_kernel_size: int = arg(unit="elements", doc="Short-conv kernel width.")
    dtype: DType = arg(doc="Activation dtype.")
    weight_dtype: DType = arg(doc="Weight storage dtype.")


DOC = KernelDoc(
    title="KDA decode (Neuron)",
    summary='The whole KDA block of one rank over a decode batch (one token per request, recurrent state update).',
    description=(
        "One compiled region of the glm5-trn2 Neuron engine on one Trainium2 LNC2 logical core, "
        "at per-rank shapes (heads / experts / vocab of this rank). Rows are imported from the "
        "engine's own benchmark (L1_CONTRACT.md); the region's TP / world collectives are separate kinds."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=("S' = decay(S) + beta k (v - S^T k)^T; y = o_proj(norm(S'^T q))",),
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
    register(neuron_spec(KIND, _impl, KdaDecodeArgs, "KDA decode (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "KdaDecodeArgs"]
