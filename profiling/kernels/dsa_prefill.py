"""DSA prefill (Neuron).

The whole DSA block of one rank over one request chunk: MLA (NoPE) with TP-sharded heads, the
replicated indexer and top-k selection, latent/indexer cache writes.

Wire string ``'dsa_prefill'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported
from the glm5-trn2
bench (bench_dsa.py kind dsa_prefill) by ``tools/neuron-bench-import`` and are never profiled live
here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "dsa_prefill"


@dataclass(frozen=True)
class DsaPrefillArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens of the one request chunk (bucket-padded).")
    context_len: int = arg(
        unit="tokens", doc="Context bucket the region reads ([:C], C >= start + T)."
    )
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    num_heads: int = arg(unit="elements", doc="MLA heads on this rank (64 / attn_tp).")
    kv_lora_rank: int = arg(unit="elements", doc="Latent KV width.")
    head_dim: int = arg(unit="elements", doc="MLA query/key (NoPE) and value head dimension.")
    index_heads: int = arg(unit="elements", doc="Indexer heads (replicated).")
    index_head_dim: int = arg(unit="elements", doc="Indexer head dimension.")
    index_topk: int = arg(unit="elements", doc="Selected tokens per query row.")
    dtype: DType = arg(doc="Activation dtype.")
    weight_dtype: DType = arg(doc="Weight storage dtype.")


DOC = KernelDoc(
    title="DSA prefill (Neuron)",
    summary=(
        "The whole DSA block of one rank over one request chunk: MLA (NoPE) with TP-sharded "
        "heads, the replicated indexer and top-k selection, latent/indexer cache writes."
    ),
    description=(
        "One compiled region of the glm5-trn2 Neuron engine on one Trainium2 LNC2 logical core, "
        "at per-rank shapes (heads / experts / vocab of this rank). Rows are imported from the "
        "engine's own benchmark (L1_CONTRACT.md); the region's TP / world collectives are"
        " separate kinds."
    ),
    category="Attention",
    subcategory="DSA",
    formula=("y = o_proj(MLA(q, latent[topk(indexer(q, keys[:C]))]))",),
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
    register(neuron_spec(KIND, _impl, DsaPrefillArgs, "DSA prefill (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "DsaPrefillArgs"]
