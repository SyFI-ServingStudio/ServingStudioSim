"""MoE partial (Neuron).

One EP rank's MoE layer over every gathered row: router, its local routed experts and its shard of the shared expert (partial sum before the world reduce-scatter).

Wire string ``'moe_partial'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported from the glm5-trn2
bench (bench_moe.py kind moe_partial (critical EP rank, real routing)) by ``tools/neuron-bench-import`` and are never profiled live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "moe_partial"


@dataclass(frozen=True)
class MoePartialArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows gathered over the world (every DP group's rows).")
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    num_experts: int = arg(unit="elements", doc="Routed experts in the model.")
    top_k: int = arg(unit="elements", doc="Experts per token.")
    local_experts: int = arg(unit="elements", doc="Routed experts on this rank (num_experts / ep).")
    intermediate_size: int = arg(unit="elements", doc="Expert intermediate width.")
    shared_intermediate_size: int = arg(unit="elements", doc="Shared-expert intermediate columns on this rank (2048 / ep).")
    dtype: DType = arg(doc="Activation dtype.")
    weight_dtype: DType = arg(doc="Expert weight storage dtype (FP8, upcast in-graph).")


DOC = KernelDoc(
    title="MoE partial (Neuron)",
    summary="One EP rank's MoE layer over every gathered row: router, its local routed experts and its shard of the shared expert (partial sum before the world reduce-scatter).",
    description=(
        "One compiled region of the glm5-trn2 Neuron engine on one Trainium2 LNC2 logical core, "
        "at per-rank shapes (heads / experts / vocab of this rank). Rows are imported from the "
        "engine's own benchmark (L1_CONTRACT.md); the region's TP / world collectives are separate kinds."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=('y = sum_{e in local, topk(x)} g_e(x) FFN_e(x) + shared_shard(x)',),
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
    register(neuron_spec(KIND, _impl, MoePartialArgs, "MoE partial (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "MoePartialArgs"]
