"""MoE router (Neuron).

The replicated fp32 router alone (logits, sigmoid, bias, top-k) over the gathered rows.

Wire string ``'moe_router'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported from the glm5-trn2
bench (bench_moe.py kind router) by ``tools/neuron-bench-import`` and are never profiled live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "moe_router"


@dataclass(frozen=True)
class MoeRouterArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows routed.")
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    num_experts: int = arg(unit="elements", doc="Routed experts.")
    top_k: int = arg(unit="elements", doc="Experts per token.")
    dtype: DType = arg(doc="Activation dtype.")


for _impl in GLM_IMPLS[KIND]:
    register(neuron_spec(KIND, _impl, MoeRouterArgs, "MoE router (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "MoeRouterArgs"]
