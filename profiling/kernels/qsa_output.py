"""QSA output on the Flash-Next Trainium engine.

Sigmoid-gate the attention output and apply the row-parallel o_proj.

Wire string ``'qsa_output'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-flashnext/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``, ``<impl>`` the
impl key the Flash-Next engine's ``qsa`` ops use; the region is built by
``flashnext.neuron.sim_l1.qsa`` (inputs as in ``bench/bench_qsa.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc
from profiling.db.registry import register
from profiling.kernels import qsa_neuron_common as c
from profiling.kernels.neuron_common import neuron_spec

KIND = "qsa_output"


@dataclass(frozen=True)
class QsaOutputArgs(KernelArgs):
    mode: str = c.mode()
    num_rows: int = c.num_rows()
    hidden_size: int = c.hidden_size()
    num_q_heads: int = c.num_q_heads()
    num_kv_heads: int = c.num_kv_heads()
    head_dim: int = c.head_dim()
    index_heads: int = c.index_heads()
    index_head_dim: int = c.index_head_dim()
    index_topk_tokens: int = c.index_topk_tokens()
    compress_ratio: int = c.compress_ratio()
    dtype: DType = c.dtype()


DOC = KernelDoc(
    title="QSA output (Neuron)",
    summary=("Sigmoid-gate the attention output and apply the row-parallel o_proj."),
    description=(
        "attn·σ(gate) over the rank's query heads, then o_proj to hidden_size. The result"
        " is the rank's TP partial sum; the all-reduce is not included."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "y = o_proj(attn · σ(gate))",
        "FLOPs = 2·num_rows·num_q_heads·head_dim·hidden_size",
    ),
    default_metric="time_ms",
    method=c.NEURON_METHOD,
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch",
        QsaOutputArgs,
        ("Sigmoid output gate and the row-parallel o_proj GEMM in plain torch."),
    )
)

__all__ = ["KIND", "QsaOutputArgs"]
