"""QSA sparse attention on the Flash-Next Trainium engine.

GQA attention of each query row over its selected groups and its open group's tail.

Wire string ``'qsa_attention'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "qsa_attention"


@dataclass(frozen=True)
class QsaAttentionArgs(KernelArgs):
    mode: str = c.mode()
    num_rows: int = c.num_rows()
    context_len: int = c.context_len()
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
    title="QSA sparse attention (Neuron)",
    summary=("GQA attention of each query row over its selected groups and its open group's tail."),
    description=(
        "The attention core of QSA on a fixed selection: each query row attends to the "
        "index_topk_tokens tokens of its selected groups plus the causal tail of its open"
        " group, read from the paged K and V caches; below the budget the selection is "
        "the whole causal context. Selections are random top-k sets over the visible "
        "groups."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "o[r] = softmax(q[r]·K[sel(r)]ᵀ / √head_dim) · V[sel(r)]",
        "S = min(context_len, index_topk_tokens + compress_ratio)",
        "FLOPs ≈ 4·num_rows·num_q_heads·head_dim·S",
    ),
    default_metric="time_ms",
    method=c.NEURON_METHOD,
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch_dense",
        QsaAttentionArgs,
        ("Masked dense attention over the whole gathered context (O(context))."),
    )
)
register(
    neuron_spec(
        KIND,
        "torch_sparse",
        QsaAttentionArgs,
        ("Gather of the selected groups' K and V per row, then masked softmax attention in torch."),
    )
)
register(
    neuron_spec(
        KIND,
        "nki",
        QsaAttentionArgs,
        (
            "nkilib attention_cte (dense prefill regime) or attention_tkg over the "
            "gathered selection."
        ),
    )
)

__all__ = ["KIND", "QsaAttentionArgs"]
