"""QSA indexer scores on the Flash-Next Trainium engine.

Score every visible compressed group for each query row.

Wire string ``'qsa_indexer_scores'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "qsa_indexer_scores"


@dataclass(frozen=True)
class QsaIndexerScoresArgs(KernelArgs):
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
    title="QSA indexer scores (Neuron)",
    summary=("Score every visible compressed group for each query row."),
    description=(
        "Gathers the request's compressed keys [context_len / compress_ratio, "
        "index_head_dim] through the block table and computes, per query row and group, "
        "the unit-weight sum over indexer heads of relu(q_h · k) / √index_head_dim."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "score[r, g] = Σ_h relu(q_idx[r, h] · comp[g]) / √d",
        "FLOPs = 2·num_rows·index_heads·index_head_dim·context_len / compress_ratio",
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
        QsaIndexerScoresArgs,
        ("fp32 matmul relu-sum over the gathered compressed keys."),
    )
)
register(
    neuron_spec(
        KIND,
        "torch_bf16",
        QsaIndexerScoresArgs,
        ("The same relu-sum with bf16 matmuls."),
    )
)

__all__ = ["KIND", "QsaIndexerScoresArgs"]
