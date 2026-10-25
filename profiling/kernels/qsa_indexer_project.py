"""QSA indexer projection on the Flash-Next Trainium engine.

Project the indexer's query heads (norm + RoPE) and the raw indexer key of each row.

Wire string ``'qsa_indexer_project'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "qsa_indexer_project"


@dataclass(frozen=True)
class QsaIndexerProjectArgs(KernelArgs):
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
    title="QSA indexer projection (Neuron)",
    summary=(
        "Project the indexer's query heads (norm + RoPE) and the raw indexer key of each row."
    ),
    description=(
        "One GEMM to index_heads·index_head_dim query features plus one raw key of "
        "index_head_dim; the query heads get a Gemma norm and RoPE, the raw key is cached"
        " unnormalized for compression. Replicated on every rank."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=("FLOPs = 2·num_rows·hidden_size·(index_heads + 1)·index_head_dim",),
    default_metric="time_ms",
    method=c.NEURON_METHOD,
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch",
        QsaIndexerProjectArgs,
        ("Indexer q projection, norm and RoPE, and the raw indexer key, in plain torch."),
    )
)

__all__ = ["KIND", "QsaIndexerProjectArgs"]
