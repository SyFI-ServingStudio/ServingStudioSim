"""QSA cache write on the Flash-Next Trainium engine.

Write the rows' K, V and raw indexer keys into the paged pools through slot mapping.

Wire string ``'qsa_cache_write'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "qsa_cache_write"


@dataclass(frozen=True)
class QsaCacheWriteArgs(KernelArgs):
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
    title="QSA cache write (Neuron)",
    summary=(
        "Write the rows' K, V and raw indexer keys into the paged pools through slot mapping."
    ),
    description=(
        "Slots computed on device from the block table and positions; K and V [num_rows, "
        "num_kv_heads, head_dim] and raw keys [num_rows, index_head_dim] copied in place "
        "into the 32-token-block pools."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=("bytes = 2·num_rows·(2·num_kv_heads·head_dim + index_head_dim)",),
    default_metric="time_ms",
    method=c.NEURON_METHOD,
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch",
        QsaCacheWriteArgs,
        ("Slot mapping from the block table and index_copy_ into the k, v and raw-key pools."),
    )
)

__all__ = ["KIND", "QsaCacheWriteArgs"]
