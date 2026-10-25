"""QSA indexer compress on the Flash-Next Trainium engine.

Mean-pool each completed group of raw indexer keys into one compressed key and write it.

Wire string ``'qsa_indexer_compress'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "qsa_indexer_compress"


@dataclass(frozen=True)
class QsaIndexerCompressArgs(KernelArgs):
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
    title="QSA indexer compress (Neuron)",
    summary=(
        "Mean-pool each completed group of raw indexer keys into one compressed key and write it."
    ),
    description=(
        "For every group of compress_ratio tokens that completes in the step: the mean of"
        " its raw keys (fp32), a Gemma norm and RoPE at the group's first position, "
        "written to the compressed pool. Prefill computes num_rows / compress_ratio + 1 "
        "groups, decode one."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=("comp_g = RoPE(norm(mean(raw[4g .. 4g + 3])))",),
    default_metric="time_ms",
    method=c.NEURON_METHOD,
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch",
        QsaIndexerCompressArgs,
        (
            "Mean of compress_ratio raw keys, Gemma norm and RoPE, written to the "
            "compressed pool, in plain torch."
        ),
    )
)

__all__ = ["KIND", "QsaIndexerCompressArgs"]
