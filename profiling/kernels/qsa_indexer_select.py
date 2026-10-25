"""QSA indexer select on the Flash-Next Trainium engine.

Pick each row's top groups by score plus its open group, and the token-slot validity mask.

Wire string ``'qsa_indexer_select'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "qsa_indexer_select"


@dataclass(frozen=True)
class QsaIndexerSelectArgs(KernelArgs):
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
    title="QSA indexer select (Neuron)",
    summary=(
        "Pick each row's top groups by score plus its open group, and the token-slot validity mask."
    ),
    description=(
        "From the group scores, select per query row the top index_topk_tokens / "
        "compress_ratio causally visible groups (all visible groups when there are no "
        "more than that) plus the open group, and build the [num_rows, 2052] token "
        "validity mask."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "k = index_topk_tokens / compress_ratio",
        "groups[r] = topk_{g visible}(score[r, g], k) ∪ {open group}",
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
        QsaIndexerSelectArgs,
        ("torch.topk (an exact two-stage top-k above 16384 groups)."),
    )
)
register(
    neuron_spec(
        KIND,
        "nki",
        QsaIndexerSelectArgs,
        ("vllm_neuron's rotational NKI top-k kernel."),
    )
)

__all__ = ["KIND", "QsaIndexerSelectArgs"]
