"""QSA projection on the Flash-Next Trainium engine.

The QSA q/gate, k and v projections with Gemma q/k norms and partial RoPE.

Wire string ``'qsa_project'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "qsa_project"


@dataclass(frozen=True)
class QsaProjectArgs(KernelArgs):
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
    title="QSA projection (Neuron)",
    summary=("The QSA q/gate, k and v projections with Gemma q/k norms and partial RoPE."),
    description=(
        "x [num_rows, hidden_size] through the rank's packed [q | gate] projection and "
        "its KV heads' k and v projections, Gemma (1 + w) norms on q and k, and NeoX RoPE"
        " on the first 64 channels."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=("FLOPs = 2·num_rows·hidden_size·head_dim·(2·num_q_heads + 2·num_kv_heads)",),
    default_metric="time_ms",
    method=c.NEURON_METHOD,
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch",
        QsaProjectArgs,
        ("q/gate, k, v projections, Gemma q/k norms and partial NeoX RoPE in plain torch."),
    )
)

__all__ = ["KIND", "QsaProjectArgs"]
