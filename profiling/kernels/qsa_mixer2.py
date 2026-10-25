"""QSA mixer, two regions on the Flash-Next Trainium engine.

One QSA layer as the engine launches it: a front region up to the top-k select and a back region
with attention and o_proj.

Wire string ``'qsa_mixer2'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "qsa_mixer2"


@dataclass(frozen=True)
class QsaMixer2Args(KernelArgs):
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
    title="QSA mixer, two regions (Neuron)",
    summary=(
        "One QSA layer as the engine launches it: a front region up to the top-k select "
        "and a back region with attention and o_proj."
    ),
    description=(
        "The Flash-Next Trainium engine's QSAMixer2 (ops/qsa.py): the front region "
        "projects q, k, v and the gate, projects the indexer, writes the paged K, V and "
        "raw-key caches, compresses completed groups, scores every visible compressed "
        "group and selects the top index_topk_tokens / compress_ratio groups (NKI "
        "rotational top-k); the back region runs the sparse attention over the selected "
        "groups plus the open group's tail and the sigmoid-gated o_proj. Splitting at the"
        " selection avoids the in-graph gather penalty of the single-region mixer. Time "
        "is the sum of both regions' device time; the launch of the second region is not "
        "included. The output is the rank's TP partial sum."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "front: q, k, v, gate, q_idx = proj(x); cache ← k, v, raw",
        "front: groups = topk(Σ_h relu(q_idx·comp))",
        "back: y = o_proj(attn(q, K[groups], V[groups]) · σ(gate))",
    ),
    default_metric="time_ms",
    method=c.NEURON_METHOD,
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch_sparse",
        QsaMixer2Args,
        (
            "QSAMixer2 with the gather + masked softmax sparse attention in the back "
            "region; NKI top-k select."
        ),
    )
)
register(
    neuron_spec(
        KIND,
        "nki",
        QsaMixer2Args,
        (
            "QSAMixer2 with nkilib attention (attention_cte in the dense prefill regime, "
            "attention_tkg otherwise) in the back region; NKI top-k select."
        ),
    )
)

__all__ = ["KIND", "QsaMixer2Args"]
