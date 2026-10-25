"""Shared pieces of the Neuron QSA kinds (Flash-Next sparse attention on Trainium2).

Not a kernel kind: the ``qsa_*`` Neuron kind modules declare their args from
these, so every kind reads the same (``doc/pr/trn2-flashnext/L1_CONTRACT.md``):
``mode`` first, then the shape args (``num_rows`` and, for a context-dependent
kind, ``context_len``), then
``hidden_size, num_q_heads, num_kv_heads, head_dim, index_heads, index_head_dim,
index_topk_tokens, compress_ratio, dtype``.

Shapes are per rank: TP splits the 24 query heads (``24 / tp``) and each rank
keeps the KV head its query heads read (2 at tp 1, else 1); the indexer is
replicated on every rank. tp 1, 2, 4, 8 (24 heads do not split 16 ways).
"""

from __future__ import annotations

from typing import Any

from profiling.db.doc import arg

CATEGORY = "Attention"
SUBCATEGORY = "Sparse GQA"

#: How every Neuron QSA kind is timed (``flashnext.neuron.sim_l1.measure``).
NEURON_METHOD = (
    "NRT system-trace device time on one Trainium2 LNC2 logical core: the "
    "median over 30 calls, summed over the NEFFs one call launches "
    "(flashnext.neuron.runtime.device_bench). Compilation, input construction "
    "and warm-up stay outside the trace. Weights are the real layer-3 QSA "
    "weights of the rank-0 TP shard; hidden states, paged KV and indexer pools "
    "are seeded random bf16, built as the project's bench_qsa.py builds them "
    "(pools capped at 262144 physical tokens, block tables over random blocks). "
    "Host launch overhead is the separate kind region_launch_overhead."
)

OPS_MODULE = "flashnext/neuron/ops/qsa.py"


def mode() -> Any:
    return arg(
        doc="prefill (one request chunk with prior context) or decode (one token per request)."
    )


def num_rows() -> Any:
    return arg(
        unit="tokens",
        doc="Query rows: tokens of the one request chunk in prefill, requests in decode.",
    )


def context_len() -> Any:
    return arg(
        unit="tokens",
        doc=(
            "Visible context of the last row: prior + num_rows in prefill, the KV length "
            "including the new token in decode."
        ),
    )


def hidden_size() -> Any:
    return arg(unit="elements", doc="Model hidden size: the layer's input and output width.")


def num_q_heads() -> Any:
    return arg(unit="heads", doc="Query heads on this rank (24 / tp for Flash-Next).")


def num_kv_heads() -> Any:
    return arg(unit="heads", doc="KV heads cached on this rank (2 at tp 1, else 1).")


def head_dim() -> Any:
    return arg(unit="elements", doc="Features in each query, key and value head.")


def index_heads() -> Any:
    return arg(unit="heads", doc="Indexer query heads (replicated on every rank).")


def index_head_dim() -> Any:
    return arg(unit="elements", doc="Features in each indexer query head and compressed key.")


def index_topk_tokens() -> Any:
    return arg(unit="tokens", doc="Token budget each query row attends to beyond its open group.")


def compress_ratio() -> Any:
    return arg(unit="tokens", doc="Tokens mean-pooled into one compressed indexer key.")


def dtype() -> Any:
    return arg(doc="Element type of activations, weights and caches.")
