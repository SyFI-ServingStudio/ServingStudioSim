"""The stock vLLM Neuron serving contract every Trainium2 consumer shares.

Llama 3.1 8B BF16 at TP4 on one Trainium2 chip (four LNC2 ranks). The engine
is started with a fixed paged KV pool of 6782 pages of 32 tokens
(``--num-gpu-blocks-override 6782``; vLLM Neuron 0.29 accepts it and reports
217,024 KV tokens; doc/pr/llama31-8b-trn2/README.md), one prefill graph per context C (the
compiled ``max_model_len`` and ``max_num_batched_tokens``) and the configured
decode graphs (``num_seqs_buckets``). The Rust side mirrors these rules in
``simulator/src/timing/kernels/neuron_nki.rs``.

Imports nothing heavy, so host-side adapters can use it.
"""

from __future__ import annotations

KV_BLOCKS = 6782
BLOCK_SIZE = 32
TP_SIZE = 4
KV_SLOTS = KV_BLOCKS * BLOCK_SIZE
# The contexts the engine is started at, one prefill graph each.
CONTEXTS = (128, 512, 1024, 2048, 4096, 8192)


def resident_capacity(context: int) -> int:
    """Requests whose full context fits the KV pool at once."""
    return KV_SLOTS // context


def largest_decode_bucket(context: int) -> int:
    """The smallest power-of-two decode bucket holding every KV-resident request.

    The stock engine's token budget is the context, so a decode batch can also
    never exceed the context (the C128 B256 slot-mapping failure).
    """
    capacity = resident_capacity(context)
    return min(1 << (capacity - 1).bit_length(), context)


# Per-rank geometry of Llama 3.1 8B at TP4: hidden 4096; 32 query and 8 KV heads of
# 128 split four ways; MLP intermediate 14336 split four ways. The standalone NKI
# leaves (nki_units, nki_bench) build and account their launch units from these.
HIDDEN = 4096
HEAD_DIM = 128
Q_HEADS_PER_RANK = 32 // TP_SIZE
KV_HEADS_PER_RANK = 8 // TP_SIZE
INTERMEDIATE_PER_RANK = 14336 // TP_SIZE
QKV_OUT_PER_RANK = (Q_HEADS_PER_RANK + 2 * KV_HEADS_PER_RANK) * HEAD_DIM  # 1536
VOCAB = 128256
VOCAB_PER_RANK = VOCAB // TP_SIZE  # 32064, the lm_head shard
