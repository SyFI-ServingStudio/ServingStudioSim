"""The stock vLLM Neuron serving contract every Trainium2 consumer shares.

Llama 3.1 8B BF16 at TP4 on one Trainium2 chip (four LNC2 ranks), with the
paged KV pool vLLM Neuron allocates there: 6782 pages of 32 tokens. A context
C is the compiled ``max_model_len``. The Rust side mirrors these rules in
``simulator/src/timing/kernels/neuron_llama_forward.rs``;
``tests/test_neuron_full_forward.py`` checks that the two agree.

Imports nothing heavy, so the alignment adapter can use it on the host.
"""

from __future__ import annotations

KV_BLOCKS = 6782
BLOCK_SIZE = 32
TP_SIZE = 4
KV_SLOTS = KV_BLOCKS * BLOCK_SIZE
# Compiled contexts with accepted whole-forward rows.
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


def prefill_validation_bucket(context: int) -> int:
    """Decode bucket of the frozen full-logit workload that accepts a prefill row:
    16, or the context's largest bucket when that is smaller."""
    return min(16, largest_decode_bucket(context))
