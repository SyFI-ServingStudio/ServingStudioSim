"""The per-step rest of the decode forward beside its layers: the
vocab-sharded embedding and its all-reduce, the RoPE tables, the final norm,
the lm_head shard and the engine's sampler across the shards, plus the
graph's own fixed cost; the intercept of the reduced-depth TP4 forward that
also gives ``neuron_decoder_layer``."""

from __future__ import annotations

from profiling.db.doc import KernelDoc
from profiling.db.registry import register
from profiling.kernels.neuron_decoder_layer import (
    FORWARD_CAVEATS,
    FORWARD_METHOD,
    NeuronDecodeForwardArgs,
    forward_specs,
)

KIND = "neuron_decode_fixed"

DOC = KernelDoc(
    title="Decode step fixed term (TP4, intercept of a reduced-depth forward)",
    summary=(
        "Everything a Llama 3.1 8B TP4 decode step runs beside its layers, per rank: embedding, "
        "RoPE, final norm, lm_head, the engine's on-device sampler, the graph's fixed cost."
    ),
    description=(
        "t(8 layers) - 8 x layer of the engine's decode forward measured on the chip's four ranks "
        "(see neuron_decoder_layer). The sampler is the engine's NF.sample with its default "
        "on-device sampling config (all_greedy false, max_top_k 256): the compiled graph runs the "
        "distributed top-k path for every row, so the term grows with the batch bucket."
    ),
    category="Other",
    formula=(
        "flops = 2*B*4096*32064 per rank (the lm_head shard)",
        "bytes = 2 * (4096*32064 + 2*B*4096)",
    ),
    default_metric="time_ms",
    method=FORWARD_METHOD,
    caveats=FORWARD_CAVEATS,
)

for _spec in forward_specs(
    KIND,
    NeuronDecodeForwardArgs,
    "The per-step fixed term of a TP4 reduced-depth decode forward on the NKI kernel attention "
    "path, on the chip's four LNC2 cores.",
    "The per-step fixed term of a TP4 reduced-depth decode forward on the stock call's torch "
    "fallback attention, on the chip's four LNC2 cores.",
):
    register(_spec)
