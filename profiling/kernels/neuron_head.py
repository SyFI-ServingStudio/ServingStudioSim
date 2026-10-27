"""The head unit of the Trainium2 Llama forward: embedding gather, final norm,
vocab-sharded lm_head and greedy argmax, timed alone per TP4 rank."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_nki_qkv import NKI_CAVEATS, NKI_METHOD, nki_leaf_spec

KIND = "neuron_head"


@dataclass(frozen=True)
class NeuronHeadArgs(KernelArgs):
    phase: str = arg(doc="prefill or decode: the forward phase whose launch this row times.")
    token_bucket: int = arg(
        unit="tokens",
        doc=(
            "Compiled variant: prefill token bucket T (the embedding gathers T rows, one row "
            "is sampled) or decode batch bucket B (B rows gathered and sampled)."
        ),
    )
    tp_size: int = arg(unit="ranks", doc="Four LNC2 ranks on one Trainium2 chip; per-rank shapes.")
    dtype: DType = arg(doc="Weights, activations and logits dtype; BF16 only.")


DOC = KernelDoc(
    title="Head unit of the Trainium2 Llama forward",
    summary=(
        "Embedding row gather, final RMSNorm of the sampled rows, the vocab-sharded lm_head "
        "GEMM and greedy argmax, compiled as one graph and timed alone per TP4 rank."
    ),
    description=(
        "The work of one forward outside the 32 layers, per rank: the embedding gather of the "
        "step's tokens ([rows, 4096] BF16 from a [128256/4, 4096] shard), the final RMSNorm of "
        "the sampled rows, lm_head [sampled, 4096] @ [4096, 32064] and an argmax over the local "
        "logits. The logits all-gather across the four ranks is a neuron_collective row. In "
        "prefill one row is sampled (the prompt's last token); in decode every request samples. "
        "Variants are the prefill token buckets of the stock context ladder and decode batch "
        "buckets (powers of two up to 512)."
    ),
    category="GEMM",
    formula=(
        "flops = 2 * sampled * 4096 * 32064 per rank (sampled = 1 in prefill, B in decode)",
        "bytes = 2 * (4096*32064 + rows*4096 + sampled*32064)",
    ),
    default_metric="time_ms",
    method=NKI_METHOD,
    caveats=NKI_CAVEATS
    + (
        (
            "An approximation of the stock forward's head: sampling here is a greedy argmax; "
            "the stock graph's on-device sampler and its host hand-off are not reproduced."
        ),
    ),
)

register(
    nki_leaf_spec(
        KIND,
        NeuronHeadArgs,
        backend_summary=(
            "Embedding, final norm, lm_head shard and argmax of vLLM Neuron 0.29, compiled as one "
            "torch.compile graph and timed alone on one LNC2 logical core."
        ),
    )
)
