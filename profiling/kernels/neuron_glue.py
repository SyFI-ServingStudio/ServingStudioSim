"""Compiler-generated torch glue of one layer: the non-kernel ops around the NKI units."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_nki_qkv import NKI_CAVEATS, NKI_METHOD, nki_leaf_spec

KIND = "neuron_glue"


@dataclass(frozen=True)
class NeuronGlueArgs(KernelArgs):
    phase: str = arg(doc="prefill or decode: the forward phase whose launch this row times.")
    token_bucket: int = arg(
        unit="tokens", doc="Rows of the launch: prefill token bucket T or decode batch bucket B."
    )
    tp_size: int = arg(unit="ranks", doc="Four LNC2 ranks on one Trainium2 chip; per-rank shapes.")
    dtype: DType = arg(doc="Weights, activations and KV dtype; BF16 only.")


DOC = KernelDoc(
    title="Compiler-generated torch glue of one layer",
    summary=(
        "The non-kernel torch ops of one layer (norms, residuals, layouts, prefill KV write), "
        "compiled and timed alone per TP4 rank."
    ),
    description=(
        "Prefill: input RMSNorm, QKV split and head layouts, the paged KV index_put_, GQA repeat, "
        "flash layouts, residual add and post-attention RMSNorm at T tokens. Decode: two RMSNorms "
        "and two residual adds at [B, 4096]. Inside the model's graph, neuronx-cc fuses the same "
        "ops differently. This row approximates the glue; it does not reproduce it. Variants are "
        "the prefill token buckets of the stock context ladder and decode batch buckets (powers of "
        "two up to 512)."
    ),
    category="Other",
    formula=(
        "flops = 0 (not a matmul)",
        "bytes = 2 * (6*rows*4096 + prefill KV write 2*rows*2*128)",
    ),
    default_metric="time_ms",
    method=NKI_METHOD,
    caveats=NKI_CAVEATS
    + (
        (
            "An approximation by construction: the model fuses the same ops into its own compiler "
            "regions. Inside the engine's graph the norms cost about half their standalone time "
            "(doc/pr/llama31-8b-trn2/README.md, 'Check 1'); the standalone prefill glue at "
            "C512 T512 is 0.134 ms on vLLM Neuron 0.29. The decode step is costed from the "
            "neuron_decoder_layer leaf by default, which carries the norms and residuals itself."
        ),
    ),
)

register(
    nki_leaf_spec(
        KIND,
        NeuronGlueArgs,
        backend_summary=(
            "The torch glue of one vLLM Neuron 0.29 Llama3 layer, compiled as one torch.compile "
            "graph and timed alone on one LNC2 logical core at the per-rank shapes of TP4."
        ),
    )
)
