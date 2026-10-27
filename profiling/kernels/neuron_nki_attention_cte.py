"""Standalone NKI prefill attention: one launch unit of the stock Llama forward."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_nki_qkv import NKI_CAVEATS, NKI_METHOD, nki_leaf_spec

KIND = "neuron_nki_attention_cte"


@dataclass(frozen=True)
class NeuronNkiAttentionCteArgs(KernelArgs):
    token_bucket: int = arg(unit="tokens", doc="Compiled prefill token bucket T (a stock context).")
    tp_size: int = arg(unit="ranks", doc="Four LNC2 ranks on one Trainium2 chip; per-rank shapes.")
    dtype: DType = arg(doc="Weights, activations and KV dtype; BF16 only.")


DOC = KernelDoc(
    title="Standalone NKI context-encoding attention (prefill)",
    summary=(
        "One prefill layer's causal attention over T tokens, the nkilib attention_cte kernel, "
        "timed alone per TP4 rank."
    ),
    description=(
        "`vllm_neuron.functional.flash_attention`: 8 query heads of 128 per rank against the "
        "layer's K/V, causal over T. The glue repeats the 2 KV heads to 8 before the call. The KV "
        "write into the paged cache is torch glue, not this kernel. Variants are the prefill token "
        "buckets of the stock context ladder."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "flops = 2 * 8 * T^2 * 128 per rank (causal half of QK^T and PV)",
        "bytes = 2 * 4 * 8 * T * 128",
    ),
    default_metric="time_ms",
    method=NKI_METHOD,
    caveats=NKI_CAVEATS,
)

register(
    nki_leaf_spec(
        KIND,
        NeuronNkiAttentionCteArgs,
        backend_summary=(
            "The launch unit of vLLM Neuron 0.29, compiled as one torch.compile graph "
            "and timed alone on one LNC2 logical core at the per-rank shapes of TP4."
        ),
    )
)
