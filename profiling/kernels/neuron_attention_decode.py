"""The decode attention block of one layer, on either of the two paths the
engine's `attention_decode` call lowers to: the NKI kernel (per-head block
table) or the stock torch fallback (2-D block table)."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_nki_qkv import (
    NEURON_TORCH_BACKEND,
    NKI_CAVEATS,
    NKI_METHOD,
    nki_leaf_spec,
)

KIND = "neuron_attention_decode"


@dataclass(frozen=True)
class NeuronAttentionDecodeArgs(KernelArgs):
    token_bucket: int = arg(
        unit="tokens", doc="Compiled decode batch bucket B (one token per request)."
    )
    max_model_len: int = arg(unit="tokens", doc="Context each request's block table spans.")
    kv_blocks: int = arg(unit="pages", doc="Total paged KV pool per rank; 6782.")
    block_size: int = arg(unit="tokens", doc="Tokens per KV page; 32.")
    tp_size: int = arg(unit="ranks", doc="Four LNC2 ranks on one Trainium2 chip; per-rank shapes.")
    dtype: DType = arg(doc="Weights, activations and KV dtype; BF16 only.")


DOC = KernelDoc(
    title="Decode attention block (attention_decode, NKI kernel or torch fallback)",
    summary=(
        "One decode layer's fused attention block over the paged KV cache "
        "(vllm_neuron.functional.attention_decode), timed alone per TP4 rank on one of the two "
        "paths the call lowers to."
    ),
    description=(
        "Fused QKV projection + RoPE, the paged K/V read over max_model_len, GQA attention for B "
        "requests, the output projection and the cache write, per rank. Backend vllm_neuron_nki: "
        "the per-head block table [B, kv_heads, blocks], which routes the call to the nkilib "
        "attention_block_tkg kernel (the kernel reads the paged cache itself; update_cache=False "
        "with the caller's scatter of the new K/V), what a Llama3 forward_decode patched to build "
        "that table launches. Backend vllm_neuron_torch: the stock Llama3 forward_decode call as "
        "is (the 2-D block table, the explicit decode mask, update_cache=True with the new "
        "token's slot), which at TP4 with two KV heads per rank the functional routes to its "
        "torch fallback (a torch gather of the K/V blocks on GpSimd, dense attention). Variants "
        "are decode batch buckets: powers of two up to the context's KV-resident capacity."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "flops = 2*B*4096*1536 + 4*B*8*C*128 + 2*B*1024*4096 per rank",
        "bytes = 2 * (4096*1536 + 1024*4096 + 2*B*C*2*128 + 2*B*4096)",
    ),
    default_metric="time_ms",
    method=NKI_METHOD,
    caveats=NKI_CAVEATS
    + (
        (
            "On the torch fallback the decode mask is built inside the unit on every call, where "
            "the engine builds it once per forward and reuses it across the 32 layers. Block "
            "tables are scattered across the pool as in live serving, with every request at full "
            "context."
        ),
        (
            "The stock vLLM Neuron 0.29 engine runs the torch fallback for Llama3 at TP4; costing "
            "a deployment on the NKI kernel rows assumes the per-head block-table change to "
            "forward_decode."
        ),
    ),
)

register(
    nki_leaf_spec(
        KIND,
        NeuronAttentionDecodeArgs,
        backend_summary=(
            "The NKI kernel path (per-head block table) of vLLM Neuron 0.29's attention_decode, "
            "compiled as one torch.compile graph and timed alone on one LNC2 logical core at the "
            "per-rank shapes of TP4."
        ),
    )
)

register(
    nki_leaf_spec(
        KIND,
        NeuronAttentionDecodeArgs,
        backend_summary=(
            "The stock Llama3 forward_decode call of vLLM Neuron 0.29 (2-D block table, the "
            "torch fallback at TP4), compiled as one torch.compile graph and timed alone on one "
            "LNC2 logical core at the per-rank shapes of TP4."
        ),
        backend=NEURON_TORCH_BACKEND,
        runner=f"profile_{KIND}_torch_batch",
    )
)
