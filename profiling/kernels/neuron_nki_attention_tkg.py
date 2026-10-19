"""Standalone NKI decode attention block: one launch unit of the Llama forward."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_nki_qkv import NKI_CAVEATS, NKI_METHOD, nki_leaf_spec

KIND = "neuron_nki_attention_tkg"


@dataclass(frozen=True)
class NeuronNkiAttentionTkgArgs(KernelArgs):
    token_bucket: int = arg(
        unit="tokens", doc="Compiled decode batch bucket B (one token per request)."
    )
    max_model_len: int = arg(unit="tokens", doc="Context each request's block table spans.")
    kv_blocks: int = arg(unit="pages", doc="Total paged KV pool per rank; 6782.")
    block_size: int = arg(unit="tokens", doc="Tokens per KV page; 32.")
    tp_size: int = arg(unit="ranks", doc="Four LNC2 ranks on one Trainium2 chip; per-rank shapes.")
    dtype: DType = arg(doc="Weights, activations and KV dtype; BF16 only.")


DOC = KernelDoc(
    title="Standalone NKI token-generation attention block (decode)",
    summary=(
        "One decode layer's fused attention block over the paged KV cache, the nkilib "
        "attention_block_tkg kernel, timed alone per TP4 rank."
    ),
    description=(
        "`vllm_neuron.functional.attention_decode` with a per-head block table: fused QKV "
        "projection + RoPE, paged K/V gather over max_model_len, GQA attention for B requests and "
        "the output projection, per rank. The stock Llama3 decode at TP4 passes a 2-D block table, "
        "so it never reaches this kernel and runs the torch fallback. This row models the per-head "
        "block table path (the patched forward_decode). Variants are decode batch buckets: powers "
        "of two up to the context's KV-resident capacity."
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
            "The KV write of the new token (scatter_new_kv) is inside the unit; the attention mask "
            "is generated in-kernel from pos_ids. Block tables are scattered across the pool as in "
            "live serving, with every request at full context."
        ),
        (
            "The stock vLLM Neuron binary does not run this kernel for Llama3 at TP4; costing a "
            "stock deployment with it assumes the per-head block table change."
        ),
    ),
)

register(
    nki_leaf_spec(
        KIND,
        NeuronNkiAttentionTkgArgs,
        backend_summary=(
            "The NKI kernel of pinned vLLM Neuron 0.24.0.1.1.0, compiled and timed alone on one "
            "LNC2 logical core at the per-rank shapes of TP4."
        ),
    )
)
