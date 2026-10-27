"""Standalone NKI output projection (prefill): one launch unit of the Llama forward."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_nki_qkv import NKI_CAVEATS, NKI_METHOD, nki_leaf_spec

KIND = "neuron_nki_output_projection"


@dataclass(frozen=True)
class NeuronNkiOutputProjectionArgs(KernelArgs):
    token_bucket: int = arg(unit="tokens", doc="Compiled prefill token bucket T (a stock context).")
    tp_size: int = arg(unit="ranks", doc="Four LNC2 ranks on one Trainium2 chip; per-rank shapes.")
    dtype: DType = arg(doc="Weights, activations and KV dtype; BF16 only.")


DOC = KernelDoc(
    title="Standalone NKI attention output projection (prefill)",
    summary=(
        "One prefill layer's O projection, the nkilib output_projection_cte kernel, timed alone "
        "per TP4 rank."
    ),
    description=(
        "`vllm_neuron.functional.o_proj`: attention output [1, 8, 128, T] @ W[1024, 4096] per "
        "rank. The TP reduce-scatter that follows is a separate collective row. Prefill only: in "
        "decode, both attention paths fuse the projection into neuron_attention_decode. "
        "Variants are the prefill token buckets of the stock context ladder."
    ),
    category="GEMM",
    formula=(
        "flops = 2 * T * 1024 * 4096 per rank",
        "bytes = 2 * (T*1024 + 1024*4096 + T*4096)",
    ),
    default_metric="time_ms",
    method=NKI_METHOD,
    caveats=NKI_CAVEATS,
)

register(
    nki_leaf_spec(
        KIND,
        NeuronNkiOutputProjectionArgs,
        backend_summary=(
            "The launch unit of vLLM Neuron 0.29, compiled as one torch.compile graph "
            "and timed alone on one LNC2 logical core at the per-rank shapes of TP4."
        ),
    )
)
