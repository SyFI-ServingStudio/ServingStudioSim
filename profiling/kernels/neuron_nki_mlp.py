"""Standalone NKI gated MLP: one launch unit of the stock Llama forward."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_nki_qkv import NKI_CAVEATS, NKI_METHOD, nki_leaf_spec

KIND = "neuron_nki_mlp"


@dataclass(frozen=True)
class NeuronNkiMlpArgs(KernelArgs):
    phase: str = arg(doc="prefill or decode: the forward phase whose launch this row times.")
    token_bucket: int = arg(
        unit="tokens", doc="Rows of the launch: prefill token bucket T or decode batch bucket B."
    )
    tp_size: int = arg(unit="ranks", doc="Four LNC2 ranks on one Trainium2 chip; per-rank shapes.")
    dtype: DType = arg(doc="Weights, activations and KV dtype; BF16 only.")


DOC = KernelDoc(
    title="Standalone NKI gated MLP",
    summary=(
        "One layer's gated MLP, the nkilib mlp kernel, timed alone per TP4 rank at the prefill or "
        "decode bucket."
    ),
    description=(
        "`vllm_neuron.functional.mlp`: [rows, 4096] with gate/up [4096, 3584] and down [3584, "
        "4096] per rank, SiLU-gated. The kernel selects its CTE (prefill) or TKG (decode) path "
        "from the row count. The row is therefore keyed by phase and bucket. Variants are the "
        "prefill token buckets of the stock context ladder and decode batch buckets (powers of two "
        "up to 512)."
    ),
    category="GEMM",
    formula=(
        "flops = 6 * rows * 4096 * 3584 per rank",
        "bytes = 2 * (2*rows*4096 + 3*4096*3584)",
    ),
    default_metric="time_ms",
    method=NKI_METHOD,
    caveats=NKI_CAVEATS,
)

register(
    nki_leaf_spec(
        KIND,
        NeuronNkiMlpArgs,
        backend_summary=(
            "The launch unit of vLLM Neuron 0.29, compiled as one torch.compile graph "
            "and timed alone on one LNC2 logical core at the per-rank shapes of TP4."
        ),
    )
)
