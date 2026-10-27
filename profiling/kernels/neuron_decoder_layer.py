"""One decoder layer of the decode forward, with its TP all-reduces, measured
inside a reduced-depth TP4 decode forward on the whole chip: the leaf that
carries the in-graph overlap and the collectives the standalone units cannot
(doc/pr/llama31-8b-trn2/README.md, "Check 1")."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc, arg
from profiling.db.registry import register
from profiling.kernels.neuron_nki_qkv import NEURON_TORCH_BACKEND, nki_leaf_spec

KIND = "neuron_decoder_layer"
RUNNER_MODULE = "profiling.runners.neuron.forward_bench"

FORWARD_METHOD = (
    "The chip's four LNC2 logical cores inside the vLLM Neuron 0.29 image (Neuron SDK 2.33), "
    "one process per TP rank. Each rank compiles nki_units.DecodeForward through the production "
    "compile path, for n = 8 and n = 32. DecodeForward is a reconstruction of the engine's "
    "decode forward from the same vllm_neuron functional calls, with random weights. The graph "
    "holds the vocab-sharded embedding and its all-reduce, the RoPE tables, n decoder layers "
    "with both block all-reduces, the final norm, the lm_head shard and the engine's own sampler "
    "with its default on-device sampling config. Each depth runs one compile call and 10 warm "
    "calls, a cross-rank fence, then 50 executions under the NRT system profile. Per execution "
    "the shortest of the four ranks is kept (the last rank to launch does not wait for the "
    "others); t(n) is the median. layer = (t(32) - t(8)) / 24; fixed = t(8) - 8 x layer. Both "
    "depths and both terms are kept in the row's provenance."
)

FORWARD_CAVEATS = (
    (
        "The decode step is fixed + num_layers x layer; it is 2-5% below the engine's measured "
        "decode step at every bucket from B1 to B128 (C512, kernel path; "
        "doc/pr/llama31-8b-trn2/README.md, 'Check 1'). Stacks of 1 and 4 layers under-read the "
        "layer by 2-6%: the compiler schedules a short graph differently."
    ),
    (
        "On backend vllm_neuron_torch each layer builds the decode mask (the engine builds it once "
        "per forward), so the torch rows carry about one mask build per layer that the engine does "
        "not."
    ),
    (
        "Rows exist only for compiled variants; the cache is a direct lookup, no interpolation. "
        "Changing a kernel inside the layer (the attention path, the MLP, the dtype) is another "
        "backend of this kind, measured as its own forward; the per-op rows of the same ops remain "
        "for attribution, not for addition."
    ),
    "FLOPs and bytes are operand-count estimates for the rate columns, not device counters.",
)


@dataclass(frozen=True)
class NeuronDecodeForwardArgs(KernelArgs):
    token_bucket: int = arg(
        unit="tokens", doc="Compiled decode batch bucket B (one token per request)."
    )
    max_model_len: int = arg(unit="tokens", doc="Context each request's block table spans.")
    kv_blocks: int = arg(unit="pages", doc="Total paged KV pool per rank; 6782.")
    block_size: int = arg(unit="tokens", doc="Tokens per KV page; 32.")
    tp_size: int = arg(unit="ranks", doc="Four LNC2 ranks on one Trainium2 chip; per-rank shapes.")
    dtype: DType = arg(doc="Weights, activations and KV dtype; BF16 only.")


DOC = KernelDoc(
    title="Decoder layer, decode (TP4, inside a reduced-depth forward)",
    summary=(
        "One Llama 3.1 8B decode layer per TP4 rank with both block all-reduces, the marginal of "
        "the engine's decode forward between 8 and 32 layers on the whole chip."
    ),
    description=(
        "The engine's decode step is one compiled graph per bucket on four ranks. Inside it the "
        "units of a layer overlap and fuse, and the block all-reduces run in the graph. The "
        "standalone rows do not add up to that. This leaf measures the layer inside such a graph. "
        "Backend "
        "vllm_neuron_nki: the attention block on the per-head block-table NKI kernel path (what a "
        "patched Llama3 forward_decode launches). Backend vllm_neuron_torch: the stock Llama3 "
        "forward_decode call, the torch fallback at TP4. Variants are decode batch buckets up to "
        "the context's KV-resident capacity."
    ),
    category="Other",
    formula=(
        "flops = 2*B*4096*1536 + 4*B*8*C*128 + 2*B*1024*4096 + 6*B*4096*3584 per rank",
        "bytes = 2 * (4096*1536 + 1024*4096 + 3*4096*3584 + 2*B*C*2*128 + 10*B*4096)",
    ),
    default_metric="time_ms",
    method=FORWARD_METHOD,
    caveats=FORWARD_CAVEATS,
)


def forward_specs(kind, args_schema, nki_summary, torch_summary):
    """The two backends of a kind measured by forward_bench on the whole chip."""
    return (
        nki_leaf_spec(kind, args_schema, nki_summary, module=RUNNER_MODULE, logical_cores=4),
        nki_leaf_spec(
            kind,
            args_schema,
            torch_summary,
            backend=NEURON_TORCH_BACKEND,
            runner=f"profile_{kind}_torch_batch",
            module=RUNNER_MODULE,
            logical_cores=4,
        ),
    )


for _spec in forward_specs(
    KIND,
    NeuronDecodeForwardArgs,
    "One decode layer of a TP4 reduced-depth forward on the NKI kernel attention path, on the "
    "chip's four LNC2 cores.",
    "One decode layer of a TP4 reduced-depth forward on the stock call's torch fallback "
    "attention, on the chip's four LNC2 cores.",
):
    register(_spec)
