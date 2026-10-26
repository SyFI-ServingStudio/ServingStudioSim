"""The stock vLLM Neuron full Llama forward with prefill segmentation."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)
from profiling.kernels.neuron_llama_forward import VLLM_NEURON_WORKER_ENV

KIND = "neuron_llama_segmented_forward"


@dataclass(frozen=True)
class NeuronLlamaSegmentedForwardArgs(KernelArgs):
    phase: str = arg(
        doc="prefill (one S-token segment) or decode; whole 32-layer model with greedy sampling."
    )
    token_bucket: int = arg(
        unit="tokens", doc="Prefill: segment size S (padded active tokens). Decode: batch bucket."
    )
    prior_tokens: int = arg(
        unit="tokens",
        doc="Prefill: cached tokens attended before this segment (k x S). Decode: 0.",
    )
    segment_size: int = arg(
        unit="tokens", doc="Prefill segment S: 512, 1024, 2048 or 4096, below the context."
    )
    max_model_len: int = arg(
        unit="tokens", doc="Allocated per-request context and block-table width."
    )
    kv_blocks: int = arg(unit="pages", doc="Total paged KV pool per rank; 6782.")
    block_size: int = arg(unit="tokens", doc="Tokens per KV page; 32.")
    tp_size: int = arg(unit="ranks", doc="Four LNC2 ranks on one Trainium2 chip.")
    dtype: DType = arg(doc="Weights, activations and KV dtype; BF16 only.")


DOC = KernelDoc(
    title="Stock Neuron Llama segmented-prefill forward",
    summary=(
        "Native device execution of the complete Llama 3.1 8B TP4 model when the stock engine "
        "prefills in S-token segments."
    ),
    description=(
        "Measures the unmodified vLLM Neuron fullgraph callables of an engine configured with "
        "max_num_batched_tokens, num_batched_tokens_buckets and kv_segment_size_buckets = S: "
        "one S-token prefill-segment graph (one sequence attending to its cached prior) and "
        "the stock decode graphs, including TP collectives, KV updates and greedy sampling."
    ),
    category="Other",
    formula=(
        "time = median of per-forward physical-core interval unions across all four ranks, "
        "per (phase, bucket, prior_tokens)",
    ),
    default_metric="time_ms",
    method=(
        "Public LLM.generate with NRT system profiling after warmup. Prefill segments are "
        "timed from dedicated batch-1 calls (one per museum subject), where the k-th prefill "
        "execution has prior k x S; their tokens must equal the numerically accepted "
        "request's. Decode rows come from the canonical calls as in neuron_llama_forward."
    ),
    caveats=(
        "Fixed original 32-layer Llama 3.1 8B base checkpoint and immutable runtime; "
        "weights and runtime versions are verified.",
        "Contexts 16384 and 32768 (8192 as a cross-check against neuron_llama_forward). "
        "Decode buckets are capped at min(next_pow2(217024 // C), C): 16 at C16384, 8 at "
        "C32768.",
        "Each run independently checks full logits with the unmodified vendor FP32/BF16 "
        "criterion on (context-8)-token museum prompts. Every prefill row is accepted by the "
        "prefill validation shape b{min(16, KV-resident capacity)}. CPU references use HF "
        "eager attention up to 8192 history tokens and SDPA above.",
        "The profiled prior grid is k x S for k < ceil((C-8)/S). The last segment of a "
        "canonical prompt is partly padding; its row is the padded S-token graph.",
        "FLOPs and bytes estimate per-rank graph contractions and persistent operands; "
        "a segment adds full attention over its prior and reads the prior KV. They omit "
        "nonmatmul arithmetic, temporaries, communication, spills and rereads. They are not "
        "device counters. Energy is unreported (zero).",
    ),
    reference="profiling.runners.neuron.vllm_forward_reference",
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_neuron_segmented",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            device_family="neuron",
            architectures=frozenset({"Trainium2"}),
        ),
        runner_ref=RunnerRef(
            "profiling.runners.neuron.vllm_segmented_forward", "profile_segmented_forward_batch"
        ),
        row_provenance_ref=RunnerRef(
            "profiling.runners.neuron.vllm_segmented_forward", "row_provenance"
        ),
        table_name=KIND,
        args_schema=NeuronLlamaSegmentedForwardArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_neuron_env",
        neuron_logical_cores=4,
        list_native=True,
        worker_env=VLLM_NEURON_WORKER_ENV,
        doc=BackendDoc(
            summary=(
                "Pinned stock vLLM Neuron 0.24.0.1.1.0 full-model execution at TP4/LNC2 with "
                "S-token prefill segments."
            ),
            url="https://github.com/vllm-project/vllm-neuron/tree/f8abae640a43824c1dc73aed3cf2f67b83bce507",
        ),
    )
)
