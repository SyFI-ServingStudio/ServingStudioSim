"""Collective-delimited segments of the unchanged stock vLLM Neuron Llama forward."""

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

KIND = "neuron_llama_segment"


@dataclass(frozen=True)
class NeuronLlamaSegmentArgs(KernelArgs):
    phase: str = arg(doc="prefill or decode of the whole 32-layer forward that owns the segment.")
    token_bucket: int = arg(
        unit="tokens", doc="Compiled prefill token bucket or decode batch bucket."
    )
    max_model_len: int = arg(
        unit="tokens", doc="Allocated per-request context and decode block-table width."
    )
    kv_blocks: int = arg(unit="pages", doc="Total paged KV pool per rank; initially 6782.")
    block_size: int = arg(unit="tokens", doc="Tokens per KV page; initially 32.")
    tp_size: int = arg(unit="ranks", doc="Four LNC2 ranks on one Trainium2 chip.")
    dtype: DType = arg(doc="Weights, activations and KV dtype; BF16 only.")
    segment: str = arg(
        doc=(
            "embedding: start to the first reduction; attention_block / mlp_block: one "
            "layer's span between its reductions (mean over the 32 layers); head: last "
            "reduction to the end (final norm, lm_head, logit gathers, sampling)."
        )
    )


DOC = KernelDoc(
    title="Stock Neuron Llama forward segments",
    summary="Wall time between the collectives of the unchanged stock Llama 3.1 8B forward.",
    description=(
        "Cuts the production whole-forward NEFF's native device timeline at its "
        "reduction collectives (AllReduce in decode, ReduceScatter in prefill): "
        "embedding, per-layer attention and MLP blocks, and the head. The binary is "
        "byte-identical to neuron_llama_forward/vllm_neuron; nothing is recompiled."
    ),
    category="Other",
    formula=(
        "time = median over captured forwards of the rank-averaged segment span; "
        "attention_block and mlp_block use each forward's mean over its 32 layers",
        "embedding + 32 x (attention_block + mlp_block) + head = execution span per forward",
    ),
    default_metric="time_ms",
    method=(
        "One group runs the unchanged stock engine: the vendor full-logit check, the "
        "untraced system-profile whole-forward timing, then a device instruction capture "
        "of the same engine configuration. Rows require: traced NEFF bytes equal to the "
        "numerically validated stock NEFFs; complete instruction traces on both physical "
        "cores of every captured rank with no lost notifications except DMA; the exact "
        "collective layout; every layer within 10% of its forward's layer mean; and the "
        "composed forward within 5% of the same-group untraced stock median."
    ),
    caveats=(
        "Segments are wall time between cross-rank synchronization points, not isolated "
        "kernel latency: about 15 us of the next sublayer's prologue overlaps each "
        "closing collective, and each collective's own time belongs to the segment it "
        "closes.",
        "Verified only at context 512, pool 6782/page 32, TP4/LNC2 BF16, prefill 512 and "
        "decode 1/16. Other shapes are refused before any device work.",
        "Device instruction tracing is active while segments are captured; the 5% gate "
        "against the untraced whole forward bounds its perturbation.",
        "FLOPs and bytes split the whole-forward estimates by semantic owner: embedding "
        "rows; one layer's norm, QKV/O, 1/32 of attention and KV bytes; one layer's norm "
        "and gated MLP; final norm and lm_head. They omit nonmatmul arithmetic, "
        "temporaries, communication, spills and rereads. They are not device counters. "
        "Energy is unreported (zero).",
        "Inputs use the neuron_llama_forward canonical workload: eight museum subjects, "
        "prompts ending eight tokens below context capacity.",
    ),
    reference="profiling.runners.neuron.vllm_forward_reference",
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_neuron_collective_segments",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            device_family="neuron",
            architectures=frozenset({"Trainium2"}),
        ),
        runner_ref=RunnerRef("profiling.runners.neuron.vllm_segment", "profile_segment_batch"),
        row_provenance_ref=RunnerRef("profiling.runners.neuron.vllm_segment", "row_provenance"),
        table_name=KIND,
        args_schema=NeuronLlamaSegmentArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_neuron_env",
        neuron_logical_cores=4,
        list_native=True,
        worker_env=VLLM_NEURON_WORKER_ENV,
        doc=BackendDoc(
            summary=(
                "Pinned stock vLLM Neuron 0.24.0.1.1.0 whole forward at TP4/LNC2, segmented "
                "at its collectives in the native device instruction trace."
            ),
            url="https://github.com/vllm-project/vllm-neuron/tree/f8abae640a43824c1dc73aed3cf2f67b83bce507",
        ),
    )
)
