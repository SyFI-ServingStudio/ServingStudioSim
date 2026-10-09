"""The stock vLLM Neuron Llama forward split at the public LlamaModel return."""

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

KIND = "neuron_llama_region"


@dataclass(frozen=True)
class NeuronLlamaRegionArgs(KernelArgs):
    phase: str = arg(doc="prefill or decode of the whole 32-layer forward that owns the region.")
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
    region: str = arg(
        doc=(
            "model: embedding, 32 decoder layers, final norm, SP gather and all KV updates; "
            "head: row selection, lm_head, logit collectives and greedy sampling."
        )
    )


DOC = KernelDoc(
    title="Stock Neuron Llama forward regions",
    summary="Model body and sampling head of the stock Llama 3.1 8B TP4 forward, timed apart.",
    description=(
        "Partitions the unmodified vLLM Neuron fullgraph FX graph at the public "
        "LlamaModel.forward return into two NEFFs compiled by the stock backend with the "
        "original flags. Every stock operation keeps its op, target, args and kwargs."
    ),
    category="Other",
    formula=(
        "time = median over forwards of the region NEFF's physical-core interval union "
        "across all four ranks",
    ),
    default_metric="time_ms",
    method=(
        "One group runs the split and the unchanged stock engine on the same canonical "
        "workload. Rows require: a structural partition proof per compiled graph; the "
        "unchanged vendor full-logit check; identical tokens and per-case "
        "RMS(split-FP32) <= 1.10 x RMS(stock-FP32); and model+head medians within 5% of "
        "the same-group stock whole-forward median."
    ),
    caveats=(
        "Experimental compilation path: the split forward is not the production stock "
        "binary. neuron_llama_forward remains the stock whole-forward measurement.",
        "Verified only at context 512, pool 6782/page 32, TP4/LNC2 BF16, prefill 512 and "
        "decode 1/16. Other shapes are refused before compilation.",
        "Splitting removes cross-boundary fusion and materializes the model output; region "
        "times are not a decomposition of the stock binary's internal schedule.",
        "FLOPs and bytes estimate per-rank graph contractions and persistent operands: head "
        "is the lm_head contraction and weight shard; model is the whole-forward estimate "
        "minus head. They omit nonmatmul arithmetic, temporaries, communication, spills and "
        "rereads. They are not device counters. Energy is unreported (zero).",
        "Inputs use the neuron_llama_forward canonical workload: eight museum subjects, "
        "prompts ending eight tokens below context capacity.",
    ),
    reference="profiling.runners.neuron.vllm_forward_reference",
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_neuron_fx_regions",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            device_family="neuron",
            architectures=frozenset({"Trainium2"}),
        ),
        runner_ref=RunnerRef("profiling.runners.neuron.vllm_region", "profile_region_batch"),
        row_provenance_ref=RunnerRef("profiling.runners.neuron.vllm_region", "row_provenance"),
        table_name=KIND,
        args_schema=NeuronLlamaRegionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_neuron_env",
        neuron_logical_cores=4,
        list_native=True,
        worker_env=VLLM_NEURON_WORKER_ENV,
        doc=BackendDoc(
            summary=(
                "Pinned stock vLLM Neuron 0.24.0.1.1.0 forward, FX-partitioned into model "
                "and head NEFFs at TP4/LNC2."
            ),
            url="https://github.com/vllm-project/vllm-neuron/tree/f8abae640a43824c1dc73aed3cf2f67b83bce507",
        ),
    )
)
