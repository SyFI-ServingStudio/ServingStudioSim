"""The stock vLLM Neuron full Llama forward, including on-device sampling."""

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

KIND = "neuron_llama_forward"
# Stock public serving: loopback rendezvous and the vLLM Neuron compile/execute timeouts.
VLLM_NEURON_WORKER_ENV = (
    ("NEURON_SKIP_EFA_AFFINITY", "1"),
    ("MASTER_ADDR", "127.0.0.1"),
    ("VLLM_HOST_IP", "127.0.0.1"),
    ("GLOO_SOCKET_IFNAME", "lo"),
    ("OMP_NUM_THREADS", "4"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
    ("DO_NOT_TRACK", "1"),
    ("VLLM_NEURON_COMPILATION_TIMEOUT", "1200"),
    ("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200"),
)


@dataclass(frozen=True)
class NeuronLlamaForwardArgs(KernelArgs):
    phase: str = arg(doc="prefill or decode; whole 32-layer model including greedy sampling.")
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


DOC = KernelDoc(
    title="Stock Neuron Llama full forward",
    summary="Native device execution of the complete Llama 3.1 8B model at TP4.",
    description=(
        "Measures the unmodified vLLM Neuron fullgraph callable, including all layers, "
        "TP collectives, KV updates and greedy on-device sampling."
    ),
    category="Other",
    formula=("time = median of per-forward physical-core interval unions across all four ranks",),
    default_metric="time_ms",
    method=(
        "Public LLM.generate with NRT system profiling after warmup; native intervals "
        "exclude engine startup, compilation, CPU reference checks and host scheduling."
    ),
    caveats=(
        "Fixed original 32-layer Llama 3.1 8B base checkpoint and immutable runtime; "
        "weights and runtime versions are verified.",
        "Decode attention is the stock compiled Torch GQA fallback; "
        "the fused MLP remains inside the whole graph.",
        "Initial context buckets are 128, 512 and 2048. Each run independently checks "
        "full logits using the unmodified vendor FP32/BF16 criterion.",
        "Inputs use homogeneous prompts ending eight tokens below context capacity. "
        "Context occupancy generalization requires separate validation.",
        "FLOPs and bytes estimate per-rank graph contractions and persistent operands; "
        "they omit nonmatmul arithmetic, temporaries, communication, spills and rereads. "
        "They are not device counters or total HBM traffic. Energy is unreported (zero).",
        "Every bucket covers the same eight prespecified museum subjects; batches below "
        "eight use multiple calls. Vendor acceptance aggregates by configuration. "
        "Individual prompts can still fail (observed for B1/context512 paintings).",
    ),
    reference="profiling.runners.neuron.vllm_forward_reference",
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_neuron",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            device_family="neuron",
            architectures=frozenset({"Trainium2"}),
        ),
        runner_ref=RunnerRef("profiling.runners.neuron.vllm_forward", "profile_forward_batch"),
        row_provenance_ref=RunnerRef("profiling.runners.neuron.vllm_forward", "row_provenance"),
        table_name=KIND,
        args_schema=NeuronLlamaForwardArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_neuron_env",
        neuron_logical_cores=4,
        list_native=True,
        worker_env=VLLM_NEURON_WORKER_ENV,
        doc=BackendDoc(
            summary="Pinned stock vLLM Neuron 0.24.0.1.1.0 full-model execution at TP4/LNC2.",
            url="https://github.com/vllm-project/vllm-neuron/tree/f8abae640a43824c1dc73aed3cf2f67b83bce507",
        ),
    )
)
