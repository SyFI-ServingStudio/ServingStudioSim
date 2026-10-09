"""One complete, stateful NxDI Llama 3.1 decoder invocation on Trainium2."""

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

KIND = "neuron_llama_decoder"


@dataclass(frozen=True)
class NeuronLlamaDecoderArgs(KernelArgs):
    phase: str = arg(doc="prefill or decode; each uses its production compiler flags.")
    batch: int = arg(unit="requests", doc="Static request batch; initially one.")
    q_tokens: int = arg(unit="tokens", doc="Query rows; decode requires one.")
    kv_capacity: int = arg(unit="tokens", doc="Allocated KV length, including masked entries.")
    hidden: int = arg(unit="elements", doc="Hidden width; initially 4096.")
    intermediate: int = arg(unit="elements", doc="Dense MLP width; initially 14336.")
    q_heads: int = arg(unit="heads", doc="Query heads; initially 32.")
    kv_heads: int = arg(unit="heads", doc="Stored KV heads; initially eight.")
    head_dim: int = arg(unit="elements", doc="Head width; initially 128.")
    dtype: DType = arg(doc="Compute, weights and KV type; BF16 initially.")


DOC = KernelDoc(
    title="Neuron Llama 3.1 decoder",
    summary="Measure a complete compiled decoder layer with aliased KV updates.",
    description=(
        "Wrap the public NxDI NeuronLlamaDecoderLayer and KVCacheManager in the "
        "per-layer cache-update configuration. Both norms, projections, scaled "
        "Llama 3.1 RoPE, attention, SiLU MLP, residual additions and cache writes "
        "execute within one compiled NEFF invocation. TP=1; NKI kernel options "
        "are disabled so this backend measures the production compiler path."
    ),
    category="Other",
    formula=(
        "time = median(union of physical-core execution intervals per invocation)",
        "TFLOPS counts dense projections and attention matmuls; internal traffic is unreported.",
    ),
    default_metric="time_ms",
    method=(
        "Median of 20 native invocation intervals, unioned across the two "
        "physical cores of one LNC2 unit after five warmups. Compilation, "
        "copies and independent correctness checks are excluded."
    ),
    caveats=(
        "Fixed Llama 3.1 8B semantics: theta=500000, scaled RoPE factor=8, "
        "low/high factors=1/4, original context=8192, RMS epsilon=1e-5.",
        "Initial cache capacity is 512, batch is one, and prefill has 1..128 rows "
        "with no cached prefix.",
        "Decode reads the allocated cache even when its runtime mask excludes entries.",
        "A separately compiled layer is a leaf in a custom composed architecture; "
        "whole-model compiler fusion and checkpoint serving are unvalidated.",
        "Synthetic weights validate execution and timing; energy is unavailable.",
        "Weights are inlined into this layer NEFF; the default whole-model NxDI "
        "weight-separation configuration is a different compilation boundary.",
    ),
    reference="profiling.runners.neuron.llama_decoder.independent_oracle",
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="nxdi_compiler",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            device_family="neuron",
            architectures=frozenset({"Trainium2"}),
        ),
        runner_ref=RunnerRef("profiling.runners.neuron.llama_decoder", "profile_llama_decoder"),
        table_name=KIND,
        args_schema=NeuronLlamaDecoderArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="neuron_trace_env",
        worker_env=(("NXD_CPU_MODE", "0"), ("PJRT_DEVICE", "CPU"), ("OMP_NUM_THREADS", "2")),
        doc=BackendDoc(
            summary="NxDI decoder plus per-layer KVCacheManager, compiled for TP1/LNC2.",
            url="https://github.com/aws-neuron/neuronx-distributed-inference/blob/4bcdc54bf3b6ccdad490e5cd0680a3dc95270b8e/src/neuronx_distributed_inference/models/llama/modeling_llama.py",
        ),
    )
)
