"""Neuron's compiled token embedding gather, with measured device latency."""

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

KIND = "neuron_embedding"


@dataclass(frozen=True)
class NeuronEmbeddingArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token indices gathered in one invocation.")
    hidden: int = arg(unit="elements", doc="Width of each vocabulary embedding.")
    vocab: int = arg(unit="tokens", doc="Rows allocated in the embedding table.")
    dtype: DType = arg(doc="Embedding weight and output type; BF16 initially.")


DOC = KernelDoc(
    title="Neuron token embedding",
    summary="Gather token embeddings from a BF16 vocabulary table.",
    description=(
        "Compile the public Torch embedding operation used by NxD ParallelEmbedding "
        "at TP=1. Seeded token indices and weights stay on device during timing. "
        "The vocabulary allocation is explicit in the profile identity."
    ),
    category="Other",
    formula=("GB/s = (4*num_tokens*hidden + 8*num_tokens) / time",),
    default_metric="memory_bandwidth_gbps",
    method=(
        "Median native nc_exec_running interval union for one LNC2 invocation. "
        "Compilation, numerical checks, host copies and warmup are excluded."
    ),
    caveats=(
        "One LNC2 unit; no tensor-parallel reduction or host tokenization is included.",
        "GB/s counts logical table reads, indices and output; energy is unavailable.",
    ),
    reference="torch.nn.functional.embedding",
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="neuron_torch",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            device_family="neuron",
            architectures=frozenset({"Trainium2"}),
        ),
        runner_ref=RunnerRef("profiling.runners.neuron.embedding", "profile_embedding"),
        table_name=KIND,
        args_schema=NeuronEmbeddingArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="neuron_trace_env",
        doc=BackendDoc(
            summary="Torch functional.embedding compiled to Trainium2 with Torch NeuronX.",
            url="https://github.com/aws-neuron/neuronx-distributed/blob/main/src/neuronx_distributed/parallel_layers/layers.py",
        ),
    )
)
