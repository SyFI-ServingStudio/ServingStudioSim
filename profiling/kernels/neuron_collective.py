"""Collectives of a Trainium2 tensor-parallel group over NeuronLink.

Wire string ``neuron_collective`` matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/neuron_collective.rs``. One kind carries the
three collectives the stock vLLM Neuron Llama forward issues (``all_gather``
and ``reduce_scatter`` around each prefill block, ``all_reduce`` after each
decode block) under an ``op`` field, so they share one runner and one method.
"""

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
from profiling.kernels.neuron_nki_qkv import NATIVE_WORKER_ENV, NKI_SOURCE_URL

KIND = "neuron_collective"
BACKEND = "vllm_neuron_dist"
OPS = ("all_reduce", "all_gather", "reduce_scatter")


@dataclass(frozen=True)
class NeuronCollectiveArgs(KernelArgs):
    op: str = arg(doc="all_reduce, all_gather or reduce_scatter.")
    num_ranks: int = arg(unit="ranks", doc="Ranks of the group, one LNC2 logical core each.")
    message_size_bytes: int = arg(
        unit="bytes",
        doc=(
            "The full buffer of the operation: the tensor every rank holds after an all_reduce, "
            "the gathered output of an all_gather, the un-scattered input of a reduce_scatter."
        ),
    )
    dtype: DType = arg(doc="Element type of the buffer; measured at bf16 only (size-keyed).")
    fabric: str = arg(doc="Interconnect label; neuron_link. A row key only.")


DOC = KernelDoc(
    title="NeuronLink collective",
    summary="One collective of a Trainium2 TP group, over the chip's NeuronLink.",
    description=(
        "The stock vLLM Neuron Llama forward runs prefill sequence-parallel (all_gather of the "
        "[T/tp, H] activation before a block's projections, reduce_scatter after them) and "
        "closes each decode block with an all_reduce of [B, H]. message_size_bytes is the full "
        "[rows x hidden] activation for every op. Measured on the four LNC2 cores of one chip."
    ),
    category="Communication",
    formula=(
        "algbw = message_size_bytes / time",
        "busbw = algbw · 2(N − 1)/N for all_reduce, algbw · (N − 1)/N otherwise",
    ),
    default_metric="busbw_gbps",
    caveats=(
        "Host-timed: the mean of 100 back-to-back calls between two device synchronizations on "
        "rank 0, so a message too small to hide the launch reads as launch-bound. Energy is not "
        "measured.",
        "Cost is keyed by bytes, not dtype: rows are measured at bf16.",
    ),
    method=(
        "torch.distributed on the lite runtime's neuron backend, one spawned process per LNC2 "
        "logical core of the reserved chip. Each size runs 50 warm-up calls, a one-element "
        "all-reduce fence and a synchronize, then 100 timed calls; the time is rank 0's mean "
        "per call."
    ),
    reference=None,
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend=BACKEND,
        # Size-keyed: one bf16 curve for every logical payload dtype.
        supports=BackendSupport(
            compute=None, device_family="neuron", architectures=frozenset({"Trainium2"})
        ),
        runner_ref=RunnerRef(
            "profiling.runners.neuron.collective_bench", "profile_neuron_collective_batch"
        ),
        row_provenance_ref=RunnerRef("profiling.runners.neuron.collective_bench", "row_provenance"),
        table_name=KIND,
        args_schema=NeuronCollectiveArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_neuron_native_env",
        # The whole chip: one rank per logical core of the TP4 group.
        neuron_logical_cores=4,
        list_native=True,
        worker_env=NATIVE_WORKER_ENV,
        doc=BackendDoc(
            summary=(
                "torch.distributed collectives on the lite runtime's neuron backend across the "
                "four LNC2 logical cores of one Trainium2 chip."
            ),
            url=NKI_SOURCE_URL,
        ),
    )
)
