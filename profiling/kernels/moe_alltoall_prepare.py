"""FlashInfer MNNVL MoE all-to-all metadata preparation kernel kind.

Wire string: ``"moe_alltoall_prepare"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/moe_alltoall_prepare.rs``.

One call to ``MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather`` decides,
from the router's expert ids alone, which rows go to which rank: it counts them
(``computeCountAndIndiceDevice``), prefix-sums the counts
(``computeCumsumDevice``), compacts the indices (``moveIndiceDevice``), clears
the expert-id scratch (``memsetExpertIdsDevice``) and exchanges the resulting
metadata (``allToAllMetadataDevice``). Five launches, one measurable call.

It is profiled as its own kind, rather than modelled, because none of it is
bandwidth-bound: at 8k tokens it moves a quarter of a megabyte and still takes
0.33 ms per layer, being dominated by atomics over `ep_size x slot_count` bins
and by the inter-rank metadata exchange. A byte-rate elementwise curve would
predict approximately zero. The payload width is absent from the key for the
same reason — nothing here touches the activations.

``metric_family=COMM``; ``gpu_count_fn`` reserves the whole EP group.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import KernelArgs
from profiling.db.doc import BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "moe_alltoall_prepare"


@dataclass(frozen=True)
class MoeAlltoallPrepareArgs(KernelArgs):
    ep_size: int = arg(unit="GPUs", doc="GPUs in the expert-parallel group.")
    tokens_per_rank: int = arg(
        unit="tokens", doc="Tokens on each GPU in the measured routing table."
    )
    top_k: int = arg(unit="experts", doc="Expert slots selected per token.")
    slot_count: int = arg(unit="experts", doc="Expert slots across the group.")
    fabric: str = arg(doc="Interconnect label used to identify the measurement.")


DOC = KernelDoc(
    title="MoE all-to-all preparation",
    summary="Build and exchange the row indices needed for expert-parallel dispatch.",
    description=(
        "Before dispatch, every GPU works out where its tokens go. FlashInfer's "
        "prepare pass reads the router's expert IDs, computes the row count and "
        "offset for each destination GPU and the send and receive row indices, "
        "and exchanges the expert IDs and routing weights across the group. No "
        "hidden states move. The measurement gives every GPU tokens_per_rank "
        "tokens, each with top_k distinct expert slots drawn uniformly from "
        "slot_count."
    ),
    category="Communication",
    formula=(
        "index bytes = tokens_per_rank · top_k · 4",
        "algbw = index bytes / time",
        "busbw = algbw",
    ),
    default_metric="time_ms",
    method=(
        "Mean time per call from CUDA events around 20 replays of a CUDA graph "
        "that captures 4 to 100 prepare calls, fewer for larger routing tables. "
        "Ten eager calls and two replays warm up first. Rank 0's time is kept."
    ),
    caveats=(
        "Uniform routing reads low: the runner's notes record a real skewed "
        "layer at 8k tokens taking about 9% longer.",
        "GB/s counts only the expert-ID table read, 4 bytes per ID; the time is "
        "the number to compare.",
    ),
    # No separate PyTorch reference implements this multi-GPU metadata pass.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_mnnvl",
        supports=BackendSupport(
            compute=None,
            gpus=frozenset({"NVIDIA H100", "NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.flashinfer_mnnvl_alltoall",
            function_name="profile_moe_alltoall_prepare_batch",
        ),
        table_name=KIND,
        args_schema=MoeAlltoallPrepareArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        gpu_count_fn=lambda spec: int(spec["ep_size"]),
        list_native=True,
        doc=BackendDoc(
            summary=(
                "FlashInfer's MNNVL prepare pass, "
                "mnnvl_moe_alltoallv_prepare_without_allgather, timed as one call."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/trtllm_alltoall.py",
        ),
    )
)
