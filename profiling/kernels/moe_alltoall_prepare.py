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
    ep_size: int
    tokens_per_rank: int
    top_k: int
    slot_count: int
    fabric: str


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
    )
)
