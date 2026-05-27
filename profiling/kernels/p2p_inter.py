"""Inter-domain point-to-point kernel kind.

Wire string ``"p2p_inter"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/p2p_inter.rs`` and the facade stem
``get_p2p_inter_times`` / ``count_missing_p2p_inter``.

The inter-NVL-domain (cross-node NIC) leg of the MoE network model (ref's
``get_inter_device_p2p_times_batch`` curve). Same shape as ``p2p_intra``; the
separate kind keeps the NIC bandwidth curve distinct from the NVLink one. Two
backends: ``nccl`` (``profiling.runners.comm.p2p``) and ``nvshmem``
(``profiling.runners.comm.p2p_nvshmem``); ``fabric`` distinguishes the
namespace. ``metric_family=COMM``; ``gpu_count_fn`` reserves 2 GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "p2p_inter"


@dataclass(frozen=True)
class P2pInterArgs(KernelArgs):
    # Bytes moved over the single cross-domain src->dst link in this transfer.
    message_size_bytes: int
    dtype: DType
    # Network fabric token (matches Rust `Fabric` serde wire form, e.g.
    # "infiniband"). A row/cache key only — the runner body does not use it.
    fabric: str


def _spec(backend: str, module_name: str) -> KernelProfilerSpec:
    return KernelProfilerSpec(
        kernel_kind=KIND,
        backend=backend,
        runner_ref=RunnerRef(module_name=module_name, function_name="profile_p2p"),
        table_name=KIND,
        args_schema=P2pInterArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        gpu_count_fn=lambda spec: 2,
    )


register(_spec("nccl", "profiling.runners.comm.p2p"))
register(_spec("nvshmem", "profiling.runners.comm.p2p_nvshmem"))
