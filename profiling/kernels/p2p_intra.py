"""Intra-domain point-to-point kernel kind.

Wire string ``"p2p_intra"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/p2p_intra.rs`` and the facade stem
``get_p2p_intra_times`` / ``count_missing_p2p_intra``.

The intra-NVL-domain leg of the MoE network model (ref's
``get_p2p_metrics_batch`` curve). A single send/recv between two ranks on the
same NVLink domain, profiled as time vs ``message_size_bytes``. Two backends:
``nccl`` (``dist.send``/``recv``) and ``nvshmem`` (native ``nc.put``/``quiet``).
``metric_family=COMM`` (algbw/busbw, no tflops); ``gpu_count_fn`` reserves 2
GPUs for the launcher. The runners are referenced lazily so the main process
never eager-imports torch.
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

KIND: str = "p2p_intra"


@dataclass(frozen=True)
class P2pIntraArgs(KernelArgs):
    # Bytes moved over the single src->dst link in this transfer.
    message_size_bytes: int
    dtype: DType
    # Network fabric token (matches Rust `Fabric` serde wire form, e.g.
    # "nvlink"). A row/cache key only — the runner body does not use it.
    fabric: str


def _spec(backend: str, module_name: str) -> KernelProfilerSpec:
    return KernelProfilerSpec(
        kernel_kind=KIND,
        backend=backend,
        runner_ref=RunnerRef(module_name=module_name, function_name="profile_p2p"),
        table_name=KIND,
        args_schema=P2pIntraArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        gpu_count_fn=lambda spec: 2,
    )


register(_spec("nccl", "profiling.runners.comm.p2p"))
register(_spec("nvshmem", "profiling.runners.comm.p2p_nvshmem"))
