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
from profiling.db.doc import BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "p2p_intra"


@dataclass(frozen=True)
class P2pIntraArgs(KernelArgs):
    # Bytes moved over the single src->dst link in this transfer.
    message_size_bytes: int = arg(unit="bytes", doc="Payload bytes sent from one GPU to its peer.")
    dtype: DType = arg(doc="Element type of the payload; bf16 is used for the sweep.")
    # Network fabric token (matches Rust `Fabric` serde wire form, e.g.
    # "nvlink"). A row/cache key only — the runner body does not use it.
    fabric: str = arg(doc="Interconnect label used to identify the measurement.")


DOC = KernelDoc(
    title="Intra-domain point-to-point transfer",
    summary="Send one payload between two GPUs in the same NVLink domain.",
    description=(
        "The simulator prices each MoE dispatch and combine transfer between "
        "GPUs in one NVLink domain by looking up its bytes on this curve; an fp8 "
        "payload is fewer bytes on the same curve. One GPU sends "
        "message_size_bytes to one peer, measured at bf16."
    ),
    category="Communication",
    formula=(
        "actual bytes = max(1, ⌊message_size_bytes / bytes per element⌋) · bytes per element",
        "algbw = actual bytes / time",
        "busbw = algbw",
    ),
    default_metric="busbw_gbps",
    method=(
        "CUDA-event time around 100 repeated transfers after 50 warm-up "
        "transfers and a barrier. Rank 0's mean per-transfer time is kept. "
        "NCCL times dist.send to rank 1; NVSHMEM times put followed by quiet "
        "on the sender's stream."
    ),
    caveats=(
        "The runner rounds the requested byte count down to whole dtype elements, "
        "with at least one element transferred.",
        "The fabric label keys the result but does not select links; the "
        "reserved GPUs determine the physical connection.",
    ),
    # Neither communication backend has a separate PyTorch reference module.
    reference=None,
)

_BACKEND_DOCS = {
    "nccl": BackendDoc(summary="torch.distributed send and recv over NCCL, rank 0 to rank 1."),
    "nvshmem": BackendDoc(
        summary=(
            "nvshmem4py put from rank 0 into rank 1's symmetric memory, then quiet "
            "to wait for completion."
        )
    ),
}


def _spec(backend: str, module_name: str) -> KernelProfilerSpec:
    return KernelProfilerSpec(
        kernel_kind=KIND,
        backend=backend,
        # Comm is size-keyed — dtype-agnostic.
        supports=BackendSupport(compute=None),
        # list_native: the runner spawns its 2-rank group once per chunk and loops
        # every size inside (no per-shape re-init). function_name points at the batch
        # entry, not the single-spec one.
        runner_ref=RunnerRef(module_name=module_name, function_name="profile_p2p_batch"),
        table_name=KIND,
        args_schema=P2pIntraArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        gpu_count_fn=lambda spec: 2,
        list_native=True,
        doc=_BACKEND_DOCS[backend],
    )


register(_spec("nccl", "profiling.runners.comm.p2p"))
register(_spec("nvshmem", "profiling.runners.comm.p2p_nvshmem"))
