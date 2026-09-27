"""Inter-domain point-to-point kernel kind.

Wire string ``"p2p_inter"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/p2p_inter.rs`` and the facade stem
``get_p2p_inter_times`` / ``count_missing_p2p_inter``.

The inter-NVL-domain (cross-node NIC) leg of the MoE network model (ref's
``get_inter_device_p2p_times_batch`` curve). Same shape as ``p2p_intra``; the
separate kind keeps the NIC bandwidth curve distinct from the NVLink one.

Cross-node p2p cannot be measured on a single-node box, so this kind does NOT
do real profiling: both backends point at ``profiling.runners.comm.p2p_inter``,
which returns a *modeled* time from a measured latency lookup table (ref's
analytical model). ``metric_family=COMM``; ``gpu_count_fn`` reserves 1 GPU only
to identify the device family for curve selection — no peer rank is spawned.
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

KIND: str = "p2p_inter"


@dataclass(frozen=True)
class P2pInterArgs(KernelArgs):
    # Bytes moved over the single cross-domain src->dst link in this transfer.
    message_size_bytes: int = arg(unit="bytes", doc="Payload bytes on one cross-domain transfer.")
    dtype: DType = arg(doc="Element type recorded for the payload; bf16 is used for the sweep.")
    # Network fabric token (matches Rust `Fabric` serde wire form, e.g.
    # "infiniband"). A row/cache key only — the runner body does not use it.
    fabric: str = arg(doc="Interconnect label used to identify the modeled curve.")


DOC = KernelDoc(
    title="Inter-domain point-to-point transfer",
    summary="Modeled time for one transfer between GPUs in different NVLink domains.",
    description=(
        "The simulator prices each MoE dispatch and combine transfer between "
        "GPUs in different NVLink domains, that is, across nodes, by looking up "
        "its bytes on this curve; an fp8 payload is fewer bytes on the same "
        "curve. A single node cannot measure such a transfer, so the rows are "
        "modeled: H100, H200 and B200 read a stored inter-node latency table, "
        "and other GPUs use a fixed bandwidth."
    ),
    category="Communication",
    formula=(
        "effective bytes = message_size_bytes (H100/H200/B200); otherwise "
        "max(message_size_bytes, 32 KiB)",
        "algbw = effective bytes / time",
        "busbw = algbw",
    ),
    default_metric="time_ms",
    method=(
        "No transfer is timed. H100, H200 and B200 times are linearly "
        "interpolated from a stored latency table; values beyond its last "
        "point are extrapolated from the final segment. Other GPUs use "
        "effective bytes divided by a fixed bandwidth."
    ),
    caveats=(
        "Both backend labels return the same modeled time; neither launches "
        "a communication library.",
        "The fabric label does not change the estimate.",
        "The latency table was ported from an earlier reference model; how it "
        "was measured is not recorded here.",
    ),
    # This analytical curve has no separate PyTorch reference implementation.
    reference=None,
)

_BACKEND_DOCS = {
    "nccl": BackendDoc(
        summary=("The modeled inter-node curve under the nccl label; nothing is launched.")
    ),
    "nvshmem": BackendDoc(summary="The same modeled curve as nccl, under the nvshmem label."),
}


def _spec(backend: str, module_name: str) -> KernelProfilerSpec:
    return KernelProfilerSpec(
        kernel_kind=KIND,
        backend=backend,
        # Comm is size-keyed — dtype-agnostic.
        supports=BackendSupport(compute=None),
        runner_ref=RunnerRef(module_name=module_name, function_name="profile_p2p"),
        table_name=KIND,
        args_schema=P2pInterArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        gpu_count_fn=lambda spec: 1,
        doc=_BACKEND_DOCS[backend],
    )


# Both backends resolve to the analytical lookup-table runner: the modeled curve
# is backend-agnostic, but the two (kind, backend) keys are kept so the Rust
# cache/facade wiring stays identical to p2p_intra.
register(_spec("nccl", "profiling.runners.comm.p2p_inter"))
register(_spec("nvshmem", "profiling.runners.comm.p2p_inter"))
