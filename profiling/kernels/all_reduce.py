"""All-reduce collective kernel kind.

All Python-side per-kernel knowledge for ``all_reduce`` lives here: the wire
string ``KIND``, the ``AllReduceArgs`` schema, and the ``register(...)`` calls
that wire the ``nccl`` and ``nvshmem`` backends into ``profiling.db.registry``.

Wire string: ``"all_reduce"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/all_reduce.rs`` and the Python facade stem used
by ``profiling.facade`` to generate ``get_all_reduce_times`` /
``count_missing_all_reduce``.

Comm-specific vs the compute kernels: ``metric_family=COMM`` (the table carries
algbw/busbw/message_size, not tflops), and ``gpu_count_fn`` tells L1b each spec
needs ``num_gpus`` real GPUs reserved for the multi-rank launcher. The runner
modules are referenced lazily via ``RunnerRef`` so the main process never
eager-imports torch / a multi-process launcher.
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

KIND: str = "all_reduce"


@dataclass(frozen=True)
class AllReduceArgs(KernelArgs):
    num_gpus: int = arg(unit="GPUs", doc="GPUs in the group, one rank each.")
    # FULL buffer each rank all-reduces (the whole tensor handed to
    # dist.all_reduce), NOT a reduce-scatter shard. The ring 2(N-1)/N movement is
    # captured in the measured time, so callers pass the complete output size.
    message_size_bytes: int = arg(
        unit="bytes",
        doc="Full buffer each GPU contributes and receives, not a per-GPU shard.",
    )
    dtype: DType = arg(doc="Element type of the buffer. The simulator profiles bf16 only.")
    # Network fabric token (matches Rust `Fabric` serde wire form, e.g.
    # "nvlink"). A row/cache key only — the runner body does not use it.
    fabric: str = arg(
        doc=(
            "Interconnect label, such as nvlink. It keys the row; the run uses "
            "whatever links the reserved GPUs have."
        )
    )


DOC = KernelDoc(
    title="All-reduce",
    summary="Sum one buffer across the GPUs of a tensor-parallel group.",
    description=(
        "Tensor-parallel attention and MLP blocks end with an all-reduce of the "
        "row-parallel projection's output, so every GPU holds the full sum. "
        "message_size_bytes is the full buffer each GPU contributes, not a shard. "
        "Measured on the GPUs of one node, over whatever links connect them."
    ),
    category="Communication",
    formula=(
        "algbw = message_size_bytes / time",
        "busbw = algbw · 2(N − 1) / N, for N = num_gpus",
    ),
    default_metric="busbw_gbps",
    caveats=(
        "busbw is the number to compare with the link: it scales algbw by the "
        "2(N − 1)/N traffic of a ring all-reduce. The NVLink figure in the GPU "
        "catalog counts both directions.",
        "Cost is keyed by bytes, not dtype: rows are measured at bf16, and the "
        "simulator reads an fp8 payload as fewer bytes on the same curve.",
    ),
    method=(
        "Wall time between two CUDA events around repeated calls. Each size runs 50 "
        "warm-up all-reduces, a barrier, then 100 timed calls back to back in the "
        "same live process group; the time is rank 0's mean per call. Energy is not "
        "measured for collectives."
    ),
    reference=None,
)

_BACKEND_DOCS = {
    "nccl": BackendDoc(
        summary="torch.distributed.all_reduce with sum, on the NCCL backend.",
        url="https://github.com/NVIDIA/nccl",
    ),
    "nvshmem": BackendDoc(
        summary=(
            "nvshmem4py reduce with sum over TEAM_WORLD, on symmetric-memory "
            "buffers, so every GPU receives the sum."
        ),
        url="https://github.com/NVIDIA/nvshmem",
    ),
    "rccl": BackendDoc(
        summary=(
            "torch.distributed.all_reduce with sum on the ROCm RCCL backend: the "
            "MI300X large all-reduce vLLM-ROCm runs over Infinity Fabric. Torch's "
            '"nccl" process-group backend aliases to librccl on a ROCm build.'
        ),
        url="https://github.com/ROCm/rccl",
    ),
}


def _spec(
    backend: str,
    module_name: str,
    *,
    supports: BackendSupport = BackendSupport(compute=None),
    subprocess_env: str | None = None,
) -> KernelProfilerSpec:
    return KernelProfilerSpec(
        kernel_kind=KIND,
        backend=backend,
        # Comm is size-keyed (fewer bytes at fp8, same bf16 curve) — dtype-agnostic.
        supports=supports,
        # list_native: the runner spawns its rank group once per chunk and loops
        # every size inside, so the worker hands it the whole spec list (no per-shape
        # re-init). function_name points at the batch entry, not the single-spec one.
        runner_ref=RunnerRef(module_name=module_name, function_name="profile_all_reduce_batch"),
        table_name=KIND,
        args_schema=AllReduceArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env=subprocess_env,
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
        doc=_BACKEND_DOCS[backend],
    )


register(_spec("nccl", "profiling.runners.comm.nccl"))
register(_spec("nvshmem", "profiling.runners.comm.nvshmem"))
# MI300X large (non-fused) all-reduce: RCCL over Infinity Fabric, a MEASURED
# multi-GPU row (GLM-5.3-Flash MI300X port). AMD-arch gated (CDNA3) and run in
# the ROCm venv; the measurement reuses the NCCL runner because Torch's "nccl"
# backend is RCCL on ROCm. B200 keeps the NVIDIA nccl backend above, byte-identical.
register(
    _spec(
        "rccl",
        "profiling.runners.comm.rccl",
        supports=BackendSupport(compute=None, arch_targets=frozenset({"CDNA3"})),
        subprocess_env="vllm_rocm_env",
    )
)
