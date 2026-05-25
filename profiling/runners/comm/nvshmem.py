"""NVSHMEM (nvshmem4py) collective runners.

L1a-only: ``NvshmemLauncher`` spawns K ranks, bootstraps NVSHMEM over a
torch.distributed group (UID broadcast, NO MPI), and calls ``nvshmem.core.init``
before this per-rank fn runs. The fn allocates symmetric tensors, times an
all-reduce (``nvshmem.core.reduce`` over ``TEAM_WORLD`` — every PE gets the sum),
and returns a ``CommMetrics``. Validated against
``ref/profile/network/allreduce_nvshmem.py`` (saturated bus BW matches to ~1%).

``fabric`` is a row/cache key field only; the runner body does not use it.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.runners.comm._launcher import NvshmemLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import CommMetrics


def profile_all_reduce(
    num_gpus: int,
    message_size_bytes: int,
    dtype: DType | str,
    fabric: str,
    *,
    warmup: int = 50,
    rep: int = 100,
) -> CommMetrics:
    """Profile one NVSHMEM all-reduce config across ``num_gpus`` ranks."""
    del fabric  # row/cache key only; not used by the kernel call
    dtype = DType.from_value(dtype)
    if num_gpus < 2:
        raise ProfilerNotImplemented("all-reduce needs at least 2 GPUs")

    payload = NvshmemLauncher(num_gpus).run(
        _all_reduce_per_rank,
        message_size_bytes=message_size_bytes,
        dtype_str=dtype.value,
        warmup=warmup,
        rep=rep,
    )
    return CommMetrics(
        time_ms=float(payload["time_ms"]),
        algbw_gbps=float(payload["algbw_gbps"]),
        busbw_gbps=float(payload["busbw_gbps"]),
        message_size_bytes=int(payload["message_size_bytes"]),
        energy_j=float(payload.get("energy_j", 0.0)),
    )


def _all_reduce_per_rank(
    *,
    rank: int,
    world_size: int,
    message_size_bytes: int,
    dtype_str: str,
    warmup: int,
    rep: int,
) -> dict | None:
    """Runs inside each NVSHMEM rank (NVSHMEM + the torch.distributed group are
    already initialized by ``NvshmemLauncher``). Only rank 0 returns the payload."""
    try:
        import nvshmem.core as nc
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "nvshmem4py + torch are required for the NVSHMEM runner"
        ) from exc

    try:
        torch_dtype = DType.from_value(dtype_str).torch()
        element_size = torch.tensor([], dtype=torch_dtype).element_size()
        num_elements = max(1, message_size_bytes // element_size)
        actual_bytes = num_elements * element_size
        # Symmetric (NVSHMEM-allocated) tensors; reduce over TEAM_WORLD is an
        # all-reduce — every PE in the team receives the sum.
        src = nc.tensor((num_elements,), dtype=torch_dtype)
        dst = nc.tensor((num_elements,), dtype=torch_dtype)
        src.fill_(1.0)
        stream = torch.cuda.current_stream()

        def collective() -> None:
            nc.reduce(nc.Teams.TEAM_WORLD, dst, src, "sum", stream=stream)

        for _ in range(warmup):
            collective()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start.record()
        for _ in range(rep):
            collective()
        end.record()
        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end) / rep

        nc.free_tensor(src)
        nc.free_tensor(dst)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    if rank != 0:
        return None

    latency_s = time_ms / 1000.0
    algbw_gbps = (actual_bytes / latency_s) / 1e9 if latency_s > 0 else 0.0
    busbw_gbps = algbw_gbps * 2 * (world_size - 1) / world_size
    return {
        "time_ms": time_ms,
        "algbw_gbps": algbw_gbps,
        "busbw_gbps": busbw_gbps,
        "message_size_bytes": actual_bytes,
    }
