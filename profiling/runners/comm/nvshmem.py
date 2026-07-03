"""NVSHMEM (nvshmem4py) collective runners.

L1a-only, list-native: ``NvshmemLauncher`` spawns K ranks **once per chunk**,
bootstraps NVSHMEM over a torch.distributed group (UID broadcast, NO MPI), and
calls ``nvshmem.core.init`` before the per-rank fn runs. The fn loops every size
in the live session, timing an all-reduce (``nvshmem.core.reduce`` over
``TEAM_WORLD`` — every PE gets the sum), and rank 0 returns one payload per size.
Validated against ``ref/profile/network/allreduce_nvshmem.py`` (saturated bus BW
matches to ~1%).

``fabric`` is a row/cache key field only; the runner body does not use it.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.runners.comm._batch import all_error, run_comm_batch
from profiling.runners.comm._launcher import NvshmemLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult


def profile_all_reduce_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    """Profile a homogeneous chunk of NVSHMEM all-reduce configs with ONE rank-group
    spawn. All specs share ``num_gpus`` (the chunk is grouped by gpu_count)."""
    if not kwargs_list:
        return []
    num_gpus = int(kwargs_list[0]["num_gpus"])
    if num_gpus < 2:
        return all_error(len(kwargs_list), "all-reduce needs at least 2 GPUs")
    launcher = NvshmemLauncher(num_gpus)
    return run_comm_batch(launcher, _all_reduce_per_rank_batch, kwargs_list)


def _all_reduce_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    """Runs inside each NVSHMEM rank (NVSHMEM + the torch.distributed group are
    already initialized by ``NvshmemLauncher``). Loops every size, allocating and
    freeing the symmetric buffers per size — a trailing ``dist.barrier`` drains
    the collective before ``free_tensor`` so no PE frees a buffer another still
    reads. No per-size try/except (the ranks stay in lockstep — see ``_batch``).
    Only rank 0 returns the ordered payloads."""
    try:
        import nvshmem.core as nc
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "nvshmem4py + torch are required for the NVSHMEM runner"
        ) from exc

    results: list[dict] = []
    try:
        for spec in specs:
            torch_dtype = DType.from_value(spec["dtype"]).torch()
            element_size = torch.tensor([], dtype=torch_dtype).element_size()
            num_elements = max(1, int(spec["message_size_bytes"]) // element_size)
            actual_bytes = num_elements * element_size
            # Symmetric (NVSHMEM-allocated) tensors; reduce over TEAM_WORLD is an
            # all-reduce — every PE in the team receives the sum.
            src = nc.tensor((num_elements,), dtype=torch_dtype)
            dst = nc.tensor((num_elements,), dtype=torch_dtype)
            src.fill_(1.0)
            stream = torch.cuda.current_stream()

            for _ in range(warmup):
                nc.reduce(nc.Teams.TEAM_WORLD, dst, src, "sum", stream=stream)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            start.record()
            for _ in range(rep):
                nc.reduce(nc.Teams.TEAM_WORLD, dst, src, "sum", stream=stream)
            end.record()
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end) / rep

            # Drain the collective on every PE before freeing this size's buffers.
            dist.barrier()
            nc.free_tensor(src)
            nc.free_tensor(dst)

            latency_s = time_ms / 1000.0
            algbw_gbps = (actual_bytes / latency_s) / 1e9 if latency_s > 0 else 0.0
            busbw_gbps = algbw_gbps * 2 * (world_size - 1) / world_size
            results.append(
                {"time_ms": time_ms, "algbw_gbps": algbw_gbps, "busbw_gbps": busbw_gbps}
            )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    if rank != 0:
        return None
    return results
