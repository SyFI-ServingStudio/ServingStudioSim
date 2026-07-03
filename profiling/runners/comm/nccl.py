"""NCCL (torch.distributed) collective runners.

L1a-only, list-native: spawn K ranks via ``TorchMpLauncher`` **once per chunk**,
time each all-reduce size on the rank-0 stream inside the live group, and return
one ``RunnerResult`` per spec. DB writes, JIT policy, and registry routing live
in L1b. Mirrors ``ref/profile/network/allreduce_nccl.py`` for the real kernel
call (``dist.all_reduce``) and the alg/bus bandwidth formulas.

``fabric`` is carried as a row/key field (it pins the cache namespace) but the
runner body does not use it — the physical fabric is whatever the reserved GPU
chunk happens to sit on (intra-node NVLink for this milestone).
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.runners.comm._batch import all_error, run_comm_batch
from profiling.runners.comm._launcher import TorchMpLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult


def profile_all_reduce_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    """Profile a homogeneous chunk of NCCL all-reduce configs with ONE rank-group
    spawn. All specs share ``num_gpus`` (the chunk is grouped by gpu_count), so
    the launcher is sized from the first spec and reused for every size."""
    if not kwargs_list:
        return []
    num_gpus = int(kwargs_list[0]["num_gpus"])
    if num_gpus < 2:
        return all_error(len(kwargs_list), "all-reduce needs at least 2 GPUs")
    launcher = TorchMpLauncher(num_gpus, backend="nccl")
    return run_comm_batch(launcher, _all_reduce_per_rank_batch, kwargs_list)


def _all_reduce_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    """Runs inside each spawned rank (torch.distributed already initialized by
    ``TorchMpLauncher``). Loops every size in the live group — allocating/freeing
    per size (the spawn is the amortized cost; alloc is µs and a fresh buffer per
    size avoids perturbing the measured BW). No per-size try/except: a failure
    raises out of the whole group so the ranks never desync (see ``_batch``).
    Only rank 0 returns the ordered timing payloads."""
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the NCCL all-reduce runner") from exc

    results: list[dict] = []
    try:
        for spec in specs:
            torch_dtype = DType.from_value(spec["dtype"]).torch()
            element_size = torch.tensor([], dtype=torch_dtype).element_size()
            num_elements = max(1, int(spec["message_size_bytes"]) // element_size)
            actual_bytes = num_elements * element_size
            tensor = torch.randn(num_elements, dtype=torch_dtype, device="cuda")

            for _ in range(warmup):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            start.record()
            for _ in range(rep):
                dist.all_reduce(tensor)
            end.record()
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end) / rep

            del tensor  # free this size before the next alloc

            latency_s = time_ms / 1000.0
            algbw_gbps = (actual_bytes / latency_s) / 1e9 if latency_s > 0 else 0.0
            # Ring all-reduce moves 2(N-1)/N of the payload across the bus.
            busbw_gbps = algbw_gbps * 2 * (world_size - 1) / world_size
            results.append(
                {"time_ms": time_ms, "algbw_gbps": algbw_gbps, "busbw_gbps": busbw_gbps}
            )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    if rank != 0:
        return None
    return results
