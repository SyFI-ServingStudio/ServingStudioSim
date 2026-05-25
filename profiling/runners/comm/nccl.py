"""NCCL (torch.distributed) collective runners.

L1a-only: spawn K ranks via ``TorchMpLauncher``, time one collective on the
rank-0 stream, and return a ``CommMetrics``. DB writes, JIT policy, and registry
routing live in L1b. Mirrors ``ref/profile/network/allreduce_nccl.py`` for the
real kernel call (``dist.all_reduce``) and the alg/bus bandwidth formulas.

``fabric`` is carried as a row/key field (it pins the cache namespace) but the
runner body does not use it — the physical fabric is whatever the reserved GPU
chunk happens to sit on (intra-node NVLink for this milestone).
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.runners.comm._launcher import TorchMpLauncher
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
    """Profile one NCCL all-reduce config across ``num_gpus`` ranks."""
    del fabric  # row/cache key only; not used by the kernel call
    dtype = DType.from_value(dtype)
    if num_gpus < 2:
        raise ProfilerNotImplemented("all-reduce needs at least 2 GPUs")

    payload = TorchMpLauncher(num_gpus, backend="nccl").run(
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
    """Runs inside each spawned rank (torch.distributed already initialized by
    ``TorchMpLauncher``). Only rank 0 returns the timing payload."""
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the NCCL all-reduce runner") from exc

    try:
        torch_dtype = DType.from_value(dtype_str).torch()
        element_size = torch.tensor([], dtype=torch_dtype).element_size()
        num_elements = max(1, message_size_bytes // element_size)
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
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    if rank != 0:
        return None

    latency_s = time_ms / 1000.0
    algbw_gbps = (actual_bytes / latency_s) / 1e9 if latency_s > 0 else 0.0
    # Ring all-reduce moves 2(N-1)/N of the payload across the bus.
    busbw_gbps = algbw_gbps * 2 * (world_size - 1) / world_size
    return {
        "time_ms": time_ms,
        "algbw_gbps": algbw_gbps,
        "busbw_gbps": busbw_gbps,
        "message_size_bytes": actual_bytes,
    }
