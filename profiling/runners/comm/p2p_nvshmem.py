"""NVSHMEM (nvshmem4py) point-to-point put runner (shared by ``p2p_intra`` /
``p2p_inter``).

The nvshmem counterpart to the NCCL ``profiling.runners.comm.p2p`` runner. Uses
the **native nvshmem4py host API** (``nc.put`` + ``nc.quiet``), NOT ref's
external on-stream benchmark binary: PE 0 writes its symmetric ``src`` into PE
1's symmetric ``dst`` and ``nc.quiet`` makes the one-sided put completion-aware
(the device on-stream path times only the issue, which is too optimistic versus
NCCL send/recv). CUDA-event timed over ``rep`` iters on PE 0's stream. One hop
moves the payload once over the link, so ``busbw == algbw`` (no ring factor).

A trailing ``dist.barrier`` is required: ``NvshmemLauncher`` calls
``nc.finalize`` the instant each rank's fn returns, so PE 1 must wait for PE 0's
puts to drain before either side tears NVSHMEM down.

``fabric`` pins the cache namespace (intra → NVLink, inter → NIC) but the runner
body does not use it — single-node profiling places both ranks on whatever GPU
chunk L1b reserved.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.runners.comm._launcher import NvshmemLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import CommMetrics


def profile_p2p(
    message_size_bytes: int,
    dtype: DType | str,
    fabric: str,
    *,
    warmup: int = 50,
    rep: int = 100,
) -> CommMetrics:
    """Profile one NVSHMEM put config between 2 PEs."""
    del fabric  # row/cache key only; not used by the kernel call
    dtype = DType.from_value(dtype)
    payload = NvshmemLauncher(2).run(
        _p2p_per_rank,
        message_size_bytes=message_size_bytes,
        dtype_str=dtype.value,
        warmup=warmup,
        rep=rep,
    )
    return CommMetrics(
        time_ms=float(payload["time_ms"]),
        algbw_gbps=float(payload["algbw_gbps"]),
        busbw_gbps=float(payload["busbw_gbps"]),
        energy_j=float(payload.get("energy_j", 0.0)),
    )


def _p2p_per_rank(
    *,
    rank: int,
    world_size: int,
    message_size_bytes: int,
    dtype_str: str,
    warmup: int,
    rep: int,
) -> dict | None:
    """Runs inside each NVSHMEM PE (NVSHMEM + the torch.distributed group are
    already initialized by ``NvshmemLauncher``). PE 0 puts to PE 1; only PE 0
    returns the timing payload (it owns the put-side stream we measure)."""
    del world_size
    try:
        import nvshmem.core as nc
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "nvshmem4py + torch are required for the NVSHMEM p2p runner"
        ) from exc

    try:
        torch_dtype = DType.from_value(dtype_str).torch()
        element_size = torch.tensor([], dtype=torch_dtype).element_size()
        num_elements = max(1, message_size_bytes // element_size)
        actual_bytes = num_elements * element_size
        # Symmetric (NVSHMEM-allocated) tensors required for one-sided put.
        src = nc.tensor((num_elements,), dtype=torch_dtype)
        dst = nc.tensor((num_elements,), dtype=torch_dtype)
        src.fill_(1.0)
        stream = torch.cuda.current_stream()
        peer = 1

        def one_hop() -> None:
            if rank == 0:
                nc.put(dst, src, peer, stream=stream)
                nc.quiet(stream=stream)

        for _ in range(warmup):
            one_hop()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start.record()
        for _ in range(rep):
            one_hop()
        end.record()
        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end) / rep

        # PE 1 must not finalize NVSHMEM until PE 0's puts have drained.
        dist.barrier()
        nc.free_tensor(src)
        nc.free_tensor(dst)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    if rank != 0:
        return None

    latency_s = time_ms / 1000.0
    # One hop moves the payload once over the link: algbw == busbw.
    bw_gbps = (actual_bytes / latency_s) / 1e9 if latency_s > 0 else 0.0
    return {
        "time_ms": time_ms,
        "algbw_gbps": bw_gbps,
        "busbw_gbps": bw_gbps,
    }
