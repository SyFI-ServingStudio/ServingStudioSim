"""NCCL point-to-point send/recv runner (shared by ``p2p_intra`` / ``p2p_inter``).

L1a-only: spawn 2 ranks via ``TorchMpLauncher``, time a one-directional
``dist.send`` / ``dist.recv`` on the rank-0 stream, and return a ``CommMetrics``.
This is the profiled primitive behind ref's ``P2pCurves`` (``common_timing.rs``):
a single src→dst transfer measured as time vs message size, used by the MoE
network model to price each dispatch/combine stage's bottleneck rank.

``fabric`` pins the cache namespace (intra → NVLink, inter → NIC) but the runner
body does not use it — single-node profiling places both ranks on whatever GPU
chunk L1b reserved, so the inter curve is the same-node bandwidth shape until a
multi-node launcher exists. p2p is one hop, so ``busbw == algbw`` (no ring
factor).
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.runners.comm._launcher import TorchMpLauncher
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
    """Profile one NCCL send/recv config between 2 ranks."""
    del fabric  # row/cache key only; not used by the kernel call
    dtype = DType.from_value(dtype)
    payload = TorchMpLauncher(2, backend="nccl").run(
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
    """Runs inside each spawned rank. rank 0 sends, rank 1 receives; only rank 0
    returns the timing payload (it owns the send-side stream we measure)."""
    del world_size
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the NCCL p2p runner") from exc

    try:
        torch_dtype = DType.from_value(dtype_str).torch()
        element_size = torch.tensor([], dtype=torch_dtype).element_size()
        num_elements = max(1, message_size_bytes // element_size)
        actual_bytes = num_elements * element_size
        tensor = torch.randn(num_elements, dtype=torch_dtype, device="cuda")

        def one_hop():
            if rank == 0:
                dist.send(tensor, dst=1)
            else:
                dist.recv(tensor, src=0)

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
