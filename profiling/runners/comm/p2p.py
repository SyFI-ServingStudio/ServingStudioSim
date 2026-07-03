"""NCCL point-to-point send/recv runner (shared by ``p2p_intra`` / ``p2p_inter``).

L1a-only, list-native: spawn 2 ranks via ``TorchMpLauncher`` **once per chunk**,
time a one-directional ``dist.send`` / ``dist.recv`` per size on the rank-0
stream, and return one ``RunnerResult`` per spec. This is the profiled primitive
behind ref's ``P2pCurves`` (``common_timing.rs``): a single src→dst transfer
measured as time vs message size, used by the MoE network model to price each
dispatch/combine stage's bottleneck rank.

``fabric`` pins the cache namespace (intra → NVLink, inter → NIC) but the runner
body does not use it — single-node profiling places both ranks on whatever GPU
chunk L1b reserved, so the inter curve is the same-node bandwidth shape until a
multi-node launcher exists. p2p is one hop, so ``busbw == algbw`` (no ring
factor).
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.runners.comm._batch import run_comm_batch
from profiling.runners.comm._launcher import TorchMpLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult


def profile_p2p_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    """Profile a homogeneous chunk of NCCL send/recv configs with ONE 2-rank spawn."""
    if not kwargs_list:
        return []
    launcher = TorchMpLauncher(2, backend="nccl")
    return run_comm_batch(launcher, _p2p_per_rank_batch, kwargs_list)


def _p2p_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    """Runs inside each spawned rank; rank 0 sends, rank 1 receives. Loops every
    size in the live group (allocate/free per size). No per-size try/except so the
    two ranks stay in lockstep (see ``_batch``). Only rank 0 returns the payloads
    (it owns the send-side stream we measure)."""
    del world_size
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the NCCL p2p runner") from exc

    results: list[dict] = []
    try:
        for spec in specs:
            torch_dtype = DType.from_value(spec["dtype"]).torch()
            element_size = torch.tensor([], dtype=torch_dtype).element_size()
            num_elements = max(1, int(spec["message_size_bytes"]) // element_size)
            actual_bytes = num_elements * element_size
            tensor = torch.randn(num_elements, dtype=torch_dtype, device="cuda")

            for _ in range(warmup):
                if rank == 0:
                    dist.send(tensor, dst=1)
                else:
                    dist.recv(tensor, src=0)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            start.record()
            for _ in range(rep):
                if rank == 0:
                    dist.send(tensor, dst=1)
                else:
                    dist.recv(tensor, src=0)
            end.record()
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end) / rep

            del tensor  # free this size before the next alloc

            latency_s = time_ms / 1000.0
            # One hop moves the payload once over the link: algbw == busbw.
            bw_gbps = (actual_bytes / latency_s) / 1e9 if latency_s > 0 else 0.0
            results.append(
                {"time_ms": time_ms, "algbw_gbps": bw_gbps, "busbw_gbps": bw_gbps}
            )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    if rank != 0:
        return None
    return results
