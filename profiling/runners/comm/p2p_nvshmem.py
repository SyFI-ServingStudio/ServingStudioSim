"""NVSHMEM (nvshmem4py) point-to-point put runner (shared by ``p2p_intra`` /
``p2p_inter``).

The nvshmem counterpart to the NCCL ``profiling.runners.comm.p2p`` runner,
list-native: spawn 2 PEs **once per chunk** and time each size in the live
session. Uses the **native nvshmem4py host API** (``nc.put`` + ``nc.quiet``), NOT
ref's external on-stream benchmark binary: PE 0 writes its symmetric ``src`` into
PE 1's symmetric ``dst`` and ``nc.quiet`` makes the one-sided put completion-aware
(the device on-stream path times only the issue, which is too optimistic versus
NCCL send/recv). CUDA-event timed over ``rep`` iters on PE 0's stream. One hop
moves the payload once over the link, so ``busbw == algbw`` (no ring factor).

A trailing ``dist.barrier`` per size is required: PE 1 must wait for PE 0's puts
to drain before either side frees the symmetric buffers or (at chunk end)
``NvshmemLauncher`` calls ``nc.finalize``.

``fabric`` pins the cache namespace (intra → NVLink, inter → NIC) but the runner
body does not use it — single-node profiling places both ranks on whatever GPU
chunk L1b reserved.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.runners.comm._batch import run_comm_batch
from profiling.runners.comm._launcher import NvshmemLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult


def profile_p2p_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    """Profile a homogeneous chunk of NVSHMEM put configs with ONE 2-PE spawn."""
    if not kwargs_list:
        return []
    launcher = NvshmemLauncher(2)
    return run_comm_batch(launcher, _p2p_per_rank_batch, kwargs_list)


def _p2p_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    """Runs inside each NVSHMEM PE; PE 0 puts to PE 1. Loops every size in the live
    session (allocate/free per size, with a drain barrier before each free). No
    per-size try/except so the PEs stay in lockstep (see ``_batch``). Only PE 0
    returns the payloads (it owns the put-side stream we measure)."""
    del world_size
    try:
        import nvshmem.core as nc
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "nvshmem4py + torch are required for the NVSHMEM p2p runner"
        ) from exc

    peer = 1
    results: list[dict] = []
    try:
        for spec in specs:
            torch_dtype = DType.from_value(spec["dtype"]).torch()
            element_size = torch.tensor([], dtype=torch_dtype).element_size()
            num_elements = max(1, int(spec["message_size_bytes"]) // element_size)
            actual_bytes = num_elements * element_size
            # Symmetric (NVSHMEM-allocated) tensors required for one-sided put.
            src = nc.tensor((num_elements,), dtype=torch_dtype)
            dst = nc.tensor((num_elements,), dtype=torch_dtype)
            src.fill_(1.0)
            stream = torch.cuda.current_stream()

            for _ in range(warmup):
                if rank == 0:
                    nc.put(dst, src, peer, stream=stream)
                    nc.quiet(stream=stream)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            start.record()
            for _ in range(rep):
                if rank == 0:
                    nc.put(dst, src, peer, stream=stream)
                    nc.quiet(stream=stream)
            end.record()
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end) / rep

            # PE 1 must not free (or later finalize) until PE 0's puts have drained.
            dist.barrier()
            nc.free_tensor(src)
            nc.free_tensor(dst)

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
