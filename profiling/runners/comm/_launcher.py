"""Multi-GPU launcher helpers for comm runners.

Comm runners choose their launcher inside the runner file. L1b only reserves a
GPU chunk and calls the runner through the execution backend.
"""

from __future__ import annotations

import os
import socket
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented


class MultiGpuLauncher(ABC):
    """K GPUs -> run a per-rank function -> return rank-0 timing payload."""

    def __init__(self, num_gpus: int):
        if num_gpus < 1:
            raise ValueError("num_gpus must be >= 1")
        self.num_gpus = num_gpus

    @abstractmethod
    def run(self, per_rank_fn: Callable[..., Any], **kwargs) -> Any:
        raise NotImplementedError


class TorchMpLauncher(MultiGpuLauncher):
    """torch.multiprocessing launcher for NCCL collectives and p2p runners."""

    def __init__(
        self,
        num_gpus: int,
        *,
        backend: str = "nccl",
        master_addr: str = "127.0.0.1",
        master_port: int | None = None,
    ):
        super().__init__(num_gpus)
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port or _find_free_port()

    def run(self, per_rank_fn: Callable[..., Any], **kwargs) -> Any:
        try:
            import torch.multiprocessing as mp
        except ImportError as exc:
            raise ProfilerNotImplemented("torch.multiprocessing is unavailable") from exc

        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        try:
            mp.spawn(
                _torch_mp_entry,
                args=(
                    self.num_gpus,
                    self.backend,
                    self.master_addr,
                    self.master_port,
                    per_rank_fn,
                    kwargs,
                    result_queue,
                ),
                nprocs=self.num_gpus,
                join=True,
            )
        except Exception as exc:
            raise KernelLaunchFailed(str(exc)) from exc

        if result_queue.empty():
            return None
        ok, payload = result_queue.get()
        if not ok:
            raise KernelLaunchFailed(str(payload))
        return payload


class NvshmemLauncher(MultiGpuLauncher):
    """torch.multiprocessing launcher for NVSHMEM (nvshmem4py) collectives.

    NVSHMEM is bootstrapped over a torch.distributed process group (NO MPI): each
    spawned rank joins a `cpu:gloo,cuda:nccl` group, rank 0 mints the NVSHMEM
    `UniqueID` and broadcasts it via `broadcast_object_list`, then every rank
    calls `nvshmem.core.init(..., initializer_method="uid")`. `per_rank_fn` runs
    inside that initialized session; teardown finalizes NVSHMEM and tears the
    group down. Only rank 0's return value is propagated back.
    """

    def __init__(
        self,
        num_gpus: int,
        *,
        master_addr: str = "127.0.0.1",
        master_port: int | None = None,
    ):
        super().__init__(num_gpus)
        self.master_addr = master_addr
        self.master_port = master_port or _find_free_port()

    def run(self, per_rank_fn: Callable[..., Any], **kwargs) -> Any:
        try:
            import torch.multiprocessing as mp
        except ImportError as exc:
            raise ProfilerNotImplemented("torch.multiprocessing is unavailable") from exc

        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        try:
            mp.spawn(
                _nvshmem_mp_entry,
                args=(
                    self.num_gpus,
                    self.master_addr,
                    self.master_port,
                    per_rank_fn,
                    kwargs,
                    result_queue,
                ),
                nprocs=self.num_gpus,
                join=True,
            )
        except Exception as exc:
            raise KernelLaunchFailed(str(exc)) from exc

        if result_queue.empty():
            return None
        ok, payload = result_queue.get()
        if not ok:
            raise KernelLaunchFailed(str(payload))
        return payload


class VllmLauncher(MultiGpuLauncher):
    """Placeholder boundary for vLLM-specific multi-GPU runner setup."""

    def run(self, per_rank_fn: Callable[..., Any], **kwargs) -> Any:
        del per_rank_fn, kwargs
        raise ProfilerNotImplemented("VllmLauncher bootstrap is not implemented yet")


def _torch_mp_entry(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: int,
    per_rank_fn: Callable[..., Any],
    fn_kwargs: dict[str, Any],
    result_queue,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    try:
        import torch
        import torch.distributed as dist

        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        try:
            result = per_rank_fn(rank=rank, world_size=world_size, **fn_kwargs)
        finally:
            dist.destroy_process_group()
    except Exception as exc:
        if rank == 0:
            result_queue.put((False, str(exc)))
        raise
    if rank == 0:
        result_queue.put((True, result))


def _nvshmem_mp_entry(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    per_rank_fn: Callable[..., Any],
    fn_kwargs: dict[str, Any],
    result_queue,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    try:
        import nvshmem.core as nc
        import torch
        import torch.distributed as dist
        from cuda.core import Device

        torch.cuda.set_device(rank)
        device = Device(rank)
        device.set_current()
        # gloo carries the CPU UniqueID broadcast; nccl is the CUDA collective backend.
        dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
        try:
            uid_box = [nc.get_unique_id() if rank == 0 else None]
            dist.broadcast_object_list(uid_box, src=0)
            dist.barrier()
            nc.init(
                device=device,
                uid=uid_box[0],
                rank=rank,
                nranks=world_size,
                initializer_method="uid",
            )
            try:
                result = per_rank_fn(rank=rank, world_size=world_size, **fn_kwargs)
            finally:
                nc.finalize()
        finally:
            dist.destroy_process_group()
    except Exception as exc:
        if rank == 0:
            result_queue.put((False, str(exc)))
        raise
    if rank == 0:
        result_queue.put((True, result))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
