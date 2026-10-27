"""NeuronLink collective runner (worker only): backend ``vllm_neuron_dist``.

List-native: one ``torch.multiprocessing`` spawn of ``num_ranks`` processes per
chunk, one per LNC2 logical core of the reserved chip, each joining a
``torch.distributed`` group on the lite runtime's ``neuron`` backend; every size of
the chunk runs inside the live group and rank 0 returns the ordered timings
(``profiling/runners/comm/_batch.py`` maps them to rows). The reservation
exposes the chip's logical cores as ``NEURON_VISIBLE_DEVICES=a-b``; each rank
pins its own core with ``NEURON_RT_VISIBLE_CORES`` and all ranks share one
``NEURON_RT_ROOT_COMM_ID`` before they import torch.

Timing is host-side: ``warmup`` calls, a fence and a synchronize, then ``rep``
back-to-back calls and a synchronize; the row is rank 0's mean per call.
Collectives are issued asynchronously, so a message large enough to keep the
device busy reads as device time; a tiny message reads as launch-bound (20-29 us
for the all-reduce over 8 KB-4 MB). Inside a compiled TP4 decode graph an
all-reduce is 13 us at 8 KB (``doc/pr/llama31-8b-trn2/README.md``, "Check 1");
a device-side timing of this leaf is not implemented. Under
``decode_granularity: layer`` no decode collective reads these rows.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

from profiling.db.args import DType
from profiling.runners.comm._batch import _REP, _WARMUP, all_error, run_comm_batch
from profiling.runners.comm._launcher import MultiGpuLauncher, _find_free_port
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult

OPS = ("all_reduce", "all_gather", "reduce_scatter")
_PROVENANCE: dict[str, str] = {}


def visible_logical_cores() -> list[int]:
    """The logical cores the reservation exposes (``NEURON_VISIBLE_DEVICES`` as
    ``a-b`` or ``a``, else ``NEURON_RT_VISIBLE_CORES``)."""
    span = os.environ.get("NEURON_VISIBLE_DEVICES") or os.environ.get("NEURON_RT_VISIBLE_CORES")
    if not span:
        raise ValueError("the Neuron reservation must set NEURON_VISIBLE_DEVICES")
    first, _, last = span.partition("-")
    return list(range(int(first), int(last or first) + 1))


def bus_bandwidth_factor(op: str, num_ranks: int) -> float:
    """Ring traffic per byte of the full buffer: 2(N-1)/N for all-reduce,
    (N-1)/N for all-gather and reduce-scatter."""
    if op == "all_reduce":
        return 2.0 * (num_ranks - 1) / num_ranks
    if op in ("all_gather", "reduce_scatter"):
        return (num_ranks - 1) / num_ranks
    raise ValueError(f"unknown Neuron collective {op!r}; expected one of {OPS}")


class NeuronMpLauncher(MultiGpuLauncher):
    """Spawn one process per logical core; each pins its core, joins the
    ``neuron`` process group and runs ``per_rank_fn``. The torch rendezvous and
    the NRT root communicator listen on two distinct free ports."""

    def __init__(
        self,
        num_ranks: int,
        cores: list[int],
        *,
        master_port: int | None = None,
        own_core_only: bool = False,
    ):
        """``own_core_only``: each rank sees its logical core as device 0 and
        not the chip's span (``NEURON_VISIBLE_DEVICES`` is removed before the
        runtime loads), for code that compiles graphs on ``neuron:0``."""
        super().__init__(num_ranks)
        if len(cores) < num_ranks:
            raise ValueError(f"{num_ranks} ranks need {num_ranks} visible logical cores")
        self.cores = cores[:num_ranks]
        self.master_port = master_port or _find_free_port()
        self.root_comm_port = _find_free_port()
        while self.root_comm_port == self.master_port:
            self.root_comm_port = _find_free_port()
        self.own_core_only = own_core_only

    def run(self, per_rank_fn: Callable[..., Any], **kwargs) -> Any:
        try:
            import torch.multiprocessing as mp
        except ImportError as exc:
            raise ProfilerNotImplemented("torch.multiprocessing is unavailable") from exc
        ctx = mp.get_context("spawn")
        queue = ctx.SimpleQueue()
        try:
            mp.spawn(
                _rank_entry,
                args=(
                    self.num_gpus,
                    self.cores,
                    self.master_port,
                    self.root_comm_port,
                    self.own_core_only,
                    per_rank_fn,
                    kwargs,
                    queue,
                ),
                nprocs=self.num_gpus,
                join=True,
            )
        except Exception as exc:
            # The failing rank's own error, when it reached the queue before
            # the spawn tore the group down.
            if not queue.empty():
                ok, payload = queue.get()
                if not ok:
                    raise KernelLaunchFailed(str(payload)) from exc
            raise KernelLaunchFailed(str(exc)) from exc
        if queue.empty():
            raise KernelLaunchFailed("Neuron rank group returned no result")
        ok, payload = queue.get()
        if not ok:
            raise KernelLaunchFailed(str(payload))
        return payload


def _rank_entry(
    rank,
    world_size,
    cores,
    master_port,
    root_comm_port,
    own_core_only,
    per_rank_fn,
    fn_kwargs,
    queue,
):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            # One logical core per rank, bound before torch loads the runtime;
            # the ranks of one group share the NRT root communicator.
            "NEURON_RT_VISIBLE_CORES": str(cores[rank]),
            "NEURON_RT_ROOT_COMM_ID": f"127.0.0.1:{root_comm_port}",
        }
    )
    if own_core_only:
        # With the chip span visible the runtime requires device index ==
        # rank; a rank that compiles on neuron:0 sees its own core alone.
        os.environ.pop("NEURON_VISIBLE_DEVICES", None)
        os.environ["LOCAL_RANK"] = "0"
    try:
        # The lite runtime registers the `neuron` device, compile backends and
        # the `neuron` process-group backend on import; a spawned rank starts
        # from a bare interpreter, so register them before the group forms.
        os.environ.setdefault("PJRT_DEVICE", "CPU")
        import torch.distributed as dist
        import vllm_neuron  # noqa: F401

        if "neuron" not in dist.Backend.backend_list:
            from libtorch_neuronx_lite._compiler.distributed.backend import (
                _register_neuron_backend,
            )

            _register_neuron_backend()
        dist.init_process_group(backend="neuron", rank=rank, world_size=world_size)
        try:
            result = per_rank_fn(rank=rank, world_size=world_size, **fn_kwargs)
        finally:
            dist.destroy_process_group()
    except Exception as exc:
        queue.put((False, f"rank {rank}: {type(exc).__name__}: {exc}"))
        raise
    if rank == 0:
        queue.put((True, result))


def profile_neuron_collective_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    """Profile a chunk of NeuronLink collectives with one rank-group spawn. The
    chunk shares ``num_ranks``; the op may vary per spec."""
    if not kwargs_list:
        return []
    num_ranks = int(kwargs_list[0]["num_ranks"])
    if num_ranks < 2:
        return all_error(len(kwargs_list), "a collective needs at least 2 ranks")
    if any(int(spec["num_ranks"]) != num_ranks for spec in kwargs_list):
        return all_error(len(kwargs_list), "a chunk must share num_ranks")
    bad = [spec["op"] for spec in kwargs_list if spec["op"] not in OPS]
    if bad:
        return all_error(len(kwargs_list), f"unknown Neuron collective {bad[0]!r}")
    try:
        cores = visible_logical_cores()
        launcher = NeuronMpLauncher(num_ranks, cores)
    except ValueError as error:
        return all_error(len(kwargs_list), str(error))
    results = run_comm_batch(launcher, _per_rank_batch, kwargs_list)
    note = (
        f"collective_bench host-timed mean of rep={_REP} after warmup={_WARMUP}; "
        f"cores={launcher.cores}"
    )
    for spec, result in zip(kwargs_list, results, strict=True):
        if result.error is None:
            _PROVENANCE[_key(spec)] = note
    return results


def _key(spec):
    return json.dumps(spec, sort_keys=True, default=str)


def row_provenance(**spec):
    return _PROVENANCE.get(_key(spec))


def _per_rank_batch(*, rank, world_size, specs, warmup, rep):
    """Inside each rank. Every size runs in the live group; a failure raises
    out of the whole group so the ranks never desync. Only rank 0 returns."""
    import torch
    import torch.distributed as dist

    device = torch.device("neuron")
    results = []
    for spec in specs:
        dtype = DType.from_value(spec["dtype"]).torch()
        element = torch.tensor([], dtype=dtype).element_size()
        op = spec["op"]
        # The full buffer, in whole per-rank shards for the sharded ops.
        shard_elements = max(1, int(spec["message_size_bytes"]) // element // world_size)
        full_elements = shard_elements * world_size
        actual_bytes = full_elements * element
        if op == "all_reduce":
            buf = torch.randn(full_elements, dtype=dtype, device=device)
            call = lambda: dist.all_reduce(buf)  # noqa: E731
        elif op == "all_gather":
            shard = torch.randn(shard_elements, dtype=dtype, device=device)
            out = torch.empty(full_elements, dtype=dtype, device=device)
            call = lambda: dist.all_gather_into_tensor(out, shard)  # noqa: E731
        else:
            full = torch.randn(full_elements, dtype=dtype, device=device)
            out = torch.empty(shard_elements, dtype=dtype, device=device)
            call = lambda: dist.reduce_scatter_tensor(out, full)  # noqa: E731
        for _ in range(warmup):
            call()
        # `dist.barrier()` asks torch for the current accelerator, which the
        # lite runtime does not register; a one-element all-reduce is the
        # fence every rank can run on the neuron device.
        fence = torch.ones(1, dtype=torch.float32, device=device)
        dist.all_reduce(fence)
        torch.neuron.synchronize()
        start = time.perf_counter()
        for _ in range(rep):
            call()
        torch.neuron.synchronize()
        time_ms = (time.perf_counter() - start) * 1e3 / rep
        algbw = actual_bytes / (time_ms * 1e6)  # GB/s
        results.append(
            {
                "time_ms": time_ms,
                "algbw_gbps": algbw,
                "busbw_gbps": algbw * bus_bandwidth_factor(op, world_size),
            }
        )
    return results if rank == 0 else None
