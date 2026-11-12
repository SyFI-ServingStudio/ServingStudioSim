"""Fixed-pattern MoE exchange over an NVLink-mapped workspace: Python side.

The kernels live in ``csrc/mnnvl_fixed_exchange.cuh`` (design, wire format and
round protocol are documented there); ``csrc/mnnvl_fixed_exchange.cu`` holds the
TVM-FFI launchers, JIT-built with FlashInfer's toolchain inside the vLLM image.

``FixedExchange`` owns one EP group's mapped workspace and exposes the legs:

* ``dispatch``        all-gather of the NVFP4 activations, block scales, top-k
                      weights and ids (one launch);
* ``quant_dispatch``  the same with the NVFP4 quantization of bf16 rows fused
                      into the push (one launch, replaces quant + dispatch);
* ``combine``         reduce-scatter of bf16 partial outputs (one launch);
* ``pull_combine``    the fused combine's consumer side: every rank's partials
                      sit in its own mapped region (``partials(set)``), and the
                      consumer reads its tokens' rows from all ranks and sums
                      them (one launch, nothing on the producer side).

Construction is collective over the launcher's process group (handle exchange).
Call ``close()`` collectively before the group is destroyed.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_JIT_MODULE_NAME = "vibesim_mnnvl_fixed_exchange_sm100a"
_CSRC = Path(__file__).resolve().parent / "csrc"
MAX_RANKS = 8
STATE_BYTES = 64
FLAGS_BYTES = 64
_ALIGN = 4096

# Which kernel `max_resident_blocks` reports on.
KERNEL_DISPATCH, KERNEL_QUANT_DISPATCH, KERNEL_COMBINE, KERNEL_PULL = 0, 1, 2, 3


# DLPack capsules of every mapped view, never freed (see FixedExchange._view).
_CAPSULES: list = []


@functools.cache
def load_module(timeline: bool = False):
    """JIT-build (once per JIT cache) and load the launchers. ``timeline`` builds the
    diagnostic variant (separate module) whose dispatch kernels record phase timestamps."""
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec, sm100a_nvcc_flags

    source = _CSRC / "mnnvl_fixed_exchange.cu"
    return gen_jit_spec(
        _JIT_MODULE_NAME + ("_timeline" if timeline else ""),
        [source],
        extra_cuda_cflags=sm100a_nvcc_flags
        + ["-lineinfo"]
        + (["-DFIXED_EXCHANGE_TIMELINE"] if timeline else []),
        extra_include_paths=[_CSRC, jit_env.FLASHINFER_CSRC_DIR],
    ).build_and_load()


def _round_up(value: int, align: int = _ALIGN) -> int:
    return (value + align - 1) // align * align


@dataclass(frozen=True)
class DispatchRow:
    """Byte widths of one dispatched row's four planes."""

    payload: int
    scales: int
    weights: int
    ids: int

    @classmethod
    def nvfp4(cls, hidden_size: int, top_k: int) -> DispatchRow:
        return cls(hidden_size // 2, hidden_size // 16, 4 * top_k, 4 * top_k)

    @property
    def total(self) -> int:
        return self.payload + self.scales + self.weights + self.ids


@dataclass(frozen=True)
class ExchangeLayout:
    state: int
    flags: int
    dispatch: int
    dispatch_set: int
    combine: int
    combine_set: int
    partials: int
    partials_set: int
    total: int

    @classmethod
    def build(cls, world: int, max_rows: int, row: DispatchRow, hidden_size: int) -> ExchangeLayout:
        dispatch_set = _round_up(world * max_rows * row.total)
        combine_set = _round_up(world * max_rows * hidden_size * 2)
        partials_set = combine_set
        state = 0
        flags = STATE_BYTES
        dispatch = _ALIGN
        combine = dispatch + 2 * dispatch_set
        partials = combine + 2 * combine_set
        total = partials + 2 * partials_set
        return cls(
            state,
            flags,
            dispatch,
            dispatch_set,
            combine,
            combine_set,
            partials,
            partials_set,
            total,
        )


class TorchDistCommBackend:
    """flashinfer `CommBackend` over a torch.distributed group (as vLLM's CustomCommunicator)."""

    def __init__(self, group):
        self._group = group

    def Get_rank(self) -> int:
        import torch.distributed as dist

        return dist.get_rank(group=self._group)

    def Get_size(self) -> int:
        import torch.distributed as dist

        return dist.get_world_size(group=self._group)

    def allgather(self, data: Any):
        import torch.distributed as dist

        gathered = [None] * self.Get_size()
        dist.all_gather_object(gathered, data, group=self._group)
        return gathered

    def bcast(self, data: Any, root: int) -> Any:
        import torch.distributed as dist

        box = [data]
        dist.broadcast_object_list(box, src=root, group=self._group)
        return box[0]

    def barrier(self) -> None:
        import torch.distributed as dist

        dist.barrier(group=self._group)

    def Split(self, color: int, key: int):
        return self


class FixedExchange:
    def __init__(
        self,
        rank: int,
        world: int,
        max_rows: int,
        hidden_size: int,
        top_k: int,
        group=None,
        timeline: bool = False,
    ):
        import torch
        import torch.distributed as dist
        from flashinfer.comm.mnnvl import SymmDeviceMemory

        if not 2 <= world <= MAX_RANKS:
            raise ValueError(f"world size {world} outside 2..{MAX_RANKS}")
        self.torch = torch
        self.rank, self.world, self.max_rows = rank, world, max_rows
        self.hidden_size, self.top_k = hidden_size, top_k
        self.row = DispatchRow.nvfp4(hidden_size, top_k)
        self.layout = ExchangeLayout.build(world, max_rows, self.row, hidden_size)
        self._group = group if group is not None else dist.group.WORLD
        self.module = load_module(timeline)
        device = torch.cuda.current_device()
        self.memory = SymmDeviceMemory(
            self.layout.total,
            world,
            rank,
            device,
            comm_backend_for_handle_transfer=TorchDistCommBackend(self._group),
            enable_multicast=False,
            allocate_signal_pads=False,
        )
        self.slices = [int(self.memory.get_unicast_ptr(r)) for r in range(world)]
        self.config = torch.tensor(self._config_words(), dtype=torch.int64)
        self.local = self._view(0, self.layout.total, torch.uint8)
        # Arm every landing word, clear the round state and the ready flags.
        self.local.fill_(0xFF)
        self.local[:_ALIGN].zero_()
        self._pull_round = 0
        self.counts = torch.empty(world, dtype=torch.int32, device="cuda")
        self.max_blocks = {k: int(self.module.max_resident_blocks(k)) for k in range(4)}
        torch.cuda.synchronize()
        dist.barrier(group=self._group)

    def _config_words(self) -> list[int]:
        lay = self.layout
        slices = self.slices + [0] * (MAX_RANKS - self.world)
        return [
            self.world,
            self.rank,
            self.max_rows,
            lay.state,
            lay.flags,
            lay.dispatch,
            lay.dispatch_set,
            lay.combine,
            lay.combine_set,
            lay.partials,
            lay.partials_set,
            *slices,
        ]

    def _view(self, offset: int, nbytes: int, dtype):
        """A tensor over this rank's mapped slice, ``[nbytes / itemsize]`` of ``dtype``.

        Wraps bytes and reinterprets: FlashInfer 0.7.3's DLPack helper labels bfloat16 as
        float16 (a bf16 view wrote fp16 bits: 100.0 landed as 0x5640). The capsule (whose
        ctypes deleter torch calls when the storage dies) is kept for the process lifetime:
        freed with the tensor's Python object it segfaulted at teardown (job 21205).
        """
        from flashinfer.comm.dlpack_utils import create_dlpack_capsule

        torch = self.torch
        wrapper = create_dlpack_capsule(
            self.slices[self.rank] + offset,
            nbytes,
            nbytes,
            1,
            torch.uint8,
            torch.cuda.current_device(),
        )
        _CAPSULES.append(wrapper)
        return torch.utils.dlpack.from_dlpack(wrapper.capsule).view(-1).view(dtype)

    def partials(self, set_index: int, total_rows: int):
        """This rank's mapped partials buffer of one set, [total_rows, hidden] bf16."""
        torch = self.torch
        if total_rows > self.world * self.max_rows:
            raise ValueError("more gathered rows than the workspace holds")
        offset = self.layout.partials + set_index * self.layout.partials_set
        nbytes = total_rows * self.hidden_size * 2
        return self._view(offset, nbytes, torch.bfloat16).view(total_rows, self.hidden_size)

    @property
    def next_pull_set(self) -> int:
        """The partials set the next ``pull_combine`` reads (host mirror of the round)."""
        return self._pull_round & 1

    def set_counts(self, counts: list[int]) -> None:
        if len(counts) != self.world or max(counts) > self.max_rows or min(counts) < 0:
            raise ValueError(f"per-rank rows {counts} outside 0..{self.max_rows}")
        self.counts.copy_(self.torch.tensor(counts, dtype=self.torch.int32))

    def _blocks(self, kernel: int, tasks: int, blocks: int | None) -> int:
        cap = self.max_blocks[kernel]
        if blocks is None:
            blocks = (tasks + 511) // 512
        return max(1, min(cap, blocks))

    def _stream(self) -> int:
        return self.torch.cuda.current_stream().cuda_stream

    def dispatch(
        self,
        payload,
        scales,
        weights,
        ids,
        out,
        *,
        blocks: int | None = None,
        tasks_hint: int | None = None,
    ) -> None:
        """``out`` = (payload, scales, weights, ids) gathered, [sum(counts), plane bytes]."""
        row = self.row
        tasks = (
            tasks_hint if tasks_hint is not None else self.world * self.max_rows * (row.total // 16)
        )
        self.module.fixed_dispatch(
            self.config,
            row.payload,
            row.scales,
            row.weights,
            row.ids,
            payload.data_ptr(),
            scales.data_ptr(),
            weights.data_ptr(),
            ids.data_ptr(),
            out[0].data_ptr(),
            out[1].data_ptr(),
            out[2].data_ptr(),
            out[3].data_ptr(),
            self.counts.data_ptr(),
            self._blocks(KERNEL_DISPATCH, tasks, blocks),
            self._stream(),
        )

    def quant_dispatch(
        self,
        hidden,
        global_scale,
        weights,
        ids,
        out,
        *,
        blocks: int | None = None,
        tasks_hint: int | None = None,
    ) -> None:
        row = self.row
        tasks = (
            tasks_hint if tasks_hint is not None else self.world * self.max_rows * (row.total // 16)
        )
        self.module.quant_dispatch(
            self.config,
            self.hidden_size,
            self.top_k,
            hidden.data_ptr(),
            global_scale.data_ptr(),
            weights.data_ptr(),
            ids.data_ptr(),
            out[0].data_ptr(),
            out[1].data_ptr(),
            out[2].data_ptr(),
            out[3].data_ptr(),
            self.counts.data_ptr(),
            self._blocks(KERNEL_QUANT_DISPATCH, tasks, blocks),
            self._stream(),
        )

    def combine(
        self, partials, out, *, blocks: int | None = None, tasks_hint: int | None = None
    ) -> None:
        tasks = (
            tasks_hint
            if tasks_hint is not None
            else self.world * self.max_rows * self.hidden_size // 8
        )
        self.module.fixed_combine(
            self.config,
            self.hidden_size,
            partials.data_ptr(),
            out.data_ptr(),
            self.counts.data_ptr(),
            self._blocks(KERNEL_COMBINE, tasks, blocks),
            self._stream(),
        )

    def pull_combine(
        self,
        out,
        *,
        partials_set: int | None = None,
        blocks: int | None = None,
        tasks_hint: int | None = None,
    ) -> None:
        """Sum this rank's rows of every rank's ``partials(partials_set)`` into ``out``.

        ``partials_set`` defaults to the host round parity (``next_pull_set``); a CUDA-graph
        caller passes the set its producer was captured writing (e.g. the layer parity)."""
        tasks = tasks_hint if tasks_hint is not None else self.max_rows * self.hidden_size // 8
        if partials_set is None:
            partials_set = self.next_pull_set
        self.module.pull_combine(
            self.config,
            self.hidden_size,
            out.data_ptr(),
            self.counts.data_ptr(),
            partials_set,
            self._blocks(KERNEL_PULL, tasks, blocks),
            self._stream(),
        )
        self._pull_round += 1

    def close(self) -> None:
        """Collective: unmap once every rank's work is done."""
        import gc

        import torch.distributed as dist

        self.torch.cuda.synchronize()
        dist.barrier(group=self._group)
        self.local = None
        self.memory = None
        gc.collect()
        dist.barrier(group=self._group)
