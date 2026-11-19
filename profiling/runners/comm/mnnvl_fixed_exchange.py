"""Fixed-pattern exchange over an NVLink-mapped workspace: Python side.

The kernels live in ``csrc/mnnvl_fixed_exchange.cuh`` (design, wire format and
round protocol are documented there); ``csrc/mnnvl_fixed_exchange.cu`` holds the
TVM-FFI launchers, JIT-built with FlashInfer's toolchain inside the vLLM image.

``FixedExchange`` owns one group's mapped workspace and exposes the legs, one
kernel launch each:

* ``dispatch``        all-gather of the NVFP4 activations, block scales, top-k
                      weights and ids;
* ``quant_dispatch``  the same with the NVFP4 quantization of bf16 rows fused
                      into the push;
* ``combine``         reduce-scatter of bf16 partial outputs, pushed to their
                      owners;
* ``pull_combine``    reduce-scatter read by the owner: every rank's partials
                      sit in its own mapped region (``partials(set)``), and the
                      owner reads its tokens' rows from all ranks and sums them;
* ``all_gather``      all-gather of plain bf16 rows (``DispatchRow.bf16``);
* ``pull_gather``     all-gather read by the receiver from every rank's
                      ``partials(set)``.

Construction is collective over the launcher's process group (handle exchange).
Call ``close()`` collectively before the group is destroyed.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

from profiling.runners.moe.flashinfer_mnnvl_alltoall import TorchDistCommBackend

_JIT_MODULE_NAME = "vibesim_mnnvl_fixed_exchange_sm100a"
_CSRC = Path(__file__).resolve().parent / "csrc"
MAX_RANKS = 8
STATE_BYTES = 64
# Round markers (Lamport, armed with the landing): kMarkerOffset in the .cuh; flags lie below.
MARKER_OFFSET = 1024
_ALIGN = 4096

# Which kernel `max_resident_blocks` reports on.
KERNEL_DISPATCH, KERNEL_QUANT_DISPATCH, KERNEL_COMBINE, KERNEL_PULL, KERNEL_PULL_GATHER = range(5)
_THREADS_PER_BLOCK = 512

# DLPack capsules of every mapped view, never freed (see FixedExchange._view).
_CAPSULES: list = []


@functools.cache
def load_module():
    """JIT-build (once per JIT cache) and load the launchers."""
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec, sm100a_nvcc_flags

    return gen_jit_spec(
        _JIT_MODULE_NAME,
        [_CSRC / "mnnvl_fixed_exchange.cu"],
        extra_cuda_cflags=sm100a_nvcc_flags + ["-lineinfo"],
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

    @classmethod
    def bf16(cls, hidden_size: int) -> DispatchRow:
        """Plain bf16 rows for the row all-gather: one plane, sent with the scales plane's
        lossless wire rule (a 0xFFFFFFFF word, two NaN lanes, becomes 0x7FFFFFFF)."""
        return cls(0, hidden_size * 2, 0, 0)

    @property
    def total(self) -> int:
        return self.payload + self.scales + self.weights + self.ids


@dataclass(frozen=True)
class ExchangeLayout:
    """Byte offsets of one rank's mapped slice (the .cuh documents each region)."""

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
        dispatch = _ALIGN
        combine = dispatch + 2 * dispatch_set
        partials = combine + 2 * combine_set
        return cls(
            state=0,
            flags=STATE_BYTES,
            dispatch=dispatch,
            dispatch_set=dispatch_set,
            combine=combine,
            combine_set=combine_set,
            partials=partials,
            partials_set=partials_set,
            total=partials + 2 * partials_set,
        )


class FixedExchange:
    def __init__(
        self,
        rank: int,
        world: int,
        max_rows: int,
        hidden_size: int,
        top_k: int,
        row: DispatchRow | None = None,
    ):
        import torch
        import torch.distributed as dist
        from flashinfer.comm.mnnvl import SymmDeviceMemory

        if not 2 <= world <= MAX_RANKS:
            raise ValueError(f"world size {world} outside 2..{MAX_RANKS}")
        self.torch = torch
        self.rank, self.world, self.max_rows = rank, world, max_rows
        self.hidden_size, self.top_k = hidden_size, top_k
        self.row = row if row is not None else DispatchRow.nvfp4(hidden_size, top_k)
        self.layout = ExchangeLayout.build(world, max_rows, self.row, hidden_size)
        self.module = load_module()
        self.memory = SymmDeviceMemory(
            self.layout.total,
            world,
            rank,
            torch.cuda.current_device(),
            comm_backend_for_handle_transfer=TorchDistCommBackend(dist.group.WORLD),
            enable_multicast=False,
            allocate_signal_pads=False,
        )
        self.slices = [int(self.memory.get_unicast_ptr(r)) for r in range(world)]
        self.config = torch.tensor(self._config_words(), dtype=torch.int64)
        self.local = self._view(0, self.layout.total, torch.uint8)
        # Arm every landing word and round marker, clear the round state and the ready flags.
        self.local.fill_(0xFF)
        self.local[:MARKER_OFFSET].zero_()
        # Host mirror of the pull round: its parity is the partials set the next pull reads.
        self._pull_round = 0
        self.counts = torch.empty(world, dtype=torch.int32, device="cuda")
        self.max_blocks = {k: int(self.module.max_resident_blocks(k)) for k in range(5)}
        torch.cuda.synchronize()
        dist.barrier()

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
        float16, so a bf16 view would write fp16 bits. The capsule (whose ctypes deleter
        torch calls when the storage dies) is kept for the process lifetime: freeing it
        with the tensor's Python object segfaults at teardown.
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
        if total_rows > self.world * self.max_rows:
            raise ValueError("more gathered rows than the workspace holds")
        offset = self.layout.partials + set_index * self.layout.partials_set
        nbytes = total_rows * self.hidden_size * 2
        return self._view(offset, nbytes, self.torch.bfloat16).view(total_rows, self.hidden_size)

    def set_counts(self, counts: list[int]) -> None:
        if len(counts) != self.world or max(counts) > self.max_rows or min(counts) < 0:
            raise ValueError(f"per-rank rows {counts} outside 0..{self.max_rows}")
        self.counts.copy_(self.torch.tensor(counts, dtype=self.torch.int32))

    def _blocks(self, kernel: int, tasks: int) -> int:
        """One block per ``_THREADS_PER_BLOCK`` tasks, within what the GPU keeps resident."""
        blocks = (tasks + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        return max(1, min(self.max_blocks[kernel], blocks))

    def _stream(self) -> int:
        return self.torch.cuda.current_stream().cuda_stream

    def dispatch(self, payload, scales, weights, ids, out) -> None:
        """``out`` = (payload, scales, weights, ids) gathered, [sum(counts), plane bytes]."""
        row = self.row
        tasks = self.world * self.max_rows * (row.total // 16)
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
            self._blocks(KERNEL_DISPATCH, tasks),
            self._stream(),
        )

    def quant_dispatch(self, hidden, global_scale, weights, ids, out) -> None:
        """``dispatch`` from bf16 ``hidden`` rows, NVFP4-quantized inside the push."""
        tasks = self.world * self.max_rows * (self.row.total // 16)
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
            self._blocks(KERNEL_QUANT_DISPATCH, tasks),
            self._stream(),
        )

    def combine(self, partials, out) -> None:
        """Push this rank's ``partials`` [sum(counts), hidden] to their owners; ``out``
        [counts[rank], hidden] is the fp32 sum of every rank's partials, in bf16."""
        tasks = self.world * self.max_rows * self.hidden_size // 8
        self.module.fixed_combine(
            self.config,
            self.hidden_size,
            partials.data_ptr(),
            out.data_ptr(),
            self.counts.data_ptr(),
            self._blocks(KERNEL_COMBINE, tasks),
            self._stream(),
        )

    def pull_combine(self, out) -> None:
        """Sum this rank's rows of every rank's ``partials(set)`` into ``out``; the set
        is the pull round's parity."""
        tasks = self.max_rows * self.hidden_size // 8
        self.module.pull_combine(
            self.config,
            self.hidden_size,
            out.data_ptr(),
            self.counts.data_ptr(),
            self._pull_round & 1,
            self._blocks(KERNEL_PULL, tasks),
            self._stream(),
        )
        self._pull_round += 1

    def all_gather(self, rows, out) -> None:
        """Push all-gather of bf16 rows (``row=DispatchRow.bf16(hidden)``): this rank's
        ``rows`` [counts[rank], hidden] into every rank's ``out`` [sum(counts), hidden]."""
        row = self.row
        if row.payload or row.weights or row.ids:
            raise ValueError("all_gather needs the bf16 row layout (DispatchRow.bf16)")
        tasks = self.world * self.max_rows * (row.total // 16)
        # The dispatch kernel with only the scales plane populated.
        self.module.fixed_dispatch(
            self.config,
            0,
            row.scales,
            0,
            0,
            0,
            rows.data_ptr(),
            0,
            0,
            0,
            out.data_ptr(),
            0,
            0,
            self.counts.data_ptr(),
            self._blocks(KERNEL_DISPATCH, tasks),
            self._stream(),
        )

    def pull_gather(self, out) -> None:
        """Pull all-gather: every rank's rows sit in its own ``partials(set)`` at their
        gathered positions; copy all of them into ``out`` [sum(counts), hidden]. Shares the
        pull round (ready flags) with ``pull_combine``."""
        tasks = self.world * self.max_rows * self.hidden_size // 8
        self.module.pull_gather(
            self.config,
            self.hidden_size,
            out.data_ptr(),
            self.counts.data_ptr(),
            self._pull_round & 1,
            self._blocks(KERNEL_PULL_GATHER, tasks),
            self._stream(),
        )
        self._pull_round += 1

    def close(self) -> None:
        """Collective: unmap once every rank's work is done."""
        import gc

        import torch.distributed as dist

        self.torch.cuda.synchronize()
        dist.barrier()
        self.local = None
        self.memory = None
        gc.collect()
        dist.barrier()
