"""Torch semantic-composite runner for Qwen GDN chunk-local cumsum.

This backend times the multiple ``torch.cumsum`` launches needed to implement
the frozen chunk-local equation. It is not vLLM's fused Triton launch, and its
logical traffic/FLOP rates do not describe a fused kernel's physical
implementation. Production simulation must select the fused backend after one
is registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import (
    canonical_balanced_chunk_boundaries as _canonical_boundaries,
)
from profiling.runners.attention._gdn_common import (
    canonical_balanced_chunk_lengths as _canonical_lengths,  # noqa: F401
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_CHUNK_SIZE = 64


@dataclass(frozen=True)
class _ValidatedArgs:
    num_tokens: int
    num_chunks: int
    num_heads: int
    dtype: DType


@dataclass(frozen=True)
class _OperandShapes:
    g: tuple[int, int]
    cu_seqlens: tuple[int]
    output: tuple[int, int]


@dataclass(frozen=True)
class _Operands:
    g: Any
    cu_seqlens: Any
    output: Any
    boundaries: tuple[int, ...]


def _validate_args(
    num_tokens: int,
    num_chunks: int,
    num_heads: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    num_tokens = int(num_tokens)
    num_chunks = int(num_chunks)
    num_heads = int(num_heads)
    dtype = DType.from_value(dtype)

    if num_tokens <= 0 or num_chunks <= 0 or num_heads <= 0:
        raise ValueError(
            "num_tokens, num_chunks, and num_heads must be > 0, got "
            f"num_tokens={num_tokens}, num_chunks={num_chunks}, num_heads={num_heads}"
        )
    minimum_chunks = (num_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    if num_chunks < minimum_chunks or num_chunks > num_tokens:
        raise ValueError(
            "num_chunks must satisfy ceil(num_tokens/64) <= num_chunks <= num_tokens, "
            f"got num_tokens={num_tokens}, num_chunks={num_chunks}"
        )
    if dtype is not DType.FP32:
        raise ValueError(f"torch gdn_chunk_local_cumsum requires dtype=fp32, got {dtype.value}")
    return _ValidatedArgs(
        num_tokens=num_tokens,
        num_chunks=num_chunks,
        num_heads=num_heads,
        dtype=dtype,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch gdn_chunk_local_cumsum backend"
        )


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    return _OperandShapes(
        g=(args.num_tokens, args.num_heads),
        cu_seqlens=(args.num_chunks + 1,),
        output=(args.num_tokens, args.num_heads),
    )


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    boundaries = _canonical_boundaries(args.num_tokens, args.num_chunks)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    g = torch.empty(shapes.g, dtype=torch.float32, device=device).uniform_(
        -0.125,
        0.0,
        generator=generator,
    )
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    output = torch.empty(shapes.output, dtype=torch.float32, device=device)
    return _Operands(
        g=g,
        cu_seqlens=cu_seqlens,
        output=output,
        boundaries=boundaries,
    )


def _chunk_local_cumsum_into(
    torch: Any,
    g: Any,
    output: Any,
    boundaries: tuple[int, ...],
) -> Any:
    """Execute the reference equation into preallocated device storage.

    Boundaries are prepared before measurement. Each independent cumsum writes
    its complete destination slice, so repeated calls need no output reset.
    """
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            torch.cumsum(
                g[chunk_start:chunk_end],
                dim=0,
                dtype=torch.float32,
                out=output[chunk_start:chunk_end],
            )
    return output


def _semantic_flops(*, num_tokens: int, num_chunks: int, num_heads: int) -> int:
    """Semantic FP32 additions, not physical Torch instruction count."""
    return num_heads * (num_tokens - num_chunks)


def _logical_bytes(*, num_tokens: int, num_heads: int) -> float:
    """Logical input/output traffic, excluding metadata and intermediates."""
    return 8 * num_tokens * num_heads


def profile_gdn_chunk_local_cumsum(
    num_tokens: int,
    num_chunks: int,
    num_heads: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile the preallocated multi-launch Torch semantic implementation."""
    args = _validate_args(
        num_tokens=num_tokens,
        num_chunks=num_chunks,
        num_heads=num_heads,
        dtype=dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch gdn_chunk_local_cumsum backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(torch, args, device=torch.device("cuda"))

        def kernel():
            return _chunk_local_cumsum_into(
                torch,
                operands.g,
                operands.output,
                operands.boundaries,
            )

        # No kernel-name filter: measure the complete multi-launch semantic
        # composite. Allocation, partitioning, and metadata construction are
        # outside timing. Every cumsum overwrites its slice, so no reset is used.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        flops = _semantic_flops(
            num_tokens=args.num_tokens,
            num_chunks=args.num_chunks,
            num_heads=args.num_heads,
        )
        logical_bytes = _logical_bytes(
            num_tokens=args.num_tokens,
            num_heads=args.num_heads,
        )
        elapsed_s = time_ms / 1000.0
        tflops = flops / elapsed_s / 1e12 if time_ms > 0.0 else 0.0
        bandwidth_gbps = logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
