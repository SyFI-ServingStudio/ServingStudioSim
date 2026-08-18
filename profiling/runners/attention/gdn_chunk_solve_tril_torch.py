"""Torch semantic-composite runner for Qwen GDN triangular inversion.

This backend times the multiple Torch launches needed to implement the frozen
chunk-local ``(I + A)^-1`` recurrence. It is not vLLM's fused Triton launch,
and its logical traffic/FLOP rates do not describe a fused kernel's physical
implementation. Production simulation must select the fused backend after one
is registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import exact_int as _exact_int
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_CHUNK_SIZE = 64


@dataclass(frozen=True)
class _ValidatedArgs:
    num_tokens: int
    num_chunks: int
    max_chunk_tokens: int
    num_heads: int
    dtype: DType


@dataclass(frozen=True)
class _OperandShapes:
    A: tuple[int, int, int]
    cu_seqlens: tuple[int]
    output: tuple[int, int, int]
    square: tuple[int, int, int]
    row: tuple[int, int]
    row_product: tuple[int, int, int]
    identity: tuple[int, int]


@dataclass(frozen=True)
class _Workspaces:
    strict_inverse: Any
    inverse: Any
    negative_input_row: Any
    solved_row: Any
    row_product: Any
    identity: Any


@dataclass(frozen=True)
class _Operands:
    A: Any
    cu_seqlens: Any
    output: Any
    boundaries: tuple[int, ...]
    workspaces: _Workspaces


def _validate_args(
    num_tokens: int,
    num_chunks: int,
    max_chunk_tokens: int,
    num_heads: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    num_tokens = _exact_int("num_tokens", num_tokens)
    num_chunks = _exact_int("num_chunks", num_chunks)
    max_chunk_tokens = _exact_int("max_chunk_tokens", max_chunk_tokens)
    num_heads = _exact_int("num_heads", num_heads)
    dtype = DType.from_value(dtype)

    dimensions = (num_tokens, num_chunks, max_chunk_tokens, num_heads)
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            f"num_tokens, num_chunks, max_chunk_tokens, and num_heads must be > 0, got {dimensions}"
        )
    if max_chunk_tokens > _CHUNK_SIZE:
        raise ValueError(f"max_chunk_tokens must be <= {_CHUNK_SIZE}, got {max_chunk_tokens}")
    minimum_tokens = max_chunk_tokens + num_chunks - 1
    maximum_tokens = num_chunks * max_chunk_tokens
    if num_tokens < minimum_tokens or num_tokens > maximum_tokens:
        raise ValueError(
            "feasible (T,C,M) requires M+C-1 <= T <= C*M, got "
            f"T={num_tokens}, C={num_chunks}, M={max_chunk_tokens}"
        )
    if dtype is not DType.BF16:
        raise ValueError(f"torch gdn_chunk_solve_tril requires dtype=bf16, got {dtype.value}")
    return _ValidatedArgs(
        num_tokens=num_tokens,
        num_chunks=num_chunks,
        max_chunk_tokens=max_chunk_tokens,
        num_heads=num_heads,
        dtype=dtype,
    )


def _canonical_lengths(
    num_tokens: int,
    num_chunks: int,
    max_chunk_tokens: int,
) -> tuple[int, ...]:
    """Construct the deterministic one-chunk-per-sequence partition."""
    if num_chunks == 1:
        lengths = (max_chunk_tokens,)
    else:
        remainder_tokens = num_tokens - (max_chunk_tokens + num_chunks - 1)
        quotient, remainder = divmod(remainder_tokens, num_chunks - 1)
        lengths = (
            (max_chunk_tokens,)
            + (quotient + 2,) * remainder
            + (quotient + 1,) * (num_chunks - 1 - remainder)
        )
    if (
        len(lengths) != num_chunks
        or sum(lengths) != num_tokens
        or min(lengths) < 1
        or max(lengths) != max_chunk_tokens
        or max_chunk_tokens > _CHUNK_SIZE
    ):
        raise ValueError("canonical partition requires M+C-1 <= T <= C*M and M <= 64")
    return lengths


def _canonical_boundaries(
    num_tokens: int,
    num_chunks: int,
    max_chunk_tokens: int,
) -> tuple[int, ...]:
    return (
        0,
        *accumulate(_canonical_lengths(num_tokens, num_chunks, max_chunk_tokens)),
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch gdn_chunk_solve_tril backend")


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    max_length = args.max_chunk_tokens
    return _OperandShapes(
        A=(args.num_tokens, args.num_heads, _CHUNK_SIZE),
        cu_seqlens=(args.num_chunks + 1,),
        output=(args.num_tokens, args.num_heads, _CHUNK_SIZE),
        square=(args.num_heads, max_length, max_length),
        row=(args.num_heads, max_length),
        row_product=(args.num_heads, 1, max_length),
        identity=(max_length, max_length),
    )


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    lengths = _canonical_lengths(
        args.num_tokens,
        args.num_chunks,
        args.max_chunk_tokens,
    )
    boundaries = (0, *accumulate(lengths))
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    A = torch.zeros(shapes.A, dtype=torch.float32, device=device)

    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        length = sequence_end - sequence_start
        values = torch.empty(
            (length, args.num_heads, length),
            dtype=torch.float32,
            device=device,
        ).uniform_(-0.02, 0.02, generator=generator)
        lower = torch.tril(values.permute(1, 0, 2), diagonal=-1).permute(1, 0, 2)
        A[sequence_start:sequence_end, :, :length].copy_(lower)

    return _Operands(
        A=A,
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device=device),
        output=torch.empty(shapes.output, dtype=torch.bfloat16, device=device),
        boundaries=boundaries,
        workspaces=_Workspaces(
            strict_inverse=torch.empty(shapes.square, dtype=torch.float32, device=device),
            inverse=torch.empty(shapes.square, dtype=torch.float32, device=device),
            negative_input_row=torch.empty(shapes.row, dtype=torch.float32, device=device),
            solved_row=torch.empty(shapes.row, dtype=torch.float32, device=device),
            row_product=torch.empty(shapes.row_product, dtype=torch.float32, device=device),
            identity=torch.eye(
                args.max_chunk_tokens,
                dtype=torch.float32,
                device=device,
            ),
        ),
    )


def _solve_tril_into(
    torch: Any,
    A: Any,
    output: Any,
    boundaries: tuple[int, ...],
    workspaces: _Workspaces,
) -> Any:
    """Execute the FP32 row recurrence into preallocated BF16 output.

    The canonical partition makes every sequence exactly one local chunk.
    Output and every mutable workspace region are deterministically overwritten,
    so repeated calls require no external reset. Allocation and metadata
    construction remain outside the measured callable.
    """
    output.zero_()
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        length = sequence_end - sequence_start
        block = A[sequence_start:sequence_end, :, :length].permute(1, 0, 2)
        strict_inverse = workspaces.strict_inverse[:, :length, :length]
        strict_inverse.zero_()

        for row in range(1, length):
            negative_input_row = workspaces.negative_input_row[:, :row]
            solved_row = workspaces.solved_row[:, :row]
            torch.neg(block[:, row, :row], out=negative_input_row)
            solved_row.copy_(negative_input_row)
            if row > 1:
                product = workspaces.row_product[:, :, :row]
                torch.bmm(
                    negative_input_row.unsqueeze(1),
                    strict_inverse[:, :row, :row],
                    out=product,
                )
                solved_row.add_(product.squeeze(1))
            strict_inverse[:, row, :row].copy_(solved_row)

        inverse = workspaces.inverse[:, :length, :length]
        torch.add(
            strict_inverse,
            workspaces.identity[:length, :length],
            out=inverse,
        )
        output[sequence_start:sequence_end, :, :length].copy_(inverse.permute(1, 0, 2))
    return output


def _semantic_flops(
    *,
    num_tokens: int,
    num_chunks: int,
    max_chunk_tokens: int,
    num_heads: int,
) -> int:
    """Semantic recurrence FLOPs, not physical Torch instruction count."""
    lengths = _canonical_lengths(num_tokens, num_chunks, max_chunk_tokens)
    per_head = sum(length * (length - 1) * (2 * length - 1) // 6 for length in lengths)
    return num_heads * per_head


def _logical_bytes(*, num_tokens: int, num_heads: int) -> int:
    """Logical FP32-A read plus BF16-output write boundary traffic."""
    return 6 * num_tokens * num_heads * _CHUNK_SIZE


def profile_gdn_chunk_solve_tril(
    num_tokens: int,
    num_chunks: int,
    max_chunk_tokens: int,
    num_heads: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile the preallocated multi-launch Torch semantic implementation."""
    args = _validate_args(
        num_tokens=num_tokens,
        num_chunks=num_chunks,
        max_chunk_tokens=max_chunk_tokens,
        num_heads=num_heads,
        dtype=dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch gdn_chunk_solve_tril backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(torch, args, device=torch.device("cuda"))

        def kernel():
            return _solve_tril_into(
                torch,
                operands.A,
                operands.output,
                operands.boundaries,
                operands.workspaces,
            )

        # No kernel-name filter: measure the complete multi-launch semantic
        # composite. Allocation, partitioning, and metadata construction are
        # outside timing. The helper overwrites all mutable state, so no reset
        # is used.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        flops = _semantic_flops(
            num_tokens=args.num_tokens,
            num_chunks=args.num_chunks,
            max_chunk_tokens=args.max_chunk_tokens,
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
