"""Torch semantic-composite runner for Qwen GDN scaled-dot KKT.

This backend times the multiple Torch launches needed to implement the frozen
positive, strict-lower chunk-local equation. It is not vLLM's fused Triton
launch, and its logical traffic/FLOP rates do not describe a fused kernel's
physical implementation. Production simulation must select the fused backend
after one is registered.
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
    canonical_balanced_chunk_lengths as _canonical_lengths,
)
from profiling.runners.attention._gdn_common import (
    exact_int as _exact_int,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_CHUNK_SIZE = 64


@dataclass(frozen=True)
class _ValidatedArgs:
    num_tokens: int
    num_chunks: int
    num_key_heads: int
    num_heads: int
    key_head_dim: int
    dtype: DType


@dataclass(frozen=True)
class _OperandShapes:
    k: tuple[int, int, int]
    beta: tuple[int, int]
    g_cumsum: tuple[int, int]
    cu_seqlens: tuple[int]
    output: tuple[int, int, int]
    key_fp32: tuple[int, int, int]
    expanded_key: tuple[int, int, int]
    scaled_key: tuple[int, int, int]
    square: tuple[int, int, int]
    head_to_key: tuple[int]
    strict_upper_with_diagonal: tuple[int, int]


@dataclass(frozen=True)
class _Workspaces:
    key_fp32: Any
    expanded_key: Any
    scaled_key: Any
    dots: Any
    decay: Any
    head_to_key: Any
    strict_upper_with_diagonal: Any


@dataclass(frozen=True)
class _Operands:
    k: Any
    beta: Any
    g_cumsum: Any
    cu_seqlens: Any
    output: Any
    boundaries: tuple[int, ...]
    workspaces: _Workspaces


def _validate_args(
    num_tokens: int,
    num_chunks: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    num_tokens = _exact_int("num_tokens", num_tokens)
    num_chunks = _exact_int("num_chunks", num_chunks)
    num_key_heads = _exact_int("num_key_heads", num_key_heads)
    num_heads = _exact_int("num_heads", num_heads)
    key_head_dim = _exact_int("key_head_dim", key_head_dim)
    dtype = DType.from_value(dtype)

    dimensions = (num_tokens, num_chunks, num_key_heads, num_heads, key_head_dim)
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            "num_tokens, num_chunks, num_key_heads, num_heads, and key_head_dim "
            f"must be > 0, got {dimensions}"
        )
    minimum_chunks = (num_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    if num_chunks < minimum_chunks or num_chunks > num_tokens:
        raise ValueError(
            "num_chunks must satisfy ceil(num_tokens/64) <= num_chunks <= num_tokens, "
            f"got num_tokens={num_tokens}, num_chunks={num_chunks}"
        )
    if num_heads % num_key_heads != 0:
        raise ValueError(
            "num_heads must be divisible by num_key_heads, got "
            f"num_heads={num_heads}, num_key_heads={num_key_heads}"
        )
    if dtype is not DType.BF16:
        raise ValueError(f"torch gdn_chunk_scaled_dot_kkt requires dtype=bf16, got {dtype.value}")
    return _ValidatedArgs(
        num_tokens=num_tokens,
        num_chunks=num_chunks,
        num_key_heads=num_key_heads,
        num_heads=num_heads,
        key_head_dim=key_head_dim,
        dtype=dtype,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch gdn_chunk_scaled_dot_kkt backend"
        )


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    max_length = max(_canonical_lengths(args.num_tokens, args.num_chunks))
    return _OperandShapes(
        k=(args.num_tokens, args.num_key_heads, args.key_head_dim),
        beta=(args.num_tokens, args.num_heads),
        g_cumsum=(args.num_tokens, args.num_heads),
        cu_seqlens=(args.num_chunks + 1,),
        output=(args.num_tokens, args.num_heads, _CHUNK_SIZE),
        key_fp32=(args.num_tokens, args.num_key_heads, args.key_head_dim),
        expanded_key=(max_length, args.num_heads, args.key_head_dim),
        scaled_key=(max_length, args.num_heads, args.key_head_dim),
        square=(args.num_heads, max_length, max_length),
        head_to_key=(args.num_heads,),
        strict_upper_with_diagonal=(max_length, max_length),
    )


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    boundaries = _canonical_boundaries(args.num_tokens, args.num_chunks)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    def bounded(shape: tuple[int, ...], low: float, high: float, *, dtype: Any):
        return torch.empty(shape, dtype=dtype, device=device).uniform_(
            low,
            high,
            generator=generator,
        )

    heads_per_key = args.num_heads // args.num_key_heads
    head_to_key = torch.arange(args.num_heads, dtype=torch.int64, device=device).div(
        heads_per_key,
        rounding_mode="floor",
    )
    max_length = shapes.expanded_key[0]
    strict_upper_with_diagonal = torch.triu(
        torch.ones((max_length, max_length), dtype=torch.bool, device=device)
    )
    return _Operands(
        k=bounded(shapes.k, -0.25, 0.25, dtype=torch.bfloat16),
        beta=bounded(shapes.beta, 0.125, 0.875, dtype=torch.float32),
        g_cumsum=bounded(shapes.g_cumsum, -0.25, 0.25, dtype=torch.float32),
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device=device),
        output=torch.empty(shapes.output, dtype=torch.float32, device=device),
        boundaries=boundaries,
        workspaces=_Workspaces(
            key_fp32=torch.empty(shapes.key_fp32, dtype=torch.float32, device=device),
            expanded_key=torch.empty(shapes.expanded_key, dtype=torch.float32, device=device),
            scaled_key=torch.empty(shapes.scaled_key, dtype=torch.float32, device=device),
            dots=torch.empty(shapes.square, dtype=torch.float32, device=device),
            decay=torch.empty(shapes.square, dtype=torch.float32, device=device),
            head_to_key=head_to_key,
            strict_upper_with_diagonal=strict_upper_with_diagonal,
        ),
    )


def _scaled_dot_kkt_into(
    torch: Any,
    k: Any,
    beta: Any,
    g_cumsum: Any,
    output: Any,
    boundaries: tuple[int, ...],
    workspaces: _Workspaces,
) -> Any:
    """Execute the positive strict-lower equation into preallocated storage.

    The canonical partition makes every sequence one local chunk of at most 64
    tokens. Every output element is overwritten, so repeated calls need no
    external reset. All allocation and metadata construction remain outside the
    measured callable.
    """
    output.zero_()
    workspaces.key_fp32.copy_(k)
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        length = sequence_end - sequence_start
        expanded = workspaces.expanded_key[:length]
        torch.index_select(
            workspaces.key_fp32[sequence_start:sequence_end],
            1,
            workspaces.head_to_key,
            out=expanded,
        )
        expanded_by_head = expanded.permute(1, 0, 2)
        scaled_by_head = workspaces.scaled_key[:length].permute(1, 0, 2)
        torch.mul(
            expanded_by_head,
            beta[sequence_start:sequence_end].transpose(0, 1).unsqueeze(-1),
            out=scaled_by_head,
        )
        dots = workspaces.dots[:, :length, :length]
        torch.bmm(scaled_by_head, expanded_by_head.transpose(-1, -2), out=dots)

        gate = g_cumsum[sequence_start:sequence_end].transpose(0, 1)
        decay = workspaces.decay[:, :length, :length]
        torch.sub(gate.unsqueeze(-1), gate.unsqueeze(-2), out=decay)
        torch.exp(decay, out=decay)
        dots.mul_(decay)
        dots.masked_fill_(workspaces.strict_upper_with_diagonal[:length, :length], 0.0)
        output[sequence_start:sequence_end, :, :length].copy_(dots.permute(1, 0, 2))
    return output


def _semantic_flops(
    *,
    num_tokens: int,
    num_chunks: int,
    num_heads: int,
    key_head_dim: int,
) -> int:
    """Nominal semantic FLOPs, not physical Torch instruction count.

    The first term counts row-owned beta scaling of expanded K. Each strict
    pair counts a ``2*K-1`` dot, gate subtraction, nominal exp, and decay
    multiply. Transcendentals count as one nominal unit.
    """
    pairs = sum(length * (length - 1) // 2 for length in _canonical_lengths(num_tokens, num_chunks))
    return num_heads * num_tokens * key_head_dim + num_heads * pairs * (2 * key_head_dim + 2)


def _logical_bytes(
    *,
    num_tokens: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
) -> float:
    """Logical boundary traffic, excluding metadata and implementation traffic."""
    k_bytes = 2 * num_tokens * num_key_heads * key_head_dim
    gate_bytes = 4 * num_tokens * num_heads * 2
    output_bytes = 4 * num_tokens * num_heads * _CHUNK_SIZE
    return k_bytes + gate_bytes + output_bytes


def profile_gdn_chunk_scaled_dot_kkt(
    num_tokens: int,
    num_chunks: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile the preallocated multi-launch Torch semantic implementation."""
    args = _validate_args(
        num_tokens=num_tokens,
        num_chunks=num_chunks,
        num_key_heads=num_key_heads,
        num_heads=num_heads,
        key_head_dim=key_head_dim,
        dtype=dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch gdn_chunk_scaled_dot_kkt backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(torch, args, device=torch.device("cuda"))

        def kernel():
            return _scaled_dot_kkt_into(
                torch,
                operands.k,
                operands.beta,
                operands.g_cumsum,
                operands.output,
                operands.boundaries,
                operands.workspaces,
            )

        # No kernel-name filter: measure the complete multi-launch semantic
        # composite. Allocation, partitioning, and metadata construction are
        # outside timing. The helper overwrites output, so no reset is used.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        flops = _semantic_flops(
            num_tokens=args.num_tokens,
            num_chunks=args.num_chunks,
            num_heads=args.num_heads,
            key_head_dim=args.key_head_dim,
        )
        logical_bytes = _logical_bytes(
            num_tokens=args.num_tokens,
            num_key_heads=args.num_key_heads,
            num_heads=args.num_heads,
            key_head_dim=args.key_head_dim,
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
