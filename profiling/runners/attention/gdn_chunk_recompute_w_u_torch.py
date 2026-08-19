"""Torch semantic-composite runner for Qwen GDN WY recomputation.

This backend times the multiple Torch launches needed to implement the frozen
chunk-local W/U equations. It is not vLLM's fused Triton launch, and its
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
    value_head_dim: int
    dtype: DType


@dataclass(frozen=True)
class _OperandShapes:
    k: tuple[int, int, int]
    v: tuple[int, int, int]
    beta: tuple[int, int]
    g_cumsum: tuple[int, int]
    A: tuple[int, int, int]
    cu_seqlens: tuple[int]
    w: tuple[int, int, int]
    u: tuple[int, int, int]
    solved_fp32: tuple[int, int, int]
    k_fp32: tuple[int, int, int]
    grouped_k_fp32: tuple[int, int, int]
    k_factor: tuple[int, int, int]
    v_factor: tuple[int, int, int]
    k_result_fp32: tuple[int, int, int]
    v_result_fp32: tuple[int, int, int]
    gate_scale: tuple[int, int]
    head_to_key: tuple[int]


@dataclass(frozen=True)
class _Workspaces:
    solved_fp32: Any
    k_fp32: Any
    grouped_k_fp32: Any
    k_factor_bf16: Any
    k_factor_fp32: Any
    v_factor_bf16: Any
    v_factor_fp32: Any
    k_result_fp32: Any
    v_result_fp32: Any
    gate_scale: Any
    head_to_key: Any


@dataclass(frozen=True)
class _Operands:
    k: Any
    v: Any
    beta: Any
    g_cumsum: Any
    A: Any
    cu_seqlens: Any
    w: Any
    u: Any
    boundaries: tuple[int, ...]
    workspaces: _Workspaces


def _validate_args(
    num_tokens: int,
    num_chunks: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    num_tokens = _exact_int("num_tokens", num_tokens)
    num_chunks = _exact_int("num_chunks", num_chunks)
    num_key_heads = _exact_int("num_key_heads", num_key_heads)
    num_heads = _exact_int("num_heads", num_heads)
    key_head_dim = _exact_int("key_head_dim", key_head_dim)
    value_head_dim = _exact_int("value_head_dim", value_head_dim)
    dtype = DType.from_value(dtype)

    dimensions = (
        num_tokens,
        num_chunks,
        num_key_heads,
        num_heads,
        key_head_dim,
        value_head_dim,
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            "num_tokens, num_chunks, num_key_heads, num_heads, key_head_dim, "
            f"and value_head_dim must be > 0, got {dimensions}"
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
        raise ValueError(f"torch gdn_chunk_recompute_w_u requires dtype=bf16, got {dtype.value}")
    return _ValidatedArgs(
        num_tokens=num_tokens,
        num_chunks=num_chunks,
        num_key_heads=num_key_heads,
        num_heads=num_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        dtype=dtype,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch gdn_chunk_recompute_w_u backend"
        )


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    max_length = max(_canonical_lengths(args.num_tokens, args.num_chunks))
    return _OperandShapes(
        k=(args.num_tokens, args.num_key_heads, args.key_head_dim),
        v=(args.num_tokens, args.num_heads, args.value_head_dim),
        beta=(args.num_tokens, args.num_heads),
        g_cumsum=(args.num_tokens, args.num_heads),
        A=(args.num_tokens, args.num_heads, _CHUNK_SIZE),
        cu_seqlens=(args.num_chunks + 1,),
        w=(args.num_tokens, args.num_heads, args.key_head_dim),
        u=(args.num_tokens, args.num_heads, args.value_head_dim),
        solved_fp32=(args.num_heads, max_length, max_length),
        k_fp32=(args.num_tokens, args.num_key_heads, args.key_head_dim),
        grouped_k_fp32=(max_length, args.num_heads, args.key_head_dim),
        k_factor=(args.num_heads, max_length, args.key_head_dim),
        v_factor=(args.num_heads, max_length, args.value_head_dim),
        k_result_fp32=(args.num_heads, max_length, args.key_head_dim),
        v_result_fp32=(args.num_heads, max_length, args.value_head_dim),
        gate_scale=(args.num_heads, max_length),
        head_to_key=(args.num_heads,),
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

    A = torch.zeros(shapes.A, dtype=torch.bfloat16, device=device)
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        length = sequence_end - sequence_start
        values = bounded((length, args.num_heads, length), -0.125, 0.125, dtype=torch.float32)
        lower = torch.tril(values.permute(1, 0, 2), diagonal=-1).permute(1, 0, 2)
        A[sequence_start:sequence_end, :, :length].copy_(lower)
        rows = torch.arange(sequence_start, sequence_end, device=device)
        local_rows = torch.arange(length, device=device)
        A[rows, :, local_rows] = 1

    heads_per_key = args.num_heads // args.num_key_heads
    head_to_key = torch.arange(args.num_heads, dtype=torch.int64, device=device).div(
        heads_per_key,
        rounding_mode="floor",
    )
    return _Operands(
        k=bounded(shapes.k, -0.25, 0.25, dtype=torch.bfloat16),
        v=bounded(shapes.v, -0.25, 0.25, dtype=torch.bfloat16),
        beta=bounded(shapes.beta, -0.5, 0.5, dtype=torch.float32),
        g_cumsum=bounded(shapes.g_cumsum, -0.25, 0.25, dtype=torch.float32),
        A=A,
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device=device),
        w=torch.empty(shapes.w, dtype=torch.bfloat16, device=device),
        u=torch.empty(shapes.u, dtype=torch.bfloat16, device=device),
        boundaries=boundaries,
        workspaces=_Workspaces(
            solved_fp32=torch.empty(shapes.solved_fp32, dtype=torch.float32, device=device),
            k_fp32=torch.empty(shapes.k_fp32, dtype=torch.float32, device=device),
            grouped_k_fp32=torch.empty(shapes.grouped_k_fp32, dtype=torch.float32, device=device),
            k_factor_bf16=torch.empty(shapes.k_factor, dtype=torch.bfloat16, device=device),
            k_factor_fp32=torch.empty(shapes.k_factor, dtype=torch.float32, device=device),
            v_factor_bf16=torch.empty(shapes.v_factor, dtype=torch.bfloat16, device=device),
            v_factor_fp32=torch.empty(shapes.v_factor, dtype=torch.float32, device=device),
            k_result_fp32=torch.empty(shapes.k_result_fp32, dtype=torch.float32, device=device),
            v_result_fp32=torch.empty(shapes.v_result_fp32, dtype=torch.float32, device=device),
            gate_scale=torch.empty(shapes.gate_scale, dtype=torch.float32, device=device),
            head_to_key=head_to_key,
        ),
    )


def _recompute_w_u_into(
    torch: Any,
    k: Any,
    v: Any,
    beta: Any,
    g_cumsum: Any,
    A: Any,
    w: Any,
    u: Any,
    boundaries: tuple[int, ...],
    workspaces: _Workspaces,
) -> tuple[Any, Any]:
    """Execute the chunk-local equations into preallocated BF16 outputs.

    Factor scaling, exp, BF16 factor rounding, FP32 matrix products, and the
    final BF16 writes are all semantic work and remain inside the measured
    callable. Every used output/workspace region is overwritten, so repeated
    calls require no external reset.
    """
    workspaces.k_fp32.copy_(k)
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        length = sequence_end - sequence_start
        solved = workspaces.solved_fp32[:, :length, :length]
        solved.copy_(A[sequence_start:sequence_end, :, :length].permute(1, 0, 2))
        row_beta = beta[sequence_start:sequence_end].transpose(0, 1).unsqueeze(-1)

        v_factor_bf16 = workspaces.v_factor_bf16[:, :length]
        torch.mul(
            v[sequence_start:sequence_end].permute(1, 0, 2),
            row_beta,
            out=v_factor_bf16,
        )
        v_factor_fp32 = workspaces.v_factor_fp32[:, :length]
        v_factor_fp32.copy_(v_factor_bf16)

        grouped_k = workspaces.grouped_k_fp32[:length]
        torch.index_select(
            workspaces.k_fp32[sequence_start:sequence_end],
            1,
            workspaces.head_to_key,
            out=grouped_k,
        )
        grouped_k_by_head = grouped_k.permute(1, 0, 2)
        gate_scale = workspaces.gate_scale[:, :length]
        torch.exp(g_cumsum[sequence_start:sequence_end].transpose(0, 1), out=gate_scale)
        gate_scale.mul_(beta[sequence_start:sequence_end].transpose(0, 1))
        k_factor_bf16 = workspaces.k_factor_bf16[:, :length]
        torch.mul(grouped_k_by_head, gate_scale.unsqueeze(-1), out=k_factor_bf16)
        k_factor_fp32 = workspaces.k_factor_fp32[:, :length]
        k_factor_fp32.copy_(k_factor_bf16)

        v_result = workspaces.v_result_fp32[:, :length]
        k_result = workspaces.k_result_fp32[:, :length]
        torch.bmm(solved, v_factor_fp32, out=v_result)
        torch.bmm(solved, k_factor_fp32, out=k_result)
        u[sequence_start:sequence_end].copy_(v_result.permute(1, 0, 2))
        w[sequence_start:sequence_end].copy_(k_result.permute(1, 0, 2))
    return w, u


def _semantic_flops(
    *,
    num_tokens: int,
    num_chunks: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> int:
    """Nominal semantic FLOPs, not physical Torch instruction count.

    Matrix products use one nominal unit per multiply-add term as frozen by
    this kernel's semantic convention. The scaling term includes V beta
    multiplies, K beta/gate multiplies, and one nominal exp per token/head.
    """
    square_rows = sum(length * length for length in _canonical_lengths(num_tokens, num_chunks))
    return num_heads * (key_head_dim + value_head_dim) * square_rows + (
        num_heads * num_tokens * (value_head_dim + 2 * key_head_dim + 1)
    )


def _logical_bytes(
    *,
    num_tokens: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> int:
    """Logical boundary traffic, excluding metadata and implementation traffic."""
    return (
        2 * num_tokens * num_key_heads * key_head_dim
        + 2 * num_tokens * num_heads * value_head_dim
        + 8 * num_tokens * num_heads
        + 2 * num_tokens * num_heads * _CHUNK_SIZE
        + 2 * num_tokens * num_heads * key_head_dim
        + 2 * num_tokens * num_heads * value_head_dim
    )


def profile_gdn_chunk_recompute_w_u(
    num_tokens: int,
    num_chunks: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile the preallocated multi-launch Torch semantic implementation."""
    args = _validate_args(
        num_tokens=num_tokens,
        num_chunks=num_chunks,
        num_key_heads=num_key_heads,
        num_heads=num_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        dtype=dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch gdn_chunk_recompute_w_u backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(torch, args, device=torch.device("cuda"))

        def kernel():
            return _recompute_w_u_into(
                torch,
                operands.k,
                operands.v,
                operands.beta,
                operands.g_cumsum,
                operands.A,
                operands.w,
                operands.u,
                operands.boundaries,
                operands.workspaces,
            )

        # No kernel-name filter: measure the complete multi-launch semantic
        # composite. Allocation, partitioning, metadata, and reference work are
        # outside timing. The helper overwrites all mutable state without reset.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        flops = _semantic_flops(
            num_tokens=args.num_tokens,
            num_chunks=args.num_chunks,
            num_heads=args.num_heads,
            key_head_dim=args.key_head_dim,
            value_head_dim=args.value_head_dim,
        )
        logical_bytes = _logical_bytes(
            num_tokens=args.num_tokens,
            num_key_heads=args.num_key_heads,
            num_heads=args.num_heads,
            key_head_dim=args.key_head_dim,
            value_head_dim=args.value_head_dim,
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
