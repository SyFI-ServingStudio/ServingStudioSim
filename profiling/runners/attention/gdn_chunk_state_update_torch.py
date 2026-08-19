"""Torch semantic-composite runner for Qwen GDN chunk state update.

This backend times the multiple Torch launches needed to implement the frozen
state-snapshot, value-residual, gate-decay, and recurrent-state equations. It is
not vLLM's fused Triton launch, and its semantic/logical rates do not describe
the physical implementation. Production simulation must select the fused
backend after one is registered.
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
    num_sequences: int
    max_chunks_per_sequence: int
    num_key_heads: int
    num_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: DType


@dataclass(frozen=True)
class _OperandShapes:
    k: tuple[int, int, int]
    w: tuple[int, int, int]
    u: tuple[int, int, int]
    g_cumsum: tuple[int, int]
    initial_state: tuple[int, int, int, int]
    cu_seqlens: tuple[int]
    chunk_indices: tuple[int, int]
    chunk_offsets: tuple[int]
    h: tuple[int, int, int, int]
    v_new: tuple[int, int, int]
    final_state: tuple[int, int, int, int]
    chunk_w: tuple[int, int, int]
    chunk_value: tuple[int, int, int]
    chunk_key: tuple[int, int, int]
    state: tuple[int, int, int, int]
    snapshot: tuple[int, int, int]
    update: tuple[int, int, int]
    decay: tuple[int, int]
    head_to_key: tuple[int]


@dataclass(frozen=True)
class _Workspaces:
    state: Any
    snapshot_bf16: Any
    snapshot_fp32: Any
    w_fp32: Any
    correction: Any
    residual: Any
    decay: Any
    state_factor_bf16: Any
    state_factor_fp32: Any
    grouped_k_bf16: Any
    grouped_k_fp32: Any
    update: Any
    end_decay: Any
    head_to_key: Any


@dataclass(frozen=True)
class _Operands:
    k: Any
    w: Any
    u: Any
    g_cumsum: Any
    initial_state: Any
    cu_seqlens: Any
    chunk_indices: Any
    chunk_offsets: Any
    h: Any
    v_new: Any
    final_state: Any
    chunk_counts: tuple[int, ...]
    lengths: tuple[int, ...]
    boundaries: tuple[int, ...]
    workspaces: _Workspaces


def _validate_args(
    num_tokens: int,
    num_chunks: int,
    num_sequences: int,
    max_chunks_per_sequence: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    num_tokens = _exact_int("num_tokens", num_tokens)
    num_chunks = _exact_int("num_chunks", num_chunks)
    num_sequences = _exact_int("num_sequences", num_sequences)
    max_chunks_per_sequence = _exact_int("max_chunks_per_sequence", max_chunks_per_sequence)
    num_key_heads = _exact_int("num_key_heads", num_key_heads)
    num_heads = _exact_int("num_heads", num_heads)
    key_head_dim = _exact_int("key_head_dim", key_head_dim)
    value_head_dim = _exact_int("value_head_dim", value_head_dim)
    dtype = DType.from_value(dtype)

    dimensions = (
        num_tokens,
        num_chunks,
        num_sequences,
        max_chunks_per_sequence,
        num_key_heads,
        num_heads,
        key_head_dim,
        value_head_dim,
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            "num_tokens, num_chunks, num_sequences, max_chunks_per_sequence, "
            "num_key_heads, num_heads, key_head_dim, and value_head_dim must be > 0, "
            f"got {dimensions}"
        )
    minimum_chunks = max_chunks_per_sequence + num_sequences - 1
    maximum_chunks = num_sequences * max_chunks_per_sequence
    if num_chunks < minimum_chunks or num_chunks > maximum_chunks:
        raise ValueError(
            "feasible (C,N,M) requires M+N-1 <= C <= N*M, got "
            f"C={num_chunks}, N={num_sequences}, M={max_chunks_per_sequence}"
        )
    minimum_tokens = _CHUNK_SIZE * (num_chunks - num_sequences) + num_sequences
    maximum_tokens = _CHUNK_SIZE * num_chunks
    if num_tokens < minimum_tokens or num_tokens > maximum_tokens:
        raise ValueError(
            "feasible (T,C,N) requires 64*(C-N)+N <= T <= 64*C, got "
            f"T={num_tokens}, C={num_chunks}, N={num_sequences}"
        )
    if num_heads % num_key_heads != 0:
        raise ValueError(
            "num_heads must be divisible by num_key_heads, got "
            f"num_heads={num_heads}, num_key_heads={num_key_heads}"
        )
    if dtype is not DType.BF16:
        raise ValueError(f"torch gdn_chunk_state_update requires dtype=bf16, got {dtype.value}")
    return _ValidatedArgs(
        num_tokens=num_tokens,
        num_chunks=num_chunks,
        num_sequences=num_sequences,
        max_chunks_per_sequence=max_chunks_per_sequence,
        num_key_heads=num_key_heads,
        num_heads=num_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        dtype=dtype,
    )


def _canonical_chunk_counts(
    num_chunks: int,
    num_sequences: int,
    max_chunks_per_sequence: int,
) -> tuple[int, ...]:
    """Construct positive chunk counts with exact sum and exact maximum."""
    if num_sequences == 1:
        counts = (max_chunks_per_sequence,)
    else:
        residual = num_chunks - (max_chunks_per_sequence + num_sequences - 1)
        quotient, remainder = divmod(residual, num_sequences - 1)
        counts = (
            (max_chunks_per_sequence,)
            + (quotient + 2,) * remainder
            + (quotient + 1,) * (num_sequences - 1 - remainder)
        )
    if (
        len(counts) != num_sequences
        or sum(counts) != num_chunks
        or min(counts) < 1
        or max(counts) != max_chunks_per_sequence
    ):
        raise ValueError("canonical chunk counts require M+N-1 <= C <= N*M")
    return counts


def _canonical_lengths(
    num_tokens: int,
    num_chunks: int,
    num_sequences: int,
    max_chunks_per_sequence: int,
) -> tuple[int, ...]:
    """Construct sequence lengths realizing the canonical chunk counts."""
    counts = _canonical_chunk_counts(
        num_chunks,
        num_sequences,
        max_chunks_per_sequence,
    )
    minimum_lengths = tuple(_CHUNK_SIZE * (count - 1) + 1 for count in counts)
    residual = num_tokens - sum(minimum_lengths)
    quotient, remainder = divmod(residual, num_sequences)
    lengths = tuple(
        minimum_length + quotient + int(sequence < remainder)
        for sequence, minimum_length in enumerate(minimum_lengths)
    )
    realized_counts = tuple((length + _CHUNK_SIZE - 1) // _CHUNK_SIZE for length in lengths)
    if (
        len(lengths) != num_sequences
        or sum(lengths) != num_tokens
        or min(lengths) < 1
        or realized_counts != counts
    ):
        raise ValueError(
            "canonical lengths require 64*(C-N)+N <= T <= 64*C and feasible chunk counts"
        )
    return lengths


def _canonical_boundaries(
    num_tokens: int,
    num_chunks: int,
    num_sequences: int,
    max_chunks_per_sequence: int,
) -> tuple[int, ...]:
    return (
        0,
        *accumulate(
            _canonical_lengths(
                num_tokens,
                num_chunks,
                num_sequences,
                max_chunks_per_sequence,
            )
        ),
    )


def _canonical_index_pairs(chunk_counts: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (sequence, local_chunk)
        for sequence, count in enumerate(chunk_counts)
        for local_chunk in range(count)
    )


def _canonical_chunk_offsets(chunk_counts: tuple[int, ...]) -> tuple[int, ...]:
    return (0, *accumulate(chunk_counts))


def _canonical_valid_chunk_lengths(lengths: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        min(_CHUNK_SIZE, length - chunk_start)
        for length in lengths
        for chunk_start in range(0, length, _CHUNK_SIZE)
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch gdn_chunk_state_update backend"
        )


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    lengths = _canonical_lengths(
        args.num_tokens,
        args.num_chunks,
        args.num_sequences,
        args.max_chunks_per_sequence,
    )
    max_valid_rows = max(_canonical_valid_chunk_lengths(lengths))
    state = (
        args.num_sequences,
        args.num_heads,
        args.value_head_dim,
        args.key_head_dim,
    )
    snapshot = (args.num_heads, args.value_head_dim, args.key_head_dim)
    return _OperandShapes(
        k=(args.num_tokens, args.num_key_heads, args.key_head_dim),
        w=(args.num_tokens, args.num_heads, args.key_head_dim),
        u=(args.num_tokens, args.num_heads, args.value_head_dim),
        g_cumsum=(args.num_tokens, args.num_heads),
        initial_state=state,
        cu_seqlens=(args.num_sequences + 1,),
        chunk_indices=(args.num_chunks, 2),
        chunk_offsets=(args.num_sequences + 1,),
        h=(args.num_chunks, args.num_heads, args.value_head_dim, args.key_head_dim),
        v_new=(args.num_tokens, args.num_heads, args.value_head_dim),
        final_state=state,
        chunk_w=(args.num_heads, max_valid_rows, args.key_head_dim),
        chunk_value=(args.num_heads, max_valid_rows, args.value_head_dim),
        chunk_key=(max_valid_rows, args.num_heads, args.key_head_dim),
        state=state,
        snapshot=snapshot,
        update=snapshot,
        decay=(args.num_heads, max_valid_rows),
        head_to_key=(args.num_heads,),
    )


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    chunk_counts = _canonical_chunk_counts(
        args.num_chunks,
        args.num_sequences,
        args.max_chunks_per_sequence,
    )
    lengths = _canonical_lengths(
        args.num_tokens,
        args.num_chunks,
        args.num_sequences,
        args.max_chunks_per_sequence,
    )
    boundaries = (0, *accumulate(lengths))
    index_pairs = _canonical_index_pairs(chunk_counts)
    chunk_offsets = _canonical_chunk_offsets(chunk_counts)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    def bounded(shape: tuple[int, ...], low: float, high: float, *, dtype: Any):
        return torch.empty(shape, dtype=dtype, device=device).uniform_(
            low,
            high,
            generator=generator,
        )

    g_cumsum = torch.empty(shapes.g_cumsum, dtype=torch.float32, device=device)
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            increments = bounded(
                (chunk_end - chunk_start, args.num_heads),
                -0.02,
                0.0,
                dtype=torch.float32,
            )
            torch.cumsum(
                increments, dim=0, dtype=torch.float32, out=g_cumsum[chunk_start:chunk_end]
            )

    heads_per_key = args.num_heads // args.num_key_heads
    head_to_key = torch.arange(args.num_heads, dtype=torch.int64, device=device).div(
        heads_per_key,
        rounding_mode="floor",
    )
    return _Operands(
        k=bounded(shapes.k, -0.08, 0.08, dtype=torch.bfloat16),
        w=bounded(shapes.w, -0.04, 0.04, dtype=torch.bfloat16),
        u=bounded(shapes.u, -0.08, 0.08, dtype=torch.bfloat16),
        g_cumsum=g_cumsum,
        initial_state=bounded(shapes.initial_state, -0.01, 0.01, dtype=torch.float32),
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device=device),
        chunk_indices=torch.tensor(index_pairs, dtype=torch.int32, device=device),
        chunk_offsets=torch.tensor(chunk_offsets, dtype=torch.int32, device=device),
        h=torch.empty(shapes.h, dtype=torch.bfloat16, device=device),
        v_new=torch.empty(shapes.v_new, dtype=torch.bfloat16, device=device),
        final_state=torch.empty(shapes.final_state, dtype=torch.float32, device=device),
        chunk_counts=chunk_counts,
        lengths=lengths,
        boundaries=boundaries,
        workspaces=_Workspaces(
            state=torch.empty(shapes.state, dtype=torch.float32, device=device),
            snapshot_bf16=torch.empty(shapes.snapshot, dtype=torch.bfloat16, device=device),
            snapshot_fp32=torch.empty(shapes.snapshot, dtype=torch.float32, device=device),
            w_fp32=torch.empty(shapes.chunk_w, dtype=torch.float32, device=device),
            correction=torch.empty(shapes.chunk_value, dtype=torch.float32, device=device),
            residual=torch.empty(shapes.chunk_value, dtype=torch.float32, device=device),
            decay=torch.empty(shapes.decay, dtype=torch.float32, device=device),
            state_factor_bf16=torch.empty(
                shapes.chunk_value,
                dtype=torch.bfloat16,
                device=device,
            ),
            state_factor_fp32=torch.empty(
                shapes.chunk_value,
                dtype=torch.float32,
                device=device,
            ),
            grouped_k_bf16=torch.empty(
                shapes.chunk_key,
                dtype=torch.bfloat16,
                device=device,
            ),
            grouped_k_fp32=torch.empty(
                shapes.chunk_key,
                dtype=torch.float32,
                device=device,
            ),
            update=torch.empty(shapes.update, dtype=torch.float32, device=device),
            end_decay=torch.empty((args.num_heads,), dtype=torch.float32, device=device),
            head_to_key=head_to_key,
        ),
    )


def _state_update_into(torch: Any, operands: _Operands) -> tuple[Any, Any, Any]:
    """Execute every frozen state-update operation into preallocated storage.

    The initial-state reset copy is deliberately part of this measured helper.
    Every output and every subsequently read mutable workspace region is
    overwritten, so repeated calls need no external reset.
    """
    workspaces = operands.workspaces
    workspaces.state.copy_(operands.initial_state)
    global_chunk = 0
    for sequence, (sequence_start, sequence_end) in enumerate(
        zip(operands.boundaries, operands.boundaries[1:])
    ):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            length = chunk_end - chunk_start

            workspaces.snapshot_bf16.copy_(workspaces.state[sequence])
            operands.h[global_chunk].copy_(workspaces.snapshot_bf16)
            workspaces.snapshot_fp32.copy_(workspaces.snapshot_bf16)

            chunk_w = workspaces.w_fp32[:, :length]
            chunk_w.copy_(operands.w[chunk_start:chunk_end].permute(1, 0, 2))
            correction = workspaces.correction[:, :length]
            torch.bmm(
                chunk_w,
                workspaces.snapshot_fp32.transpose(1, 2),
                out=correction,
            )
            residual = workspaces.residual[:, :length]
            residual.copy_(operands.u[chunk_start:chunk_end].permute(1, 0, 2))
            residual.sub_(correction)
            operands.v_new[chunk_start:chunk_end].copy_(residual.permute(1, 0, 2))

            decay = workspaces.decay[:, :length]
            torch.sub(
                operands.g_cumsum[chunk_end - 1].unsqueeze(1),
                operands.g_cumsum[chunk_start:chunk_end].transpose(0, 1),
                out=decay,
            )
            decay.exp_()
            state_factor_bf16 = workspaces.state_factor_bf16[:, :length]
            torch.mul(residual, decay.unsqueeze(-1), out=state_factor_bf16)
            state_factor_fp32 = workspaces.state_factor_fp32[:, :length]
            state_factor_fp32.copy_(state_factor_bf16)

            grouped_k_bf16 = workspaces.grouped_k_bf16[:length]
            torch.index_select(
                operands.k[chunk_start:chunk_end],
                1,
                workspaces.head_to_key,
                out=grouped_k_bf16,
            )
            grouped_k_fp32 = workspaces.grouped_k_fp32[:length]
            grouped_k_fp32.copy_(grouped_k_bf16)
            torch.bmm(
                state_factor_fp32.transpose(1, 2),
                grouped_k_fp32.permute(1, 0, 2),
                out=workspaces.update,
            )

            torch.exp(operands.g_cumsum[chunk_end - 1], out=workspaces.end_decay)
            workspaces.state[sequence].mul_(workspaces.end_decay[:, None, None])
            workspaces.state[sequence].add_(workspaces.update)
            global_chunk += 1
        operands.final_state[sequence].copy_(workspaces.state[sequence])
    return operands.h, operands.v_new, operands.final_state


def _semantic_flops(
    *,
    num_tokens: int,
    num_chunks: int,
    num_sequences: int,
    max_chunks_per_sequence: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> int:
    """Nominal semantic FLOPs, excluding padded/physical Torch work."""
    lengths = _canonical_lengths(
        num_tokens,
        num_chunks,
        num_sequences,
        max_chunks_per_sequence,
    )
    chunk_lengths = _canonical_valid_chunk_lengths(lengths)
    per_head = sum(
        4 * length * key_head_dim * value_head_dim
        + length * value_head_dim
        + key_head_dim * value_head_dim
        + 2 * length
        + 1
        for length in chunk_lengths
    )
    return num_heads * per_head


def _logical_bytes(
    *,
    num_tokens: int,
    num_chunks: int,
    num_sequences: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> int:
    """Logical tensor-boundary traffic, not physical Torch traffic."""
    return (
        2 * num_tokens * num_key_heads * key_head_dim
        + 2 * num_tokens * num_heads * key_head_dim
        + 2 * num_tokens * num_heads * value_head_dim
        + 4 * num_tokens * num_heads
        + 4 * num_sequences * num_heads * value_head_dim * key_head_dim
        + 2 * num_chunks * num_heads * value_head_dim * key_head_dim
        + 2 * num_tokens * num_heads * value_head_dim
        + 4 * num_sequences * num_heads * value_head_dim * key_head_dim
    )


def profile_gdn_chunk_state_update(
    num_tokens: int,
    num_chunks: int,
    num_sequences: int,
    max_chunks_per_sequence: int,
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
        num_sequences=num_sequences,
        max_chunks_per_sequence=max_chunks_per_sequence,
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
            "torch is required for the torch gdn_chunk_state_update backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(torch, args, device=torch.device("cuda"))

        def kernel():
            return _state_update_into(torch, operands)

        # No kernel-name filter: measure the complete multi-launch semantic
        # composite. Allocation, partition/metadata construction, reference
        # work, and comparisons are outside timing. The initial-state reset and
        # every semantic operation/output write are inside this callable.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        flops = _semantic_flops(
            num_tokens=args.num_tokens,
            num_chunks=args.num_chunks,
            num_sequences=args.num_sequences,
            max_chunks_per_sequence=args.max_chunks_per_sequence,
            num_heads=args.num_heads,
            key_head_dim=args.key_head_dim,
            value_head_dim=args.value_head_dim,
        )
        logical_bytes = _logical_bytes(
            num_tokens=args.num_tokens,
            num_chunks=args.num_chunks,
            num_sequences=args.num_sequences,
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
