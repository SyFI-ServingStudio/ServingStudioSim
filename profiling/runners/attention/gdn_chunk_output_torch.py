"""Multi-launch Torch semantic baseline for Qwen GDN chunk output."""

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
    num_key_heads: int
    num_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: DType


@dataclass(frozen=True)
class _Workspaces:
    q_group_bf16: Any
    k_group_bf16: Any
    q_group_fp32: Any
    k_group_fp32: Any
    snapshot_fp32: Any
    value_fp32: Any
    state_fp32: Any
    score_fp32: Any
    score_bf16: Any
    causal_fp32: Any
    result_fp32: Any
    gate_vector: Any
    gate_matrix: Any
    lower_mask: Any
    head_to_key: Any


@dataclass(frozen=True)
class _Operands:
    q: Any
    k: Any
    v_new: Any
    h: Any
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
    value_head_dim: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    values = tuple(
        _exact_int(name, value)
        for name, value in (
            ("num_tokens", num_tokens),
            ("num_chunks", num_chunks),
            ("num_key_heads", num_key_heads),
            ("num_heads", num_heads),
            ("key_head_dim", key_head_dim),
            ("value_head_dim", value_head_dim),
        )
    )
    if any(value <= 0 for value in values):
        raise ValueError(f"all dimensions must be > 0, got {values}")
    num_tokens, num_chunks, num_key_heads, num_heads, key_head_dim, value_head_dim = values
    minimum_chunks = (num_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    if not minimum_chunks <= num_chunks <= num_tokens:
        raise ValueError("num_chunks must satisfy ceil(num_tokens/64) <= num_chunks <= num_tokens")
    if num_heads % num_key_heads:
        raise ValueError("num_heads must be divisible by num_key_heads")
    dtype = DType.from_value(dtype)
    if dtype is not DType.BF16:
        raise ValueError(f"torch gdn_chunk_output requires dtype=bf16, got {dtype.value}")
    return _ValidatedArgs(*values, dtype)


def _canonical_lengths(num_tokens: int, num_chunks: int) -> tuple[int, ...]:
    """Use one sequence at the minimum chunk count; otherwise C one-chunk sequences."""
    minimum = (num_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    if num_chunks == minimum:
        lengths = (num_tokens,)
    else:
        quotient, remainder = divmod(num_tokens, num_chunks)
        lengths = (quotient + 1,) * remainder + (quotient,) * (num_chunks - remainder)
    if (
        sum(lengths) != num_tokens
        or any(length <= 0 for length in lengths)
        or sum((length + 63) // 64 for length in lengths) != num_chunks
    ):
        raise ValueError("failed to construct exact canonical ragged geometry")
    return lengths


def _canonical_boundaries(num_tokens: int, num_chunks: int) -> tuple[int, ...]:
    return (0, *accumulate(_canonical_lengths(num_tokens, num_chunks)))


def _operand_shapes(args: _ValidatedArgs) -> dict[str, tuple[int, ...]]:
    tile = min(_CHUNK_SIZE, args.num_tokens)
    return {
        "q": (args.num_tokens, args.num_key_heads, args.key_head_dim),
        "k": (args.num_tokens, args.num_key_heads, args.key_head_dim),
        "v_new": (args.num_tokens, args.num_heads, args.value_head_dim),
        "h": (args.num_chunks, args.num_heads, args.value_head_dim, args.key_head_dim),
        "g_cumsum": (args.num_tokens, args.num_heads),
        "cu_seqlens": (len(_canonical_lengths(args.num_tokens, args.num_chunks)) + 1,),
        "output": (args.num_tokens, args.num_heads, args.value_head_dim),
        "grouped_key": (args.num_heads, tile, args.key_head_dim),
        "snapshot": (args.num_heads, args.value_head_dim, args.key_head_dim),
        "value": (args.num_heads, tile, args.value_head_dim),
        "head_value": (args.num_heads, tile, args.value_head_dim),
        "scores": (args.num_heads, tile, tile),
        "gate_vector": (args.num_heads, tile),
        "gate_matrix": (args.num_heads, tile, tile),
        "head_to_key": (args.num_heads,),
    }


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    boundaries = _canonical_boundaries(args.num_tokens, args.num_chunks)
    generator = torch.Generator(device=device).manual_seed(42)

    def bounded(shape: tuple[int, ...], dtype: Any):
        return torch.empty(shape, dtype=dtype, device=device).uniform_(
            -0.125, 0.125, generator=generator
        )

    heads_per_key = args.num_heads // args.num_key_heads
    head_to_key = torch.arange(args.num_heads, dtype=torch.int64, device=device).div(
        heads_per_key, rounding_mode="floor"
    )
    return _Operands(
        q=bounded(shapes["q"], torch.bfloat16),
        k=bounded(shapes["k"], torch.bfloat16),
        v_new=bounded(shapes["v_new"], torch.bfloat16),
        h=bounded(shapes["h"], torch.bfloat16),
        g_cumsum=bounded(shapes["g_cumsum"], torch.float32),
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device=device),
        output=torch.empty(shapes["output"], dtype=torch.bfloat16, device=device),
        boundaries=boundaries,
        workspaces=_Workspaces(
            q_group_bf16=torch.empty(shapes["grouped_key"], dtype=torch.bfloat16, device=device),
            k_group_bf16=torch.empty(shapes["grouped_key"], dtype=torch.bfloat16, device=device),
            q_group_fp32=torch.empty(shapes["grouped_key"], dtype=torch.float32, device=device),
            k_group_fp32=torch.empty(shapes["grouped_key"], dtype=torch.float32, device=device),
            snapshot_fp32=torch.empty(shapes["snapshot"], dtype=torch.float32, device=device),
            value_fp32=torch.empty(shapes["value"], dtype=torch.float32, device=device),
            state_fp32=torch.empty(shapes["head_value"], dtype=torch.float32, device=device),
            score_fp32=torch.empty(shapes["scores"], dtype=torch.float32, device=device),
            score_bf16=torch.empty(shapes["scores"], dtype=torch.bfloat16, device=device),
            causal_fp32=torch.empty(shapes["head_value"], dtype=torch.float32, device=device),
            result_fp32=torch.empty(shapes["head_value"], dtype=torch.float32, device=device),
            gate_vector=torch.empty(shapes["gate_vector"], dtype=torch.float32, device=device),
            gate_matrix=torch.empty(shapes["gate_matrix"], dtype=torch.float32, device=device),
            lower_mask=torch.ones(shapes["scores"][1:], dtype=torch.bool, device=device).tril(),
            head_to_key=head_to_key,
        ),
    )


def _chunk_output_into(
    torch: Any,
    q: Any,
    k: Any,
    v_new: Any,
    h: Any,
    g_cumsum: Any,
    output: Any,
    boundaries: tuple[int, ...],
    workspaces: _Workspaces,
) -> Any:
    """Overwrite output/workspaces with the complete chunk-output semantics."""
    scale = q.shape[-1] ** -0.5
    global_chunk = 0
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            length = chunk_end - chunk_start
            q_bf16 = workspaces.q_group_bf16[:, :length]
            k_bf16 = workspaces.k_group_bf16[:, :length]
            torch.index_select(
                q[chunk_start:chunk_end], 1, workspaces.head_to_key, out=q_bf16.transpose(0, 1)
            )
            torch.index_select(
                k[chunk_start:chunk_end], 1, workspaces.head_to_key, out=k_bf16.transpose(0, 1)
            )
            q_fp32 = workspaces.q_group_fp32[:, :length]
            q_fp32.copy_(q_bf16)
            k_fp32 = workspaces.k_group_fp32[:, :length]
            k_fp32.copy_(k_bf16)
            snapshot = workspaces.snapshot_fp32
            snapshot.copy_(h[global_chunk])
            state = workspaces.state_fp32[:, :length]
            torch.bmm(q_fp32, snapshot.transpose(1, 2), out=state)
            gate = g_cumsum[chunk_start:chunk_end].transpose(0, 1)
            gate_vector = workspaces.gate_vector[:, :length]
            torch.exp(gate, out=gate_vector)
            state.mul_(gate_vector.unsqueeze(-1))
            scores = workspaces.score_fp32[:, :length, :length]
            torch.bmm(q_fp32, k_fp32.transpose(1, 2), out=scores)
            gate_matrix = workspaces.gate_matrix[:, :length, :length]
            torch.sub(gate.unsqueeze(2), gate.unsqueeze(1), out=gate_matrix)
            torch.exp(gate_matrix, out=gate_matrix)
            scores.mul_(gate_matrix)
            scores.masked_fill_(~workspaces.lower_mask[:length, :length], 0)
            score_bf16 = workspaces.score_bf16[:, :length, :length]
            score_bf16.copy_(scores)
            scores.copy_(score_bf16)
            values = workspaces.value_fp32[:, :length]
            values.copy_(v_new[chunk_start:chunk_end].transpose(0, 1))
            causal = workspaces.causal_fp32[:, :length]
            torch.bmm(scores, values, out=causal)
            result = workspaces.result_fp32[:, :length]
            result.copy_(state)
            result.add_(causal).mul_(scale)
            output[chunk_start:chunk_end].copy_(result.transpose(0, 1))
            global_chunk += 1
    return output


def _semantic_flops(
    *, num_tokens: int, num_chunks: int, num_heads: int, key_head_dim: int, value_head_dim: int
) -> int:
    """Valid-row semantic FLOPs; excludes padded and physical Torch work."""
    total = 0
    for sequence_length in _canonical_lengths(num_tokens, num_chunks):
        for start in range(0, sequence_length, _CHUNK_SIZE):
            p = min(_CHUNK_SIZE, sequence_length - start)
            pairs = p * (p + 1) // 2
            total += num_heads * (
                2 * p * key_head_dim * value_head_dim
                + 2 * p * p * key_head_dim
                + p
                + p * value_head_dim
                + 2 * pairs
                + 2 * pairs * value_head_dim
                + 2 * p * value_head_dim
            )
    return total


def _logical_bytes(
    *,
    num_tokens: int,
    num_chunks: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> int:
    """Read q/k/v/H/g and write output once; metadata/workspaces are excluded."""
    return (
        4 * num_tokens * num_key_heads * key_head_dim
        + 2 * num_tokens * num_heads * value_head_dim
        + 2 * num_chunks * num_heads * value_head_dim * key_head_dim
        + 4 * num_tokens * num_heads
        + 2 * num_tokens * num_heads * value_head_dim
    )


def profile_gdn_chunk_output(
    num_tokens: int,
    num_chunks: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    args = _validate_args(
        num_tokens, num_chunks, num_key_heads, num_heads, key_head_dim, value_head_dim, dtype
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for gdn_chunk_output") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for torch gdn_chunk_output")
    try:
        operands = _build_operands(torch, args, device=torch.device("cuda"))

        def kernel():
            return _chunk_output_into(
                torch,
                operands.q,
                operands.k,
                operands.v_new,
                operands.h,
                operands.g_cumsum,
                operands.output,
                operands.boundaries,
                operands.workspaces,
            )

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
            num_chunks=args.num_chunks,
            num_key_heads=args.num_key_heads,
            num_heads=args.num_heads,
            key_head_dim=args.key_head_dim,
            value_head_dim=args.value_head_dim,
        )
        elapsed = time_ms / 1000
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(flops / elapsed / 1e12 if elapsed else 0),
            memory_bandwidth_gbps=float(logical_bytes / elapsed / 1e9 if elapsed else 0),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
