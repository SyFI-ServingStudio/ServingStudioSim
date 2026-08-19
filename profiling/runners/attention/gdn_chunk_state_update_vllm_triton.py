"""One-launch vLLM Triton runner for Qwen GDN chunk state update.

Construction, packed-layout validation, the bounded CPU-reference guard, and
autotune warmup stay outside measurement. The measured callable is one vLLM
wrapper invocation. CUPTI selects only the fused state-update kernel; energy
also includes the wrapper's three fresh output allocations. Reported FLOPs and
bytes are semantic/logical counts, not physical Triton work or traffic.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import (
    exact_int as _exact_int,
)
from profiling.runners.attention._gdn_common import (
    load_required_callable,
    require_exact_gpu,
)
from profiling.runners.attention.gdn_chunk_state_update_torch import (
    _canonical_chunk_counts,
    _canonical_chunk_offsets,
    _canonical_index_pairs,
    _canonical_lengths,
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gdn_chunk_state_update:vllm_triton"
_CALLABLE_MODULE = "vllm.model_executor.layers.fla.ops.chunk_delta_h"
_CALLABLE_NAME = "chunk_gated_delta_rule_fwd_h"
_KERNEL_NAME = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
_REQUIRED_GPU = "NVIDIA H200"
_CHUNK_SIZE = 64
_SUPPORTED_HEAD_DIM = 128
_MAX_GUARD_ELEMENTS = 4_500_000
_BF16_ATOL = 1e-2
_BF16_RTOL = 1e-2
_FP32_ATOL = 2e-5
_FP32_RTOL = 1e-2
_QWEN_GUARD_GEOMETRY = (128, 2, 1, 2, 16, 32, 128, 128, DType.BF16)
_WITNESS_GEOMETRY = (128, 2, 1, 2)


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
    k: tuple[int, int, int, int]
    w: tuple[int, int, int, int]
    u: tuple[int, int, int, int]
    g_cumsum: tuple[int, int, int]
    initial_state: tuple[int, int, int, int]
    cu_seqlens: tuple[int]
    chunk_indices: tuple[int, int]
    chunk_offsets: tuple[int]
    h: tuple[int, int, int, int, int]
    v_new: tuple[int, int, int, int]
    final_state: tuple[int, int, int, int]


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
    chunk_counts: tuple[int, ...]
    lengths: tuple[int, ...]
    boundaries: tuple[int, ...]


def _validate_environment() -> None:
    for name in ("FLA_USE_FAST_OPS", "FLA_USE_CUDA_GRAPH"):
        value = os.environ.get(name, "")
        if value.strip().lower() not in {"", "0", "false"}:
            raise ValueError(
                f"vllm_triton gdn_chunk_state_update requires {name} to be "
                f"absent, empty, false, or 0, got {value!r}"
            )


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
    _validate_environment()
    args = _ValidatedArgs(
        num_tokens=_exact_int("num_tokens", num_tokens),
        num_chunks=_exact_int("num_chunks", num_chunks),
        num_sequences=_exact_int("num_sequences", num_sequences),
        max_chunks_per_sequence=_exact_int("max_chunks_per_sequence", max_chunks_per_sequence),
        num_key_heads=_exact_int("num_key_heads", num_key_heads),
        num_heads=_exact_int("num_heads", num_heads),
        key_head_dim=_exact_int("key_head_dim", key_head_dim),
        value_head_dim=_exact_int("value_head_dim", value_head_dim),
        dtype=DType.from_value(dtype),
    )
    dimensions = (
        args.num_tokens,
        args.num_chunks,
        args.num_sequences,
        args.max_chunks_per_sequence,
        args.num_key_heads,
        args.num_heads,
        args.key_head_dim,
        args.value_head_dim,
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(f"all scalar dimensions must be > 0, got {dimensions}")
    minimum_chunks = args.max_chunks_per_sequence + args.num_sequences - 1
    maximum_chunks = args.num_sequences * args.max_chunks_per_sequence
    if args.num_chunks < minimum_chunks or args.num_chunks > maximum_chunks:
        raise ValueError(
            "feasible (C,N,M) requires M+N-1 <= C <= N*M, got "
            f"C={args.num_chunks}, N={args.num_sequences}, "
            f"M={args.max_chunks_per_sequence}"
        )
    minimum_tokens = _CHUNK_SIZE * (args.num_chunks - args.num_sequences) + args.num_sequences
    maximum_tokens = _CHUNK_SIZE * args.num_chunks
    if args.num_tokens < minimum_tokens or args.num_tokens > maximum_tokens:
        raise ValueError(
            "feasible (T,C,N) requires 64*(C-N)+N <= T <= 64*C, got "
            f"T={args.num_tokens}, C={args.num_chunks}, N={args.num_sequences}"
        )
    if args.num_heads % args.num_key_heads != 0:
        raise ValueError(
            "num_heads must be divisible by num_key_heads, got "
            f"num_heads={args.num_heads}, num_key_heads={args.num_key_heads}"
        )
    if args.dtype is not DType.BF16:
        raise ValueError(f"{_BACKEND} requires dtype=bf16, got {args.dtype.value}")
    if args.key_head_dim != _SUPPORTED_HEAD_DIM or args.value_head_dim != _SUPPORTED_HEAD_DIM:
        raise ValueError(
            f"{_BACKEND} requires key_head_dim=value_head_dim={_SUPPORTED_HEAD_DIM}, "
            f"got {args.key_head_dim} and {args.value_head_dim}"
        )
    return args


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    return _OperandShapes(
        k=(1, args.num_tokens, args.num_key_heads, args.key_head_dim),
        w=(1, args.num_tokens, args.num_heads, args.key_head_dim),
        u=(1, args.num_tokens, args.num_heads, args.value_head_dim),
        g_cumsum=(1, args.num_tokens, args.num_heads),
        initial_state=(
            args.num_sequences,
            args.num_heads,
            args.value_head_dim,
            args.key_head_dim,
        ),
        cu_seqlens=(args.num_sequences + 1,),
        chunk_indices=(args.num_chunks, 2),
        chunk_offsets=(args.num_sequences + 1,),
        h=(
            1,
            args.num_chunks,
            args.num_heads,
            args.value_head_dim,
            args.key_head_dim,
        ),
        v_new=(1, args.num_tokens, args.num_heads, args.value_head_dim),
        final_state=(
            args.num_sequences,
            args.num_heads,
            args.value_head_dim,
            args.key_head_dim,
        ),
    )


def _guard_elements(args: _ValidatedArgs) -> int:
    shapes = _operand_shapes(args)
    return sum(
        _numel(shape)
        for shape in (
            shapes.k,
            shapes.w,
            shapes.u,
            shapes.g_cumsum,
            shapes.initial_state,
            shapes.h,
            shapes.v_new,
            shapes.final_state,
        )
    )


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _guard_args(args: _ValidatedArgs) -> _ValidatedArgs:
    """Select one bounded witness for the shared static kernel specialization.

    N and exact M remain measured timing/metadata axes; dedicated direct H200
    audits validate their routing and loop behavior.  The per-call reference
    guard instead checks the shared H/K/V/BT specialization and recurrence
    semantics.  Exact Qwen is already bounded and guards itself.  Every other
    request uses one deterministic two-chunk, one-sequence witness while
    preserving the requested head geometry and dtype.
    """
    geometry = (
        args.num_tokens,
        args.num_chunks,
        args.num_sequences,
        args.max_chunks_per_sequence,
        args.num_key_heads,
        args.num_heads,
        args.key_head_dim,
        args.value_head_dim,
        args.dtype,
    )
    if geometry == _QWEN_GUARD_GEOMETRY:
        return args

    guard = replace(
        args,
        num_tokens=_WITNESS_GEOMETRY[0],
        num_chunks=_WITNESS_GEOMETRY[1],
        num_sequences=_WITNESS_GEOMETRY[2],
        max_chunks_per_sequence=_WITNESS_GEOMETRY[3],
    )
    if _guard_elements(guard) > _MAX_GUARD_ELEMENTS:
        raise ValueError(
            "bounded correctness witness preserving static Hg, H, K, V, and dtype "
            "exceeds the 4.5M-element cap"
        )
    return guard


def _require_h200(torch: Any) -> None:
    require_exact_gpu(torch, backend=_BACKEND, required_gpu=_REQUIRED_GPU)


def _load_fused_callable() -> Any:
    return load_required_callable(
        importlib.import_module,
        backend=_BACKEND,
        module_name=_CALLABLE_MODULE,
        callable_name=_CALLABLE_NAME,
    )


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    chunk_counts = _canonical_chunk_counts(
        args.num_chunks, args.num_sequences, args.max_chunks_per_sequence
    )
    lengths = _canonical_lengths(
        args.num_tokens,
        args.num_chunks,
        args.num_sequences,
        args.max_chunks_per_sequence,
    )
    boundaries = (0, *accumulate(lengths))
    index_pairs = _canonical_index_pairs(chunk_counts)
    offsets = _canonical_chunk_offsets(chunk_counts)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    def bounded(shape: tuple[int, ...], low: float, high: float, *, dtype: Any):
        return torch.empty(shape, dtype=dtype, device=device).uniform_(
            low, high, generator=generator
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
                increments,
                dim=0,
                dtype=torch.float32,
                out=g_cumsum[0, chunk_start:chunk_end],
            )

    # Build exact metadata on CPU and transfer once before guard/warmup/timing.
    metadata = (
        torch.tensor(boundaries, dtype=torch.int32, device="cpu"),
        torch.tensor(index_pairs, dtype=torch.int32, device="cpu"),
        torch.tensor(offsets, dtype=torch.int32, device="cpu"),
    )
    return _Operands(
        k=bounded(shapes.k, -0.08, 0.08, dtype=torch.bfloat16).contiguous(),
        w=bounded(shapes.w, -0.04, 0.04, dtype=torch.bfloat16).contiguous(),
        u=bounded(shapes.u, -0.08, 0.08, dtype=torch.bfloat16).contiguous(),
        g_cumsum=g_cumsum.contiguous(),
        initial_state=bounded(shapes.initial_state, -0.01, 0.01, dtype=torch.float32).contiguous(),
        cu_seqlens=metadata[0].to(device=device).contiguous(),
        chunk_indices=metadata[1].to(device=device).contiguous(),
        chunk_offsets=metadata[2].to(device=device).contiguous(),
        chunk_counts=chunk_counts,
        lengths=lengths,
        boundaries=boundaries,
    )


def _validate_operands(
    torch: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    require_cuda: bool = True,
) -> None:
    shapes = _operand_shapes(args)
    expected = {
        "k": (shapes.k, torch.bfloat16),
        "w": (shapes.w, torch.bfloat16),
        "u": (shapes.u, torch.bfloat16),
        "g_cumsum": (shapes.g_cumsum, torch.float32),
        "initial_state": (shapes.initial_state, torch.float32),
        "cu_seqlens": (shapes.cu_seqlens, torch.int32),
        "chunk_indices": (shapes.chunk_indices, torch.int32),
        "chunk_offsets": (shapes.chunk_offsets, torch.int32),
    }
    devices = set()
    for name, (shape, dtype) in expected.items():
        tensor = getattr(operands, name)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype is not dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous() or tensor.stride() != _packed_stride(shape):
            raise ValueError(f"{name} must have exact packed contiguous strides")
        if require_cuda and not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        devices.add(tensor.device)
        if name not in {"cu_seqlens", "chunk_indices", "chunk_offsets"}:
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")
    if len(devices) != 1:
        raise ValueError("operands and metadata must be on exactly one device")

    expected_boundaries = (0, *accumulate(operands.lengths))
    boundaries = tuple(int(value) for value in operands.cu_seqlens.tolist())
    if operands.boundaries != expected_boundaries or boundaries != expected_boundaries:
        raise ValueError("cu_seqlens must contain the exact canonical boundaries")
    if boundaries[0] != 0 or boundaries[-1] != args.num_tokens:
        raise ValueError("cu_seqlens endpoints must be 0 and num_tokens")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("cu_seqlens must be strictly increasing")
    expected_counts = _canonical_chunk_counts(
        args.num_chunks, args.num_sequences, args.max_chunks_per_sequence
    )
    expected_indices = _canonical_index_pairs(expected_counts)
    indices = tuple(tuple(int(item) for item in pair) for pair in operands.chunk_indices.tolist())
    if operands.chunk_counts != expected_counts or indices != expected_indices:
        raise ValueError("chunk_indices must contain the exact canonical mapping")
    expected_offsets = _canonical_chunk_offsets(expected_counts)
    offsets = tuple(int(value) for value in operands.chunk_offsets.tolist())
    if offsets != expected_offsets:
        raise ValueError("chunk_offsets must contain the exact canonical offsets")


def _packed_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return tuple(reversed(result))


def _invoke_fused(fused_callable: Any, operands: _Operands) -> tuple[Any, Any, Any]:
    return fused_callable(
        k=operands.k,
        w=operands.w,
        u=operands.u,
        g=operands.g_cumsum,
        gk=None,
        initial_state=operands.initial_state,
        output_final_state=True,
        chunk_size=_CHUNK_SIZE,
        save_new_value=True,
        cu_seqlens=operands.cu_seqlens,
        chunk_indices=operands.chunk_indices,
        chunk_offsets=operands.chunk_offsets,
        use_exp2=False,
    )


def _shares_storage(left: Any, right: Any) -> bool:
    return (
        left.device == right.device
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _check_correctness(
    torch: Any,
    fused_callable: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    synchronize: Any | None = None,
) -> None:
    from profiling.runners.attention.gdn_chunk_state_update_reference import (
        gdn_chunk_state_update_reference,
    )

    _validate_operands(torch, operands, args, require_cuda=operands.k.is_cuda)
    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    names = (
        "k",
        "w",
        "u",
        "g_cumsum",
        "initial_state",
        "cu_seqlens",
        "chunk_indices",
        "chunk_offsets",
    )
    snapshots = {name: getattr(operands, name).clone() for name in names}
    expected = gdn_chunk_state_update_reference(
        snapshots["k"].squeeze(0).cpu(),
        snapshots["w"].squeeze(0).cpu(),
        snapshots["u"].squeeze(0).cpu(),
        snapshots["g_cumsum"].squeeze(0).cpu(),
        snapshots["initial_state"].cpu(),
        snapshots["cu_seqlens"].cpu(),
    )
    actual = _invoke_fused(fused_callable, operands)
    synchronize()
    shapes = _operand_shapes(args)
    inputs = tuple(getattr(operands, name) for name in names)
    contracts = (
        ("h", actual[0], expected[0], shapes.h, torch.bfloat16, _BF16_ATOL, _BF16_RTOL),
        (
            "v_new",
            actual[1],
            expected[1],
            shapes.v_new,
            torch.bfloat16,
            _BF16_ATOL,
            _BF16_RTOL,
        ),
        (
            "final_state",
            actual[2],
            expected[2],
            shapes.final_state,
            torch.float32,
            _FP32_ATOL,
            _FP32_RTOL,
        ),
    )
    for name, output, reference, shape, dtype, atol, rtol in contracts:
        if output is None or tuple(output.shape) != shape:
            raise AssertionError(f"unexpected {name} shape")
        if output.dtype is not dtype:
            raise AssertionError(f"unexpected {name} dtype {output.dtype}")
        if not output.is_contiguous() or not torch.isfinite(output).all():
            raise AssertionError(f"{name} must be finite and contiguous")
        if any(_shares_storage(output, tensor) for tensor in inputs):
            raise AssertionError(f"{name} must have fresh non-aliased storage")
        expected_view = reference.unsqueeze(0) if name in {"h", "v_new"} else reference
        torch.testing.assert_close(output.cpu(), expected_view, atol=atol, rtol=rtol)
    if len({output.untyped_storage().data_ptr() for output in actual}) != 3:
        raise AssertionError("state-update outputs must not alias each other")
    for name, snapshot in snapshots.items():
        if not torch.equal(getattr(operands, name), snapshot):
            raise AssertionError(f"fused state update mutated {name}")


def profile_gdn_chunk_state_update_vllm_triton(
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
    """Profile vLLM's fused chunk-state-update launch on an NVIDIA H200."""
    args = _validate_args(
        num_tokens,
        num_chunks,
        num_sequences,
        max_chunks_per_sequence,
        num_key_heads,
        num_heads,
        key_head_dim,
        value_head_dim,
        dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc

    try:
        _require_h200(torch)
        fused_callable = _load_fused_callable()
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, args, device=device)
        _validate_operands(torch, operands, args)
        guard_args = _guard_args(args)
        guard_operands = (
            operands if guard_args == args else _build_operands(torch, guard_args, device=device)
        )
        # The guard call also warms the shared H/K/V/BT autotune key.
        _check_correctness(torch, fused_callable, guard_operands, guard_args)

        def kernel() -> tuple[Any, Any, Any]:
            return _invoke_fused(fused_callable, operands)

        # One wrapper invocation per call. CUPTI selects only the fused state
        # update (never downstream chunk_fwd_kernel_o). No reset is needed:
        # inputs are immutable and all three outputs are freshly allocated.
        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

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
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / elapsed_s / 1e12 if time_ms > 0.0 else 0.0,
        memory_bandwidth_gbps=(logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0),
    )
