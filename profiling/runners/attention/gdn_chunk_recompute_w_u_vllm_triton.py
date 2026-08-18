"""One-launch vLLM Triton runner for Qwen GDN WY recomputation.

The measured callable is exactly vLLM's fused ``recompute_w_u_fwd`` wrapper
with precomputed canonical operands and ragged metadata. Construction, tensor
validation, the bounded CPU-reference guard, and autotune warmup stay outside
measurement. CUPTI selects only ``recompute_w_u_fwd_kernel``; energy covers the
whole wrapper call, including its two fresh output allocations. Reported FLOPs
and bytes are semantic/logical counts, not physical Triton work or traffic.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, replace
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import (
    canonical_balanced_chunk_boundaries as _canonical_boundaries,
)
from profiling.runners.attention._gdn_common import (
    canonical_single_chunk_index_pairs as _canonical_index_pairs,
)
from profiling.runners.attention._gdn_common import (
    exact_int as _exact_int,
)
from profiling.runners.attention._gdn_common import (
    load_required_callable,
    require_exact_gpu,
)
from profiling.runners.attention.gdn_chunk_recompute_w_u_torch import (
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

_BACKEND = "gdn_chunk_recompute_w_u:vllm_triton"
_CALLABLE_MODULE = "vllm.model_executor.layers.fla.ops.wy_fast"
_CALLABLE_NAME = "recompute_w_u_fwd"
_KERNEL_NAME = "recompute_w_u_fwd_kernel"
_REQUIRED_GPU = "NVIDIA H200"
_CHUNK_SIZE = 64
_SUPPORTED_HEAD_DIM = 128
_MAX_GUARD_ELEMENTS = 2_500_000
_ATOL = 1e-2
_RTOL = 1e-2


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
    k: tuple[int, int, int, int]
    v: tuple[int, int, int, int]
    beta: tuple[int, int, int]
    g_cumsum: tuple[int, int, int]
    A: tuple[int, int, int, int]
    cu_seqlens: tuple[int]
    chunk_indices: tuple[int, int]
    w: tuple[int, int, int, int]
    u: tuple[int, int, int, int]


@dataclass(frozen=True)
class _Operands:
    k: Any
    v: Any
    beta: Any
    g_cumsum: Any
    A: Any
    cu_seqlens: Any
    chunk_indices: Any
    boundaries: tuple[int, ...]


def _validate_fast_ops() -> None:
    value = os.environ.get("FLA_USE_FAST_OPS", "")
    if value.strip().lower() not in {"", "0", "false"}:
        raise ValueError(
            "vllm_triton gdn_chunk_recompute_w_u requires FLA_USE_FAST_OPS "
            f"to be absent, empty, false, or 0, got {value!r}"
        )


def _validate_args(
    num_tokens: int,
    num_chunks: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    _validate_fast_ops()
    validated = _ValidatedArgs(
        num_tokens=_exact_int("num_tokens", num_tokens),
        num_chunks=_exact_int("num_chunks", num_chunks),
        num_key_heads=_exact_int("num_key_heads", num_key_heads),
        num_heads=_exact_int("num_heads", num_heads),
        key_head_dim=_exact_int("key_head_dim", key_head_dim),
        value_head_dim=_exact_int("value_head_dim", value_head_dim),
        dtype=DType.from_value(dtype),
    )
    dimensions = (
        validated.num_tokens,
        validated.num_chunks,
        validated.num_key_heads,
        validated.num_heads,
        validated.key_head_dim,
        validated.value_head_dim,
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            "num_tokens, num_chunks, num_key_heads, num_heads, key_head_dim, and "
            f"value_head_dim must be > 0, got {dimensions}"
        )
    minimum_chunks = (validated.num_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    if validated.num_chunks < minimum_chunks or validated.num_chunks > validated.num_tokens:
        raise ValueError(
            "num_chunks must satisfy ceil(num_tokens/64) <= num_chunks <= num_tokens, "
            f"got num_tokens={validated.num_tokens}, num_chunks={validated.num_chunks}"
        )
    if validated.num_heads % validated.num_key_heads != 0:
        raise ValueError(
            "num_heads must be divisible by num_key_heads, got "
            f"num_heads={validated.num_heads}, num_key_heads={validated.num_key_heads}"
        )
    if validated.dtype is not DType.BF16:
        raise ValueError(
            f"vllm_triton gdn_chunk_recompute_w_u requires dtype=bf16, got {validated.dtype.value}"
        )
    if (
        validated.key_head_dim != _SUPPORTED_HEAD_DIM
        or validated.value_head_dim != _SUPPORTED_HEAD_DIM
    ):
        raise ValueError(
            "vllm_triton gdn_chunk_recompute_w_u requires "
            f"key_head_dim=value_head_dim={_SUPPORTED_HEAD_DIM}, got "
            f"{validated.key_head_dim} and {validated.value_head_dim}"
        )
    return validated


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    return _OperandShapes(
        k=(1, args.num_tokens, args.num_key_heads, args.key_head_dim),
        v=(1, args.num_tokens, args.num_heads, args.value_head_dim),
        beta=(1, args.num_tokens, args.num_heads),
        g_cumsum=(1, args.num_tokens, args.num_heads),
        A=(1, args.num_tokens, args.num_heads, _CHUNK_SIZE),
        cu_seqlens=(args.num_chunks + 1,),
        chunk_indices=(args.num_chunks, 2),
        w=(1, args.num_tokens, args.num_heads, args.key_head_dim),
        u=(1, args.num_tokens, args.num_heads, args.value_head_dim),
    )


def _guard_elements_per_token(args: _ValidatedArgs) -> int:
    # K + V + beta/g + solved-A + W + U, counted in logical tensor elements.
    return args.num_key_heads * args.key_head_dim + args.num_heads * (
        2 * args.value_head_dim + args.key_head_dim + _CHUNK_SIZE + 2
    )


def _guard_args(args: _ValidatedArgs) -> _ValidatedArgs:
    """Bound guard storage while preserving all static head/dimension axes.

    A two-token, one-chunk guard is the smallest geometry that exercises a
    nontrivial lower entry. Configurations whose static axes alone make that
    geometry exceed the cap are rejected explicitly.
    """
    elements_per_token = _guard_elements_per_token(args)
    if 2 * elements_per_token > _MAX_GUARD_ELEMENTS:
        raise ValueError("minimum nontrivial correctness guard exceeds the 2.5M-element cap")
    if (
        args.num_tokens * elements_per_token <= _MAX_GUARD_ELEMENTS
        and max(_canonical_lengths(args.num_tokens, args.num_chunks)) >= 2
    ):
        return args
    guard_tokens = max(2, _MAX_GUARD_ELEMENTS // elements_per_token)
    guard_chunks = min(args.num_chunks, guard_tokens - 1)
    minimum_chunks = (guard_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    guard_chunks = max(guard_chunks, minimum_chunks)
    return replace(args, num_tokens=guard_tokens, num_chunks=guard_chunks)


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
    boundaries = _canonical_boundaries(args.num_tokens, args.num_chunks)
    index_pairs = _canonical_index_pairs(args.num_chunks)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    def bounded(shape: tuple[int, ...], low: float, high: float, *, dtype: Any):
        return torch.empty(shape, dtype=dtype, device=device).uniform_(
            low, high, generator=generator
        )

    A = torch.zeros(shapes.A, dtype=torch.bfloat16, device=device)
    for start, end in zip(boundaries, boundaries[1:]):
        length = end - start
        values = bounded((length, args.num_heads, length), -0.125, 0.125, dtype=torch.float32)
        lower = torch.tril(values.permute(1, 0, 2), diagonal=-1).permute(1, 0, 2)
        A[0, start:end, :, :length].copy_(lower)
        rows = torch.arange(start, end, device=device)
        local_rows = torch.arange(length, device=device)
        A[0, rows, :, local_rows] = 1

    # Match production policy: build exact int32 metadata on CPU and transfer
    # it once before correctness/autotune warmup and measurement.
    cu_seqlens_cpu = torch.tensor(boundaries, dtype=torch.int32, device="cpu")
    chunk_indices_cpu = torch.tensor(index_pairs, dtype=torch.int32, device="cpu")
    return _Operands(
        k=bounded(shapes.k, -0.25, 0.25, dtype=torch.bfloat16).contiguous(),
        v=bounded(shapes.v, -0.25, 0.25, dtype=torch.bfloat16).contiguous(),
        beta=bounded(shapes.beta, -0.5, 0.5, dtype=torch.float32).contiguous(),
        g_cumsum=bounded(shapes.g_cumsum, -0.25, 0.25, dtype=torch.float32).contiguous(),
        A=A.contiguous(),
        cu_seqlens=cu_seqlens_cpu.to(device=device).contiguous(),
        chunk_indices=chunk_indices_cpu.to(device=device).contiguous(),
        boundaries=boundaries,
    )


def _validate_A_structure(torch: Any, A: Any, boundaries: tuple[int, ...]) -> None:
    for start, end in zip(boundaries, boundaries[1:]):
        length = end - start
        block = A[0, start:end, :, :length].permute(1, 0, 2)
        diagonal = block.diagonal(dim1=-2, dim2=-1)
        if not torch.equal(diagonal, torch.ones_like(diagonal)):
            raise ValueError("A must have exact unit diagonal in every canonical chunk")
        if torch.count_nonzero(torch.triu(block, diagonal=1)).item() != 0:
            raise ValueError("A must have exact-zero upper triangle")
        if torch.count_nonzero(A[0, start:end, :, length:]).item() != 0:
            raise ValueError("A must have exact-zero unused partial columns")


def _validate_operands(
    torch: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    require_cuda: bool = True,
) -> None:
    """Close upstream dtype, layout, device, metadata, and structure gaps."""
    shapes = _operand_shapes(args)
    expected = {
        "k": (shapes.k, torch.bfloat16),
        "v": (shapes.v, torch.bfloat16),
        "beta": (shapes.beta, torch.float32),
        "g_cumsum": (shapes.g_cumsum, torch.float32),
        "A": (shapes.A, torch.bfloat16),
        "cu_seqlens": (shapes.cu_seqlens, torch.int32),
        "chunk_indices": (shapes.chunk_indices, torch.int32),
    }
    devices = set()
    for name, (shape, dtype) in expected.items():
        tensor = getattr(operands, name)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype is not dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous() or tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous with unit feature stride")
        if require_cuda and not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        devices.add(tensor.device)
        if name not in {"cu_seqlens", "chunk_indices"} and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
    if len(devices) != 1:
        raise ValueError("K, V, gates, solved A, and metadata must be on one device")

    expected_boundaries = _canonical_boundaries(args.num_tokens, args.num_chunks)
    actual_boundaries = tuple(int(value) for value in operands.cu_seqlens.tolist())
    if operands.boundaries != expected_boundaries or actual_boundaries != expected_boundaries:
        raise ValueError("cu_seqlens must contain the exact canonical boundaries")
    if actual_boundaries[0] != 0 or actual_boundaries[-1] != args.num_tokens:
        raise ValueError("cu_seqlens endpoints must be 0 and num_tokens")
    if any(left >= right for left, right in zip(actual_boundaries, actual_boundaries[1:])):
        raise ValueError("cu_seqlens must be strictly increasing")

    expected_indices = _canonical_index_pairs(args.num_chunks)
    actual_indices = tuple(
        (int(sequence), int(local_chunk))
        for sequence, local_chunk in operands.chunk_indices.tolist()
    )
    if actual_indices != expected_indices:
        raise ValueError("chunk_indices must contain the exact canonical sequence/chunk mapping")
    _validate_A_structure(torch, operands.A, operands.boundaries)


def _invoke_fused(fused_callable: Any, operands: _Operands) -> tuple[Any, Any]:
    return fused_callable(
        operands.k,
        operands.v,
        operands.beta,
        operands.g_cumsum,
        operands.A,
        operands.cu_seqlens,
        operands.chunk_indices,
    )


def _shares_storage(left: Any, right: Any) -> bool:
    return (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.device == right.device
    )


def _check_correctness(
    torch: Any,
    fused_callable: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    synchronize: Any | None = None,
) -> None:
    """Warm the shared autotune key and check both outputs against the reference."""
    from profiling.runners.attention.gdn_chunk_recompute_w_u_reference import (
        gdn_chunk_recompute_w_u_reference,
    )

    _validate_operands(torch, operands, args, require_cuda=operands.k.is_cuda)
    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    names = ("k", "v", "beta", "g_cumsum", "A", "cu_seqlens", "chunk_indices")
    snapshots = {name: getattr(operands, name).clone() for name in names}
    expected_w, expected_u = gdn_chunk_recompute_w_u_reference(
        snapshots["k"].squeeze(0).cpu(),
        snapshots["v"].squeeze(0).cpu(),
        snapshots["beta"].squeeze(0).cpu(),
        snapshots["g_cumsum"].squeeze(0).cpu(),
        snapshots["A"].squeeze(0).cpu(),
        snapshots["cu_seqlens"].cpu(),
    )

    w, u = _invoke_fused(fused_callable, operands)
    synchronize()
    shapes = _operand_shapes(args)
    inputs = tuple(getattr(operands, name) for name in names)
    for name, output, expected, shape in (
        ("w", w, expected_w, shapes.w),
        ("u", u, expected_u, shapes.u),
    ):
        if tuple(output.shape) != shape:
            raise AssertionError(f"unexpected {name} shape {tuple(output.shape)}")
        if output.dtype is not torch.bfloat16:
            raise AssertionError(f"unexpected {name} dtype {output.dtype}")
        if not output.is_contiguous() or not torch.isfinite(output).all():
            raise AssertionError(f"{name} must be finite and contiguous")
        if any(_shares_storage(output, tensor) for tensor in inputs):
            raise AssertionError(f"{name} must have fresh non-aliased storage")
        torch.testing.assert_close(output.squeeze(0).cpu(), expected, atol=_ATOL, rtol=_RTOL)
    if _shares_storage(w, u):
        raise AssertionError("W and U must not alias each other")
    for name, snapshot in snapshots.items():
        if not torch.equal(getattr(operands, name), snapshot):
            raise AssertionError(f"fused WY recomputation mutated {name}")


def profile_gdn_chunk_recompute_w_u_vllm_triton(
    num_tokens: int,
    num_chunks: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile vLLM's fused WY recomputation launch on an NVIDIA H200."""
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
        # The guard call warms H/K/V/BT/BK/BV/varlen. T/C are not autotune keys.
        _check_correctness(torch, fused_callable, guard_operands, guard_args)

        def kernel() -> tuple[Any, Any]:
            return _invoke_fused(fused_callable, operands)

        # One wrapper invocation per logical call. CUPTI includes only the WY
        # compute kernel. Inputs are immutable and W/U are freshly allocated,
        # so there is no reset. Energy includes the two wrapper allocations.
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
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / elapsed_s / 1e12 if time_ms > 0.0 else 0.0,
        memory_bandwidth_gbps=(logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0),
    )
