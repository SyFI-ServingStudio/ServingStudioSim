"""One-launch vLLM Triton runner for Qwen GDN chunk-local cumsum.

The measured callable is exactly vLLM's scalar chunk-local cumulative-sum
wrapper with precomputed canonical metadata. Allocation, metadata preparation,
the semantic correctness guard, and autotune warmup remain outside CUPTI and
energy measurement. Reported FLOPs and bytes are semantic logical counts, not
physical Triton instructions or device traffic.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
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
from profiling.runners.attention.gdn_chunk_local_cumsum_torch import (
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gdn_chunk_local_cumsum:vllm_triton"
_CALLABLE_MODULE = "vllm.model_executor.layers.fla.ops.cumsum"
_CALLABLE_NAME = "chunk_local_cumsum"
_KERNEL_NAME = "chunk_local_cumsum_scalar_kernel"
_REQUIRED_GPU = "NVIDIA H200"
_CHUNK_SIZE = 64
_MAX_GUARD_ELEMENTS = 1_048_576
_ATOL = 1e-6
_RTOL = 1e-6


@dataclass(frozen=True)
class _ValidatedArgs:
    num_tokens: int
    num_chunks: int
    num_heads: int
    dtype: DType


@dataclass(frozen=True)
class _OperandShapes:
    g: tuple[int, int, int]
    cu_seqlens: tuple[int]
    chunk_indices: tuple[int, int]
    output: tuple[int, int, int]


@dataclass(frozen=True)
class _Operands:
    g: Any
    cu_seqlens: Any
    chunk_indices: Any
    boundaries: tuple[int, ...]


def _validate_args(
    num_tokens: int,
    num_chunks: int,
    num_heads: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    validated = _ValidatedArgs(
        num_tokens=_exact_int("num_tokens", num_tokens),
        num_chunks=_exact_int("num_chunks", num_chunks),
        num_heads=_exact_int("num_heads", num_heads),
        dtype=DType.from_value(dtype),
    )
    if validated.num_tokens <= 0 or validated.num_chunks <= 0 or validated.num_heads <= 0:
        raise ValueError(
            "num_tokens, num_chunks, and num_heads must be > 0, got "
            f"num_tokens={validated.num_tokens}, num_chunks={validated.num_chunks}, "
            f"num_heads={validated.num_heads}"
        )
    minimum_chunks = (validated.num_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    if validated.num_chunks < minimum_chunks or validated.num_chunks > validated.num_tokens:
        raise ValueError(
            "num_chunks must satisfy ceil(num_tokens/64) <= num_chunks <= num_tokens, "
            f"got num_tokens={validated.num_tokens}, num_chunks={validated.num_chunks}"
        )
    if validated.dtype is not DType.FP32:
        raise ValueError(
            f"vllm_triton gdn_chunk_local_cumsum requires dtype=fp32, got {validated.dtype.value}"
        )
    return validated


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    return _OperandShapes(
        g=(1, args.num_tokens, args.num_heads),
        cu_seqlens=(args.num_chunks + 1,),
        chunk_indices=(args.num_chunks, 2),
        output=(1, args.num_tokens, args.num_heads),
    )


def _guard_args(args: _ValidatedArgs) -> _ValidatedArgs:
    """Preserve heads, cap elements, then retain a feasible canonical pair."""
    if args.num_tokens * args.num_heads <= _MAX_GUARD_ELEMENTS:
        return args
    guard_tokens = max(1, _MAX_GUARD_ELEMENTS // args.num_heads)
    guard_chunks = min(args.num_chunks, guard_tokens)
    minimum_chunks = (guard_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    if guard_chunks < minimum_chunks:
        # This is unreachable for a feasible requested pair because guard T is
        # no larger than requested T, but retain a deterministic safety seam.
        guard_chunks = minimum_chunks
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
    g = torch.empty(shapes.g, dtype=torch.float32, device=device).uniform_(
        -0.125,
        -0.001,
        generator=generator,
    )
    # Match Qwen metadata preparation: construct int32 metadata on CPU once,
    # then copy it to the selected CUDA device before any measured invocation.
    cu_seqlens_cpu = torch.tensor(boundaries, dtype=torch.int32, device="cpu")
    chunk_indices_cpu = torch.tensor(index_pairs, dtype=torch.int32, device="cpu")
    return _Operands(
        g=g.contiguous(),
        cu_seqlens=cu_seqlens_cpu.to(device=device).contiguous(),
        chunk_indices=chunk_indices_cpu.to(device=device).contiguous(),
        boundaries=boundaries,
    )


def _validate_operands(
    torch: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    require_cuda: bool = True,
) -> None:
    """Close upstream's dtype, shape, layout, device, and metadata gaps."""
    shapes = _operand_shapes(args)
    expected = {
        "g": (shapes.g, torch.float32),
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
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if require_cuda and not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        devices.add(tensor.device)
    if len(devices) != 1:
        raise ValueError("g and chunk metadata must be on one device")

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


def _invoke_fused(fused_callable: Any, operands: _Operands) -> Any:
    return fused_callable(
        operands.g,
        chunk_size=_CHUNK_SIZE,
        reverse=False,
        cu_seqlens=operands.cu_seqlens,
        chunk_indices=operands.chunk_indices,
        head_first=False,
        output_dtype=operands.g.dtype,
    )


def _shares_storage(left: Any, right: Any) -> bool:
    return (
        left.data_ptr() == right.data_ptr()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
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
    """Warm the autotune key and compare with the accepted CPU reference."""
    from profiling.runners.attention.gdn_chunk_local_cumsum_reference import (
        gdn_chunk_local_cumsum_reference,
    )

    _validate_operands(torch, operands, args, require_cuda=operands.g.is_cuda)
    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    g_snapshot = operands.g.clone()
    cu_snapshot = operands.cu_seqlens.clone()
    indices_snapshot = operands.chunk_indices.clone()
    expected = gdn_chunk_local_cumsum_reference(
        g_snapshot.squeeze(0).cpu(),
        cu_snapshot.cpu(),
    ).unsqueeze(0)

    output = _invoke_fused(fused_callable, operands)
    synchronize()
    shapes = _operand_shapes(args)
    if tuple(output.shape) != shapes.output:
        raise AssertionError(f"unexpected fused output shape {tuple(output.shape)}")
    if output.dtype is not torch.float32:
        raise AssertionError(f"unexpected fused output dtype {output.dtype}")
    if not output.is_contiguous():
        raise AssertionError("fused output must be contiguous")
    if not torch.isfinite(output).all():
        raise AssertionError("fused output must remain finite")
    if any(
        _shares_storage(output, tensor)
        for tensor in (operands.g, operands.cu_seqlens, operands.chunk_indices)
    ):
        raise AssertionError("fused output must have fresh non-aliased storage")
    torch.testing.assert_close(output.cpu(), expected, atol=_ATOL, rtol=_RTOL)
    if not torch.equal(operands.g, g_snapshot):
        raise AssertionError("fused cumsum mutated g")
    if not torch.equal(operands.cu_seqlens, cu_snapshot):
        raise AssertionError("fused cumsum mutated cu_seqlens")
    if not torch.equal(operands.chunk_indices, indices_snapshot):
        raise AssertionError("fused cumsum mutated chunk_indices")


def profile_gdn_chunk_local_cumsum_vllm_triton(
    num_tokens: int,
    num_chunks: int,
    num_heads: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile vLLM's fused scalar chunk-local cumsum on an H200."""
    args = _validate_args(
        num_tokens=num_tokens,
        num_chunks=num_chunks,
        num_heads=num_heads,
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
        # The guard call is also the autotune warmup. It preserves H, while T/C
        # are not autotune keys, so it warms the measured invocation's key.
        _check_correctness(torch, fused_callable, guard_operands, guard_args)

        def kernel() -> Any:
            return _invoke_fused(fused_callable, operands)

        # Each callable contains only the one vLLM wrapper call. CUPTI selects
        # only the scalar Triton kernel. The wrapper's fresh output allocation
        # is internal; immutable inputs mean no reset is needed.
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
    )
    logical_bytes = _logical_bytes(
        num_tokens=args.num_tokens,
        num_heads=args.num_heads,
    )
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / elapsed_s / 1e12 if time_ms > 0.0 else 0.0,
        memory_bandwidth_gbps=(logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0),
    )
