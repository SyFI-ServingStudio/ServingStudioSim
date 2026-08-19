"""One-launch vLLM Triton runner for Qwen GDN triangular inversion.

The measured callable is exactly vLLM's BT64 ``solve_tril`` wrapper with
precomputed canonical metadata. Operand construction, metadata preparation,
the semantic correctness guard, and autotune warmup remain outside CUPTI and
energy measurement. CUPTI selects only the solve kernel; the wrapper's output
allocation/fill is not part of the kernel-only duration. Reported FLOPs and
bytes are semantic logical counts, not physical Triton instructions or traffic.
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
    canonical_single_chunk_index_pairs as _canonical_index_pairs,
)
from profiling.runners.attention._gdn_common import (
    exact_int as _exact_int,
)
from profiling.runners.attention._gdn_common import (
    load_required_callable,
    require_exact_gpu,
)
from profiling.runners.attention.gdn_chunk_solve_tril_torch import (
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

_BACKEND = "gdn_chunk_solve_tril:vllm_triton"
_CALLABLE_MODULE = "vllm.model_executor.layers.fla.ops.solve_tril"
_CALLABLE_NAME = "solve_tril"
_KERNEL_NAME = "merge_16x16_to_64x64_inverse_kernel"
_REQUIRED_GPU = "NVIDIA H200"
_CHUNK_SIZE = 64
_MAX_GUARD_ELEMENTS = 1_048_576
_ATOL = 1e-2
_RTOL = 1e-2


@dataclass(frozen=True)
class _ValidatedArgs:
    num_tokens: int
    num_chunks: int
    max_chunk_tokens: int
    num_heads: int
    dtype: DType


@dataclass(frozen=True)
class _OperandShapes:
    A: tuple[int, int, int, int]
    cu_seqlens: tuple[int]
    chunk_indices: tuple[int, int]
    output: tuple[int, int, int, int]


@dataclass(frozen=True)
class _Operands:
    A: Any
    cu_seqlens: Any
    chunk_indices: Any
    boundaries: tuple[int, ...]


def _validate_fla_environment() -> None:
    precision = os.environ.get("FLA_TRIL_PRECISION")
    if precision not in {None, "", "ieee"}:
        raise ValueError(
            "vllm_triton gdn_chunk_solve_tril requires FLA_TRIL_PRECISION "
            f"to be absent, empty, or ieee, got {precision!r}"
        )
    for name in ("FLA_USE_TMA", "FLA_USE_FAST_OPS"):
        value = os.environ.get(name)
        if value not in {None, "", "0"}:
            raise ValueError(
                f"vllm_triton gdn_chunk_solve_tril requires {name} "
                f"to be absent, empty, or 0, got {value!r}"
            )


def _validate_args(
    num_tokens: int,
    num_chunks: int,
    max_chunk_tokens: int,
    num_heads: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    _validate_fla_environment()
    validated = _ValidatedArgs(
        num_tokens=_exact_int("num_tokens", num_tokens),
        num_chunks=_exact_int("num_chunks", num_chunks),
        max_chunk_tokens=_exact_int("max_chunk_tokens", max_chunk_tokens),
        num_heads=_exact_int("num_heads", num_heads),
        dtype=DType.from_value(dtype),
    )
    dimensions = (
        validated.num_tokens,
        validated.num_chunks,
        validated.max_chunk_tokens,
        validated.num_heads,
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            f"num_tokens, num_chunks, max_chunk_tokens, and num_heads must be > 0, got {dimensions}"
        )
    if validated.max_chunk_tokens > _CHUNK_SIZE:
        raise ValueError(
            f"max_chunk_tokens must be <= {_CHUNK_SIZE}, got {validated.max_chunk_tokens}"
        )
    minimum_tokens = validated.max_chunk_tokens + validated.num_chunks - 1
    maximum_tokens = validated.num_chunks * validated.max_chunk_tokens
    if validated.num_tokens < minimum_tokens or validated.num_tokens > maximum_tokens:
        raise ValueError(
            "feasible (T,C,M) requires M+C-1 <= T <= C*M, got "
            f"T={validated.num_tokens}, C={validated.num_chunks}, "
            f"M={validated.max_chunk_tokens}"
        )
    if validated.dtype is not DType.BF16:
        raise ValueError(
            f"vllm_triton gdn_chunk_solve_tril requires dtype=bf16, got {validated.dtype.value}"
        )
    return validated


def _canonical_boundaries(args: _ValidatedArgs) -> tuple[int, ...]:
    return (
        0,
        *accumulate(
            _canonical_lengths(
                args.num_tokens,
                args.num_chunks,
                args.max_chunk_tokens,
            )
        ),
    )


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    shape = (1, args.num_tokens, args.num_heads, _CHUNK_SIZE)
    return _OperandShapes(
        A=shape,
        cu_seqlens=(args.num_chunks + 1,),
        chunk_indices=(args.num_chunks, 2),
        output=shape,
    )


def _guard_args(args: _ValidatedArgs) -> _ValidatedArgs:
    """Bound A+output elements while preserving H and feasible nontrivial work."""
    elements_per_token = 2 * _CHUNK_SIZE * args.num_heads
    if args.num_tokens * elements_per_token <= _MAX_GUARD_ELEMENTS:
        return args

    # Two tokens are the smallest nontrivial strict-lower solve. For realistic
    # supported head counts this remains below the cap; keeping the floor makes
    # the rule deterministic even for unusually large requested H.
    guard_tokens = max(2, _MAX_GUARD_ELEMENTS // elements_per_token)
    guard_max = min(args.max_chunk_tokens, guard_tokens)
    minimum_chunks = (guard_tokens + guard_max - 1) // guard_max
    maximum_chunks = guard_tokens - guard_max + 1
    guard_chunks = min(max(args.num_chunks, minimum_chunks), maximum_chunks)
    return replace(
        args,
        num_tokens=guard_tokens,
        num_chunks=guard_chunks,
        max_chunk_tokens=guard_max,
    )


def _require_h200(torch: Any) -> None:
    require_exact_gpu(torch, backend=_BACKEND, required_gpu=_REQUIRED_GPU)


def _load_fused_callable() -> Any:
    # The upstream module treats an absent precision variable as IEEE but an
    # explicitly empty value as invalid. Normalize the contract's empty/default
    # spelling inside this isolated worker before the first vLLM import.
    if os.environ.get("FLA_TRIL_PRECISION") == "":
        os.environ.pop("FLA_TRIL_PRECISION")
    return load_required_callable(
        importlib.import_module,
        backend=_BACKEND,
        module_name=_CALLABLE_MODULE,
        callable_name=_CALLABLE_NAME,
    )


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    boundaries = _canonical_boundaries(args)
    index_pairs = _canonical_index_pairs(args.num_chunks)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    A = torch.zeros(shapes.A, dtype=torch.float32, device=device)
    for start, end in zip(boundaries, boundaries[1:]):
        length = end - start
        values = torch.empty(
            (length, args.num_heads, length),
            dtype=torch.float32,
            device=device,
        ).uniform_(-0.02, 0.02, generator=generator)
        lower = torch.tril(values.permute(1, 0, 2), diagonal=-1).permute(1, 0, 2)
        A[0, start:end, :, :length].copy_(lower)

    # Materialize audited metadata on CPU and transfer it once, before any
    # guard, autotuning, or measured invocation.
    cu_cpu = torch.tensor(boundaries, dtype=torch.int32, device="cpu")
    indices_cpu = torch.tensor(index_pairs, dtype=torch.int32, device="cpu")
    return _Operands(
        A=A.contiguous(),
        cu_seqlens=cu_cpu.to(device=device).contiguous(),
        chunk_indices=indices_cpu.to(device=device).contiguous(),
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
        "A": (shapes.A, torch.float32),
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
            raise ValueError(f"{name} must be contiguous with unit last-dimension stride")
        if require_cuda and not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        devices.add(tensor.device)
    if len(devices) != 1:
        raise ValueError("A and chunk metadata must be on one device")

    expected_boundaries = _canonical_boundaries(args)
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
    if not torch.isfinite(operands.A).all():
        raise ValueError("A must contain only finite values")
    for start, end in zip(expected_boundaries, expected_boundaries[1:]):
        length = end - start
        local = operands.A[0, start:end]
        rows = torch.arange(length, device=local.device)
        columns = torch.arange(_CHUNK_SIZE, device=local.device)
        forbidden = columns.unsqueeze(0) >= rows.unsqueeze(1)
        if torch.count_nonzero(local.masked_select(forbidden.unsqueeze(1))).item() != 0:
            raise ValueError(
                "A must be exactly strict-lower; diagonal, upper, and unused columns must be zero"
            )


def _invoke_fused(torch: Any, fused_callable: Any, operands: _Operands) -> Any:
    return fused_callable(
        A=operands.A,
        cu_seqlens=operands.cu_seqlens,
        chunk_indices=operands.chunk_indices,
        output_dtype=torch.bfloat16,
    )


def _shares_storage(left: Any, right: Any) -> bool:
    return (
        left.data_ptr() == right.data_ptr()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.device == right.device
    )


def _check_output_structure(torch: Any, output: Any, boundaries: tuple[int, ...]) -> None:
    for start, end in zip(boundaries, boundaries[1:]):
        length = end - start
        local = output[0, start:end]
        rows = torch.arange(length, device=output.device)
        if not torch.equal(
            local[rows, :, rows],
            torch.ones((length, output.shape[2]), dtype=torch.bfloat16, device=output.device),
        ):
            raise AssertionError("fused inverse diagonal must be exactly one")
        columns = torch.arange(_CHUNK_SIZE, device=output.device)
        forbidden = columns.unsqueeze(0) > rows.unsqueeze(1)
        if torch.count_nonzero(local.masked_select(forbidden.unsqueeze(1))).item() != 0:
            raise AssertionError("fused inverse upper and unused columns must be exactly zero")


def _check_correctness(
    torch: Any,
    fused_callable: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    synchronize: Any | None = None,
) -> None:
    """Warm the autotune key and compare with the accepted CPU reference."""
    from profiling.runners.attention.gdn_chunk_solve_tril_reference import (
        gdn_chunk_solve_tril_reference,
    )

    _validate_operands(torch, operands, args, require_cuda=operands.A.is_cuda)
    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    A_snapshot = operands.A.clone()
    cu_snapshot = operands.cu_seqlens.clone()
    indices_snapshot = operands.chunk_indices.clone()
    expected = gdn_chunk_solve_tril_reference(
        A_snapshot.squeeze(0).cpu(),
        cu_snapshot.cpu(),
    )

    output = _invoke_fused(torch, fused_callable, operands)
    synchronize()
    expected_shape = _operand_shapes(args).output
    if tuple(output.shape) != expected_shape:
        raise AssertionError(f"unexpected fused output shape {tuple(output.shape)}")
    if output.dtype is not torch.bfloat16:
        raise AssertionError(f"unexpected fused output dtype {output.dtype}")
    if not output.is_contiguous():
        raise AssertionError("fused output must be contiguous")
    if not torch.isfinite(output).all():
        raise AssertionError("fused output must remain finite")
    if any(
        _shares_storage(output, tensor)
        for tensor in (operands.A, operands.cu_seqlens, operands.chunk_indices)
    ):
        raise AssertionError("fused output must have fresh non-aliased storage")
    torch.testing.assert_close(output.squeeze(0).cpu(), expected, atol=_ATOL, rtol=_RTOL)
    _check_output_structure(torch, output, operands.boundaries)
    for name, snapshot in (
        ("A", A_snapshot),
        ("cu_seqlens", cu_snapshot),
        ("chunk_indices", indices_snapshot),
    ):
        if not torch.equal(getattr(operands, name), snapshot):
            raise AssertionError(f"fused triangular solve mutated {name}")


def profile_gdn_chunk_solve_tril_vllm_triton(
    num_tokens: int,
    num_chunks: int,
    max_chunk_tokens: int,
    num_heads: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile vLLM's BT64 triangular-solve launch on an NVIDIA H200."""
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
        # The guard call also warms the shared (H, BT64, varlen) autotune key.
        # T is not specialized, so a bounded guard warms the measured route.
        _check_correctness(torch, fused_callable, guard_operands, guard_args)

        def kernel() -> Any:
            return _invoke_fused(torch, fused_callable, operands)

        # The wrapper allocates and zero-fills a fresh output, but CUPTI selects
        # exactly the BT64 inverse kernel. Inputs are immutable and no reset is
        # needed between calls.
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
        max_chunk_tokens=args.max_chunk_tokens,
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
