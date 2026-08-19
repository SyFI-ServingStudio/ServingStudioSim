"""One-launch vLLM Triton runner for Qwen GDN chunk output."""

from __future__ import annotations

import importlib
import math
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
from profiling.runners.attention.gdn_chunk_output_torch import (
    _canonical_lengths,
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gdn_chunk_output:vllm_triton"
_MODULE = "vllm.model_executor.layers.fla.ops.chunk_o"
_CALLABLE = "chunk_fwd_o"
_KERNEL = "chunk_fwd_kernel_o"
_GPU = "NVIDIA H200"
_CHUNK_SIZE = 64
_HEAD_DIM = 128
_GUARD_CAP = 3_000_000
_ATOL = _RTOL = 1e-2


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
class _Operands:
    q: Any
    k: Any
    v_new: Any
    h: Any
    g_cumsum: Any
    cu_seqlens: Any
    chunk_indices: Any
    boundaries: tuple[int, ...]


def _validate_fast_ops() -> None:
    value = os.environ.get("FLA_USE_FAST_OPS", "")
    if value.strip().lower() not in {"", "0", "false"}:
        raise ValueError(f"{_BACKEND} requires FLA_USE_FAST_OPS absent/empty/false/0")


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
    args = _ValidatedArgs(
        _exact_int("num_tokens", num_tokens),
        _exact_int("num_chunks", num_chunks),
        _exact_int("num_key_heads", num_key_heads),
        _exact_int("num_heads", num_heads),
        _exact_int("key_head_dim", key_head_dim),
        _exact_int("value_head_dim", value_head_dim),
        DType.from_value(dtype),
    )
    dimensions = (
        args.num_tokens,
        args.num_chunks,
        args.num_key_heads,
        args.num_heads,
        args.key_head_dim,
        args.value_head_dim,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"all dimensions must be > 0, got {dimensions}")
    minimum = (args.num_tokens + 63) // 64
    if not minimum <= args.num_chunks <= args.num_tokens:
        raise ValueError("num_chunks must satisfy ceil(num_tokens/64) <= C <= T")
    if args.num_heads % args.num_key_heads:
        raise ValueError("num_heads must be divisible by num_key_heads")
    if args.dtype is not DType.BF16:
        raise ValueError(f"{_BACKEND} requires dtype=bf16")
    if args.key_head_dim != _HEAD_DIM or args.value_head_dim != _HEAD_DIM:
        raise ValueError(f"{_BACKEND} requires key_head_dim=value_head_dim=128")
    return args


def _boundaries(args: _ValidatedArgs) -> tuple[int, ...]:
    return (0, *accumulate(_canonical_lengths(args.num_tokens, args.num_chunks)))


def _index_pairs(args: _ValidatedArgs) -> tuple[tuple[int, int], ...]:
    return tuple(
        (sequence, local)
        for sequence, length in enumerate(_canonical_lengths(args.num_tokens, args.num_chunks))
        for local in range((length + 63) // 64)
    )


def _operand_shapes(args: _ValidatedArgs) -> dict[str, tuple[int, ...]]:
    sequences = len(_canonical_lengths(args.num_tokens, args.num_chunks))
    return {
        "q": (1, args.num_tokens, args.num_key_heads, args.key_head_dim),
        "k": (1, args.num_tokens, args.num_key_heads, args.key_head_dim),
        "v_new": (1, args.num_tokens, args.num_heads, args.value_head_dim),
        "h": (1, args.num_chunks, args.num_heads, args.value_head_dim, args.key_head_dim),
        "g_cumsum": (1, args.num_tokens, args.num_heads),
        "cu_seqlens": (sequences + 1,),
        "chunk_indices": (args.num_chunks, 2),
        "output": (1, args.num_tokens, args.num_heads, args.value_head_dim),
    }


def _guard_elements(args: _ValidatedArgs) -> int:
    shapes = _operand_shapes(args)
    return sum(math.prod(shapes[name]) for name in ("q", "k", "v_new", "h", "g_cumsum", "output"))


def _guard_args(args: _ValidatedArgs) -> _ValidatedArgs:
    qwen = (128, 2, 16, 32, 128, 128)
    if (
        args.num_tokens,
        args.num_chunks,
        args.num_key_heads,
        args.num_heads,
        args.key_head_dim,
        args.value_head_dim,
    ) == qwen:
        guard = args
    else:
        guard = replace(args, num_tokens=2, num_chunks=1)
    if _guard_elements(guard) > _GUARD_CAP:
        raise ValueError("bounded correctness witness exceeds 3M logical elements")
    return guard


def _require_h200(torch: Any) -> None:
    require_exact_gpu(torch, backend=_BACKEND, required_gpu=_GPU)


def _load_callable() -> Any:
    return load_required_callable(
        importlib.import_module,
        backend=_BACKEND,
        module_name=_MODULE,
        callable_name=_CALLABLE,
        environment_label="repository vllm_env",
        missing_message=f"missing {_MODULE}.{_CALLABLE}",
    )


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    boundaries = _boundaries(args)
    indices = _index_pairs(args)
    generator = torch.Generator(device=device).manual_seed(42)

    def bounded(name: str, dtype: Any):
        return (
            torch.empty(shapes[name], dtype=dtype, device=device)
            .uniform_(-0.125, 0.125, generator=generator)
            .contiguous()
        )

    return _Operands(
        q=bounded("q", torch.bfloat16),
        k=bounded("k", torch.bfloat16),
        v_new=bounded("v_new", torch.bfloat16),
        h=bounded("h", torch.bfloat16),
        g_cumsum=bounded("g_cumsum", torch.float32),
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device=device).contiguous(),
        chunk_indices=torch.tensor(indices, dtype=torch.int32, device=device).contiguous(),
        boundaries=boundaries,
    )


def _validate_operands(
    torch: Any, operands: _Operands, args: _ValidatedArgs, *, require_cuda: bool = True
) -> None:
    shapes = _operand_shapes(args)
    expected = {
        "q": torch.bfloat16,
        "k": torch.bfloat16,
        "v_new": torch.bfloat16,
        "h": torch.bfloat16,
        "g_cumsum": torch.float32,
        "cu_seqlens": torch.int32,
        "chunk_indices": torch.int32,
    }
    devices = set()
    for name, dtype in expected.items():
        tensor = getattr(operands, name)
        if tuple(tensor.shape) != shapes[name] or tensor.dtype is not dtype:
            raise ValueError(f"invalid {name} shape/dtype")
        if (
            (require_cuda and not tensor.is_cuda)
            or not tensor.is_contiguous()
            or tensor.stride(-1) != 1
        ):
            raise ValueError(f"{name} must be packed contiguous storage on the required device")
        if name not in {"cu_seqlens", "chunk_indices"} and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite")
        devices.add(tensor.device)
    if len(devices) != 1:
        raise ValueError("all operands and metadata must share one CUDA device")
    if tuple(operands.cu_seqlens.tolist()) != _boundaries(args):
        raise ValueError("cu_seqlens must contain exact canonical boundaries")
    if tuple(map(tuple, operands.chunk_indices.tolist())) != _index_pairs(args):
        raise ValueError("chunk_indices must contain exact canonical mapping")


def _invoke(callable_: Any, operands: _Operands) -> Any:
    return callable_(
        q=operands.q,
        k=operands.k,
        v=operands.v_new,
        h=operands.h,
        g=operands.g_cumsum,
        scale=operands.q.shape[-1] ** -0.5,
        cu_seqlens=operands.cu_seqlens,
        chunk_indices=operands.chunk_indices,
        chunk_size=_CHUNK_SIZE,
        core_attn_out=None,
    )


def _shares(left: Any, right: Any) -> bool:
    return (
        left.device == right.device
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _check_correctness(
    torch: Any,
    callable_: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    synchronize: Any | None = None,
) -> None:
    from profiling.runners.attention.gdn_chunk_output_reference import gdn_chunk_output_reference

    _validate_operands(torch, operands, args, require_cuda=operands.q.is_cuda)
    names = ("q", "k", "v_new", "h", "g_cumsum", "cu_seqlens", "chunk_indices")
    snapshots = {name: getattr(operands, name).clone() for name in names}
    expected = gdn_chunk_output_reference(
        snapshots["q"].squeeze(0).cpu(),
        snapshots["k"].squeeze(0).cpu(),
        snapshots["v_new"].squeeze(0).cpu(),
        snapshots["h"].squeeze(0).cpu(),
        snapshots["g_cumsum"].squeeze(0).cpu(),
        snapshots["cu_seqlens"].cpu(),
    )
    output = _invoke(callable_, operands)
    (synchronize or torch.cuda.synchronize)()
    if tuple(output.shape) != _operand_shapes(args)["output"]:
        raise AssertionError("unexpected output shape")
    if (
        output.dtype is not torch.bfloat16
        or not output.is_contiguous()
        or not torch.isfinite(output).all()
    ):
        raise AssertionError("output must be finite contiguous BF16")
    if any(_shares(output, getattr(operands, name)) for name in names):
        raise AssertionError("output must use fresh non-aliased storage")
    torch.testing.assert_close(output.squeeze(0).cpu(), expected, atol=_ATOL, rtol=_RTOL)
    for name, snapshot in snapshots.items():
        if not torch.equal(getattr(operands, name), snapshot):
            raise AssertionError(f"chunk output mutated {name}")


def profile_gdn_chunk_output_vllm_triton(
    num_tokens: int,
    num_chunks: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    args = _validate_args(
        num_tokens,
        num_chunks,
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
        callable_ = _load_callable()
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, args, device=device)
        _validate_operands(torch, operands, args)
        guard_args = _guard_args(args)
        guard = (
            operands if guard_args == args else _build_operands(torch, guard_args, device=device)
        )
        _check_correctness(torch, callable_, guard, guard_args)

        def kernel():
            return _invoke(callable_, operands)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL)
        # Energy covers the one wrapper call, including its fresh output allocation.
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
        num_chunks=args.num_chunks,
        num_key_heads=args.num_key_heads,
        num_heads=args.num_heads,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
    )
    elapsed = time_ms / 1000
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / elapsed / 1e12 if elapsed else 0,
        memory_bandwidth_gbps=logical_bytes / elapsed / 1e9 if elapsed else 0,
    )
