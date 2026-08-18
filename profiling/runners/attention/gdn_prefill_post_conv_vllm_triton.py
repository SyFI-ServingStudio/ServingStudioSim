"""One-launch vLLM Triton runner for Qwen GDN prefill post-conv prep.

The timed callable is exactly vLLM's fused post-convolution wrapper. Operand
allocation and the semantic correctness guard remain outside CUPTI and energy
measurement. Reported FLOPs and bytes are semantic logical counts, not physical
Triton instructions or device traffic.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import load_required_callable, require_exact_gpu
from profiling.runners.attention.gdn_prefill_post_conv_torch import (
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gdn_prefill_post_conv:vllm_triton"
_CALLABLE_MODULE = "vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv"
_CALLABLE_NAME = "fused_post_conv_prep"
_KERNEL_NAME = "_fused_post_conv_kernel"
_REQUIRED_GPU = "NVIDIA H200"
_SUPPORTED_HEAD_DIM_PAIRS = frozenset({(64, 64), (128, 128)})
_MAX_GUARD_PACKED_ELEMENTS = 1_048_576
_QK_ATOL = 1e-3
_QK_RTOL = 1e-2
_GATE_ATOL = 1e-6
_GATE_RTOL = 1e-6


@dataclass(frozen=True)
class _ValidatedArgs:
    num_tokens: int
    num_qk_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: DType


@dataclass(frozen=True)
class _OperandShapes:
    conv_output: tuple[int, int]
    a: tuple[int, int]
    b: tuple[int, int]
    A_log: tuple[int]
    dt_bias: tuple[int]
    q: tuple[int, int, int]
    k: tuple[int, int, int]
    v: tuple[int, int, int]
    g: tuple[int, int]
    beta: tuple[int, int]


@dataclass(frozen=True)
class _Operands:
    conv_output: Any
    a: Any
    b: Any
    A_log: Any
    dt_bias: Any


def _validate_args(
    num_tokens: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    validated = _ValidatedArgs(
        num_tokens=int(num_tokens),
        num_qk_heads=int(num_qk_heads),
        num_value_heads=int(num_value_heads),
        key_head_dim=int(key_head_dim),
        value_head_dim=int(value_head_dim),
        dtype=DType.from_value(dtype),
    )
    dimensions = (
        validated.num_tokens,
        validated.num_qk_heads,
        validated.num_value_heads,
        validated.key_head_dim,
        validated.value_head_dim,
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            "num_tokens, num_qk_heads, num_value_heads, key_head_dim, and "
            f"value_head_dim must be > 0, got {dimensions}"
        )
    if validated.dtype is not DType.BF16:
        raise ValueError(
            f"vllm_triton gdn_prefill_post_conv requires dtype=bf16, got {validated.dtype.value}"
        )
    head_dims = (validated.key_head_dim, validated.value_head_dim)
    if head_dims not in _SUPPORTED_HEAD_DIM_PAIRS:
        supported = sorted(_SUPPORTED_HEAD_DIM_PAIRS)
        raise ValueError(
            "vllm_triton gdn_prefill_post_conv requires an established "
            f"(key_head_dim, value_head_dim) pair in {supported}, got {head_dims}"
        )
    return validated


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    packed_width = (
        2 * args.num_qk_heads * args.key_head_dim + args.num_value_heads * args.value_head_dim
    )
    gates = (args.num_tokens, args.num_value_heads)
    return _OperandShapes(
        conv_output=(args.num_tokens, packed_width),
        a=gates,
        b=gates,
        A_log=(args.num_value_heads,),
        dt_bias=(args.num_value_heads,),
        q=(args.num_tokens, args.num_qk_heads, args.key_head_dim),
        k=(args.num_tokens, args.num_qk_heads, args.key_head_dim),
        v=(args.num_tokens, args.num_value_heads, args.value_head_dim),
        g=gates,
        beta=gates,
    )


def _guard_args(args: _ValidatedArgs) -> _ValidatedArgs:
    """Preserve head geometry and bound only guard rows by packed elements."""
    packed_width = _operand_shapes(args).conv_output[1]
    max_tokens = max(1, _MAX_GUARD_PACKED_ELEMENTS // packed_width)
    return replace(args, num_tokens=min(args.num_tokens, max_tokens))


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
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    activation_dtype = args.dtype.torch()

    def bounded(shape: tuple[int, ...], low: float, high: float, *, dtype: Any):
        return torch.empty(shape, dtype=dtype, device=device).uniform_(
            low,
            high,
            generator=generator,
        )

    return _Operands(
        conv_output=bounded(
            shapes.conv_output,
            -0.5,
            0.5,
            dtype=activation_dtype,
        ).contiguous(),
        a=bounded(shapes.a, -0.5, 0.5, dtype=activation_dtype).contiguous(),
        b=bounded(shapes.b, -0.5, 0.5, dtype=activation_dtype).contiguous(),
        A_log=bounded(
            shapes.A_log,
            -0.25,
            0.25,
            dtype=torch.float32,
        ).contiguous(),
        dt_bias=bounded(
            shapes.dt_bias,
            -0.25,
            0.25,
            dtype=torch.float32,
        ).contiguous(),
    )


def _validate_operands(
    torch: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    require_cuda: bool = True,
) -> None:
    """Close upstream's shape/layout/dtype/device validation gaps."""
    shapes = _operand_shapes(args)
    expected = {
        "conv_output": (shapes.conv_output, torch.bfloat16),
        "a": (shapes.a, torch.bfloat16),
        "b": (shapes.b, torch.bfloat16),
        "A_log": (shapes.A_log, torch.float32),
        "dt_bias": (shapes.dt_bias, torch.float32),
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
    if len(devices) != 1:
        raise ValueError("all fused post-conv operands must be on one device")


def _invoke_fused(fused_callable: Any, operands: _Operands, args: _ValidatedArgs) -> Any:
    return fused_callable(
        operands.conv_output,
        operands.a,
        operands.b,
        operands.A_log,
        operands.dt_bias,
        num_k_heads=args.num_qk_heads,
        head_k_dim=args.key_head_dim,
        head_v_dim=args.value_head_dim,
        apply_l2norm=True,
        output_g_exp=False,
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
    """Compare all fused outputs with the accepted semantic reference."""
    from profiling.runners.attention.gdn_prefill_post_conv_reference import (
        gdn_prefill_post_conv_reference,
    )

    _validate_operands(torch, operands, args, require_cuda=operands.conv_output.is_cuda)
    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    names = ("conv_output", "a", "b", "A_log", "dt_bias")
    snapshots = {name: getattr(operands, name).clone() for name in names}
    expected = gdn_prefill_post_conv_reference(
        snapshots["conv_output"],
        snapshots["a"],
        snapshots["b"],
        snapshots["A_log"],
        snapshots["dt_bias"],
        num_qk_heads=args.num_qk_heads,
        num_value_heads=args.num_value_heads,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
    )
    returned = _invoke_fused(fused_callable, operands, args)
    synchronize()
    if not isinstance(returned, tuple) or len(returned) != 5:
        raise AssertionError("fused post-conv must return exactly five tensors")

    shapes = _operand_shapes(args)
    output_names = ("q", "k", "v", "g", "beta")
    expected_shapes = (shapes.q, shapes.k, shapes.v, shapes.g, shapes.beta)
    expected_dtypes = (
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
    )
    for name, output, shape, dtype in zip(
        output_names,
        returned,
        expected_shapes,
        expected_dtypes,
        strict=True,
    ):
        if tuple(output.shape) != shape:
            raise AssertionError(f"unexpected {name} shape {tuple(output.shape)}")
        if output.dtype is not dtype:
            raise AssertionError(f"unexpected {name} dtype {output.dtype}")
        if not output.is_contiguous():
            raise AssertionError(f"{name} output must be contiguous")
        if not torch.isfinite(output).all():
            raise AssertionError(f"{name} output must remain finite")

    for index, output in enumerate(returned):
        if any(_shares_storage(output, getattr(operands, name)) for name in names):
            raise AssertionError(f"{output_names[index]} output must have fresh storage")
        if any(_shares_storage(output, other) for other in returned[index + 1 :]):
            raise AssertionError("fused outputs must not alias one another")

    torch.testing.assert_close(
        returned[0].float(), expected[0].float(), atol=_QK_ATOL, rtol=_QK_RTOL
    )
    torch.testing.assert_close(
        returned[1].float(), expected[1].float(), atol=_QK_ATOL, rtol=_QK_RTOL
    )
    if not torch.equal(returned[2], expected[2]):
        raise AssertionError("fused V output must match the reference exactly")
    torch.testing.assert_close(returned[3], expected[3], atol=_GATE_ATOL, rtol=_GATE_RTOL)
    torch.testing.assert_close(returned[4], expected[4], atol=_GATE_ATOL, rtol=_GATE_RTOL)
    for name, snapshot in snapshots.items():
        if not torch.equal(getattr(operands, name), snapshot):
            raise AssertionError(f"fused post-conv mutated {name}")


def profile_gdn_prefill_post_conv_vllm_triton(
    num_tokens: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile vLLM's fused prefill post-conv launch on an NVIDIA H200."""
    args = _validate_args(
        num_tokens=num_tokens,
        num_qk_heads=num_qk_heads,
        num_value_heads=num_value_heads,
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
        _check_correctness(torch, fused_callable, guard_operands, guard_args)

        def kernel() -> Any:
            return _invoke_fused(fused_callable, operands, args)

        # Only fused_post_conv_prep runs in each measurement callable. Its five
        # output allocations are wrapper internals and add no neighboring GPU
        # compute launch. Inputs are immutable, so no reset is required.
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
        num_qk_heads=args.num_qk_heads,
        num_value_heads=args.num_value_heads,
        key_head_dim=args.key_head_dim,
    )
    logical_bytes = _logical_bytes(
        num_tokens=args.num_tokens,
        num_qk_heads=args.num_qk_heads,
        num_value_heads=args.num_value_heads,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
        dtype=args.dtype,
    )
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / elapsed_s / 1e12 if time_ms > 0.0 else 0.0,
        memory_bandwidth_gbps=(logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0),
    )
