"""One-launch vLLM Triton runner for Qwen GDN gated RMS normalization.

The timed callable is exactly vLLM's fused gated RMSNorm wrapper. Operand
allocation and the semantic correctness guard remain outside CUPTI and energy
measurement. Reported FLOPs and bytes are semantic logical counts, not physical
Triton instructions or device traffic.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import load_required_callable, require_exact_gpu
from profiling.runners.attention.gdn_gated_rms_norm_torch import (
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gdn_gated_rms_norm:vllm_triton"
_CALLABLE_MODULE = "vllm.model_executor.layers.fla.ops.layernorm_guard"
_CALLABLE_NAME = "rmsnorm_fn"
_KERNEL_NAME = "layer_norm_fwd_kernel"
_REQUIRED_GPU = "NVIDIA H200"
_MAX_BF16_HIDDEN = 32768
_GUARD_MAX_ROWS = 512
_GUARD_MAX_ELEMENTS = 65536
_OUTPUT_ATOL = 1e-2
_OUTPUT_RTOL = 1e-2


@dataclass(frozen=True)
class _ValidatedArgs:
    m: int
    hidden: int
    dtype: DType


@dataclass(frozen=True)
class _OperandShapes:
    x: tuple[int, int]
    z: tuple[int, int]
    weight: tuple[int]
    output: tuple[int, int]


@dataclass(frozen=True)
class _Operands:
    x: Any
    z: Any
    weight: Any


def _validate_args(m: int, hidden: int, dtype: DType | str) -> _ValidatedArgs:
    validated = _ValidatedArgs(
        m=int(m),
        hidden=int(hidden),
        dtype=DType.from_value(dtype),
    )
    if validated.m <= 0 or validated.hidden <= 0:
        raise ValueError(
            f"m and hidden must be > 0, got m={validated.m}, hidden={validated.hidden}"
        )
    if validated.dtype is not DType.BF16:
        raise ValueError(
            f"vllm_triton gdn_gated_rms_norm requires dtype=bf16, got {validated.dtype.value}"
        )
    if validated.hidden > _MAX_BF16_HIDDEN:
        raise ValueError(
            "vllm_triton gdn_gated_rms_norm requires hidden<=32768 for the "
            f"64 KiB BF16 fused-feature limit, got {validated.hidden}"
        )
    return validated


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    matrix_shape = (args.m, args.hidden)
    return _OperandShapes(
        x=matrix_shape,
        z=matrix_shape,
        weight=(args.hidden,),
        output=matrix_shape,
    )


def _correctness_guard_args(args: _ValidatedArgs) -> _ValidatedArgs:
    """Keep the requested hidden size while bounding guard-only FP32 work."""
    max_rows_for_elements = max(1, _GUARD_MAX_ELEMENTS // args.hidden)
    guard_m = min(args.m, _GUARD_MAX_ROWS, max_rows_for_elements)
    return _ValidatedArgs(m=guard_m, hidden=args.hidden, dtype=args.dtype)


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
    x = torch.empty(shapes.x, dtype=activation_dtype, device=device).uniform_(
        -0.5, 0.5, generator=generator
    )
    z = torch.empty(shapes.z, dtype=activation_dtype, device=device).uniform_(
        -0.5, 0.5, generator=generator
    )
    weight = torch.empty(shapes.weight, dtype=activation_dtype, device=device).uniform_(
        0.75, 1.25, generator=generator
    )
    return _Operands(x=x.contiguous(), z=z.contiguous(), weight=weight.contiguous())


def _invoke_fused(fused_callable: Any, operands: _Operands) -> Any:
    return fused_callable(
        operands.x,
        operands.weight,
        None,
        z=operands.z,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
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
    """Compare fused output with the semantic reference without mutating inputs."""
    from profiling.runners.attention.gdn_gated_rms_norm_reference import (
        gdn_gated_rms_norm_reference,
    )

    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    snapshots = {
        "x": operands.x.clone(),
        "z": operands.z.clone(),
        "weight": operands.weight.clone(),
    }
    expected = gdn_gated_rms_norm_reference(snapshots["x"], snapshots["z"], snapshots["weight"])
    returned = _invoke_fused(fused_callable, operands)
    synchronize()

    if tuple(returned.shape) != _operand_shapes(args).output:
        raise AssertionError(f"unexpected fused output shape {tuple(returned.shape)}")
    if returned.dtype is not torch.bfloat16:
        raise AssertionError(f"unexpected fused output dtype {returned.dtype}")
    if not returned.is_contiguous():
        raise AssertionError("fused output must be contiguous")
    if any(_shares_storage(returned, getattr(operands, name)) for name in snapshots):
        raise AssertionError("fused output must have fresh storage")
    if not torch.isfinite(returned).all():
        raise AssertionError("fused output must remain finite")
    torch.testing.assert_close(
        returned.float(),
        expected.float(),
        atol=_OUTPUT_ATOL,
        rtol=_OUTPUT_RTOL,
    )
    for name, snapshot in snapshots.items():
        operand = getattr(operands, name)
        if not torch.equal(operand, snapshot):
            raise AssertionError(f"fused gated RMSNorm mutated {name}")
        if not operand.is_contiguous() or operand.stride(-1) != 1:
            raise AssertionError(f"{name} must be contiguous for a copy-free wrapper call")
        if operand.contiguous().data_ptr() != operand.data_ptr():
            raise AssertionError(f"{name} would require a wrapper copy")


def profile_gdn_gated_rms_norm_vllm_triton(
    m: int,
    hidden: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile vLLM's fused gated RMSNorm on an NVIDIA H200."""
    args = _validate_args(m=m, hidden=hidden, dtype=dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc

    try:
        _require_h200(torch)
        fused_callable = _load_fused_callable()
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, args, device=device)
        guard_args = _correctness_guard_args(args)
        guard_operands = (
            operands if guard_args == args else _build_operands(torch, guard_args, device=device)
        )
        _check_correctness(torch, fused_callable, guard_operands, guard_args)

        def kernel() -> Any:
            return _invoke_fused(fused_callable, operands)

        # Each measurement callable contains only rmsnorm_fn. Its output/rstd
        # allocations are part of the upstream wrapper and do not add another
        # GPU launch. Inputs are immutable, so no reset is needed.
        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    flops = _semantic_flops(m=args.m, hidden=args.hidden)
    logical_bytes = _logical_bytes(m=args.m, hidden=args.hidden, dtype=args.dtype)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / elapsed_s / 1e12 if time_ms > 0.0 else 0.0,
        memory_bandwidth_gbps=(logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0),
    )
