"""Torch semantic-composite runner for Qwen GDN gated RMS normalization.

This backend times every launch made by the standalone Torch reference. It is
not vLLM's fused Triton normalization kernel, and its logical traffic/FLOP rates
do not describe a fused kernel's physical implementation. Production simulation
must select the fused backend once one is registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


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
    m = int(m)
    hidden = int(hidden)
    dtype = DType.from_value(dtype)
    if m <= 0 or hidden <= 0:
        raise ValueError(f"m and hidden must be > 0, got m={m}, hidden={hidden}")
    if dtype is not DType.BF16:
        raise ValueError(f"torch gdn_gated_rms_norm requires dtype=bf16, got {dtype.value}")
    return _ValidatedArgs(m=m, hidden=hidden, dtype=dtype)


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch gdn_gated_rms_norm backend")


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    matrix_shape = (args.m, args.hidden)
    return _OperandShapes(
        x=matrix_shape,
        z=matrix_shape,
        weight=(args.hidden,),
        output=matrix_shape,
    )


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    # Bounded deterministic operands keep FP32 statistics and SiLU finite.
    x = torch.empty(shapes.x, dtype=args.dtype.torch(), device=device).uniform_(
        -0.5,
        0.5,
        generator=generator,
    )
    z = torch.empty(shapes.z, dtype=args.dtype.torch(), device=device).uniform_(
        -0.5,
        0.5,
        generator=generator,
    )
    weight = torch.empty(
        shapes.weight,
        dtype=args.dtype.torch(),
        device=device,
    ).uniform_(0.75, 1.25, generator=generator)
    return _Operands(x=x, z=z, weight=weight)


def _semantic_flops(*, m: int, hidden: int) -> int:
    """Nominal semantic FLOPs, not physical Torch instruction count.

    Per element: square, two RMS/affine multiplies, sigmoid, SiLU multiply,
    and output-gate multiply. Per row: ``hidden-1`` reduction additions plus
    mean division, epsilon addition, and rsqrt. Casts count as zero.
    """
    return 7 * m * hidden + 2 * m


def _logical_bytes(*, m: int, hidden: int, dtype: DType) -> float:
    """Logical boundary traffic, excluding Torch FP32 intermediates.

    Counts reads of x, z, and the shared weight plus one output write. Allocation
    and all composite temporaries are intentionally excluded.
    """
    elements = 3 * m * hidden + hidden
    return elements * dtype.size_bytes()


def profile_gdn_gated_rms_norm(
    m: int,
    hidden: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile the complete multi-launch Torch gated-RMSNorm reference."""
    args = _validate_args(m=m, hidden=hidden, dtype=dtype)
    try:
        import torch

        from profiling.runners.attention.gdn_gated_rms_norm_reference import (
            gdn_gated_rms_norm_reference,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch and the semantic reference are required for the torch gdn_gated_rms_norm backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(torch, args, device=torch.device("cuda"))

        def kernel():
            return gdn_gated_rms_norm_reference(
                operands.x,
                operands.z,
                operands.weight,
            )

        # No kernel-name filter: the measured object is the full multi-launch
        # semantic composite. Inputs are immutable, so no reset is required.
        # Operand allocation remains outside both measurement regions.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        flops = _semantic_flops(m=args.m, hidden=args.hidden)
        logical_bytes = _logical_bytes(m=args.m, hidden=args.hidden, dtype=args.dtype)
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
