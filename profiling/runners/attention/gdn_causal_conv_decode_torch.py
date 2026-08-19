"""Torch semantic-composite runner for Qwen GDN causal-convolution decode.

This backend times every launch made by the standalone Torch reference. It is
not vLLM's fused causal-convolution update kernel, and its logical traffic/FLOP
rates do not describe the fused kernel's physical implementation. Production
simulation must select the fused backend once one is registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SUPPORTED_KERNEL_SIZES = frozenset(range(2, 7))


@dataclass(frozen=True)
class _ValidatedArgs:
    batch_size: int
    channels: int
    kernel_size: int
    dtype: DType
    state_dtype: DType


@dataclass(frozen=True)
class _OperandShapes:
    x: tuple[int, int]
    weight: tuple[int, int]
    state_storage: tuple[int, int, int]
    slot_indices: tuple[int]
    output: tuple[int, int]


@dataclass(frozen=True)
class _Operands:
    x: Any
    weight: Any
    state: Any
    slot_indices: Any


def _validate_args(
    batch_size: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> _ValidatedArgs:
    batch_size = int(batch_size)
    channels = int(channels)
    kernel_size = int(kernel_size)
    dtype = DType.from_value(dtype)
    state_dtype = DType.from_value(state_dtype)

    if batch_size <= 0 or channels <= 0:
        raise ValueError(
            f"batch_size and channels must be > 0, got batch_size={batch_size}, channels={channels}"
        )
    if kernel_size not in _SUPPORTED_KERNEL_SIZES:
        raise ValueError(
            "kernel_size must be supported by the production Triton kernel "
            f"({min(_SUPPORTED_KERNEL_SIZES)}..{max(_SUPPORTED_KERNEL_SIZES)}), "
            f"got {kernel_size}"
        )
    if dtype is not DType.BF16 or state_dtype is not DType.BF16:
        raise ValueError(
            "torch gdn_causal_conv_decode requires dtype=bf16 and "
            f"state_dtype=bf16, got {dtype.value} and {state_dtype.value}"
        )
    return _ValidatedArgs(
        batch_size=batch_size,
        channels=channels,
        kernel_size=kernel_size,
        dtype=dtype,
        state_dtype=state_dtype,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch gdn_causal_conv_decode backend"
        )


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    return _OperandShapes(
        x=(args.batch_size, args.channels),
        weight=(args.channels, args.kernel_size),
        state_storage=(
            args.batch_size + 1,
            args.channels,
            args.kernel_size - 1,
        ),
        slot_indices=(args.batch_size,),
        output=(args.batch_size, args.channels),
    )


def _valid_slot_indices(batch_size: int) -> tuple[int, ...]:
    return tuple(range(1, int(batch_size) + 1))


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    # Bounded operands keep the FP32 dot and SiLU finite while repeated decode
    # calls shift the same token into every selected state window.
    x = torch.empty(shapes.x, dtype=args.dtype.torch(), device=device).uniform_(
        -0.25,
        0.25,
        generator=generator,
    )
    weight = torch.empty(
        shapes.weight,
        dtype=args.dtype.torch(),
        device=device,
    ).uniform_(-0.125, 0.125, generator=generator)
    state = torch.empty(
        shapes.state_storage,
        dtype=args.state_dtype.torch(),
        device=device,
    ).uniform_(-0.25, 0.25, generator=generator)
    slot_indices = torch.arange(
        1,
        args.batch_size + 1,
        dtype=torch.int32,
        device=device,
    )
    return _Operands(
        x=x,
        weight=weight,
        state=state,
        slot_indices=slot_indices,
    )


def _semantic_flops(*, batch_size: int, channels: int, kernel_size: int) -> int:
    """Nominal semantic FLOPs, not physical Torch instruction count.

    Each channel's FP32 dot counts ``W`` multiplies and ``W-1`` additions. SiLU
    counts one sigmoid and one multiply. Casts and the state shift count as zero.
    """
    return batch_size * channels * (2 * kernel_size + 1)


def _logical_bytes(
    *,
    batch_size: int,
    channels: int,
    kernel_size: int,
    dtype: DType,
    state_dtype: DType,
) -> float:
    """Logical boundary traffic, excluding Torch composite intermediates.

    Counts input/weight/state reads, output/state writes, and fixed-int32 slot
    indices. The shared depthwise weight is counted once rather than per row.
    """
    data_elements = 2 * batch_size * channels + channels * kernel_size
    state_elements = 2 * batch_size * channels * (kernel_size - 1)
    slot_index_bytes = batch_size * 4
    return (
        data_elements * dtype.size_bytes()
        + state_elements * state_dtype.size_bytes()
        + slot_index_bytes
    )


def profile_gdn_causal_conv_decode(
    batch_size: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile the complete multi-launch Torch causal-conv decode reference."""
    args = _validate_args(
        batch_size=batch_size,
        channels=channels,
        kernel_size=kernel_size,
        dtype=dtype,
        state_dtype=state_dtype,
    )
    try:
        import torch

        from profiling.runners.attention.gdn_causal_conv_decode_reference import (
            gdn_causal_conv_decode_reference,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch and the semantic reference are required for the torch "
            "gdn_causal_conv_decode backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(torch, args, device=torch.device("cuda"))

        def kernel():
            return gdn_causal_conv_decode_reference(
                operands.x,
                operands.weight,
                operands.state,
                operands.slot_indices,
            )

        # Deliberately no reset and no kernel-name filter. Each measured call is
        # one valid state-mutating decode step on the current bounded state, and
        # the multi-launch semantic composite is the object being measured.
        # Operand allocation remains outside both measurement regions.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )
        flops = _semantic_flops(
            batch_size=args.batch_size,
            channels=args.channels,
            kernel_size=args.kernel_size,
        )
        logical_bytes = _logical_bytes(
            batch_size=args.batch_size,
            channels=args.channels,
            kernel_size=args.kernel_size,
            dtype=args.dtype,
            state_dtype=args.state_dtype,
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
