"""Torch semantic-composite runner for Qwen GDN prefill post-conv prep.

This backend times every launch made by the standalone Torch reference. It is
not vLLM's fused Triton post-convolution launch, and its logical traffic/FLOP
rates do not describe a fused kernel's physical implementation. Production
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
    num_tokens = int(num_tokens)
    num_qk_heads = int(num_qk_heads)
    num_value_heads = int(num_value_heads)
    key_head_dim = int(key_head_dim)
    value_head_dim = int(value_head_dim)
    dtype = DType.from_value(dtype)

    dimensions = (
        num_tokens,
        num_qk_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            "num_tokens, num_qk_heads, num_value_heads, key_head_dim, and "
            f"value_head_dim must be > 0, got {dimensions}"
        )
    if dtype is not DType.BF16:
        raise ValueError(f"torch gdn_prefill_post_conv requires dtype=bf16, got {dtype.value}")
    return _ValidatedArgs(
        num_tokens=num_tokens,
        num_qk_heads=num_qk_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        dtype=dtype,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch gdn_prefill_post_conv backend")


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    packed_width = (
        2 * args.num_qk_heads * args.key_head_dim + args.num_value_heads * args.value_head_dim
    )
    gate_shape = (args.num_tokens, args.num_value_heads)
    return _OperandShapes(
        conv_output=(args.num_tokens, packed_width),
        a=gate_shape,
        b=gate_shape,
        A_log=(args.num_value_heads,),
        dt_bias=(args.num_value_heads,),
        q=(args.num_tokens, args.num_qk_heads, args.key_head_dim),
        k=(args.num_tokens, args.num_qk_heads, args.key_head_dim),
        v=(args.num_tokens, args.num_value_heads, args.value_head_dim),
        g=gate_shape,
        beta=gate_shape,
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

    # These bounded inputs keep normalization and gate math finite and make
    # allocation tests deterministic across repeated construction.
    return _Operands(
        conv_output=bounded(
            shapes.conv_output,
            -0.5,
            0.5,
            dtype=activation_dtype,
        ),
        a=bounded(shapes.a, -0.5, 0.5, dtype=activation_dtype),
        b=bounded(shapes.b, -0.5, 0.5, dtype=activation_dtype),
        A_log=bounded(shapes.A_log, -0.25, 0.25, dtype=torch.float32),
        dt_bias=bounded(shapes.dt_bias, -0.25, 0.25, dtype=torch.float32),
    )


def _semantic_flops(
    *,
    num_tokens: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
) -> int:
    """Nominal semantic FLOPs, not physical Torch instruction count.

    Each Q/K vector counts square, reduction, epsilon add, rsqrt, and scale as
    ``3*K+1`` units. Each token/value-head gate pair counts add, softplus,
    decay multiply, and sigmoid; ``exp(A_log)`` counts once per value head.
    Transcendentals are one nominal unit each. Copies and casts count as zero.
    """
    qk_norm = 2 * num_tokens * num_qk_heads * (3 * key_head_dim + 1)
    gates = 4 * num_tokens * num_value_heads + num_value_heads
    return qk_norm + gates


def _logical_bytes(
    *,
    num_tokens: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType,
) -> float:
    """Logical operation-boundary traffic, not physical Torch traffic.

    Counts BF16 conv/a/b reads and q/k/v writes, FP32 parameter reads, and FP32
    g/beta writes. Composite intermediates and allocator traffic are excluded.
    """
    packed_width = 2 * num_qk_heads * key_head_dim + num_value_heads * value_head_dim
    bf16_elements = 2 * num_tokens * packed_width + 2 * num_tokens * num_value_heads
    fp32_elements = 2 * num_value_heads + 2 * num_tokens * num_value_heads
    return bf16_elements * dtype.size_bytes() + fp32_elements * DType.FP32.size_bytes()


def profile_gdn_prefill_post_conv(
    num_tokens: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile the complete multi-launch Torch post-conv semantic reference."""
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

        from profiling.runners.attention.gdn_prefill_post_conv_reference import (
            gdn_prefill_post_conv_reference,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch and the semantic reference are required for the torch "
            "gdn_prefill_post_conv backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(torch, args, device=torch.device("cuda"))

        def kernel():
            return gdn_prefill_post_conv_reference(
                operands.conv_output,
                operands.a,
                operands.b,
                operands.A_log,
                operands.dt_bias,
                num_qk_heads=args.num_qk_heads,
                num_value_heads=args.num_value_heads,
                key_head_dim=args.key_head_dim,
                value_head_dim=args.value_head_dim,
            )

        # No kernel-name filter: measure the entire multi-launch semantic
        # composite. Inputs are immutable, so no reset is required. Operand
        # allocation stays outside both timing and energy regions.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
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
