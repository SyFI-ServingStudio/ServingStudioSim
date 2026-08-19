"""Torch semantic-composite runner for Qwen GDN recurrent decode.

This backend times every launch made by the standalone Torch reference. It is
not vLLM's one-launch packed recurrent kernel, and its logical traffic/FLOP
rates do not describe the fused kernel's physical implementation.
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
class _Operands:
    query: Any
    key: Any
    value: Any
    a: Any
    b: Any
    A_log: Any
    dt_bias: Any
    state: Any


def _validate_args(
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> tuple[int, int, int, int, int, DType, DType]:
    batch_size = int(batch_size)
    num_qk_heads = int(num_qk_heads)
    num_value_heads = int(num_value_heads)
    key_head_dim = int(key_head_dim)
    value_head_dim = int(value_head_dim)
    dtype = DType.from_value(dtype)
    state_dtype = DType.from_value(state_dtype)

    dimensions = (
        batch_size,
        num_qk_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            "batch_size, num_qk_heads, num_value_heads, key_head_dim, and "
            f"value_head_dim must be > 0, got {dimensions}"
        )
    if num_value_heads % num_qk_heads != 0:
        raise ValueError(
            "num_value_heads must be divisible by num_qk_heads, "
            f"got {num_value_heads} and {num_qk_heads}"
        )
    if dtype is not DType.BF16 or state_dtype is not DType.FP32:
        raise ValueError(
            "torch gdn_recurrent_decode requires dtype=bf16 and "
            f"state_dtype=fp32, got {dtype.value} and {state_dtype.value}"
        )
    return (
        batch_size,
        num_qk_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        dtype,
        state_dtype,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch gdn_recurrent_decode backend")


def _build_operands(
    torch: Any,
    *,
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    activation_dtype: Any,
    state_dtype: Any,
) -> _Operands:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)
    activation_kwargs = {
        "dtype": activation_dtype,
        "device": "cuda",
        "generator": generator,
    }
    query = torch.randn(
        (batch_size, num_qk_heads, key_head_dim),
        **activation_kwargs,
    )
    key = torch.randn(
        (batch_size, num_qk_heads, key_head_dim),
        **activation_kwargs,
    )
    value = torch.randn(
        (batch_size, num_value_heads, value_head_dim),
        **activation_kwargs,
    )
    # Bounded raw gates keep decay/update values representative and finite over
    # the many state-mutating repetitions used by CUPTI and energy measurement.
    a = torch.empty(
        (batch_size, num_value_heads),
        dtype=activation_dtype,
        device="cuda",
    ).uniform_(-0.5, 0.5, generator=generator)
    b = torch.empty(
        (batch_size, num_value_heads),
        dtype=activation_dtype,
        device="cuda",
    ).uniform_(-0.5, 0.5, generator=generator)
    A_log = torch.zeros(
        num_value_heads,
        dtype=state_dtype,
        device="cuda",
    )
    dt_bias = torch.zeros(
        num_value_heads,
        dtype=state_dtype,
        device="cuda",
    )
    state = torch.empty(
        (batch_size, num_value_heads, key_head_dim, value_head_dim),
        dtype=state_dtype,
        device="cuda",
    ).normal_(mean=0.0, std=0.01, generator=generator)
    return _Operands(
        query=query,
        key=key,
        value=value,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        state=state,
    )


def _semantic_flops(
    *,
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> int:
    """Nominal semantic FLOPs, not a sum of physical Torch instructions.

    Transcendentals count as one operation. Copies, casts, and head expansion
    count as zero. The dominant recurrence counts decay plus memory/update/
    readout arithmetic over the semantic FP32 state.
    """
    state_elements = batch_size * num_value_heads * key_head_dim * value_head_dim
    qk_norm = 2 * batch_size * num_qk_heads * (3 * key_head_dim + 1)
    query_scale = batch_size * num_value_heads * key_head_dim
    gates = 4 * batch_size * num_value_heads + num_value_heads
    recurrence = 7 * state_elements
    return qk_norm + query_scale + gates + recurrence


def _logical_bytes(
    *,
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType,
    state_dtype: DType,
) -> float:
    """Logical boundary traffic, excluding all Torch intermediate tensors."""
    activation_elements = (
        2 * batch_size * num_qk_heads * key_head_dim
        + 2 * batch_size * num_value_heads * value_head_dim
        + 2 * batch_size * num_value_heads
    )
    state_elements = batch_size * num_value_heads * key_head_dim * value_head_dim
    fp32_elements = 2 * num_value_heads + 2 * state_elements
    return activation_elements * dtype.size_bytes() + fp32_elements * state_dtype.size_bytes()


def profile_gdn_recurrent_decode(
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile the complete multi-launch Torch GDN decode reference."""
    (
        batch_size,
        num_qk_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        dtype,
        state_dtype,
    ) = _validate_args(
        batch_size,
        num_qk_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        dtype,
        state_dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch gdn_recurrent_decode backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        from profiling.runners.attention.gdn_recurrent_decode_reference import (
            gdn_recurrent_decode_reference,
        )

        operands = _build_operands(
            torch,
            batch_size=batch_size,
            num_qk_heads=num_qk_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            activation_dtype=dtype.torch(),
            state_dtype=state_dtype.torch(),
        )

        def kernel():
            return gdn_recurrent_decode_reference(
                operands.query,
                operands.key,
                operands.value,
                operands.a,
                operands.b,
                operands.A_log,
                operands.dt_bias,
                operands.state,
            )

        # Deliberately no name filter: measure every launch in the standalone
        # Torch semantic composite, not a surrogate production kernel.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )
        flops = _semantic_flops(
            batch_size=batch_size,
            num_qk_heads=num_qk_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
        )
        logical_bytes = _logical_bytes(
            batch_size=batch_size,
            num_qk_heads=num_qk_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            dtype=dtype,
            state_dtype=state_dtype,
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
