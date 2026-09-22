"""SGLang Triton causal-convolution decode runner for Kimi-K3 GDN/KDA."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SUPPORTED_CHANNELS = frozenset({4608, 36864})


def _validate_args(
    batch_size: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> tuple[int, int, int, DType, DType]:
    values = (int(batch_size), int(channels), int(kernel_size))
    activation = DType.from_value(dtype)
    state = DType.from_value(state_dtype)
    if min(values) <= 0:
        raise ValueError("batch_size, channels, and kernel_size must be positive")
    if channels not in _SUPPORTED_CHANNELS:
        raise ValueError("SGLang K3 causal-conv channels must be 4608 or 36864")
    if kernel_size != 4:
        raise ValueError("SGLang K3 causal-conv decode requires kernel_size=4")
    if activation is not DType.BF16 or state is not DType.BF16:
        raise ValueError("SGLang K3 causal-conv decode requires bf16 activations and state")
    return *values, activation, state


def profile_gdn_causal_conv_decode_sglang_triton(
    batch_size: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    batch_size, channels, kernel_size, dtype, state_dtype = _validate_args(
        batch_size, channels, kernel_size, dtype, state_dtype
    )
    try:
        import torch
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang K3 causal_conv1d_update is required") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for SGLang causal-conv profiling")
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    x = torch.randn(
        (batch_size, channels),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    # SGLang stores the K3 state as [slot, window, channel], while the update
    # callable consumes the channel-major [slot, channel, window] view.
    conv_state = (
        torch.randn(
            (batch_size + 1, kernel_size - 1, channels),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        .transpose(-1, -2)
        .contiguous()
    )
    weight = torch.randn(
        (channels, kernel_size), dtype=torch.float32, device=device, generator=generator
    ).contiguous()
    bias = torch.zeros(channels, dtype=torch.float32, device=device)
    indices = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)

    def kernel() -> Any:
        return causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias=bias,
            activation="silu",
            conv_state_indices=indices,
            validate_data=False,
        )

    try:
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    seconds = time_ms / 1000.0
    flops = 2 * batch_size * channels * kernel_size + 8 * batch_size * channels
    bytes_accessed = (
        (batch_size * channels + channels * kernel_size) * dtype.size_bytes()
        + (batch_size + 1) * channels * (kernel_size - 1) * state_dtype.size_bytes()
        + batch_size * 4
    )
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / seconds / 1e12 if seconds else 0.0,
        memory_bandwidth_gbps=bytes_accessed / seconds / 1e9 if seconds else 0.0,
        energy_j=float(energy_j),
    )
