"""Fused vLLM CUDA runner for residual-add RMSNorm.

The timed callable is exactly ``vllm._custom_ops.fused_add_rms_norm`` and CUPTI
selects its single ``fused_add_rms_norm_kernel`` launch. Logical traffic and
nominal FLOPs describe the semantic operation, not undocumented device traffic.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_EPS = 1e-5
_KERNEL_NAME = "fused_add_rms_norm_kernel"
_SUPPORTED_GPUS = frozenset({"NVIDIA H200", "NVIDIA B200"})
_SUPPORTED_DTYPES = frozenset({DType.BF16, DType.FP16})


def _validate_args(
    m: int,
    hidden: int,
    dtype: DType | str,
) -> tuple[int, int, DType]:
    m = int(m)
    hidden = int(hidden)
    dtype = DType.from_value(dtype)
    if m <= 0 or hidden <= 0:
        raise ValueError(f"m and hidden must be > 0, got m={m}, hidden={hidden}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            "vllm_cuda residual_rms_norm supports only bf16 and fp16, "
            f"got {dtype.value}"
        )
    return m, hidden, dtype


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the residual_rms_norm vllm_cuda backend"
        )
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in _SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            "residual_rms_norm vllm_cuda is verified only on H200/B200, "
            f"got {gpu_name}"
        )


def profile_residual_rms_norm_vllm_cuda(
    m: int,
    hidden: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile vLLM's one-launch, in-place fused residual-add RMSNorm."""
    m, hidden, dtype = _validate_args(m, hidden, dtype)
    try:
        import torch
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the instrumented vLLM environment is required for "
            "the residual_rms_norm vllm_cuda backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        torch_dtype = dtype.torch()
        # The vLLM op mutates both activations. Zeros remain zeros across the
        # repeated timing and energy calls, preventing FP16 accumulation
        # overflow without changing the launch or memory-access shape.
        input_tensor = torch.zeros(
            (m, hidden),
            dtype=torch_dtype,
            device="cuda",
        )
        residual = torch.zeros_like(input_tensor)
        weight = torch.ones(
            (hidden,),
            dtype=torch_dtype,
            device="cuda",
        )

        def kernel() -> None:
            ops.fused_add_rms_norm(
                input_tensor,
                residual,
                weight,
                _EPS,
            )

        # Allocation stays outside the callable. The name filter selects only
        # vLLM's single fused device launch and excludes CUPTI's L2 clear.
        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )

        # Logical traffic: read input, residual, and weight, then write the
        # normalized input and residual sum back in place.
        logical_elements = 4 * m * hidden + hidden
        bytes_accessed = int(logical_elements * dtype.size_bytes())
        bandwidth_gbps = (
            (bytes_accessed / (time_ms / 1000.0)) / 1e9
            if time_ms > 0.0
            else 0.0
        )

        # Nominal semantic FLOPs: residual add, square/reduction contribution,
        # normalization multiply, and weight multiply, plus rowwise mean,
        # epsilon, and rsqrt. Casts count as zero.
        flops = 5 * m * hidden + 2 * m
        tflops = (
            (flops / (time_ms / 1000.0)) / 1e12
            if time_ms > 0.0
            else 0.0
        )
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
