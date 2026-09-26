"""vLLM CUDA runner for plain (non-residual) RMSNorm.

The timed callable is exactly ``vllm._custom_ops.rms_norm(out, input, weight,
eps)`` -- what ``vllm.model_executor.layers.layernorm.RMSNorm.forward_cuda``
launches without a residual (e.g. GLM-5.3's final model norm) -- and CUPTI
selects its single ``rms_norm_kernel`` launch. Logical traffic and nominal
FLOPs describe the semantic operation, not undocumented device traffic.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_EPS = 1e-5
_KERNEL_NAME = "rms_norm_kernel"
_SUPPORTED_COMPUTE_GPU_PAIRS = frozenset({(DType.BF16, "NVIDIA B200")})


def _validate_args(m: int, hidden: int, dtype: DType | str) -> tuple[int, int, DType]:
    m = int(m)
    hidden = int(hidden)
    dtype = DType.from_value(dtype)
    if m <= 0 or hidden <= 0:
        raise ValueError(f"m and hidden must be > 0, got m={m}, hidden={hidden}")
    if dtype is not DType.BF16:
        raise ValueError(f"vllm_cuda rms_norm supports only bf16, got {dtype.value}")
    return m, hidden, dtype


def _validate_cuda_device(torch: Any, dtype: DType) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the rms_norm vllm_cuda backend")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if (dtype, gpu_name) not in _SUPPORTED_COMPUTE_GPU_PAIRS:
        raise ProfilerNotImplemented(
            f"rms_norm vllm_cuda has no verified {dtype.value}/{gpu_name} implementation"
        )


def profile_rms_norm_vllm_cuda(m: int, hidden: int, dtype: DType | str) -> ComputeMetrics:
    """Profile vLLM's one-launch out-of-place RMSNorm."""
    m, hidden, dtype = _validate_args(m, hidden, dtype)
    try:
        import torch
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the instrumented vLLM environment is required for the rms_norm vllm_cuda backend"
        ) from exc

    _validate_cuda_device(torch, dtype)

    try:
        torch_dtype = dtype.torch()
        generator = torch.Generator(device="cuda").manual_seed(17)
        input_tensor = torch.randn((m, hidden), dtype=torch_dtype, device="cuda", generator=generator)
        weight = torch.randn((hidden,), dtype=torch_dtype, device="cuda", generator=generator)
        output = torch.empty_like(input_tensor)

        def kernel() -> None:
            ops.rms_norm(output, input_tensor, weight, _EPS)

        kernel()
        torch.cuda.synchronize()
        values = input_tensor.float()
        expected = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + _EPS)
        torch.testing.assert_close(
            output.float(), (expected * weight.float()).to(torch_dtype).float(), rtol=2e-2, atol=2e-2
        )

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)

        # Logical traffic: read input and weight, write output.
        bytes_accessed = int((2 * m * hidden + hidden) * dtype.size_bytes())
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0.0 else 0.0
        flops = 4 * m * hidden + 2 * m
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0.0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
