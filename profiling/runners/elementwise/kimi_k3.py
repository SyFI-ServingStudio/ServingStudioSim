"""Profilers for Kimi-K3 fused eager elementwise calls."""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_K3_MERGED_FRONT_WIDTH = 15_984


def profile_k3_situ_and_mul_prefill(
    num_tokens: int,
    hidden_size: int,
    input_dtype: DType | str,
    output_dtype: DType | str,
    beta: float,
    linear_beta: float,
) -> ComputeMetrics:
    if type(num_tokens) is not int or num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer")
    if DType.from_value(input_dtype) is not DType.FP32:
        raise ValueError("K3 SiTU prefill expects FP32 input")
    if DType.from_value(output_dtype) is not DType.BF16:
        raise ValueError("K3 SiTU prefill expects BF16 output")
    if float(beta) != 4.0 or float(linear_beta) != 25.0:
        raise ValueError("K3 SiTU prefill uses beta=4 and linear_beta=25")

    try:
        import torch
        from sglang.kernels.ops.kimi_k3 import situ_and_mul
    except ImportError as exc:
        raise ProfilerNotImplemented("the SGLang K3 environment is required") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("K3 SiTU prefill profiling requires CUDA")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented("K3 SiTU prefill profiling requires SM100")

    try:
        generator = torch.Generator(device="cuda").manual_seed(31)
        merged_front = torch.randn(
            (num_tokens, max(_K3_MERGED_FRONT_WIDTH, 2 * hidden_size)),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        input_tensor = merged_front[:, : 2 * hidden_size]
        output = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

        def run_once():
            return situ_and_mul(input_tensor, output, float(beta), float(linear_beta))

        run_once()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError("K3 SiTU prefill ran out of CUDA memory") from exc
    except RuntimeError as exc:
        raise KernelLaunchFailed(f"K3 SiTU prefill failed: {exc}") from exc

    seconds = time_ms / 1000.0
    bytes_accessed = input_tensor.numel() * 4 + output.numel() * 2
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(bytes_accessed / seconds / 1e9 if seconds else 0.0),
        energy_j=float(energy_j),
    )


def profile_k3_add3_prefill(
    num_tokens: int,
    hidden_size: int,
    dtype: DType | str,
) -> ComputeMetrics:
    if type(num_tokens) is not int or num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer")
    if DType.from_value(dtype) is not DType.BF16:
        raise ValueError("K3 add3 prefill expects BF16 tensors")

    try:
        import torch
        from sglang.kernels.ops.elementwise import add3
    except ImportError as exc:
        raise ProfilerNotImplemented("the SGLang K3 environment is required") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("K3 add3 prefill profiling requires CUDA")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented("K3 add3 prefill profiling requires SM100")

    try:
        generator = torch.Generator(device="cuda").manual_seed(37)
        a = torch.randn(
            (num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        b = torch.randn_like(a)
        c = torch.randn_like(a)

        def run_once():
            # Production _add3 lets the fused op own its output allocation.
            # Preserve that boundary so its launch/tactic selection matches
            # the eager layer call.
            return add3.add3(a, b, c, prefetch_bc=True)

        run_once()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError("K3 add3 prefill ran out of CUDA memory") from exc
    except RuntimeError as exc:
        raise KernelLaunchFailed(f"K3 add3 prefill failed: {exc}") from exc

    seconds = time_ms / 1000.0
    elements = 4 * a.numel()
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(elements * 2 / seconds / 1e9 if seconds else 0.0),
        energy_j=float(energy_j),
    )


__all__ = ["profile_k3_add3_prefill", "profile_k3_situ_and_mul_prefill"]
