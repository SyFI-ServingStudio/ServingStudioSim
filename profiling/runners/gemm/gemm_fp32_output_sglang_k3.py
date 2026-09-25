"""Profile Kimi-K3's raw BF16-input/FP32-output GEMM path."""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gemm_fp32_output:sglang_k3_fp32_auto"


def profile_gemm_fp32_output_sglang_k3(
    m: int,
    n: int,
    k: int,
    input_dtype: DType | str,
) -> ComputeMetrics:
    for name, value in (("m", m), ("n", n), ("k", k)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if DType.from_value(input_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires input_dtype=bf16")

    try:
        import torch
        from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
            cutedsl_bf16_gemm_out,
            use_cutedsl_bf16_gemm,
        )
        from sglang.srt.layers.quantization.unquant import get_bf16_gemm_backend
    except ImportError as exc:
        raise ProfilerNotImplemented("the SGLang environment is required") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented(f"{_BACKEND} requires SM100")

    try:
        generator = torch.Generator(device="cuda").manual_seed(31)
        hidden = torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=generator)
        weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda", generator=generator)
        output = torch.empty((m, n), dtype=torch.float32, device="cuda")

        use_cutedsl = get_bf16_gemm_backend().is_cutedsl() and use_cutedsl_bf16_gemm(m, n, k)

        def launch():
            # Match sglang.srt.models.kimi_k3._k3_bf16_gemm: the FP32-output
            # branch still takes the cutedsl path when the shape is eligible.
            if use_cutedsl:
                return cutedsl_bf16_gemm_out(hidden, weight, output)
            return torch.mm(hidden, weight.t(), out=output, out_dtype=torch.float32)

        first = launch()
        torch.cuda.synchronize()
        if first.dtype is not torch.float32:
            raise AssertionError(f"expected FP32 output, got {first.dtype}")
        time_ms = Timer.cupti(launch, interval_union=True)
        energy_j = Energy.perf(launch, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    elapsed_s = time_ms / 1000.0
    flops = 2 * m * n * k
    logical_bytes = 2 * m * k + 2 * n * k + 4 * m * n
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_s / 1e12 if elapsed_s else 0.0),
        memory_bandwidth_gbps=float(logical_bytes / elapsed_s / 1e9 if elapsed_s else 0.0),
        energy_j=float(energy_j),
    )


__all__ = ["profile_gemm_fp32_output_sglang_k3"]
