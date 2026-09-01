"""Profile SGLang's BF16-input, FP32-output MoE router dispatch."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gemm_fp32_output:sglang_router_auto"
_SUPPORTED_N = frozenset({256, 384})


def _device_sm(torch: Any) -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _max_router_gemm_tokens(sm: int) -> int:
    return 4 if sm in (100, 103) else 16


def profile_gemm_fp32_output_sglang_router(
    m: int,
    n: int,
    k: int,
    input_dtype: DType | str,
) -> ComputeMetrics:
    if type(m) is not int or m <= 0:
        raise ValueError("m must be a positive integer")
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if DType.from_value(input_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires input_dtype=bf16")

    try:
        import torch
        from sglang.kernels.ops.attention.dsv4 import linear_bf16_fp32
        from sglang.kernels.ops.gemm import dsv3_router_gemm
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the SGLang environment") from exc

    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    sm = _device_sm(torch)
    if sm < 90:
        raise ProfilerNotImplemented(f"{_BACKEND} requires SM90+, got sm{sm}")

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(31)
        hidden = torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=generator)
        weight = torch.randn((n, k), dtype=torch.bfloat16, device=device, generator=generator)
        use_dedicated = m <= _max_router_gemm_tokens(sm) and k % 1024 == 0 and n in _SUPPORTED_N
        if use_dedicated:

            def launch():
                return dsv3_router_gemm(hidden, weight, out_dtype=torch.float32)

        else:

            def launch():
                return linear_bf16_fp32(hidden, weight)

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


__all__ = ["profile_gemm_fp32_output_sglang_router"]
