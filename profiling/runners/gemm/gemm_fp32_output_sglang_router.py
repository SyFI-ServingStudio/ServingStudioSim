"""Profile SGLang's BF16-input, FP32-output MoE router dispatch."""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gemm_fp32_output:sglang_router_auto"
# SGLang's own router dispatch, not a measured set (python/sglang/srt/models/
# deepseek_v2.py, MoEGate.forward): tiny_gemm_bf16 serves up to
# tiny_router_gemm_max_tokens(...) rows; every other shape takes
# linear_bf16_fp32. The threshold comes from SGLang's own helper so the runner
# cannot disagree with the pinned checkout.


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
        from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
        from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
        from sglang.srt.models.deepseek_common.utils import tiny_router_gemm_max_tokens
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the SGLang environment") from exc

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(31)
        hidden = torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=generator)
        weight = torch.randn((n, k), dtype=torch.bfloat16, device=device, generator=generator)
        tiny_max_tokens = tiny_router_gemm_max_tokens(
            num_experts=n, hidden_size=k, weight_dtype=weight.dtype
        )
        if m <= tiny_max_tokens:

            def launch():
                return tiny_gemm_bf16(hidden, weight, out_dtype=torch.float32)

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
