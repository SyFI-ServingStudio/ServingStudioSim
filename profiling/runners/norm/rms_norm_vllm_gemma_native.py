"""vLLM ``GemmaRMSNorm.forward_native`` run eagerly, as a no-Inductor vLLM model launches it.

``GemmaRMSNorm.forward_cuda`` calls ``forward_native``: it builds an fp32
``1 + weight`` and calls the ``vllm.ir`` ``rms_norm`` op. The vllm_c and oink
providers require the weight dtype to match the input, so a bf16 input with that
fp32 weight dispatches to the native PyTorch implementation, about ten eager
launches (weight cast and offset, input cast, square, mean, add, rsqrt, scale,
weight multiply, output cast). CUPTI counts every launch of the call.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.mhc.hc_reference import grouped_rms_norm_reference

_BACKEND = "rms_norm:vllm_gemma_native"
_EPS = 1e-6


def profile_rms_norm_vllm_gemma_native(m: int, hidden: int, dtype: DType | str) -> ComputeMetrics:
    for name, value in (("m", m), ("hidden", hidden)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16, got {dtype}")
    try:
        import torch
        from vllm import ir
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    native = ir.ops.rms_norm.impls["native"].impl_fn
    try:
        generator = torch.Generator(device="cuda").manual_seed(17)
        x = torch.randn((m, hidden), device="cuda", generator=generator).to(torch.bfloat16)
        weight = (0.1 * torch.randn((hidden,), device="cuda", generator=generator)).to(
            torch.bfloat16
        )

        def launch() -> object:
            return native(x, weight.float() + 1.0, _EPS)

        torch.testing.assert_close(
            launch(), grouped_rms_norm_reference(torch, x, weight, _EPS, 1), rtol=2e-2, atol=2e-2
        )
        time_ms = Timer.cupti(launch, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, AssertionError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    # Logical traffic as for vllm_cuda: read input and weight, write output.
    seconds = time_ms / 1000.0
    bytes_accessed = 2 * (2 * m * hidden + hidden)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float((4 * m * hidden + 2 * m) / seconds / 1e12),
        memory_bandwidth_gbps=float(bytes_accessed / seconds / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_rms_norm_vllm_gemma_native"]
