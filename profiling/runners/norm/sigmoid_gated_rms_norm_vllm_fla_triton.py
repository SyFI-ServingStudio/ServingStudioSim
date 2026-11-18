"""FLA ``rms_norm_gated(activation="sigmoid")`` runner (vLLM's ``FusedRMSNormGated.forward_cuda``).

The timed callable is the wrapper itself, as the model calls it: x and g contiguous
``[M, hidden]``, a BF16 weight, no bias or residual, eps 1e-5. The wrapper writes the
output into x; the check runs on clones before timing.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "sigmoid_gated_rms_norm:vllm_fla_triton"
_KERNEL_NAME = "layer_norm_gated_fwd_kernel"
_ATOL = 1e-2
_RTOL = 1e-2


def _invoke(rms_norm_gated: Any, x: Any, g: Any, weight: Any) -> Any:
    from profiling.runners.norm.sigmoid_gated_rms_norm_reference import EPSILON

    return rms_norm_gated(x, g, weight, None, "sigmoid", eps=EPSILON)


def profile_sigmoid_gated_rms_norm_vllm_fla_triton(
    m: int, hidden: int, dtype: DType | str
) -> ComputeMetrics:
    m, hidden, dtype = int(m), int(hidden), DType.from_value(dtype)
    if m <= 0 or hidden <= 0:
        raise ValueError(f"m and hidden must be > 0, got m={m}, hidden={hidden}")
    if dtype is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND}: bf16 only, got {dtype.value}")
    try:
        import torch
        from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires vLLM's third-party FLA kda ops") from exc
    from profiling.runners.norm.sigmoid_gated_rms_norm_reference import (
        sigmoid_gated_rms_norm_reference,
    )

    try:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(42)
        x = torch.empty((m, hidden), dtype=torch.bfloat16, device="cuda").uniform_(
            -0.5, 0.5, generator=gen
        )
        g = torch.empty((m, hidden), dtype=torch.bfloat16, device="cuda").uniform_(
            -3.0, 3.0, generator=gen
        )
        weight = torch.empty((hidden,), dtype=torch.bfloat16, device="cuda").uniform_(
            0.75, 1.25, generator=gen
        )
        rows = min(m, 4096)
        expected = sigmoid_gated_rms_norm_reference(x[:rows], g[:rows], weight)
        got = _invoke(rms_norm_gated, x[:rows].clone(), g[:rows].clone(), weight)
        torch.cuda.synchronize()
        torch.testing.assert_close(got.float(), expected.float(), atol=_ATOL, rtol=_RTOL)

        def kernel() -> Any:
            return _invoke(rms_norm_gated, x, g, weight)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    seconds = time_ms / 1000.0
    flops = 6 * m * hidden + 2 * m
    logical_bytes = (3 * m * hidden + hidden) * 2
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / seconds / 1e12 if time_ms > 0 else 0.0,
        memory_bandwidth_gbps=logical_bytes / seconds / 1e9 if time_ms > 0 else 0.0,
    )


__all__ = ["profile_sigmoid_gated_rms_norm_vllm_fla_triton"]
