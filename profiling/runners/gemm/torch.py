"""Torch GEMM runners.

This file is L1a-only: it allocates tensors, times one kernel, and returns
metrics. DB writes, JIT policy, subprocess selection, and registry routing all
live in L1b.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


def profile_single_gemm(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    dtype = DType.from_value(dtype)

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the torch GEMM runner") from exc

    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch GEMM runner")

    try:
        torch_dtype = dtype.torch()
        a = torch.randn(m, k, dtype=torch_dtype, device="cuda")
        b = torch.randn(k, n, dtype=torch_dtype, device="cuda")

        def kernel():
            return torch.mm(a, b)

        warmup = 10
        time_ms = Timer.do_bench(kernel, warmup=warmup, rep=1000)
        energy_j = Energy.perf(
            kernel,
            warmup=min(warmup, 5),
            min_duration_ms=1000,
            per_iter_time_ms=time_ms,
        )

        flops = 2 * m * n * k
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        bytes_accessed = int((a.numel() + b.numel() + m * n) * dtype.size_bytes())
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
