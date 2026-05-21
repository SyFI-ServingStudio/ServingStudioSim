"""FlashInfer RMSNorm runner.

This file is L1a-only: it allocates tensors, times one kernel, and returns
metrics. DB writes, JIT policy, subprocess selection, and registry routing all
live in L1b. Mirrors ``ref/profile/norm/rmsnorm_flashinfer.py`` — the real
kernel is ``flashinfer.norm.rmsnorm`` and timing is CUPTI kernel-only (matching
the ``RMSNormKernel`` device kernel), not a hand-rolled torch RMSNorm.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

# CUPTI matches this device kernel name (see ref/profile/norm/rmsnorm_flashinfer.py).
_KERNEL_NAME = "RMSNormKernel"
_EPS = 1e-6


def profile_rms_norm(
    m: int,
    hidden: int,
    dtype: DType | str,
) -> ComputeMetrics:
    dtype = DType.from_value(dtype)

    try:
        import flashinfer.norm
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch + flashinfer are required for the flashinfer RMSNorm runner"
        ) from exc

    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the flashinfer RMSNorm runner")

    try:
        torch_dtype = dtype.torch()
        x = torch.randn(m, hidden, dtype=torch_dtype, device="cuda")
        weight = torch.randn(hidden, dtype=torch_dtype, device="cuda")

        def kernel():
            return flashinfer.norm.rmsnorm(x, weight, eps=_EPS)

        # cupti samples adaptively until convergence with no warmup; energy keeps
        # a light warmup before its NVML window.
        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )

        # RMSNorm is bandwidth-bound: read input, write output (weight is
        # negligible). Mirrors the ref's 2 * batch * hidden * element_size.
        bytes_accessed = int(2 * m * hidden * dtype.size_bytes())
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        # ~5 flops/element (square, reduce, rsqrt, two muls); not the bound, kept
        # only so ComputeMetrics is fully populated.
        flops = 5 * m * hidden
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
