"""Torch semantic-composite runner for residual-add RMSNorm.

This backend times the complete multi-launch Torch reference with CUPTI. It is
not the fused vLLM kernel and its logical traffic/FLOP estimates do not describe
vLLM fusion internals.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_EPS = 1e-5
_SUPPORTED_DTYPES = frozenset({DType.BF16, DType.FP16})


def profile_residual_rms_norm(
    m: int,
    hidden: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile the full Torch residual-add RMSNorm semantic callable."""
    m = int(m)
    hidden = int(hidden)
    dtype = DType.from_value(dtype)
    if m <= 0 or hidden <= 0:
        raise ValueError(f"m and hidden must be > 0, got m={m}, hidden={hidden}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"torch residual_rms_norm supports only bf16 and fp16, got {dtype.value}")

    try:
        import torch

        from profiling.runners.norm.residual_rms_norm_reference import (
            residual_rms_norm_reference,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the residual_rms_norm Torch backend"
        ) from exc

    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the residual_rms_norm Torch backend")

    try:
        torch_dtype = dtype.torch()
        x = torch.randn(m, hidden, dtype=torch_dtype, device="cuda")
        residual = torch.randn(m, hidden, dtype=torch_dtype, device="cuda")
        weight = torch.randn(hidden, dtype=torch_dtype, device="cuda")

        def kernel():
            return residual_rms_norm_reference(
                x,
                residual,
                weight,
                eps=_EPS,
            )

        # Deliberately no kernel-name filter: the backend measures every GPU
        # launch implementing the standalone Torch semantic composite.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )

        # Logical traffic: read x, residual, and weight; write normalized output
        # and residual_out. Torch's FP32 temporaries are intentionally excluded,
        # so this is not an estimate of fused-vLLM physical traffic.
        logical_elements = 4 * m * hidden + hidden
        bytes_accessed = int(logical_elements * dtype.size_bytes())
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0.0 else 0.0

        # Nominal semantic FLOPs: per-element residual add, square, reduction
        # contribution, normalization multiply, and weight multiply, plus one
        # mean division, epsilon add, and rsqrt per row (casts count as zero).
        flops = 5 * m * hidden + 2 * m
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0.0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
