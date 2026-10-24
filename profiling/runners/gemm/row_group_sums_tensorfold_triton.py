"""TensorFold's Triton ``_group_sums`` as the ``row_group_sums`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.cuda.kernels.qmm.group_sums(x, gs)`` (perf-study 6469730a):
``x`` ``(M, K)`` bf16 contiguous, returning ``(M, K / gs)`` fp32. The DFlash2
drafter's tiled 4-bit ``matmul`` launches it for every projection whose caller
passes no sums: per layer the attention and MLP kernel projections, qkv and o,
the selector's hidden projection, and the tap path's ``fc`` and kv projections.
One ``_group_sums`` launch on grid ``(M, cdiv(K / gs, 16))``, two warps.
"""

from __future__ import annotations

import importlib
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.cuda.kernels.qmm"
_KERNEL_NAME = "_group_sums"
_LABEL = "row_group_sums:tensorfold_triton"
_GUARD_ROWS = 37
RMSE_RATIO_TOL = 1e-5


def validate_args(num_tokens: int, hidden: int, group_size: int, dtype: DType | str) -> None:
    for name, value in (("num_tokens", num_tokens), ("hidden", hidden), ("group_size", group_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} sums bf16 rows only, got {dtype}")
    if hidden % group_size or group_size & (group_size - 1):
        raise ValueError(
            f"{_LABEL} needs a power-of-two group_size dividing hidden, got {group_size}, {hidden}"
        )


def profile_row_group_sums_tensorfold_triton(
    num_tokens: int, hidden: int, group_size: int, dtype: DType | str
) -> ComputeMetrics:
    validate_args(num_tokens, hidden, group_size, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        group_sums = importlib.import_module(_MODULE).group_sums
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"{_LABEL} needs {_MODULE}.group_sums (tensorfold_env)"
        ) from exc
    from profiling.runners.gemm.row_group_sums_reference import row_group_sums_reference

    try:
        guard = torch.randn(_GUARD_ROWS, hidden, device="cuda").to(torch.bfloat16)
        got = group_sums(guard, group_size)
        want = row_group_sums_reference(torch, guard, group_size)
        if got.shape != want.shape or got.dtype != torch.float32:
            raise AssertionError(f"{_LABEL} returned {tuple(got.shape)} {got.dtype}")
        err = float((got - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt())
        if not err < RMSE_RATIO_TOL:
            raise AssertionError(f"{_LABEL} rmse ratio {err:.2e} (tol {RMSE_RATIO_TOL:.0e})")
        x = torch.randn(num_tokens, hidden, device="cuda").to(torch.bfloat16)
        torch.cuda.synchronize()

        def kernel() -> Any:
            return group_sums(x, group_size)

        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, AssertionError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc

    nbytes = num_tokens * (2 * hidden + 4 * (hidden // group_size))
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(nbytes / seconds / 1e9) if time_ms > 0 else 0.0,
        energy_j=float(energy_j),
    )


__all__ = ["profile_row_group_sums_tensorfold_triton"]
