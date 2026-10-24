"""TensorFold's Triton ``_add_rmsnorm`` without a residual as an ``rms_norm`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.families.qwen3_5.cuda.glue.add_rmsnorm(x, None, w, eps)``
(TensorFold perf-study 6469730a), the first layer's input norm: ``forward.py``
(verify) and ``prefill.py`` through ``prefill_bf16`` call it with
``pending=None`` before layer 0, since there is no previous block output yet.

- ``x`` is the embedded ``(m, hidden)`` bf16 rows, contiguous; ``w`` the
  ``(hidden,)`` bf16 weight.
- With ``r=None`` the kernel compiles the ``HAS_R=False`` specialization: it
  reads x only and returns ``(x, y, xs)`` with ``h`` aliasing ``x`` (no write),
  ``y = bf16(rmsnorm(x) * w)`` and ``xs``, fp32 sums of ``y`` over each
  64-element group (1/32 of one bf16 row, always written).

One launch of ``_add_rmsnorm`` (one program a row, 8 warps). Triton
JIT-compiles on first call, which the correctness check absorbs.
"""

from __future__ import annotations

import importlib
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.families.qwen3_5.cuda.glue"
_CALLABLE = "add_rmsnorm"
_KERNEL_NAME = "_add_rmsnorm"
_LABEL = "rms_norm:tensorfold_triton"
# TensorFold passes the checkpoint's rms_norm_eps (1e-6 for Qwen3.8-27B).
_EPS = 1e-6
_GROUP = 64
_GUARD_ROWS = 67
# The kernel rounds once after the weight multiply; the reference may round
# in another order: a couple of bf16 ulps.
Y_MAX_REL_TOL = 2e-2
XS_RMSE_RATIO_TOL = 1e-3


def validate_args(m: int, hidden: int, dtype: DType | str) -> None:
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} is bf16 only; got dtype={dtype}")
    if m <= 0 or hidden <= 0:
        raise ValueError(f"{_LABEL} needs m and hidden > 0, got m={m}, hidden={hidden}")
    if hidden % _GROUP:
        raise ProfilerNotImplemented(f"{_LABEL} needs hidden a multiple of {_GROUP}, got {hidden}")


def rms_norm_reference(torch: Any, x: Any, w: Any, eps: float) -> Any:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).to(
        torch.bfloat16
    )


def build_operands(torch: Any, m: int, hidden: int) -> tuple[Any, Any]:
    x = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda")
    w = (1.0 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
    return x, w


def check_correctness(torch: Any, add_rmsnorm: Any, hidden: int) -> tuple[float, float]:
    """One call on guard rows against the Torch oracle; return (y max rel err, xs rmse ratio)."""
    x, w = build_operands(torch, _GUARD_ROWS, hidden)
    before = x.clone()
    h, y, xs = add_rmsnorm(x, None, w, _EPS)
    torch.cuda.synchronize()
    if h is not x or not torch.equal(x, before):
        raise AssertionError(f"{_LABEL} no-residual call must return x itself, unmodified")
    if y.shape != x.shape or y.dtype != torch.bfloat16:
        raise AssertionError(f"{_LABEL} returned y {tuple(y.shape)} {y.dtype}")
    if xs.shape != (_GUARD_ROWS, hidden // _GROUP) or xs.dtype != torch.float32:
        raise AssertionError(f"{_LABEL} returned xs {tuple(xs.shape)} {xs.dtype}")
    want = rms_norm_reference(torch, x, w, _EPS).float()
    y_err = float(((y.float() - want).abs() / want.abs().clamp_min(1e-2)).max())
    if not y_err < Y_MAX_REL_TOL:
        raise AssertionError(f"{_LABEL} y max rel err {y_err:.2e} (tol {Y_MAX_REL_TOL:.0e})")
    want_xs = y.float().reshape(_GUARD_ROWS, -1, _GROUP).sum(-1)
    xs_err = float((xs - want_xs).pow(2).mean().sqrt() / want_xs.pow(2).mean().sqrt())
    if not xs_err < XS_RMSE_RATIO_TOL:
        raise AssertionError(f"{_LABEL} xs rmse ratio {xs_err:.2e} (tol {XS_RMSE_RATIO_TOL:.0e})")
    return y_err, xs_err


def profile_rms_norm_tensorfold_triton(m: int, hidden: int, dtype: DType | str) -> ComputeMetrics:
    m, hidden = int(m), int(hidden)
    validate_args(m, hidden, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        add_rmsnorm = getattr(importlib.import_module(_MODULE), _CALLABLE)
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"{_LABEL} needs {_MODULE}.{_CALLABLE} (tensorfold_env)"
        ) from exc

    try:
        check_correctness(torch, add_rmsnorm, hidden)
        x, w = build_operands(torch, m, hidden)

        def kernel() -> Any:
            return add_rmsnorm(x, None, w, _EPS)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc

    # vllm_cuda's counts: x read, y written, weight read; xs not counted.
    bytes_accessed = (2 * m * hidden + hidden) * DType.BF16.size_bytes()
    flops = 4 * m * hidden + 2 * m
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / seconds / 1e12) if time_ms > 0 else 0.0,
        memory_bandwidth_gbps=float(bytes_accessed / seconds / 1e9) if time_ms > 0 else 0.0,
        energy_j=float(energy_j),
    )


__all__ = ["profile_rms_norm_tensorfold_triton"]
