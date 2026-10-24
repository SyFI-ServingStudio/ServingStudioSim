"""TensorFold's Triton residual-add RMSNorm as a ``residual_rms_norm`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.families.qwen3_5.cuda.glue.add_rmsnorm(x, r, w, eps)``
(TensorFold perf-study 6469730a), the call ``forward.py`` (verify) and, through
``prefill_bf16``, ``prefill.py`` (bf16 prompts, the default) make before every
attention/GDN block and every MLP, and for the final norm:

- ``x`` is the running residual stream and ``r`` the previous block's output
  (the kind's ``residual`` / ``x``; the sum is symmetric), both ``(m, hidden)``
  bf16, contiguous; ``w`` is the ``(hidden,)`` bf16 norm weight.
- It returns ``(h, y, xs)``: ``h = bf16(x + r)`` (the new residual, the kind's
  ``s``), ``y = bf16(rmsnorm(h) * w)``, and ``xs``, fp32 sums of ``y`` over each
  64-element group, ``(m, hidden / 64)``. ``xs`` feeds the affine-4-bit matmul
  path; checkpoint (NVFP4/FP8) linears ignore it. It adds ``hidden / 64`` fp32
  writes per row (1/32 of one bf16 row), which the kernel always does.
- ``h``, ``y`` and ``xs`` are fresh ``torch.empty`` tensors (no launch), so the
  call is out of place and repeated calls see the same inputs.

The call is one launch of the Triton ``_add_rmsnorm`` (one program per row,
8 warps). The first layer's call (``r=None``) compiles a different
specialization that skips the residual read and the ``h`` write; that is a
plain RMSNorm and is not this kind. Triton JIT-compiles on first call, which the
correctness check absorbs before timing.
"""

from __future__ import annotations

import importlib
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.families.qwen3_5.cuda.glue"
_CALLABLE = "add_rmsnorm"
_KERNEL_NAME = "_add_rmsnorm"
_LABEL = "residual_rms_norm:tensorfold_triton"
# TensorFold passes the checkpoint's rms_norm_eps (1e-6 for Qwen3.8-27B); eps
# is a scalar argument and does not change the launch.
_EPS = 1e-6
_GROUP = 64
# Guard rows for the oracle: enough to cover several programs and keep the
# fp32 reference small whatever m is profiled.
_GUARD_ROWS = 67
# h is the bf16 rounding of an fp32 sum, so it matches exactly. y differs from
# the reference by bf16 rounding order (the kernel rounds once after the
# weight multiply, the reference before it): at most a couple of bf16 ulps.
Y_MAX_REL_TOL = 2e-2
XS_RMSE_RATIO_TOL = 1e-3


def validate_args(m: int, hidden: int, dtype: DType | str) -> None:
    """The kernel's contract: bf16 rows (it casts to ``tl.bfloat16``), hidden a multiple of 64."""
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} is bf16 only; got dtype={dtype}")
    if m <= 0 or hidden <= 0:
        raise ValueError(f"{_LABEL} needs m and hidden > 0, got m={m}, hidden={hidden}")
    if hidden % _GROUP:
        # The group-sum store indexes hidden // 64 groups a row; a ragged tail
        # would be dropped, which TensorFold never runs.
        raise ProfilerNotImplemented(f"{_LABEL} needs hidden a multiple of {_GROUP}, got {hidden}")


def build_operands(torch: Any, m: int, hidden: int) -> tuple[Any, Any, Any]:
    x = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda")
    r = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda")
    w = (1.0 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
    return x, r, w


def check_correctness(torch: Any, add_rmsnorm: Any, hidden: int) -> tuple[float, float]:
    """One call on guard rows against the Torch oracle; return (y max rel err, xs rmse ratio)."""
    from profiling.runners.norm.residual_rms_norm_reference import residual_rms_norm_reference

    x, r, w = build_operands(torch, _GUARD_ROWS, hidden)
    h, y, xs = add_rmsnorm(x, r, w, _EPS)
    torch.cuda.synchronize()
    if h.shape != x.shape or y.shape != x.shape or h.dtype != torch.bfloat16:
        raise AssertionError(f"{_LABEL} returned h {tuple(h.shape)} y {tuple(y.shape)} {h.dtype}")
    if xs.shape != (_GUARD_ROWS, hidden // _GROUP) or xs.dtype != torch.float32:
        raise AssertionError(f"{_LABEL} returned xs {tuple(xs.shape)} {xs.dtype}")
    want_y, want_h = residual_rms_norm_reference(x, r, w, _EPS)
    if not torch.equal(h, want_h):
        raise AssertionError(f"{_LABEL} residual sum differs from bf16(x + r)")
    y_err = float(((y.float() - want_y.float()).abs() / want_y.float().abs().clamp_min(1e-2)).max())
    if not y_err < Y_MAX_REL_TOL:
        raise AssertionError(f"{_LABEL} y max rel err {y_err:.2e} (tol {Y_MAX_REL_TOL:.0e})")
    want_xs = y.float().reshape(_GUARD_ROWS, hidden // _GROUP, _GROUP).sum(-1)
    xs_err = float((xs - want_xs).pow(2).mean().sqrt() / want_xs.pow(2).mean().sqrt())
    if not xs_err < XS_RMSE_RATIO_TOL:
        raise AssertionError(f"{_LABEL} xs rmse ratio {xs_err:.2e} (tol {XS_RMSE_RATIO_TOL:.0e})")
    return y_err, xs_err


def profile_residual_rms_norm_tensorfold_triton(
    m: int,
    hidden: int,
    dtype: DType | str,
) -> ComputeMetrics:
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
        x, r, w = build_operands(torch, m, hidden)

        def kernel() -> Any:
            return add_rmsnorm(x, r, w, _EPS)

        # One launch per call; the name filter also keeps CUPTI's L2 clear out.
        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)

        # The kind's logical traffic (x, residual, weight read; y and s
        # written), the same formula as the other backends. xs (m * hidden / 64
        # fp32) is a TensorFold-only extra and not counted.
        bytes_accessed = (4 * m * hidden + hidden) * DType.BF16.size_bytes()
        flops = 5 * m * hidden + 2 * m
        seconds = time_ms / 1000.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(flops / seconds / 1e12) if time_ms > 0 else 0.0,
            memory_bandwidth_gbps=float(bytes_accessed / seconds / 1e9) if time_ms > 0 else 0.0,
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc
