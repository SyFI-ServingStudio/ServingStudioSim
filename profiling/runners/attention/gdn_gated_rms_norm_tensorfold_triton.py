"""TensorFold's Triton GDN gated RMSNorm as a ``gdn_gated_rms_norm`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.families.qwen3_5.cuda.glue.gated_norm(y, z, w, eps)``
(TensorFold perf-study 6469730a), the call ``forward.py`` makes after the GDN
tree recurrence in verify (``tree_forward`` and ``multi_tree_forward``) and,
through ``prefill_bf16`` (bf16 prompts, the default), ``prefill.py`` makes
after the chunked recurrence:

- ``y`` is the recurrence output and ``z`` the output gate, both
  ``(W, VH, DV)`` bf16, contiguous; ``w`` is the ``(DV,)`` bf16 per-head norm
  weight. The kind's ``m`` is ``W * VH`` rows and ``hidden`` is ``DV``.
- It returns ``(out, xs)``: ``out = bf16(silu(z) * rmsnorm(y) * w)`` as
  ``(W, VH * DV)`` and ``xs``, fp32 sums of ``out`` over each 64-element group,
  ``DV / 64`` floats a row. ``xs`` feeds the next matmul's activation path; it
  is 1/32 of one bf16 row of extra writes, which the kernel always does.
- Both outputs are fresh ``torch.empty`` tensors (no launch), so the call is out
  of place and repeated calls see the same inputs.

The call is one launch of the Triton ``_gated_norm`` on grid ``(W, VH)``, one
single-warp program a head row. Production rows are therefore visited head-
strided: neighbouring programs read rows ``VH`` apart. A flat ``(m, 1, DV)``
launch runs the same code with contiguous neighbours and is measurably faster
once traffic leaves L2 (about 10-15% at 12k-49k rows on GB10), while the
head count itself barely matters (16 to 64 heads within a few percent). The
kind has no head field, so the runner launches ``(m / VH, VH, DV)`` with VH
the largest power of two up to 64 that divides ``m``: production's head-strided
order without a model constant. Triton JIT-compiles on first call, which the
correctness check absorbs before timing.
"""

from __future__ import annotations

import importlib
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.gdn_gated_rms_norm_torch import (
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.families.qwen3_5.cuda.glue"
_CALLABLE = "gated_norm"
_KERNEL_NAME = "_gated_norm"
_LABEL = "gdn_gated_rms_norm:tensorfold_triton"
# TensorFold passes the checkpoint's rms_norm_eps (1e-6 for Qwen3.8-27B), the
# same epsilon the kind's reference freezes; eps does not change the launch.
_EPS = 1e-6
_GROUP = 64
_MAX_HEADS = 64
# Guard rows for the oracle: several token rows of a full head split, small
# enough to keep the fp32 reference cheap whatever m is profiled.
_GUARD_ROWS = 3 * _MAX_HEADS
# out differs from the fp32 reference only by bf16 rounding of the final value.
OUT_MAX_REL_TOL = 2e-2
XS_RMSE_RATIO_TOL = 1e-3


def validate_args(m: int, hidden: int, dtype: DType | str) -> None:
    """The kernel's contract: bf16 rows of a power-of-two width that is a multiple of 64."""
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} is bf16 only; got dtype={dtype}")
    if m <= 0 or hidden <= 0:
        raise ValueError(f"{_LABEL} needs m and hidden > 0, got m={m}, hidden={hidden}")
    if hidden % _GROUP or hidden & (hidden - 1):
        # tl.arange(0, DV) needs a power of two, and the group-sum reshape needs
        # DV // 64 whole groups; TensorFold never runs anything else.
        raise ProfilerNotImplemented(
            f"{_LABEL} needs hidden a power of two and a multiple of {_GROUP}, got {hidden}"
        )


def heads_per_row(m: int) -> int:
    """The largest power of two up to 64 that divides ``m`` (1 for odd ``m``)."""
    heads = 1
    while heads < _MAX_HEADS and m % (2 * heads) == 0:
        heads *= 2
    return heads


def build_operands(torch: Any, m: int, hidden: int) -> tuple[Any, Any, Any]:
    heads = heads_per_row(m)
    shape = (m // heads, heads, hidden)
    y = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    z = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    w = (1.0 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
    return y, z, w


def check_correctness(torch: Any, gated_norm: Any, hidden: int) -> tuple[float, float]:
    """One call on guard rows against the Torch oracle; return (out max rel err, xs rmse ratio)."""
    from profiling.runners.attention.gdn_gated_rms_norm_reference import (
        gdn_gated_rms_norm_reference,
    )

    y, z, w = build_operands(torch, _GUARD_ROWS, hidden)
    tokens, heads, _ = y.shape
    out, xs = gated_norm(y, z, w, _EPS)
    torch.cuda.synchronize()
    if out.shape != (tokens, heads * hidden) or out.dtype != torch.bfloat16:
        raise AssertionError(f"{_LABEL} returned out {tuple(out.shape)} {out.dtype}")
    if xs.shape != (tokens, heads * hidden // _GROUP) or xs.dtype != torch.float32:
        raise AssertionError(f"{_LABEL} returned xs {tuple(xs.shape)} {xs.dtype}")
    rows = out.reshape(_GUARD_ROWS, hidden)
    want = gdn_gated_rms_norm_reference(
        y.reshape(_GUARD_ROWS, hidden), z.reshape(_GUARD_ROWS, hidden), w
    ).float()
    out_err = float(((rows.float() - want).abs() / want.abs().clamp_min(1e-2)).max())
    if not out_err < OUT_MAX_REL_TOL:
        raise AssertionError(f"{_LABEL} out max rel err {out_err:.2e} (tol {OUT_MAX_REL_TOL:.0e})")
    want_xs = rows.float().reshape(tokens, heads * hidden // _GROUP, _GROUP).sum(-1)
    xs_err = float((xs - want_xs).pow(2).mean().sqrt() / want_xs.pow(2).mean().sqrt())
    if not xs_err < XS_RMSE_RATIO_TOL:
        raise AssertionError(f"{_LABEL} xs rmse ratio {xs_err:.2e} (tol {XS_RMSE_RATIO_TOL:.0e})")
    return out_err, xs_err


def profile_gdn_gated_rms_norm_tensorfold_triton(
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
        gated_norm = getattr(importlib.import_module(_MODULE), _CALLABLE)
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(f"{_LABEL} needs {_MODULE}.{_CALLABLE} (tensorfold_env)") from exc

    try:
        check_correctness(torch, gated_norm, hidden)
        y, z, w = build_operands(torch, m, hidden)

        def kernel() -> Any:
            return gated_norm(y, z, w, _EPS)

        # One launch per call; the name filter also keeps CUPTI's L2 clear out.
        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)

        # The kind's logical traffic (y, z, weight read; out written), the same
        # formula as the other backends. xs (m * hidden / 64 fp32) is a
        # TensorFold-only extra and not counted.
        flops = _semantic_flops(m=m, hidden=hidden)
        bytes_accessed = _logical_bytes(m=m, hidden=hidden, dtype=DType.BF16)
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
