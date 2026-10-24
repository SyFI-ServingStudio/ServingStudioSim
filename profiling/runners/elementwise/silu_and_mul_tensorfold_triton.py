"""TensorFold's Triton ``_swiglu`` on a stacked projection as a ``silu_and_mul`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.families.qwen3_5.cuda.glue.swiglu_stacked(gu)``
(TensorFold perf-study 6469730a), the call ``forward.py`` ``_mlp`` makes in
verify after the stacked gate|up lane GEMM (prefill uses the fused
``gemm_swiglu_quant`` instead):

- ``gu`` is the ``(W, 2 * hidden_dim)`` bf16 GEMM output, gate half then up
  half, exactly the kind's input row; the kernel reads both halves in place
  (row stride ``2 * hidden_dim``).
- It returns ``(out, xs)``: ``out = bf16(silu(gate) * up)`` computed in fp32,
  ``(W, hidden_dim)``, and ``xs``, fp32 sums of ``out`` over each 64-element
  group, which the kernel always writes (1/32 of one bf16 output row).

One launch of ``_swiglu`` on grid ``(W, cdiv(hidden_dim, 1024))``, four warps a
program. Triton JIT-compiles on first call, which the correctness check
absorbs before timing.
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
_CALLABLE = "swiglu_stacked"
_KERNEL_NAME = "_swiglu"
_LABEL = "silu_and_mul:tensorfold_triton"
_GROUP = 64
# The kernel's BLOCK; the group-sum reshape needs it to be whole groups.
_BLOCK = 1024
_GUARD_ROWS = 37
# The kernel addresses row `row * LD` with LD = 2 * hidden_dim in int32
# (program_id times a constexpr), so a stacked input past 2**31 elements wraps
# and faults (65536 rows at hidden 17408 hit cudaErrorIllegalAddress on GB10).
# TensorFold's verify windows are at most 256 rows.
MAX_STACKED_ELEMENTS = 2**31 - 1
XS_RMSE_RATIO_TOL = 1e-3


def validate_args(num_tokens: int, hidden_dim: int, dtype: DType | str) -> None:
    if num_tokens <= 0 or hidden_dim <= 0:
        raise ValueError(
            f"{_LABEL} needs num_tokens and hidden_dim > 0, got {num_tokens}, {hidden_dim}"
        )
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} is bf16 only; got dtype={dtype}")
    if hidden_dim % _GROUP:
        # xs has hidden_dim // 64 groups a row; a ragged tail group would be
        # dropped, which TensorFold never runs.
        raise ProfilerNotImplemented(
            f"{_LABEL} needs hidden_dim a multiple of {_GROUP}, got {hidden_dim}"
        )
    if num_tokens * 2 * hidden_dim > MAX_STACKED_ELEMENTS:
        raise ProfilerNotImplemented(
            f"{_LABEL} indexes rows in int32: num_tokens * 2 * hidden_dim must stay "
            f"under 2**31, got {num_tokens} x {2 * hidden_dim}"
        )


def check_correctness(torch: Any, swiglu_stacked: Any, hidden_dim: int) -> tuple[float, float]:
    """One call on guard rows against the Torch oracle; return (out max abs err, xs rmse ratio)."""
    from profiling.runners.elementwise.silu_and_mul_reference import silu_and_mul_reference

    gu = torch.randn(_GUARD_ROWS, 2 * hidden_dim, dtype=torch.bfloat16, device="cuda")
    out, xs = swiglu_stacked(gu)
    torch.cuda.synchronize()
    if out.shape != (_GUARD_ROWS, hidden_dim) or out.dtype != torch.bfloat16:
        raise AssertionError(f"{_LABEL} returned out {tuple(out.shape)} {out.dtype}")
    if xs.shape != (_GUARD_ROWS, hidden_dim // _GROUP) or xs.dtype != torch.float32:
        raise AssertionError(f"{_LABEL} returned xs {tuple(xs.shape)} {xs.dtype}")
    want = silu_and_mul_reference(torch, gu)
    out_err = float((out.float() - want.float()).abs().max())
    if not out_err < 5e-2:
        raise AssertionError(f"{_LABEL} out max abs err {out_err:.2e}")
    want_xs = out.float().reshape(_GUARD_ROWS, -1, _GROUP).sum(-1)
    xs_err = float((xs - want_xs).pow(2).mean().sqrt() / want_xs.pow(2).mean().sqrt())
    if not xs_err < XS_RMSE_RATIO_TOL:
        raise AssertionError(f"{_LABEL} xs rmse ratio {xs_err:.2e} (tol {XS_RMSE_RATIO_TOL:.0e})")
    return out_err, xs_err


def profile_silu_and_mul_tensorfold_triton(
    num_tokens: int,
    hidden_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    num_tokens, hidden_dim = int(num_tokens), int(hidden_dim)
    validate_args(num_tokens, hidden_dim, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        swiglu_stacked = getattr(importlib.import_module(_MODULE), _CALLABLE)
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"{_LABEL} needs {_MODULE}.{_CALLABLE} (tensorfold_env)"
        ) from exc

    try:
        check_correctness(torch, swiglu_stacked, hidden_dim)
        gu = torch.randn(num_tokens, 2 * hidden_dim, dtype=torch.bfloat16, device="cuda")

        def kernel() -> Any:
            return swiglu_stacked(gu)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc

    # The kind's count: both input halves and the output once (bf16).
    nbytes = 6 * num_tokens * hidden_dim
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(nbytes / seconds / 1e9) if time_ms > 0 else 0.0,
        energy_j=float(energy_j),
    )


__all__ = ["profile_silu_and_mul_tensorfold_triton"]
