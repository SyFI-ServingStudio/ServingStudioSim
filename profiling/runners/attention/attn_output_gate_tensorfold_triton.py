"""TensorFold's Triton ``_gate_mul`` as the ``attn_output_gate`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.families.qwen3_5.cuda.glue.gate_mul(o, qg, heads=,
head_dim=)`` (TensorFold perf-study 6469730a), the call every gated-attention
layer makes between attention and the o projection, in verify (``forward.py``)
and, through ``prefill_bf16.gate_mul``, prefill (``prefill.py``):

- ``o`` is the attention output ``(W, heads, head_dim)`` bf16, contiguous;
  ``qg`` is the q projection output ``(W, heads * 2 * head_dim)``, whose gate
  halves the kernel reads in place.
- It returns ``(out, xs)``: ``out = bf16(o * sigmoid(gate))`` as
  ``(W, heads * head_dim)`` and ``xs``, fp32 sums of ``out`` over each
  64-element group. ``xs`` feeds the affine-4-bit matmul path; the kernel
  always writes it (1/32 of one bf16 row).

One launch of ``_gate_mul`` on grid ``(W, heads)``, two warps a program.
Triton JIT-compiles on first call, which the correctness check absorbs.
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
_CALLABLE = "gate_mul"
_KERNEL_NAME = "_gate_mul"
_LABEL = "attn_output_gate:tensorfold_triton"
_GROUP = 64
_GUARD_ROWS = 37
XS_RMSE_RATIO_TOL = 1e-3


def validate_args(num_tokens: int, num_heads: int, head_dim: int, dtype: DType | str) -> None:
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_heads", num_heads),
        ("head_dim", head_dim),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} is bf16 only; got dtype={dtype}")
    if head_dim % _GROUP or head_dim & (head_dim - 1):
        # tl.arange(0, D) needs a power of two and the group-sum reshape whole
        # 64-element groups; TensorFold never runs anything else.
        raise ProfilerNotImplemented(
            f"{_LABEL} needs head_dim a power of two and a multiple of {_GROUP}, got {head_dim}"
        )


def build_operands(torch: Any, rows: int, heads: int, head_dim: int) -> tuple[Any, Any]:
    o = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    qg = torch.randn(rows, heads * 2 * head_dim, dtype=torch.bfloat16, device="cuda")
    return o, qg


def check_correctness(torch: Any, gate_mul: Any, heads: int, head_dim: int) -> tuple[float, float]:
    """One call on guard rows against the Torch oracle; return (out max abs err, xs rmse ratio)."""
    from profiling.runners.attention.attn_output_gate_reference import attn_output_gate_reference

    o, qg = build_operands(torch, _GUARD_ROWS, heads, head_dim)
    out, xs = gate_mul(o, qg, heads=heads, head_dim=head_dim)
    torch.cuda.synchronize()
    if out.shape != (_GUARD_ROWS, heads * head_dim) or out.dtype != torch.bfloat16:
        raise AssertionError(f"{_LABEL} returned out {tuple(out.shape)} {out.dtype}")
    if xs.shape != (_GUARD_ROWS, heads * head_dim // _GROUP) or xs.dtype != torch.float32:
        raise AssertionError(f"{_LABEL} returned xs {tuple(xs.shape)} {xs.dtype}")
    want = attn_output_gate_reference(torch, o, qg, head_dim)
    # One fp32 multiply and one bf16 rounding on both sides: equal up to a bf16 ulp.
    out_err = float((out.float() - want.float()).abs().max())
    if not out_err < 2e-2:
        raise AssertionError(f"{_LABEL} out max abs err {out_err:.2e}")
    want_xs = out.float().reshape(_GUARD_ROWS, -1, _GROUP).sum(-1)
    xs_err = float((xs - want_xs).pow(2).mean().sqrt() / want_xs.pow(2).mean().sqrt())
    if not xs_err < XS_RMSE_RATIO_TOL:
        raise AssertionError(f"{_LABEL} xs rmse ratio {xs_err:.2e} (tol {XS_RMSE_RATIO_TOL:.0e})")
    return out_err, xs_err


def profile_attn_output_gate_tensorfold_triton(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    validate_args(num_tokens, num_heads, head_dim, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        gate_mul = getattr(importlib.import_module(_MODULE), _CALLABLE)
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"{_LABEL} needs {_MODULE}.{_CALLABLE} (tensorfold_env)"
        ) from exc

    try:
        check_correctness(torch, gate_mul, num_heads, head_dim)
        o, qg = build_operands(torch, num_tokens, num_heads, head_dim)

        def kernel() -> Any:
            return gate_mul(o, qg, heads=num_heads, head_dim=head_dim)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc

    # Output, gate and result once each (bf16); xs is a TensorFold-only extra.
    nbytes = 6 * num_tokens * num_heads * head_dim
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(nbytes / seconds / 1e9) if time_ms > 0 else 0.0,
        energy_j=float(energy_j),
    )


__all__ = ["profile_attn_output_gate_tensorfold_triton"]
