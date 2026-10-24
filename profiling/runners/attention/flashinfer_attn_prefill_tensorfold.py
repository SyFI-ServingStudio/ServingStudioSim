"""TensorFold prompt attention as a ``flashinfer_attn_prefill`` backend.

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.cuda.kernels.prefill_attention.attention(q, k_cache,
v_cache, p0, scale=)`` (TensorFold perf-study 6469730a), called the way
``families/qwen3_5/cuda/prefill.py`` calls it for each stream of a prefill step:

- ``q`` is ``(append_len, num_qo_heads, head_dim)`` bf16, contiguous, the rows
  at positions ``[p0, p0 + append_len)`` (already normed and roped; the output
  gate is a separate kernel and is not here).
- ``k_cache`` / ``v_cache`` are the stream's contiguous bf16 caches
  ``(cap, num_kv_heads, head_dim)``. The step writes the chunk's keys and values
  into rows ``[p0, p0 + append_len)`` by torch slicing before the call; that
  write is not part of this kernel. The kernel reads keys below
  ``p0 + append_len`` only, so ``cap = p0 + append_len`` here.
- ``p0 = prefix_len``, the committed length before the chunk; query row ``i``
  attends causally to keys ``[0, p0 + i]``.
- ``scale = head_dim ** -0.5``.

For head_dim 128 and 256 the call is one launch of the CUDA ``pattn_kernel``
(``prefill_attention.cu``); for head_dim 64 it is the Triton ``_attend``. Both
are inside the public callable, so the time is the CUPTI sum over every launch
of one call (``kernel_name=None``); the output allocation launches nothing.
The extension JIT-compiles on first call, which the correctness check absorbs
before timing.
"""

from __future__ import annotations

import importlib
from typing import Any

from profiling.db.args import DType
from profiling.runners.attention import _common
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.cuda.kernels.prefill_attention"
_CALLABLE = "attention"
_HEAD_DIMS = (64, 128, 256)
# Guard shape for the oracle: crosses the 64-key tile and the 64-row CTA block
# boundaries, and keeps the fp32 reference small whatever the profiled shape.
_GUARD_PREFIX = 131
_GUARD_APPEND = 77
# The kernel rounds P to bf16 for the PV product and stores bf16 output; on
# randn inputs the measured RMSE ratio against the fp32 oracle is ~3e-3.
RMSE_RATIO_TOL = 1e-2


def validate_args(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
    prefix_len: int,
    append_len: int,
) -> None:
    """The public callable's contract (``prefill_attention._check``): bf16 throughout."""
    label = "flashinfer_attn_prefill:tensorfold_pattn"
    for name, dtype in (("q_dtype", q_dtype), ("kv_dtype", kv_dtype), ("o_dtype", o_dtype)):
        if DType.from_value(dtype) is not DType.BF16:
            raise ProfilerNotImplemented(f"{label} is bf16 only; got {name}={dtype}")
    if head_dim not in _HEAD_DIMS:
        raise ProfilerNotImplemented(f"{label} supports head_dim in {_HEAD_DIMS}, got {head_dim}")
    if num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError(f"{label} needs num_qo_heads a multiple of num_kv_heads")
    if prefix_len < 0 or append_len < 1:
        raise ValueError(f"{label} needs prefix_len >= 0 and append_len >= 1")


def build_operands(
    torch: Any,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    prefix_len: int,
    append_len: int,
) -> tuple[Any, Any, Any]:
    """``q`` at the chunk's rows and caches holding every key through ``p0 + W - 1``."""
    kv_rows = prefix_len + append_len
    q = torch.randn(append_len, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(kv_rows, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn(kv_rows, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    return q, k_cache, v_cache


def reference(torch: Any, q: Any, k_cache: Any, v_cache: Any, p0: int, scale: float) -> Any:
    """Independent fp32 oracle: SDPA with an explicit causal-with-prefix mask."""
    import torch.nn.functional as F

    w, h, _ = q.shape
    hk = k_cache.shape[1]
    kv_rows = p0 + w
    group = h // hk
    # Query head h reads KV head h // group (GQA, heads grouped contiguously).
    k = k_cache[:kv_rows].float().repeat_interleave(group, dim=1).transpose(0, 1)
    v = v_cache[:kv_rows].float().repeat_interleave(group, dim=1).transpose(0, 1)
    qf = q.float().transpose(0, 1)
    rows = torch.arange(w, device=q.device)[:, None] + p0
    keys = torch.arange(kv_rows, device=q.device)[None, :]
    mask = keys <= rows  # (W, kv_rows): row i sees keys [0, p0 + i]
    out = F.scaled_dot_product_attention(qf, k, v, attn_mask=mask, scale=scale)
    return out.transpose(0, 1)


def check_correctness(
    torch: Any,
    attention: Any,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    prefix_len: int,
    append_len: int,
) -> float:
    """One call on the guard shape against the oracle; return its RMSE ratio."""
    p0 = min(prefix_len, _GUARD_PREFIX)
    w = min(append_len, _GUARD_APPEND)
    q, k_cache, v_cache = build_operands(
        torch,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        prefix_len=p0,
        append_len=w,
    )
    scale = head_dim**-0.5
    out = attention(q, k_cache, v_cache, p0, scale=scale)
    torch.cuda.synchronize()
    if out.shape != q.shape or out.dtype != torch.bfloat16:
        raise AssertionError(f"tensorfold attention returned {tuple(out.shape)} {out.dtype}")
    if not torch.isfinite(out).all():
        raise AssertionError("tensorfold attention returned non-finite values")
    expected = reference(torch, q, k_cache, v_cache, p0, scale)
    err = float(
        (out.float() - expected).pow(2).mean().sqrt()
        / expected.pow(2).mean().sqrt().clamp_min(1e-12)
    )
    if not err < RMSE_RATIO_TOL:
        raise AssertionError(
            f"tensorfold attention mismatch vs fp32 SDPA oracle at p0={p0}, W={w}: "
            f"rmse ratio {err:.2e} (tol {RMSE_RATIO_TOL:.0e})"
        )
    return err


def profile_flashinfer_attn_prefill_tensorfold_pattn(
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
    prefix_len: int,
    append_len: int,
) -> ComputeMetrics:
    num_qo_heads, num_kv_heads, head_dim = int(num_qo_heads), int(num_kv_heads), int(head_dim)
    prefix_len, append_len = int(prefix_len), int(append_len)
    validate_args(
        num_qo_heads, num_kv_heads, head_dim, q_dtype, kv_dtype, o_dtype, prefix_len, append_len
    )
    label = "flashinfer_attn_prefill:tensorfold_pattn"
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {label}") from exc
    try:
        attention = getattr(importlib.import_module(_MODULE), _CALLABLE)
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"{label} needs {_MODULE}.{_CALLABLE} (tensorfold_env)"
        ) from exc

    dims = dict(num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
    try:
        check_correctness(torch, attention, **dims, prefix_len=prefix_len, append_len=append_len)
        q, k_cache, v_cache = build_operands(
            torch, **dims, prefix_len=prefix_len, append_len=append_len
        )
        scale = head_dim**-0.5

        def kernel() -> Any:
            return attention(q, k_cache, v_cache, prefix_len, scale=scale)

        flops = _common.attention_flops(
            q_len=append_len,
            kv_len=prefix_len + append_len,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            causal=True,
        )
        bytes_accessed = 2 * (2 * q.numel() + k_cache.numel() + v_cache.numel())
        return _common.measure(kernel, flops=flops, bytes_accessed=bytes_accessed)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{label} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{label} failed: {exc}") from exc
