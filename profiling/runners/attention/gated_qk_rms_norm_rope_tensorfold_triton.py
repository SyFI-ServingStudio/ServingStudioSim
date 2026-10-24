"""TensorFold's Triton ``_attn_prep`` as the ``gated_qk_rms_norm_rope`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.families.qwen3_5.cuda.glue.attn_prep(qg, k, q_norm,
k_norm, pos, inv_freq, eps, heads=, kv_heads=, head_dim=, mrope_section=)``
(TensorFold perf-study 6469730a), the call every gated-attention layer makes
after its q and k projections, in verify (``forward.py`` tree_forward /
multi_tree_forward) and prefill (``prefill.py``):

- ``qg`` is the q projection output ``(W, heads * 2 * head_dim)`` bf16, each
  head's query beside its gate; ``k`` is ``(W, kv_heads * head_dim)``; both are
  contiguous GEMM outputs. ``q_norm``/``k_norm`` are ``(head_dim,)`` bf16.
- ``pos`` is ``(W,)`` int32 for text or ``(3, W)`` for multimodal rows;
  ``inv_freq`` is the fp32 ``(rotary_dim / 2,)`` vector, so the kernel's
  ``HALF`` (and the rotated width) comes from its length.
- It returns fresh ``(W, heads, head_dim)`` q and ``(W, kv_heads, head_dim)``
  k. The gate is not read or copied.

One launch of ``_attn_prep`` on grid ``(W, heads + kv_heads)``, two warps a
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
_CALLABLE = "attn_prep"
_KERNEL_NAME = "_attn_prep"
_LABEL = "gated_qk_rms_norm_rope:tensorfold_triton"
# TensorFold passes the checkpoint's rms_norm_eps (1e-6 for Qwen3.8-27B) and
# rope_theta (1e7); both are scalars or data and do not change the launch.
_EPS = 1e-6
_ROPE_THETA = 10_000_000.0
# Verify positions sit deep in a context; the angle is computed per element,
# so the value does not change the time.
_POSITION_BASE = 30_000
_GUARD_ROWS = 37
# q/k are bf16 roundings of an fp32 rotation; sin/cos and rsqrt differ from
# Torch's by a few fp32 ulps, which can flip a bf16 rounding.
MAX_ABS_TOL = 3e-2


def validate_args(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    dtype: DType | str,
) -> None:
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_q_heads", num_q_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("rotary_dim", rotary_dim),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and <= head_dim, got {rotary_dim}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} is bf16 only; got dtype={dtype}")
    if head_dim & (head_dim - 1):
        # tl.arange(0, D) needs a power of two; TensorFold never runs anything else.
        raise ProfilerNotImplemented(f"{_LABEL} needs head_dim a power of two, got {head_dim}")


def mrope_section(rotary_dim: int) -> tuple[int, int, int]:
    """A split summing to rotary_dim / 2 (Qwen3.5's (11, 11, 10) at rotary_dim 64)."""
    half = rotary_dim // 2
    return half - 2 * (half // 3), half // 3, half // 3


def build_operands(
    torch: Any, rows: int, heads: int, kv_heads: int, head_dim: int, rotary_dim: int, mrope: bool
) -> tuple[Any, ...]:
    dev = "cuda"
    qg = torch.randn(rows, heads * 2 * head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(rows, kv_heads * head_dim, dtype=torch.bfloat16, device=dev)
    q_norm = (1.0 + 0.1 * torch.randn(head_dim, device=dev)).to(torch.bfloat16)
    k_norm = (1.0 + 0.1 * torch.randn(head_dim, device=dev)).to(torch.bfloat16)
    text = torch.arange(_POSITION_BASE, _POSITION_BASE + rows, device=dev, dtype=torch.int32)
    # Text rows carry the same position in all three multimodal rows.
    pos = text.expand(3, rows).contiguous() if mrope else text
    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=dev) / rotary_dim
    inv_freq = (1.0 / _ROPE_THETA**exponents).float()
    return qg, k, q_norm, k_norm, pos, inv_freq


def check_correctness(
    torch: Any,
    attn_prep: Any,
    heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope: bool,
) -> float:
    """One call on guard rows against the Torch oracle; return the max abs error."""
    from profiling.runners.attention.gated_qk_rms_norm_rope_reference import (
        gated_qk_rms_norm_rope_reference,
    )

    qg, k, q_norm, k_norm, pos, inv_freq = build_operands(
        torch, _GUARD_ROWS, heads, kv_heads, head_dim, rotary_dim, mrope
    )
    # Distinct multimodal rows so a wrong section mapping shows up.
    if mrope:
        pos = pos + torch.tensor([[0], [7], [13]], device=pos.device, dtype=pos.dtype)
    section = mrope_section(rotary_dim) if mrope else None
    q, key = attn_prep(
        qg,
        k,
        q_norm,
        k_norm,
        pos,
        inv_freq,
        _EPS,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        **({"mrope_section": section} if mrope else {}),
    )
    torch.cuda.synchronize()
    want_q, want_k = gated_qk_rms_norm_rope_reference(
        torch, qg, k, q_norm, k_norm, pos, inv_freq, _EPS, head_dim, section
    )
    if q.shape != want_q.shape or key.shape != want_k.shape or q.dtype != torch.bfloat16:
        raise AssertionError(f"{_LABEL} returned q {tuple(q.shape)} k {tuple(key.shape)} {q.dtype}")
    err = max(
        float((q.float() - want_q.float()).abs().max()),
        float((key.float() - want_k.float()).abs().max()),
    )
    if not err < MAX_ABS_TOL:
        raise AssertionError(f"{_LABEL} max abs err {err:.2e} (tol {MAX_ABS_TOL:.0e})")
    return err


def logical_bytes(num_tokens: int, num_q_heads: int, num_kv_heads: int, head_dim: int) -> int:
    """q and k heads read once and written once, bf16."""
    return 4 * num_tokens * head_dim * (num_q_heads + num_kv_heads)


def profile_gated_qk_rms_norm_rope_tensorfold_triton(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope: bool,
    dtype: DType | str,
) -> ComputeMetrics:
    validate_args(num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim, dtype)
    mrope = bool(mrope)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        attn_prep = getattr(importlib.import_module(_MODULE), _CALLABLE)
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"{_LABEL} needs {_MODULE}.{_CALLABLE} (tensorfold_env)"
        ) from exc

    try:
        check_correctness(torch, attn_prep, num_q_heads, num_kv_heads, head_dim, rotary_dim, mrope)
        qg, k, q_norm, k_norm, pos, inv_freq = build_operands(
            torch, num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim, mrope
        )
        extra = {"mrope_section": mrope_section(rotary_dim)} if mrope else {}

        def kernel() -> Any:
            return attn_prep(
                qg,
                k,
                q_norm,
                k_norm,
                pos,
                inv_freq,
                _EPS,
                heads=num_q_heads,
                kv_heads=num_kv_heads,
                head_dim=head_dim,
                **extra,
            )

        # One launch per call; the name filter also keeps CUPTI's L2 clear out.
        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc

    seconds = time_ms / 1000.0
    nbytes = logical_bytes(num_tokens, num_q_heads, num_kv_heads, head_dim)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(nbytes / seconds / 1e9) if time_ms > 0 else 0.0,
        energy_j=float(energy_j),
    )


__all__ = ["profile_gated_qk_rms_norm_rope_tensorfold_triton"]
