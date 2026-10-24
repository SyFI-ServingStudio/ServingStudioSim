"""TensorFold's Triton ``_prep_kernel`` as the ``head_major_qk_norm_rope`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``DFlash2._prep(qkv, layer, cos, sin, heads)``
(``tensorfold/families/qwen3_5/cuda/dflash2.py``, perf-study 6469730a) on a
stand-in drafter that holds only what the method reads (``device``,
``kv_local``, ``head_dim``, ``eps`` and the layer's q/k norm weights):

- drafting (``_layer_fast``): ``qkv`` is the fused qkv projection
  ``(R, (heads + 2 kv) · D)`` and ``heads`` the query heads;
- the context taps (``add_taps_streams``, ``add_taps``): ``qkv`` is the kv
  projection ``(R, 2 kv · D)`` and ``heads`` = 0, so only k and v are written;
- ``cos``/``sin`` are the fp32 ``(R, D / 2)`` rotary tables.

One ``_prep_kernel`` launch on grid ``(R, heads + 2 kv)``, one warp a program.
"""

from __future__ import annotations

import importlib
import types
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.families.qwen3_5.cuda.dflash2"
_KERNEL_NAME = "_prep_kernel"
_LABEL = "head_major_qk_norm_rope:tensorfold_triton"
# The drafter's rms_norm_eps and rope_theta (Qwen3.8-27B-DFlash2 config).
_EPS = 1e-6
_THETA = 10_000_000.0
_GUARD_ROWS = 37


def validate_args(
    num_tokens: int, num_q_heads: int, num_kv_heads: int, head_dim: int, dtype: DType | str
) -> None:
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if type(num_q_heads) is not int or num_q_heads < 0:
        raise ValueError(f"num_q_heads must be a non-negative integer, got {num_q_heads!r}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} is bf16 only, got {dtype}")
    half = head_dim // 2
    if head_dim % 2 or half & (half - 1):
        raise ProfilerNotImplemented(f"{_LABEL} needs head_dim / 2 a power of two, got {head_dim}")


def build(torch: Any, rows: int, num_q_heads: int, num_kv_heads: int, head_dim: int):
    owner = types.SimpleNamespace(
        device=torch.device("cuda"),
        kv_local=num_kv_heads,
        head_dim=head_dim,
        eps=_EPS,
        weights={
            "layers.0.self_attn.q_norm.weight": (1 + 0.1 * torch.randn(head_dim, device="cuda")).to(
                torch.bfloat16
            ),
            "layers.0.self_attn.k_norm.weight": (1 + 0.1 * torch.randn(head_dim, device="cuda")).to(
                torch.bfloat16
            ),
        },
    )
    qkv = torch.randn(rows, (num_q_heads + 2 * num_kv_heads) * head_dim, device="cuda").to(
        torch.bfloat16
    )
    inv_freq = 1.0 / _THETA ** (
        torch.arange(head_dim // 2, device="cuda", dtype=torch.float32) * 2 / head_dim
    )
    phase = (
        torch.arange(4096, 4096 + rows, device="cuda", dtype=torch.float32)[:, None]
        * inv_freq[None, :]
    )
    return owner, qkv, phase.cos().contiguous(), phase.sin().contiguous()


def profile_head_major_qk_norm_rope_tensorfold_triton(
    num_tokens: int, num_q_heads: int, num_kv_heads: int, head_dim: int, dtype: DType | str
) -> ComputeMetrics:
    validate_args(num_tokens, num_q_heads, num_kv_heads, head_dim, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        prep = importlib.import_module(_MODULE).DFlash2._prep
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"{_LABEL} needs {_MODULE}.DFlash2._prep (tensorfold_env)"
        ) from exc
    from profiling.runners.attention.head_major_qk_norm_rope_reference import (
        head_major_qk_norm_rope_reference,
    )

    try:
        owner, qkv, cos, sin = build(torch, _GUARD_ROWS, num_q_heads, num_kv_heads, head_dim)
        q, k, v = prep(owner, qkv, 0, cos, sin, num_q_heads)
        wq, wk, wv = head_major_qk_norm_rope_reference(
            torch,
            qkv,
            owner.weights["layers.0.self_attn.q_norm.weight"],
            owner.weights["layers.0.self_attn.k_norm.weight"],
            cos,
            sin,
            _EPS,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
        pairs = [(k, wk), (v, wv)] + ([(q, wq)] if num_q_heads else [])
        for got, want in pairs:
            if got.shape != want.shape:
                raise AssertionError(
                    f"{_LABEL} returned {tuple(got.shape)}, want {tuple(want.shape)}"
                )
            err = float((got.float() - want.float()).abs().max())
            if not err < 5e-2:
                raise AssertionError(f"{_LABEL} max abs err {err:.2e}")
        owner, qkv, cos, sin = build(torch, num_tokens, num_q_heads, num_kv_heads, head_dim)
        torch.cuda.synchronize()

        def kernel() -> Any:
            return prep(owner, qkv, 0, cos, sin, num_q_heads)

        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, AssertionError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc

    nbytes = 4 * num_tokens * (num_q_heads + 2 * num_kv_heads) * head_dim
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(nbytes / seconds / 1e9) if time_ms > 0 else 0.0,
        energy_j=float(energy_j),
    )


__all__ = ["profile_head_major_qk_norm_rope_tensorfold_triton"]
