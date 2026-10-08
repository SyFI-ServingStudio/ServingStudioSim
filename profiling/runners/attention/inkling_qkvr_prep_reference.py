"""Independent Torch reference for Inkling's fused QKVR prep.

Written from the layer math in the alignment fork
(``vllm/models/inkling/nvidia/attention.py``: ``InklingShortConv`` on K and V,
``InklingRMSNorm`` on q and k, ``compute_log_scaling_tau``, ``RelLogitsProj``)
and not from ``ops/qkvr_prep.py``. It shares no code with the Triton kernels:
the tap source is decided by absolute position (a tap at or after the
request's first new position comes from this forward's ``qkvr``, an earlier one
from the conv-state cache), where the kernels compare row indices.

One call per layer, per token ``t`` of request ``r`` at position ``p``:

- ``k' = bf16(k_t + sum_{i<W} w_k[:, i] * k(p - W + 1 + i))`` and the same for
  ``v``, with zero taps before position 0 (depthwise causal conv + residual);
- ``k_hat = bf16(rmsnorm(k') * k_norm_w)`` and
  ``q_hat = bf16(rmsnorm(q) * q_norm_w)``, then on global layers
  ``q_hat = bf16(q_hat * tau_t)``;
- ``rel[t, h, :] = bf16(r[t, h, :] @ W_rel)`` and on global layers
  ``bf16(rel * tau_t)``;
- the raw ``k_t, v_t`` go to the conv-state cache at the token's conv slot and
  ``k_hat, v'`` go to the paged attention cache at its attention slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrepShape:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    sconv_kernel_size: int

    @property
    def q_width(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def row_width(self) -> int:
        """Elements of one ``qkvr`` row: q | k | v | r."""
        return self.q_width + 2 * self.kv_width + self.num_q_heads * self.d_rel


def _rmsnorm(torch: Any, x: Any, weight: Any, eps: float) -> Any:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()


def reference_conv(
    torch: Any,
    shape: PrepShape,
    qkvr: Any,
    k_weight: Any,
    v_weight: Any,
    positions: Any,
    first_new_position: Any,
    first_row: Any,
    cached_tap: Any,
) -> tuple[Any, Any]:
    """K and V after the short conv and residual, bf16, shape ``[T, kv_width]``.

    ``first_new_position[t]`` / ``first_row[t]`` are the position and row of the
    first token its request contributes to this forward. ``cached_tap(pos, t)``
    returns the pre-forward conv-cache ``(k, v)`` rows at absolute positions
    ``pos`` of token ``t``'s request (only called where ``pos`` is cached).
    """

    q_w, kv_w, width = shape.q_width, shape.kv_width, shape.sconv_kernel_size
    k = qkvr[:, q_w : q_w + kv_w].float()
    v = qkvr[:, q_w + kv_w : q_w + 2 * kv_w].float()
    acc_k = torch.zeros_like(k)
    acc_v = torch.zeros_like(v)
    rows = torch.arange(qkvr.shape[0], device=qkvr.device)
    for tap in range(width):
        source = positions - (width - 1) + tap
        exists = source >= 0
        in_forward = exists & (source >= first_new_position)
        cached = exists & ~in_forward
        src_k = torch.zeros_like(k)
        src_v = torch.zeros_like(v)
        if bool(in_forward.any()):
            src_row = first_row + (source - first_new_position)
            idx = rows[in_forward]
            src_k[idx] = k[src_row[in_forward]]
            src_v[idx] = v[src_row[in_forward]]
        if bool(cached.any()):
            idx = rows[cached]
            ck, cv = cached_tap(source[cached], idx)
            src_k[idx] = ck.float()
            src_v[idx] = cv.float()
        acc_k += src_k * k_weight[:, tap].float()
        acc_v += src_v * v_weight[:, tap].float()
    return (acc_k + k).to(torch.bfloat16), (acc_v + v).to(torch.bfloat16)


def reference_prep(
    torch: Any,
    shape: PrepShape,
    qkvr: Any,
    k_conv: Any,
    v_conv: Any,
    q_norm_weight: Any,
    k_norm_weight: Any,
    rel_proj: Any,
    tau: Any | None,
    eps: float,
) -> dict[str, Any]:
    """q, rel logits and the attention-cache K/V rows for every token."""

    tokens = qkvr.shape[0]
    hq, hkv, hd = shape.num_q_heads, shape.num_kv_heads, shape.head_dim
    q = qkvr[:, : shape.q_width].float().view(tokens, hq, hd)
    q_hat = _rmsnorm(torch, q, q_norm_weight, eps).to(torch.bfloat16)
    if tau is not None:
        q_hat = (q_hat.float() * tau.float()[:, None, None]).to(torch.bfloat16)

    k_hat = _rmsnorm(torch, k_conv.float().view(tokens, hkv, hd), k_norm_weight, eps).to(
        torch.bfloat16
    )

    r_off = shape.q_width + 2 * shape.kv_width
    r = qkvr[:, r_off : r_off + hq * shape.d_rel].float().view(tokens, hq, shape.d_rel)
    rel = torch.einsum("thd,de->the", r, rel_proj.float()).to(torch.bfloat16)
    if tau is not None:
        rel = (rel.float() * tau.float()[:, None, None]).to(torch.bfloat16)
    return {
        "q": q_hat.reshape(tokens, shape.q_width),
        "rel": rel,
        "k_cache_rows": k_hat,
        "v_cache_rows": v_conv.view(tokens, hkv, hd),
    }


def log_scaling_tau(torch: Any, positions: Any, n_floor: int, alpha: float) -> Any:
    """Inkling's per-token log-length scale (fp32), as the model computes it."""
    n = (positions + 1).float() / float(n_floor)
    return 1.0 + alpha * torch.log(torch.clamp(n, min=1.0))


def compare(torch: Any, got: Any, want: Any) -> dict[str, float]:
    """Max absolute error and max error relative to the reference's max magnitude."""
    g = got.float()
    w = want.float()
    diff = (g - w).abs()
    scale = float(w.abs().max().item()) or 1.0
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "max_rel": (float(diff.max().item()) / scale) if diff.numel() else 0.0,
        "mismatch_fraction": float((diff > 0).float().mean().item()) if diff.numel() else 0.0,
    }


__all__ = [
    "PrepShape",
    "compare",
    "log_scaling_tau",
    "reference_conv",
    "reference_prep",
]
