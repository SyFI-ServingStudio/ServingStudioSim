"""Torch semantics for the sigmoid-gated RMSNorm: y = rmsnorm(x) * weight * sigmoid(g)."""

from __future__ import annotations

import torch

__all__ = ["sigmoid_gated_rms_norm_reference"]

EPSILON = 1e-5


def sigmoid_gated_rms_norm_reference(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """``x`` and gate ``g`` are ``[M, hidden]``, ``weight`` is ``[hidden]``.

    Statistics, the affine scale and the sigmoid gate are computed in FP32 and the
    result is rounded to ``x.dtype`` (vLLM's ``FusedRMSNormGated.forward_native``
    with activation="sigmoid", eps 1e-5). Inputs are not mutated.
    """
    if x.ndim != 2 or g.shape != x.shape or weight.shape != (x.shape[-1],):
        raise ValueError(f"x {tuple(x.shape)}, g {tuple(g.shape)}, weight {tuple(weight.shape)}")
    x_fp32 = x.float()
    rstd = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + EPSILON)
    return (x_fp32 * rstd * weight.float() * torch.sigmoid(g.float())).to(x.dtype)
