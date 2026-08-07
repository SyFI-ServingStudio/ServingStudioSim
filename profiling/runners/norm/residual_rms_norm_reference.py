"""Torch semantic reference for residual-add RMSNorm.

The residual sum and RMS reduction are evaluated in FP32. The function returns
both the normalized output and the input-dtype residual sum without mutating
the caller's tensors.
"""

from __future__ import annotations

import math

import torch

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def residual_rms_norm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted RMSNorm output and residual sum for two 2-D inputs."""
    # Source semantics: vLLM RMSNorm.forward_static/fused_add_rms_norm.
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not isinstance(residual, torch.Tensor):
        raise TypeError("residual must be a torch.Tensor")
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")

    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got rank {x.ndim}")
    if residual.ndim != 2:
        raise ValueError(f"residual must be 2-D, got rank {residual.ndim}")
    if x.shape != residual.shape:
        raise ValueError(
            f"x and residual must have the same shape, got {x.shape} and {residual.shape}"
        )

    m, hidden = x.shape
    if m <= 0 or hidden <= 0:
        raise ValueError(f"x and residual dimensions must be positive, got {(m, hidden)}")

    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"x dtype must be one of torch.bfloat16, torch.float16, or torch.float32, got {x.dtype}"
        )
    if residual.dtype != x.dtype:
        raise TypeError(
            f"x and residual must have the same dtype, got {x.dtype} and {residual.dtype}"
        )
    if weight.ndim != 1 or weight.shape != (hidden,):
        raise ValueError(f"weight must have shape ({hidden},), got {tuple(weight.shape)}")
    if weight.dtype != x.dtype:
        raise TypeError(f"weight dtype must match input dtype {x.dtype}, got {weight.dtype}")
    if residual.device != x.device or weight.device != x.device:
        raise ValueError("x, residual, and weight must be on the same device")

    if isinstance(eps, bool):
        raise TypeError("eps must be a positive finite number")
    try:
        eps_value = float(eps)
    except (TypeError, ValueError) as exc:
        raise TypeError("eps must be a positive finite number") from exc
    if not math.isfinite(eps_value) or eps_value <= 0:
        raise ValueError(f"eps must be positive and finite, got {eps}")

    summed_fp32 = x.float() + residual.float()
    residual_out = summed_fp32.to(x.dtype)
    variance_fp32 = summed_fp32.square().mean(dim=-1, keepdim=True)
    normalized_fp32 = summed_fp32 * torch.rsqrt(variance_fp32 + eps_value)
    normalized_out = normalized_fp32.to(x.dtype) * weight
    return normalized_out, residual_out
