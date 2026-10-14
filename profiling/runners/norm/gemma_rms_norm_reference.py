"""Value reference for Gemma-style RMSNorm, ``x / rms(x) * (1 + w)`` in FP32.

``num_groups`` splits the last dimension into equal groups with one RMS each;
the hyper-connection norms use one group per stream.
"""

from __future__ import annotations

from typing import Any


def gemma_rms_norm_reference(
    torch: Any, x: Any, weight: Any, eps: float, num_groups: int = 1
) -> Any:
    values = x.float().unflatten(-1, (num_groups, -1))
    normed = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (normed.flatten(-2) * (1.0 + weight.float())).to(x.dtype)


__all__ = ["gemma_rms_norm_reference"]
