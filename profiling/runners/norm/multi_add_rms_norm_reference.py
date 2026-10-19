"""Independent value reference for the summed RMSNorm with fused outputs."""

from typing import Any

FP8_E4M3_MAX = 448.0


def multi_add_rms_norm_reference(
    torch: Any,
    addends: list[Any],
    weight: Any,
    scale: Any,
    eps: float,
) -> tuple[Any, Any, Any]:
    """Return (sum, normalized, fp8) for row-aligned addends; all three planes, always."""
    total = addends[0].float()
    for addend in addends[1:]:
        total = total + addend.float()
    normalized = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + eps) * weight.float()
    fp8 = (normalized / scale.float()).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    dtype = addends[0].dtype
    return total.to(dtype), normalized.to(dtype), fp8


__all__ = ["multi_add_rms_norm_reference"]
