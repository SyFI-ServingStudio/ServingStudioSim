"""Torch reference for the MXFP4-weight / MXFP8-activation routed MoE.

Independent of FlashInfer: it dequantizes the checkpoint-order tensors
(contiguous ``[gate; up]`` W13 rows, packed E2M1 nibbles, UE8M0 per-32 scales),
runs one FP32 matmul pair per local expert, applies DeepSeek-V4's clamped
SwiGLU, and combines the rows with the precomputed top-k weights. Remote
experts (outside ``[local_offset, local_offset + num_local)``) contribute
nothing, as on one EP rank.
"""

from __future__ import annotations

from typing import Any

MX_BLOCK = 32
# OCP MX E2M1 code points, indexed by the 4-bit code (sign in bit 3).
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0


def _ue8m0(torch: Any, scale: Any) -> Any:
    return torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0)


def dequantize_mxfp4(torch: Any, packed: Any, scale: Any) -> Any:
    """``[..., K/2]`` uint8 (low nibble first) + ``[..., K/32]`` UE8M0 -> FP32."""

    lut = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES), device=packed.device)
    codes = packed.view(torch.uint8)
    values = torch.stack((lut[(codes & 0xF).long()], lut[(codes >> 4).long()]), dim=-1)
    values = values.reshape(*codes.shape[:-1], codes.shape[-1] * 2)
    blocks = values.reshape(*values.shape[:-1], -1, MX_BLOCK)
    return (blocks * _ue8m0(torch, scale).unsqueeze(-1)).reshape(values.shape)


def dequantize_mxfp8(torch: Any, values: Any, scale: Any) -> Any:
    """``[..., K]`` E4M3 + ``[..., K/32]`` UE8M0 -> FP32."""

    blocks = values.to(torch.float32).reshape(*values.shape[:-1], -1, MX_BLOCK)
    return (blocks * _ue8m0(torch, scale).unsqueeze(-1)).reshape(values.shape)


def quantize_dequantize_mxfp8(torch: Any, x: Any) -> Any:
    """Round-trip FP32 rows through OCP MXFP8 (power-of-two scale >= amax/448)."""

    blocks = x.reshape(*x.shape[:-1], -1, MX_BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=2.0**-127)
    scale = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX)))
    quantized = (blocks / scale).to(torch.float8_e4m3fn).to(torch.float32)
    return (quantized * scale).reshape(x.shape)


def clamped_swiglu(torch: Any, gate: Any, up: Any, limit: float | None) -> Any:
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


def mxfp4_fused_moe_reference(
    torch: Any,
    *,
    hidden: Any,
    hidden_scale: Any,
    w13: Any,
    w13_scale: Any,
    w2: Any,
    w2_scale: Any,
    topk_ids: Any,
    topk_weights: Any,
    local_offset: int,
    num_local: int,
    clamp_limit: float | None,
    requantize_intermediate: bool,
) -> Any:
    """FP32 output ``[T, H]`` of one EP rank's routed experts."""

    x = dequantize_mxfp8(torch, hidden, hidden_scale)
    out = torch.zeros_like(x)
    intermediate = w13.shape[1] // 2
    for local in range(num_local):
        rows, slots = torch.nonzero(topk_ids == local_offset + local, as_tuple=True)
        if rows.numel() == 0:
            continue
        w13_f = dequantize_mxfp4(torch, w13[local], w13_scale[local])
        h = x[rows] @ w13_f.t()
        act = clamped_swiglu(torch, h[:, :intermediate], h[:, intermediate:], clamp_limit)
        if requantize_intermediate:
            act = quantize_dequantize_mxfp8(torch, act)
        y = act @ dequantize_mxfp4(torch, w2[local], w2_scale[local]).t()
        out.index_add_(0, rows, y * topk_weights[rows, slots].float().unsqueeze(-1))
    return out


__all__ = [
    "clamped_swiglu",
    "dequantize_mxfp4",
    "dequantize_mxfp8",
    "mxfp4_fused_moe_reference",
    "quantize_dequantize_mxfp8",
]
