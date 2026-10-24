"""Independent PyTorch reference for ``gemm_swiglu_quant``.

Dequantize the NVFP4 activation rows and the checkpoint NVFP4 weights, compute
SiLU(x · Wg^T) · (x · Wu^T) in fp32, and quantize the product to NVFP4 with the
checkpoint's activation rules: per 16 elements an e4m3 scale of
``qg · amax / 6`` (saturating), each element times ``qg / scale`` rounded to the
nearest e2m1 magnitude (ties to the even code, saturating at 6). ``qg`` is the
consumer's global scale, 1 / its static input scale.

Layouts are the checkpoint's and TensorFold's, read from their writers:

- activation codes ``[m, k/2]`` uint8, element ``2j`` in byte ``j``'s low nibble;
  scales ``[k/64, mpad, 4]`` e4m3 bytes, ``[g, r, b]`` = row ``r``'s 16-element
  block ``4g + b`` (``checkpoint.quant4`` with ``tb = 0``);
- weight ``[n, k/2]`` uint8 (low nibble first) with e4m3 ``weight_scale``
  ``[n, k/16]`` and an fp32 ``weight_scale_2`` (``Fp4Linear.from_checkpoint``);
- the output uses the activation layout with ``k`` replaced by ``n``.

Pure Torch: no TensorFold import, so it checks the kernel, not a copy of it.
"""

from __future__ import annotations

from typing import Any

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0
BLOCK = 16


def e2m1_decode(torch: Any, packed: Any) -> Any:
    """uint8 ``[..., c]`` of two e2m1 codes each -> fp32 ``[..., 2c]`` (low nibble first)."""
    table = torch.tensor(
        list(E2M1_VALUES) + [-v for v in E2M1_VALUES], dtype=torch.float32, device=packed.device
    )
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    return torch.stack((table[lo], table[hi]), dim=-1).flatten(-2)


def e2m1_encode(torch: Any, v: Any) -> Any:
    """fp32 -> 4-bit e2m1 codes (int64), nearest magnitude, ties to the even code."""
    a = v.abs()
    code = torch.full_like(a, 7, dtype=torch.int64)
    # Upper bounds of codes 0..6; ``<=`` where the tie rounds down to an even
    # code, ``<`` where it rounds up to one.
    for c, (bound, inclusive) in reversed(
        list(
            enumerate(
                [(0.25, True), (0.75, False), (1.25, True), (1.75, False),
                 (2.5, True), (3.5, False), (5.0, True)]
            )
        )
    ):
        hit = a <= bound if inclusive else a < bound
        code = torch.where(hit, torch.full_like(code, c), code)
    negative = (v < 0) & (code != 0)
    return torch.where(negative, code | 8, code)


def e4m3_byte(torch: Any, v: Any) -> Any:
    """fp32 (non-negative) -> e4m3 bytes, saturating at 448."""
    return v.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_value(torch: Any, b: Any) -> Any:
    return b.view(torch.float8_e4m3fn).to(torch.float32)


def dequantize_rows(torch: Any, codes: Any, scales: Any, act: float) -> Any:
    """NVFP4 rows ``[m, k/2]`` + scales ``[k/64, mpad, 4]`` under input scale ``act`` -> fp32 ``[m, k]``."""
    m, k = codes.shape[0], codes.shape[1] * 2
    block = e4m3_value(torch, scales[:, :m, :]).permute(1, 0, 2).reshape(m, k // BLOCK)
    values = e2m1_decode(torch, codes).view(m, k // BLOCK, BLOCK)
    return (values * block[..., None] * act).view(m, k)


def dequantize_weight(torch: Any, weight: Any, weight_scale: Any, global_scale: float) -> Any:
    """Checkpoint NVFP4 ``[n, k/2]`` + e4m3 ``[n, k/16]`` + fp32 global -> fp32 ``[n, k]``."""
    n, k = weight.shape[0], weight.shape[1] * 2
    block = e4m3_value(torch, weight_scale.view(torch.uint8).view(n, k // BLOCK))
    values = e2m1_decode(torch, weight).view(n, k // BLOCK, BLOCK)
    return (values * block[..., None] * global_scale).view(n, k)


def quantize_rows(torch: Any, h: Any, qg: float) -> tuple[Any, Any]:
    """fp32 ``[m, n]`` -> NVFP4 codes ``[m, n/2]`` and scales ``[n/64, m, 4]`` under global ``qg``."""
    m, n = h.shape
    blocks = h.view(m, n // BLOCK, BLOCK)
    g = torch.tensor(qg, dtype=torch.float32, device=h.device)
    amax = blocks.abs().amax(-1)
    sf8 = e4m3_byte(torch, g * (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)))
    sf = e4m3_value(torch, sf8)
    mul = torch.where(sf != 0, g / sf, torch.zeros_like(sf))
    code = e2m1_encode(torch, blocks * mul[..., None]).view(m, n // 2, 2)
    codes = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    scales = sf8.view(m, n // 64, 4).permute(1, 0, 2).contiguous()
    return codes, scales


def swiglu_fp32(torch: Any, x: Any, wg: Any, wu: Any) -> Any:
    """SiLU(x · Wg^T) · (x · Wu^T) in fp32 (no TF32)."""
    gate = x @ wg.t()
    up = x @ wu.t()
    return gate / (1.0 + torch.exp(-gate)) * up


def gemm_swiglu_quant_reference(
    torch: Any, x: Any, wg: Any, wu: Any, qg: float
) -> tuple[Any, Any, Any]:
    """Dequantized fp32 operands -> (codes, scales, fp32 SwiGLU) of the fused launch's output."""
    h = swiglu_fp32(torch, x, wg, wu)
    codes, scales = quantize_rows(torch, h, qg)
    return codes, scales, h


__all__ = [
    "dequantize_rows",
    "dequantize_weight",
    "e2m1_decode",
    "e2m1_encode",
    "gemm_swiglu_quant_reference",
    "quantize_rows",
    "swiglu_fp32",
]
