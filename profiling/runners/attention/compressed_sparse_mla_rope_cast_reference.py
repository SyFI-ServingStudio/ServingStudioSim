"""Torch semantics for DeepSeek V4.1 FlashMLA mega attention.

The contract comes from the alignment fork (``servingstudio-alignment-v41``):
``vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py``, which calls
``torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_{decode,fwd}``. The call
passes ``enable_q_norm=False``, so this reference applies no Q RMSNorm. It
applies GPT-J (interleaved-pair) RoPE to Q's last 64 dims, runs one-KV-head
sparse attention with an attention sink, applies the inverse RoPE to the
output's last 64 dims, and casts the output to FP8 E4M3 with UE8M0 scales per
32 dims. Keys and values are the same 512-dim cache row, which already carries
RoPE.

Two paged cache records, documented in the fork's
``common/ops/cache_utils.py``. Each page puts its whole data region before
its whole scale region:

- ``mxfp8`` (528 B/token): 512 FP8 E4M3 values, then 16 UE8M0 scales of 32 dims;
- ``nvfp4`` (288 B/token): 512 E2M1 values packed two per byte (even element in
  the low nibble), then 32 FP8 E4M3 scales of 16 dims.

This module never imports vLLM, FlashMLA or custom CUDA. The layout helpers
(``q_to_fused_layout`` and ``output_from_fused_layout``) re-derive the kernel's
documented chunk-interleaved transport layouts. Only the production runner
uses them, to feed the kernel and read its output back.
"""

from __future__ import annotations

import torch

__all__ = [
    "RECORD_BYTES",
    "decode_records",
    "compressed_sparse_mla_rope_cast_reference",
    "encode_records",
    "output_from_fused_layout",
    "q_to_fused_layout",
    "quantize_output",
]

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
OUT_GROUP = 32
Q_CHUNK = 16
WV_GROUP_SIZE = 8
FP8_MAX = 448.0
RECORD_BYTES = {"mxfp8": 528, "nvfp4": 288}
_DATA_BYTES = {"mxfp8": 512, "nvfp4": 256}
_SCALE_BYTES = {"mxfp8": 16, "nvfp4": 32}
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_ROW_CHUNK = 64


def _page_regions(cache: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``[blocks, page, bytes]`` -> data ``[slots, data]`` and scale ``[slots, sf]`` views."""
    blocks, page, width = cache.shape
    if width != RECORD_BYTES[fmt]:
        raise ValueError(f"{fmt} cache needs {RECORD_BYTES[fmt]} B/token, got {width}")
    flat = cache.reshape(blocks, page * width)
    data_bytes = page * _DATA_BYTES[fmt]
    data = flat[:, :data_bytes].reshape(blocks * page, _DATA_BYTES[fmt])
    scale = flat[:, data_bytes:].reshape(blocks * page, _SCALE_BYTES[fmt])
    return data, scale


def decode_records(cache: torch.Tensor, slots: torch.Tensor, fmt: str) -> torch.Tensor:
    """Decode cache rows ``slots`` (any shape, all valid) to fp32 ``[..., 512]``."""
    data, scale = _page_regions(cache, fmt)
    flat_slots = slots.reshape(-1).long()
    raw = data[flat_slots]
    sf = scale[flat_slots]
    if fmt == "mxfp8":
        values = raw.contiguous().view(torch.float8_e4m3fn).float().view(-1, 16, 32)
        factors = torch.exp2(sf.float() - 127.0)
    else:
        packed = raw.long()
        codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).view(-1, HEAD_DIM)
        table = torch.tensor(_E2M1, dtype=torch.float32, device=cache.device)
        magnitude = table[codes & 7]
        values = torch.where(codes >= 8, -magnitude, magnitude).view(-1, 32, 16)
        factors = sf.contiguous().view(torch.float8_e4m3fn).float()
    return (values * factors[..., None]).view(*slots.shape, HEAD_DIM)


def _ue8m0_exponent(amax: torch.Tensor, max_value: float) -> torch.Tensor:
    """Smallest power-of-two scale exponent with ``amax / 2**e <= max_value``."""
    ratio = torch.clamp(amax / max_value, min=2.0**-126)
    return torch.ceil(torch.log2(ratio))


def encode_records(rows: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode fp32/bf16 ``[N, 512]`` rows into (data bytes, scale bytes) of ``fmt``."""
    rows = rows.float()
    n = rows.shape[0]
    if fmt == "mxfp8":
        groups = rows.view(n, 16, 32)
        exponent = _ue8m0_exponent(groups.abs().amax(-1), FP8_MAX)
        scaled = groups / torch.exp2(exponent)[..., None]
        data = scaled.to(torch.float8_e4m3fn).view(torch.uint8).view(n, 512)
        scale = (exponent + 127.0).to(torch.uint8)
        return data, scale
    groups = rows.view(n, 32, 16)
    scale_fp8 = (groups.abs().amax(-1) / 6.0).clamp(min=2.0**-9).to(torch.float8_e4m3fn)
    scaled = groups / scale_fp8.float()[..., None]
    table = torch.tensor(_E2M1, dtype=torch.float32, device=rows.device)
    code = (scaled.abs()[..., None] - table).abs().argmin(-1)
    code = code | ((scaled < 0) & (code > 0)).long() << 3
    code = code.view(n, 256, 2)
    data = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    return data, scale_fp8.view(torch.uint8)


def write_records(cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor, fmt: str) -> None:
    """Store ``rows`` at ``slots`` of a paged ``fmt`` cache (Torch encoder)."""
    data, scale = _page_regions(cache, fmt)
    encoded, factors = encode_records(rows, fmt)
    data[slots.long()] = encoded
    scale[slots.long()] = factors


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """GPT-J rotation of the last 64 dims; ``cos``/``sin`` broadcast ``[T, 1, 32]``."""
    even = x[..., NOPE_DIM::2]
    odd = x[..., NOPE_DIM + 1 :: 2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return torch.cat((x[..., :NOPE_DIM], rotated.flatten(-2)), dim=-1)


def compressed_sparse_mla_rope_cast_reference(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    swa_cache: torch.Tensor,
    swa_slots: torch.Tensor,
    extra_cache: torch.Tensor | None = None,
    extra_slots: torch.Tensor | None = None,
    extra_format: str = "nvfp4",
) -> torch.Tensor:
    """Return the fp32 ``[T, H, 512]`` output before the FP8 cast.

    ``q`` is the standard-layout ``[T, H, 512]`` query before RoPE. ``swa_slots``
    and ``extra_slots`` are ``[T, K]`` physical slot ids, and ``-1`` marks an
    unused entry. The SWA cache is ``mxfp8``, and the compressed cache is
    ``extra_format``. A row with no valid key and a finite sink returns zeros.
    Duplicate slots keep their repeated softmax mass, as in the kernel.
    """
    if q.dim() != 3 or q.shape[-1] != HEAD_DIM:
        raise ValueError(f"q must be [T, H, {HEAD_DIM}], got {tuple(q.shape)}")
    if (extra_cache is None) != (extra_slots is None):
        raise ValueError("extra_cache and extra_slots must be given together")
    output = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    table = cos_sin_cache.float()[positions.long()]
    cos, sin = table[:, None, : ROPE_DIM // 2], table[:, None, ROPE_DIM // 2 :]
    sink = attn_sink.float()
    sources = [(swa_cache, swa_slots, "mxfp8")]
    if extra_cache is not None:
        sources.append((extra_cache, extra_slots, extra_format))
    for start in range(0, q.shape[0], _ROW_CHUNK):
        end = min(start + _ROW_CHUNK, q.shape[0])
        query = _rope(q[start:end].float(), cos[start:end], sin[start:end])
        keys, valid = [], []
        for cache, slots, fmt in sources:
            chunk_slots = slots[start:end]
            mask = chunk_slots >= 0
            keys.append(decode_records(cache, chunk_slots.clamp(min=0), fmt) * mask[..., None])
            valid.append(mask)
        key = torch.cat(keys, dim=1)
        mask = torch.cat(valid, dim=1)
        scores = torch.einsum("thd,tkd->thk", query, key) * softmax_scale
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        row_max = torch.maximum(scores.amax(-1), sink[None, :])
        weights = torch.exp(scores - row_max[..., None])
        denominator = weights.sum(-1) + torch.exp(sink[None, :] - row_max)
        attended = torch.einsum("thk,tkd->thd", weights, key) / denominator[..., None]
        output[start:end] = _rope(attended, cos[start:end], -sin[start:end])
    return output


def quantize_output(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 E4M3 values ``[T, H, 512]`` and UE8M0 exponent bytes ``[T, H, 16]``."""
    t, h, _ = output.shape
    groups = output.float().view(t, h, HEAD_DIM // OUT_GROUP, OUT_GROUP)
    exponent = _ue8m0_exponent(groups.abs().amax(-1), FP8_MAX)
    values = (groups / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
    return values.view(t, h, HEAD_DIM), (exponent + 127.0).to(torch.uint8)


def dequantize_output(values: torch.Tensor, exponent_bytes: torch.Tensor) -> torch.Tensor:
    t, h, _ = values.shape
    groups = values.float().view(t, h, HEAD_DIM // OUT_GROUP, OUT_GROUP)
    return (groups * torch.exp2(exponent_bytes.float() - 127.0)[..., None]).view(t, h, HEAD_DIM)


def q_to_fused_layout(q: torch.Tensor) -> torch.Tensor:
    """Standard ``[T, H, 512]`` -> the kernel's Q layout (16-dim chunks across heads).

    ``fused[(d // 16) * H * 16 + h * 16 + d % 16] = standard[h * 512 + d]``.
    """
    t, h, d = q.shape
    return q.view(t, h, d // Q_CHUNK, Q_CHUNK).transpose(1, 2).reshape(t, h, d)


def output_from_fused_layout(
    data: torch.Tensor, packed_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel output -> standard ``[T, H, 512]`` FP8 values and ``[T, H, 16]`` exponents.

    Per 8-head group, ``fused[(c * 8 + h) * 32 + j] = standard[h * 512 + c * 32 + j]``,
    and the int32 scale words each pack four UE8M0 bytes, one per fused chunk.
    """
    t, groups, width = data.shape
    chunks = HEAD_DIM // OUT_GROUP
    values = (
        data.view(t, groups, chunks, WV_GROUP_SIZE, OUT_GROUP)
        .transpose(2, 3)
        .reshape(t, groups * WV_GROUP_SIZE, HEAD_DIM)
    )
    exponents = (
        packed_scale.contiguous()
        .view(torch.uint8)
        .view(t, groups, chunks, WV_GROUP_SIZE)
        .transpose(2, 3)
        .reshape(t, groups * WV_GROUP_SIZE, chunks)
    )
    if width != WV_GROUP_SIZE * HEAD_DIM:
        raise ValueError(f"unexpected fused output width {width}")
    return values, exponents
