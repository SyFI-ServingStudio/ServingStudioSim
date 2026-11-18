"""Torch semantics for the pooled index-key decode/verify cache update.

Source: the vLLM fork's
``vllm.models.glm5next.nvidia.ops.kpool_compress.kpool_decode_update_and_maybe_write_cache_batched``
(fork 5629b2614). Each request feeds ``next_n`` new tokens in position order.
Every token is stashed (raw key and gate score, BF16) in its request's tail
ring; a token whose position closes a pool (``pos % pool_size == pool_size - 1``)
and whose pool slot is mapped first compresses the pool: a per-dimension
softmax over ``score + ape`` weights the pool's keys, the BF16-rounded sum is
rotated by the orthonormal Hadamard-128 and quantized to FP8 with one
power-of-two scale per pooled key, written into the page-planar index cache.

This is written independently of the Triton kernel, from the operation's
definition (Sylvester Hadamard matrix, explicit softmax), and is used only to
check the production callable outside timing. Requests are independent, so
the reference runs them together and steps through the ``next_n`` tokens in
order, which is the ordering the operation requires.
"""

from __future__ import annotations

import functools

import torch

_FP8_E4M3_MAX = 448.0
_AMAX_FLOOR = 1e-4


def hadamard_matrix(n: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix of order ``n`` (a power of two), FP32."""
    return _hadamard_matrix(n, str(device))


@functools.lru_cache(maxsize=8)
def _hadamard_matrix(n: int, device: str) -> torch.Tensor:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.kron(h, base)
    return h / (n**0.5)


def compress_pool(
    keys: torch.Tensor, scores: torch.Tensor, ape: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress pools ``[..., pool_size, dim]`` into fp32 values ``[..., dim]`` and scales."""
    logits = scores.float() + ape.float()
    weights = torch.softmax(logits, dim=-2)
    pooled = (weights * keys.float()).sum(dim=-2)
    pooled = pooled.to(torch.bfloat16).float()
    h = hadamard_matrix(pooled.shape[-1], device=pooled.device)
    rotated = (pooled @ h.T).to(torch.bfloat16).float()
    amax = torch.clamp_min(rotated.abs().amax(dim=-1), _AMAX_FLOOR)
    scale = torch.exp2(torch.ceil(torch.log2(amax / _FP8_E4M3_MAX)))
    quantized = torch.clamp(rotated / scale[..., None], -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return quantized, scale


def dsa_pooled_index_cache_update_reference(
    tail: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the update to ``tail`` in place; return the pooled keys it writes.

    Returns ``(cache_slots [C] int64, quantized values [C, dim] fp32, scales [C] fp32)``
    for every completed pool, so the caller can compare against whatever scale
    encoding the device's cache uses.
    """
    num_requests, next_n, _ = key.shape
    ring = tail.shape[2]
    rows = torch.arange(num_requests)
    offsets = torch.arange(pool_size)
    slots, values, scales = [], [], []
    for t in range(next_n):
        pos = positions[:, t].long()
        loc = slot_mapping[:, t].long()
        tslot = tail_slot_mapping[:, t].long()
        block = tslot.clamp_min(0) // ring
        complete = (pos >= 0) & (loc >= 0) & (pos % pool_size == pool_size - 1)
        if bool(complete.any()):
            r = rows[complete]
            start = pos[complete] - (pool_size - 1)
            phys = (start[:, None] + offsets[None, :]) % ring
            b = block[complete][:, None]
            keys = tail[b, 0, phys].clone()
            scores = tail[b, 1, phys].clone()
            keys[:, -1] = key[r, t]
            scores[:, -1] = slot_score[r, t]
            quantized, scale = compress_pool(keys, scores, ape)
            slots.append(loc[complete])
            values.append(quantized)
            scales.append(scale)
        stash = (pos >= 0) & (tslot >= 0)
        tail[block[stash], 0, pos[stash] % ring] = key[stash, t].to(tail.dtype)
        tail[block[stash], 1, pos[stash] % ring] = slot_score[stash, t].to(tail.dtype)
    if not slots:
        dim = key.shape[2]
        return torch.empty(0, dtype=torch.long), torch.empty(0, dim), torch.empty(0)
    return torch.cat(slots), torch.cat(values), torch.cat(scales)


__all__ = [
    "compress_pool",
    "dsa_pooled_index_cache_update_reference",
    "hadamard_matrix",
]
