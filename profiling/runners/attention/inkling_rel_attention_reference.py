"""PyTorch reference for Inkling relative-position attention.

Independent of tml_fa4: it reads keys and values through the page table and
applies the relative logits by distance, the way the model defines them
(``fa4_rel_attention.py`` score_mod and ``tml_fa4/testing.py attention_ref``):

    s[h, i, j] = (q[i, h] . k[j, h // G]) * softmax_scale + B[i, h, j]
    B[i, h, j] = rel_logits[i, h, pos_i - j]   if 0 <= pos_i - j < rel_extent
               = 0                            if pos_i - j >= rel_extent (global)
    keys j > pos_i are masked; local layers also mask pos_i - j >= rel_extent.

Everything runs in float32 on whatever device the inputs live on.
"""

from __future__ import annotations

from typing import Any


def gather_request_kv(key_cache: Any, value_cache: Any, block_row: Any, context: int) -> tuple:
    """Keys and values ``[context, kv_heads, head_dim]`` of one request.

    ``key_cache`` / ``value_cache`` are ``[num_pages, page_size, kv_heads, head_dim]``
    views (any strides); ``block_row`` maps a request page index to a physical page.
    """
    import torch

    page_size = key_cache.shape[1]
    tokens = torch.arange(context, device=key_cache.device)
    pages = block_row.to(key_cache.device).long()[tokens // page_size]
    offsets = tokens % page_size
    return key_cache[pages, offsets].float(), value_cache[pages, offsets].float()


def rel_attention_reference(
    q: Any,
    positions: Any,
    keys: Any,
    values: Any,
    rel_logits: Any,
    *,
    softmax_scale: float,
    is_local: bool,
) -> Any:
    """Attention output ``[rows, heads, head_dim]`` (float32) for one request.

    ``q`` is ``[rows, heads, head_dim]`` at absolute ``positions`` ``[rows]``;
    ``keys`` / ``values`` hold the request's first ``context`` tokens
    ``[context, kv_heads, head_dim]``; ``rel_logits`` is ``[rows, heads, rel_extent]``.
    """
    import torch

    rows, heads, _ = q.shape
    kv_heads = keys.shape[1]
    rel_extent = rel_logits.shape[-1]
    group = heads // kv_heads
    k = keys.float().repeat_interleave(group, dim=1)  # [S, H, D]
    v = values.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), k) * softmax_scale
    distance = positions.long()[:, None] - torch.arange(keys.shape[0], device=q.device)[None, :]
    in_extent = (distance >= 0) & (distance < rel_extent)
    index = distance.clamp(0, rel_extent - 1)  # [rows, S]
    bias = torch.gather(
        rel_logits.float(), -1, index[:, None, :].expand(rows, heads, index.shape[1])
    )  # [rows, H, S]
    bias = bias.masked_fill(~in_extent[:, None, :], 0.0).permute(1, 0, 2)
    scores = scores + bias
    visible = in_extent if is_local else distance >= 0
    scores = scores.masked_fill(~visible[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hts,shd->thd", probs, v)
