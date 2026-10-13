"""PyTorch reference for FA4 varlen attention with sinks and a separate V dim.

Independent of FlashAttention: it reads keys and values through the page table
and applies the definition directly. For a query at absolute position ``p`` in
its request and query head ``h`` (KV head ``h // G``):

    s[h, j] = (q[h] . k[j, h // G]) * softmax_scale       for visible keys j
    visible: j <= p, and p - j < window when the layer is windowed
    P = softmax([s[h, visible], sink[h]])                 (sink only if present)
    o[h] = sum_j P[j] * v[j, h // G]                      (the sink adds no value)

The sink is a per-head logit that joins the denominator unscaled, as FA4's
``s_aux`` / ``learnable_sink`` does. Everything runs in float32 on whatever
device the inputs live on.
"""

from __future__ import annotations

from typing import Any


def gather_request_kv(key_cache: Any, value_cache: Any, block_row: Any, context: int) -> tuple:
    """Keys ``[context, kv_heads, head_dim_qk]`` and values ``[context, kv_heads, head_dim_v]``.

    ``key_cache`` / ``value_cache`` are ``[num_pages, page_size, kv_heads, dim]``
    views (any strides); ``block_row`` maps a request page index to a physical page.
    """
    import torch

    page_size = key_cache.shape[1]
    tokens = torch.arange(context, device=key_cache.device)
    pages = block_row.to(key_cache.device).long()[tokens // page_size]
    offsets = tokens % page_size
    return key_cache[pages, offsets].float(), value_cache[pages, offsets].float()


def varlen_attention_reference(
    q: Any,
    positions: Any,
    keys: Any,
    values: Any,
    *,
    softmax_scale: float,
    window: int,
    sinks: Any = None,
) -> Any:
    """Attention output ``[rows, heads, head_dim_v]`` (float32) for one request.

    ``q`` is ``[rows, heads, head_dim_qk]`` at absolute ``positions`` ``[rows]``;
    ``keys`` / ``values`` hold the request's first ``context`` tokens;
    ``window`` is the number of keys a query sees including itself (0 = all
    earlier keys); ``sinks`` is ``[heads]`` or None.
    """
    import torch

    rows, heads, _ = q.shape
    group = heads // keys.shape[1]
    k = keys.float().repeat_interleave(group, dim=1)  # [S, H, Dqk]
    v = values.float().repeat_interleave(group, dim=1)  # [S, H, Dv]
    scores = torch.einsum("thd,shd->hts", q.float(), k) * softmax_scale
    distance = positions.long()[:, None] - torch.arange(keys.shape[0], device=q.device)[None, :]
    visible = distance >= 0
    if window > 0:
        visible &= distance < window
    scores = scores.masked_fill(~visible[None], float("-inf"))
    if sinks is not None:
        sink = sinks.float().to(q.device)[:, None, None].expand(heads, rows, 1)
        scores = torch.cat([scores, sink], dim=-1)
    probs = torch.softmax(scores, dim=-1)[..., : keys.shape[0]]
    return torch.einsum("hts,shd->thd", probs, v)
