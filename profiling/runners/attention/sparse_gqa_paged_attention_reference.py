"""Independent PyTorch reference for sparse GQA paged attention with a sigmoid gate."""

from __future__ import annotations

from typing import Any


def sparse_gqa_paged_attention_reference(
    torch: Any,
    q: Any,
    k_cache: Any,
    v_cache: Any,
    indices: Any,
    block_table: Any,
    token_to_req: Any,
    gate: Any,
    rows: list[int],
    *,
    output_scale: float = 1.0,
) -> Any:
    """Return gated BF16 outputs ``[len(rows), heads, dim]`` for the given rows.

    ``indices`` is the packed ``[rows, width + 1]`` buffer: logical token ids
    (``-1`` padding) followed by the row's valid count. ``k_cache``/``v_cache``
    are ``[pages, page_size, kv_heads, dim]``.
    """
    page_size = k_cache.shape[1]
    num_heads, head_dim = q.shape[1], q.shape[2]
    group = num_heads // k_cache.shape[2]
    width = indices.shape[1] - 1
    out = []
    for row in rows:
        count = int(indices[row, width])
        tokens = indices[row, :count].long()
        tokens = tokens[tokens >= 0]
        request = int(token_to_req[row])
        pages = block_table[request, tokens // page_size].long()
        offsets = tokens % page_size
        keys = k_cache[pages, offsets].float()  # [S, kv_heads, dim]
        values = v_cache[pages, offsets].float()
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        query = q[row].float()  # [heads, dim]
        scores = torch.einsum("hd,shd->hs", query, keys) / head_dim**0.5
        probs = torch.softmax(scores, dim=-1)
        attended = torch.einsum("hs,shd->hd", probs, values) * output_scale
        attended = attended.to(torch.bfloat16).float()
        gated = attended * torch.sigmoid(gate.view(q.shape)[row].float())
        out.append(gated.to(torch.bfloat16))
    return torch.stack(out)


__all__ = ["sparse_gqa_paged_attention_reference"]
