"""Pure-Torch reference for ``sliding_block_attention``.

Independent of TensorFold: one stream's block rows attend, in fp32, to its
context keys inside the window and to its block keys (all of them, or the
earlier ones when causal), as ``dflash2.py`` ``DFlash2._attention`` masks them.
"""

from __future__ import annotations

from typing import Any


def sliding_block_attention_reference(
    torch: Any,
    q: Any,
    k_block: Any,
    v_block: Any,
    k_ctx: Any,
    v_ctx: Any,
    *,
    window: int,
    scale: float,
    causal: bool,
) -> Any:
    """``q`` ``(H, L, D)``, block/context K and V ``(Hkv, L or n, D)`` -> fp32 ``(L, H * D)``."""

    heads, length, dim = q.shape
    group = heads // k_block.shape[0]
    n = k_ctx.shape[1]
    keys = torch.cat((k_ctx, k_block), dim=1).float().repeat_interleave(group, dim=0)
    values = torch.cat((v_ctx, v_block), dim=1).float().repeat_interleave(group, dim=0)
    row = torch.arange(length, device=q.device)[:, None]
    key = torch.arange(n + length, device=q.device)[None, :]
    allowed = ((key < n) & (n + row - key < window + 1)) | (key >= n)
    if causal:
        allowed = allowed & ((key < n) | (key <= n + row))
    scores = torch.einsum("hld,hkd->hlk", q.float(), keys) * scale
    scores = scores.masked_fill(~allowed[None], float("-inf"))
    out = torch.einsum("hlk,hkd->hld", torch.softmax(scores, dim=-1), values)
    return out.transpose(0, 1).reshape(length, heads * dim)
