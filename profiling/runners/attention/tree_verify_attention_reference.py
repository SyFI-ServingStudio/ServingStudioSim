"""Pure-Torch reference for ``tree_verify_attention``.

Independent of TensorFold: per stream, fp32 SDPA over ``[committed keys; window
keys]`` with an explicit boolean mask. A window row sees every committed key of
its stream and the window rows on its own root-to-row path (ancestors and
itself). GQA groups query heads contiguously: query head ``h`` reads KV head
``h // (num_qo_heads // num_kv_heads)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def path_mask(torch: Any, parents: Sequence[int], device: Any = None) -> Any:
    """``(w, w)`` bool: ``mask[r, c]`` when window row ``c`` is ``r`` or an ancestor of it.

    ``parents`` are window-local row indices, ``-1`` at a root, each parent before its child.
    """

    w = len(parents)
    mask = torch.zeros((w, w), dtype=torch.bool, device=device)
    for row, parent in enumerate(parents):
        if parent >= row:
            raise ValueError("a parent must precede its child in the window")
        mask[row, row] = True
        if parent >= 0:
            mask[row] |= mask[parent]
    return mask


def tree_verify_attention_reference(
    torch: Any,
    q: Any,
    k_nodes: Any,
    v_nodes: Any,
    caches: Sequence[tuple[Any, Any]],
    parents: Sequence[Sequence[int]],
    committed: Sequence[int],
    scale: float,
) -> Any:
    """fp32 ``(W, H, D)`` for streams back to back in ``q`` / ``k_nodes`` / ``v_nodes``.

    ``caches[s]`` holds at least ``committed[s]`` rows ``(rows, HK, D)``; ``parents[s]`` is
    stream ``s``'s window-local tree.
    """

    import torch.nn.functional as F

    h, hk = q.shape[1], k_nodes.shape[1]
    group = h // hk
    out = []
    start = 0
    for (k_cache, v_cache), local, p in zip(caches, parents, committed):
        w = len(local)
        rows = slice(start, start + w)
        keys = torch.cat([k_cache[:p].float(), k_nodes[rows].float()])  # (p + w, HK, D)
        values = torch.cat([v_cache[:p].float(), v_nodes[rows].float()])
        keys = keys.repeat_interleave(group, dim=1).transpose(0, 1)  # (H, p + w, D)
        values = values.repeat_interleave(group, dim=1).transpose(0, 1)
        mask = torch.cat(
            [
                torch.ones((w, p), dtype=torch.bool, device=q.device),
                path_mask(torch, local, q.device),
            ],
            dim=1,
        )
        qf = q[rows].float().transpose(0, 1)  # (H, w, D)
        o = F.scaled_dot_product_attention(qf, keys, values, attn_mask=mask, scale=scale)
        out.append(o.transpose(0, 1))
        start += w
    return torch.cat(out)
