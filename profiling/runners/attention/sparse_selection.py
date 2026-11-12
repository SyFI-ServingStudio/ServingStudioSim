"""Synthetic indexer selections shared by the sparse-attention runners.

A row at logical position ``context_len - 1`` sees ``context_len // ratio``
complete compressed groups (vLLM fork ``qsa_cache.py`` ``visible_blocks``) and
selects up to ``token_topk // ratio`` of them. The selection here is a uniform
random subset in random order, the shape a top-k over unstructured scores gives.
"""

from __future__ import annotations

from typing import Any

_ROW_CHUNK = 1024


def random_group_selection(
    torch: Any,
    *,
    num_rows: int,
    context_len: int,
    token_topk: int,
    compress_ratio: int,
    device: Any,
    seed: int = 0,
) -> tuple[Any, Any, Any]:
    """Return ``(block_indices, positions, visible)`` for ``num_rows`` rows.

    ``block_indices`` is int32 ``[rows, token_topk // ratio]`` with ``-1`` padding,
    ``positions`` int64 and ``visible`` int32 ``[rows]``.
    """
    block_topk = token_topk // compress_ratio
    visible_groups = context_len // compress_ratio
    selected = min(visible_groups, block_topk)
    blocks = torch.full((num_rows, block_topk), -1, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    if selected:
        for start in range(0, num_rows, _ROW_CHUNK):
            stop = min(start + _ROW_CHUNK, num_rows)
            scores = torch.rand((stop - start, visible_groups), device=device, generator=generator)
            blocks[start:stop, :selected] = scores.topk(selected, dim=1).indices.to(torch.int32)
            del scores
    positions = torch.full((num_rows,), context_len - 1, dtype=torch.int64, device=device)
    visible = torch.full((num_rows,), visible_groups, dtype=torch.int32, device=device)
    return blocks, positions, visible


def selected_tokens(context_len: int, token_topk: int, compress_ratio: int) -> int:
    """Tokens a row attends: selected complete groups plus its open causal tail."""
    groups = min(context_len // compress_ratio, token_topk // compress_ratio)
    return groups * compress_ratio + context_len % compress_ratio


__all__ = ["random_group_selection", "selected_tokens"]
