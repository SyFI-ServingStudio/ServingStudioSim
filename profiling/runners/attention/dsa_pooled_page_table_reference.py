"""Torch semantics for the pooled DSA indexer's post-top-k page table.

The pooled (kpool) DSA indexer selects ``top_k`` pools of ``kpool`` tokens per
query row. Sparse MLA then needs the physical cache slots of those tokens plus
the row's trailing partial pool, packed as one ``page_table_width`` wide row.
This module states that mapping in plain Torch, independent of vLLM, Triton
and CUDA extensions. It follows the vLLM fork's decode chain (fork 5629b26):

1. ``expand_pools_and_append_tail`` (``models/glm5next/nvidia/ops/
   kpool_compress.py``): history column ``c < top_k * kpool`` is token
   ``pool[c // kpool] * kpool + c % kpool`` or -1 when the pool id is -1;
   tail column ``top_k * kpool + j`` (``j < kpool - 1``) is token
   ``(seq // kpool) * kpool + j`` when ``j < seq % kpool``, else -1;
2. that row copied into a -1 filled ``page_table_width`` buffer;
3. ``triton_convert_req_index_to_global_index(..., return_valid_counts=True)``
   (``v1/attention/backends/mla/sparse_utils.py``): a token whose block is
   inside the block table becomes ``block_table[req, tok // block_size] *
   block_stride + tok % block_size``; the valid slots are packed to the front
   in column order (``COMPACT_TO_FRONT``), the rest -1, with one count per row.

Token ids are int32 as in the fork (the pool id is cast to int64, scaled, and
the expanded row stored as int32, so ``pool * kpool`` wraps like int32).
"""

from __future__ import annotations

import torch

__all__ = ["pooled_page_table_reference", "pooled_topk_reference_values"]


def _expand(pool: torch.Tensor, seq: torch.Tensor, kpool: int) -> torch.Tensor:
    rows, top_k = pool.shape
    pid = pool.to(torch.int64)
    cols = torch.arange(top_k * kpool, device=pool.device)
    hist = pid[:, cols // kpool] * kpool + (cols % kpool)
    # int32 store of the int64 product (two's complement wrap)
    hist = ((hist + 2**31) % 2**32 - 2**31).to(torch.int64)
    hist = torch.where(pid[:, cols // kpool] >= 0, hist, torch.full_like(hist, -1))
    seq64 = seq.to(torch.int64)
    tail_start = (seq64 // kpool) * kpool
    tail_count = seq64 - tail_start
    j = torch.arange(kpool - 1, device=pool.device)
    tail = tail_start[:, None] + j[None, :]
    tail = torch.where(j[None, :] < tail_count[:, None], tail, torch.full_like(tail, -1))
    return torch.cat([hist, tail], dim=1)


def pooled_page_table_reference(
    pool: torch.Tensor,
    seq_lens: torch.Tensor,
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    *,
    kpool: int,
    block_size: int,
    block_stride: int,
    page_table_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(page_table [rows, width] int32, valid_counts [rows] int32)``."""
    if pool.ndim != 2 or seq_lens.shape != (pool.shape[0],) or req_id.shape != (pool.shape[0],):
        raise ValueError("pool [rows, top_k], seq_lens [rows] and req_id [rows] required")
    tokens = _expand(pool, seq_lens, kpool)
    if tokens.shape[1] > page_table_width:
        raise ValueError("page_table_width must hold top_k * kpool + kpool - 1 tokens")
    blocks = torch.div(tokens, block_size, rounding_mode="floor")
    valid = (tokens >= 0) & (blocks < block_table.shape[1])
    safe_blocks = torch.where(valid, blocks, torch.zeros_like(blocks))
    pages = block_table.to(torch.int64)[req_id.to(torch.int64)[:, None], safe_blocks]
    phys = pages * block_stride + tokens % block_size
    phys = ((phys + 2**31) % 2**32 - 2**31)
    rows = tokens.shape[0]
    counts = valid.sum(dim=1)
    # stable compaction: destination of each valid slot = its rank among the row's valid slots
    dest = torch.cumsum(valid.to(torch.int64), dim=1) - 1
    out = torch.full((rows, page_table_width), -1, dtype=torch.int64, device=pool.device)
    r_idx, c_idx = valid.nonzero(as_tuple=True)
    out[r_idx, dest[r_idx, c_idx]] = phys[r_idx, c_idx]
    return out.to(torch.int32), counts.to(torch.int32)


def pooled_topk_reference_values(logits: torch.Tensor, lengths: torch.Tensor, top_k: int) -> list[torch.Tensor]:
    """Per row, the sorted values of the ``min(top_k, length)`` largest valid logits."""
    out = []
    for row in range(logits.shape[0]):
        length = int(lengths[row])
        k = min(top_k, length)
        out.append(torch.sort(torch.topk(logits[row, :length], k).values).values)
    return out
