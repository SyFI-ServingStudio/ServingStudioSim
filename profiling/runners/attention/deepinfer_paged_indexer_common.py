"""Shared operand builders for DeepInfer's decode (paged) indexer runners.

DeepInfer's decode indexer (``deepinfer/model_executor/models/v4_modules/index_attn.py``
``_topk_decode``) launches every query row as its own one-token query:
``q.unsqueeze(1)`` gives ``[rows, 1, heads, head_dim / 2]`` FP4 E2M1 with a
packed UE8M0 scale per 32 elements, and ``indices=input_metadata.query_indices``
names each row's request (``prepare_v4_decode_metadata``: ``seq_idx`` per row).
A request's ``next_n`` verify rows are consecutive, so DeepGEMM's SM100 paged
scheduler groups them back into one request (``RequestInfo::from_q_token``).
``block_tables`` is ``block_tables_out [rows, max_num_blocks]``: every row
carries a copy of its request's block table.

The index-K cache (``FP4_INDEX_KV_FORMAT``) is ``uint8 [pages, block_size, 1,
head_dim / 2 + 4]`` with page-planar storage: ``block_size * head_dim / 2``
packed E2M1 bytes, then one int32 of four packed UE8M0 scales per key
(``fp8_fp4_paged_mqa_logits`` derives exactly these two views).
``block_size`` is the main KV block (128 tokens, ``ModelMetadata.
resolved_tokens_per_block``) over the layer's compress ratio.

Operand values are uniform random E2M1 codes with power-of-two scales in
[2^-2, 2^1]; the logits kernels' time does not depend on them.

Pure-Python helpers here have no torch import so CPU tests can use them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.runners.attention import deepinfer_indexer_common as common

# UE8M0 exponent bytes 125..128 = 2^-2 .. 2^1.
_SF_LOW, _SF_HIGH = 125, 129
_SEED = 0
# DeepGEMM SM100 paged MQA logits: page sizes it compiles (attention.hpp).
DENSE_PAGE_SIZES = (32, 64, 128)


def pages_per_request(context_len: int, block_size: int) -> int:
    return -(-context_len // block_size)


def page_bytes(block_size: int, head_dim: int) -> int:
    """Bytes of one FP4 index-K page: packed values then one int32 of scales per key."""
    return block_size * (head_dim // 2 + 4)


def scattered_page_count(batch_size: int, context_len: int, block_size: int) -> int:
    """Physical pages the unique-scattered mapping allocates (every other page used)."""
    return 2 * batch_size * pages_per_request(context_len, block_size) + 1


def request_indices(batch_size: int, next_n: int) -> list[int]:
    """``query_indices``: the request of each row, rows of a request consecutive."""
    return [row // next_n for row in range(batch_size * next_n)]


@dataclass
class PagedFp4Operands:
    """FP4 queries, weights and a unique-scattered paged FP4 index-K cache."""

    q_fp: Any  # int8 [rows, 1, heads, head_dim / 2]
    q_sf: Any  # int32 [rows, 1, heads]
    q_ref: Any  # fp32 [rows, heads, head_dim]
    weights: Any  # bf16 [rows, heads], 16-byte rows
    cache: Any  # uint8 [pages, block_size, 1, head_dim / 2 + 4]
    context_lens: Any  # int32 [rows, 1]
    block_table: Any  # int32 [rows, pages per request], each row its request's table
    indices: Any  # int32 [rows]
    batch_size: int
    next_n: int
    context_len: int
    block_size: int
    head_dim: int

    @property
    def rows(self) -> int:
        return self.batch_size * self.next_n

    def request_keys(self, torch: Any, request: int) -> Any:
        """fp32 dequantized keys [context_len, head_dim] of one request."""
        pages = self.block_table[request * self.next_n].long()
        flat = self.cache.view(self.cache.shape[0], -1)
        value_bytes = self.block_size * self.head_dim // 2
        values = flat[pages, :value_bytes].reshape(-1, self.head_dim // 2).view(torch.int8)
        scales = flat[pages, value_bytes:].reshape(-1, 4).contiguous().view(torch.int32)
        keys = common.fp4_dequant(torch, values.contiguous(), scales.view(-1))
        return keys[: self.context_len]


def _random_sf(torch: Any, shape: tuple[int, ...], generator: Any) -> Any:
    sf = torch.empty((*shape, 4), dtype=torch.uint8, device="cuda")
    sf.random_(_SF_LOW, _SF_HIGH, generator=generator)
    return sf.view(torch.int32).squeeze(-1)


def build_paged_fp4(
    torch: Any,
    *,
    batch_size: int,
    next_n: int,
    context_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
) -> PagedFp4Operands:
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    rows = batch_size * next_n
    q_fp = torch.empty((rows, num_heads, head_dim // 2), dtype=torch.uint8, device="cuda")
    q_fp.random_(0, 256, generator=generator)
    q_fp = q_fp.view(torch.int8)
    q_sf = _random_sf(torch, (rows, num_heads), generator)
    q_ref = common.fp4_dequant(
        torch, q_fp.reshape(rows * num_heads, head_dim // 2), q_sf.reshape(rows * num_heads)
    ).view(rows, num_heads, head_dim)
    weights = common.bf16_weights(torch, rows, num_heads)

    per_request = pages_per_request(context_len, block_size)
    num_pages = scattered_page_count(batch_size, context_len, block_size)
    cache = torch.empty(
        (num_pages, block_size, 1, head_dim // 2 + 4), dtype=torch.uint8, device="cuda"
    )
    flat = cache.view(num_pages, -1)
    value_bytes = block_size * head_dim // 2
    flat[:, :value_bytes].random_(0, 256, generator=generator)
    flat[:, value_bytes:].random_(_SF_LOW, _SF_HIGH, generator=generator)

    physical = 2 * torch.arange(batch_size * per_request, dtype=torch.int32, device="cuda") + 1
    request_table = physical.view(batch_size, per_request)
    block_table = request_table.repeat_interleave(next_n, dim=0).contiguous()
    indices = torch.tensor(request_indices(batch_size, next_n), dtype=torch.int32, device="cuda")
    context_lens = torch.full((rows, 1), context_len, dtype=torch.int32, device="cuda")
    return PagedFp4Operands(
        q_fp=q_fp.view(rows, 1, num_heads, head_dim // 2),
        q_sf=q_sf.view(rows, 1, num_heads),
        q_ref=q_ref,
        weights=weights,
        cache=cache,
        context_lens=context_lens,
        block_table=block_table,
        indices=indices,
        batch_size=batch_size,
        next_n=next_n,
        context_len=context_len,
        block_size=block_size,
        head_dim=head_dim,
    )


__all__ = [
    "DENSE_PAGE_SIZES",
    "PagedFp4Operands",
    "build_paged_fp4",
    "page_bytes",
    "pages_per_request",
    "request_indices",
    "scattered_page_count",
]
