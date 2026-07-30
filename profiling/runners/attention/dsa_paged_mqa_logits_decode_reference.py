"""Torch semantics for the DSA paged decode MQA-logits operation."""

from __future__ import annotations

import torch

_SUPPORTED_Q_DTYPES = (
    torch.float8_e4m3fn,
    torch.bfloat16,
    torch.float16,
    torch.float32,
)
_INDEX_DTYPES = (torch.int32, torch.int64)


def dsa_paged_mqa_logits_decode_reference(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_size: int,
    max_model_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Return FP32 DSA decode logits resolved from a page-planar cache.

    Each cache page stores all E4M3 key bytes first, followed by raw FP32
    scales. Invalid output tails are ``-inf`` when ``clean_logits`` is true and
    NaN sentinels otherwise; only valid positions are contractual in the latter
    case.

    This models the math and valid-output contract of vLLM v0.23.0 commit
    0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 and DeepGEMM commit
    891d57b4db1071624b5c8fa0d1e51cb317fa709f,
    ``fp8_fp4_paged_mqa_logits``. Later profiling backends construct the exact
    DeepGEMM layouts and time the production kernel.
    """
    _validate(
        q,
        cache,
        weights,
        context_lens,
        block_table,
        block_size,
        max_model_len,
        clean_logits,
    )

    batch_size, next_n, _, head_dim = q.shape
    invalid_value = float("-inf") if clean_logits else float("nan")
    output = torch.full(
        (batch_size * next_n, max_model_len),
        invalid_value,
        dtype=torch.float32,
        device=q.device,
    )
    page_bytes = block_size * (head_dim + 4)

    for batch in range(batch_size):
        for prediction in range(next_n):
            context_len = int(context_lens[batch, prediction].item())
            if context_len == 0:
                continue

            keys = torch.empty((context_len, head_dim), dtype=torch.float32, device=q.device)
            scales = torch.empty(context_len, dtype=torch.float32, device=q.device)
            for position in range(context_len):
                logical_page, page_offset = divmod(position, block_size)
                physical_page = int(block_table[batch, logical_page].item())
                page = cache[physical_page].reshape(page_bytes)
                key_start = page_offset * head_dim
                keys[position] = (
                    page[key_start : key_start + head_dim].view(torch.float8_e4m3fn).float()
                )
                scale_start = block_size * head_dim + page_offset * 4
                scales[position] = page[scale_start : scale_start + 4].view(torch.float32)[0]

            dots = torch.einsum("hd,nd->hn", q[batch, prediction].float(), keys)
            row = batch * next_n + prediction
            output[row, :context_len] = (torch.relu(dots) * weights[row].unsqueeze(1)).sum(
                dim=0
            ) * scales

    return output


def _validate(
    q: object,
    cache: object,
    weights: object,
    context_lens: object,
    block_table: object,
    block_size: object,
    max_model_len: object,
    clean_logits: object,
) -> None:
    tensors = (
        ("q", q),
        ("cache", cache),
        ("weights", weights),
        ("context_lens", context_lens),
        ("block_table", block_table),
    )
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise TypeError("block_size must be an integer")
    if not isinstance(max_model_len, int) or isinstance(max_model_len, bool):
        raise TypeError("max_model_len must be an integer")
    if not isinstance(clean_logits, bool):
        raise TypeError("clean_logits must be a bool")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")

    assert isinstance(q, torch.Tensor)
    assert isinstance(cache, torch.Tensor)
    assert isinstance(weights, torch.Tensor)
    assert isinstance(context_lens, torch.Tensor)
    assert isinstance(block_table, torch.Tensor)

    expected_ranks = {
        "q": 4,
        "cache": 4,
        "weights": 2,
        "context_lens": 2,
        "block_table": 2,
    }
    for name, tensor in tensors:
        expected_rank = expected_ranks[name]
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise ValueError(f"{name} dimensions must be positive, got {tuple(tensor.shape)}")

    batch_size, next_n, num_heads, head_dim = q.shape
    num_pages = cache.shape[0]
    if cache.shape[1] != block_size:
        raise ValueError(
            f"cache block dimension must equal block_size {block_size}, got {cache.shape[1]}"
        )
    if cache.shape[2] != 1 or cache.shape[3] != head_dim + 4:
        raise ValueError(
            f"cache must have shape [P, block_size, 1, D+4] with D+4={head_dim + 4}, "
            f"got {tuple(cache.shape)}"
        )
    if weights.shape != (batch_size * next_n, num_heads):
        raise ValueError(
            f"weights must have shape ({batch_size * next_n}, {num_heads}), "
            f"got {tuple(weights.shape)}"
        )
    if context_lens.shape != (batch_size, next_n):
        raise ValueError(
            f"context_lens must have shape ({batch_size}, {next_n}), "
            f"got {tuple(context_lens.shape)}"
        )
    if block_table.shape[0] != batch_size:
        raise ValueError(
            f"block_table first dimension must equal batch size {batch_size}, "
            f"got {block_table.shape[0]}"
        )

    if q.dtype not in _SUPPORTED_Q_DTYPES:
        raise TypeError(
            "q dtype must be torch.float8_e4m3fn, torch.bfloat16, "
            f"torch.float16, or torch.float32, got {q.dtype}"
        )
    if cache.dtype is not torch.uint8:
        raise TypeError(f"cache dtype must be torch.uint8, got {cache.dtype}")
    if weights.dtype is not torch.float32:
        raise TypeError(f"weights dtype must be torch.float32, got {weights.dtype}")
    for name, tensor in (("context_lens", context_lens), ("block_table", block_table)):
        if tensor.dtype not in _INDEX_DTYPES:
            raise TypeError(f"{name} dtype must be torch.int32 or torch.int64")

    if q.device.type == "meta":
        raise ValueError("meta tensors are not supported")
    for name, tensor in tensors[1:]:
        if tensor.device != q.device:
            raise ValueError(
                f"{name} must be on the same device as q, got {tensor.device} and {q.device}"
            )

    if q.layout is not torch.strided:
        raise ValueError("q must have torch.strided layout")
    if weights.layout is not torch.strided:
        raise ValueError("weights must have torch.strided layout")
    if cache.layout is not torch.strided or not cache.is_contiguous():
        raise ValueError("cache must be contiguous")
    if context_lens.layout is not torch.strided or not context_lens.is_contiguous():
        raise ValueError("context_lens must be contiguous")
    if block_table.layout is not torch.strided or block_table.stride(-1) != 1:
        raise ValueError("block_table innermost dimension must be contiguous")

    page_key_bytes = block_size * head_dim
    page_bytes = block_size * (head_dim + 4)
    if cache.storage_offset() % 4 or page_key_bytes % 4 or page_bytes % 4:
        raise ValueError("cache page key/scale planes must be FP32-byte aligned")

    if not bool(torch.isfinite(q.float()).all().item()):
        raise ValueError("q must contain only finite values")
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("weights must contain only finite values")
    if bool(torch.any(context_lens < 0).item()):
        raise ValueError("context_lens values must be nonnegative")
    if bool(torch.any(context_lens > max_model_len).item()):
        raise ValueError(f"context_lens values must not exceed max_model_len {max_model_len}")

    required_blocks = torch.div(
        context_lens + block_size - 1,
        block_size,
        rounding_mode="floor",
    )
    max_required = int(required_blocks.max().item())
    if block_table.shape[1] < max_required:
        raise ValueError(
            f"block_table needs at least {max_required} columns, got {block_table.shape[1]}"
        )

    page_width = cache.shape[1] * cache.shape[2] * cache.shape[3]
    for batch in range(batch_size):
        batch_required = int(required_blocks[batch].max().item())
        referenced_pages = block_table[batch, :batch_required]
        if referenced_pages.numel() and (
            bool(torch.any(referenced_pages < 0).item())
            or bool(torch.any(referenced_pages >= num_pages).item())
        ):
            raise ValueError(f"referenced block_table page IDs must be in [0, {num_pages})")

        max_context = int(context_lens[batch].max().item())
        for position in range(max_context):
            logical_page, page_offset = divmod(position, block_size)
            physical_page = int(block_table[batch, logical_page].item())
            page = cache[physical_page].reshape(page_width)
            scale_start = page_key_bytes + page_offset * 4
            scale = page[scale_start : scale_start + 4].view(torch.float32)[0]
            if not bool(torch.isfinite(scale).item()) or not bool((scale > 0).item()):
                raise ValueError("referenced cache scales must be finite and positive")
