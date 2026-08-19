"""Torch CPU semantics for vLLM's MoE block-alignment operation."""

from __future__ import annotations

import torch

__all__ = ["moe_align_block_size_reference"]

_INT32_MAX = torch.iinfo(torch.int32).max


def moe_align_block_size_reference(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group flattened token/slot assignments into padded expert blocks.

    Valid assignments are ordered by ascending expert ID and, within an expert,
    ascending flattened assignment ID ``token * K + slot``. Production CUDA
    atomics do not promise a semantic within-expert order, so this reference
    deliberately chooses the deterministic order without changing the grouped
    assignment multiset consumed by the expert GEMMs.
    """
    num_tokens, top_k, assignments, capacity, num_blocks = _validate(
        topk_ids, num_experts, block_size
    )

    sorted_token_ids = torch.full((capacity,), assignments, dtype=torch.int32, device="cpu")
    expert_ids = torch.full((num_blocks,), -1, dtype=torch.int32, device="cpu")
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device="cpu")

    flat_experts = topk_ids.reshape(num_tokens * top_k)
    offset = 0
    for expert_tensor in torch.unique(flat_experts, sorted=True):
        expert = int(expert_tensor.item())
        assignment_ids = torch.nonzero(flat_experts == expert, as_tuple=False).flatten()
        count = assignment_ids.numel()
        if count == 0:
            continue
        padded_count = _round_up(count, block_size)
        sorted_token_ids[offset : offset + count].copy_(assignment_ids.to(torch.int32))
        first_block = offset // block_size
        expert_ids[first_block : first_block + padded_count // block_size].fill_(expert)
        offset += padded_count

    num_tokens_post_pad[0] = offset
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _validate(
    topk_ids: object,
    num_experts: object,
    block_size: object,
) -> tuple[int, int, int, int, int]:
    if not isinstance(topk_ids, torch.Tensor):
        raise TypeError("topk_ids must be a torch.Tensor")
    if topk_ids.layout is not torch.strided:
        raise ValueError("topk_ids must have torch.strided layout")
    if topk_ids.device.type == "meta":
        raise ValueError("meta tensors are not supported")
    if topk_ids.device.type != "cpu":
        raise ValueError(f"semantic reference requires CPU topk_ids, got {topk_ids.device.type}")
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank 2, got rank {topk_ids.ndim}")
    if topk_ids.dtype is not torch.int32:
        raise TypeError(f"topk_ids dtype must be torch.int32, got {topk_ids.dtype}")

    num_tokens, top_k = topk_ids.shape
    if num_tokens <= 0 or top_k <= 0:
        raise ValueError("topk_ids token and top-k dimensions must be positive")
    for name, value in (("num_experts", num_experts), ("block_size", block_size)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an exact integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        if value > _INT32_MAX:
            raise ValueError(f"{name} exceeds the int32 range")

    assignments = _checked_mul(num_tokens, top_k, "assignment count")
    if assignments > _INT32_MAX:
        raise ValueError("assignment sentinel exceeds the int32 range")
    padding_capacity = _checked_mul(num_experts, block_size - 1, "padding capacity")
    capacity = _checked_add(assignments, padding_capacity, "sorted-token capacity")
    if assignments < num_experts:
        capacity = min(
            _checked_mul(assignments, block_size, "small-assignment capacity"),
            capacity,
        )
    if capacity > _INT32_MAX:
        raise ValueError("sorted-token capacity exceeds the int32 range")
    num_blocks = _ceil_div(capacity, block_size)
    if num_blocks > _INT32_MAX:
        raise ValueError("expert block count exceeds the int32 range")

    minimum = int(topk_ids.min().item())
    maximum = int(topk_ids.max().item())
    if minimum < 0 or maximum >= num_experts:
        raise ValueError(f"expert IDs must be in [0, {num_experts})")
    sorted_rows = topk_ids.sort(dim=1).values
    if top_k > 1 and torch.any(sorted_rows[:, 1:] == sorted_rows[:, :-1]).item():
        raise ValueError("expert IDs must be unique within each token row")

    counts = torch.unique(topk_ids, return_counts=True)[1]
    padded = sum(_round_up(int(count), block_size) for count in counts.tolist())
    if padded > capacity:
        raise ValueError("padded assignment count exceeds production capacity")
    if padded > _INT32_MAX:
        raise ValueError("padded assignment count exceeds the int32 range")
    return num_tokens, top_k, assignments, capacity, num_blocks


def _round_up(value: int, multiple: int) -> int:
    return _checked_mul(_ceil_div(value, multiple), multiple, "padded assignment count")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _checked_add(left: int, right: int, label: str) -> int:
    value = left + right
    if value > _INT32_MAX:
        raise ValueError(f"{label} exceeds the int32 range")
    return value


def _checked_mul(left: int, right: int, label: str) -> int:
    value = left * right
    if value > _INT32_MAX:
        raise ValueError(f"{label} exceeds the int32 range")
    return value
