"""Torch CPU semantics for vLLM's fused MoE top-k router selection."""

from __future__ import annotations

import torch

__all__ = ["moe_fused_topk_reference"]

_INT32_MAX = torch.iinfo(torch.int32).max


def moe_fused_topk_reference(
    logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select and renormalize experts from dense BF16 router logits.

    Each row is converted to FP32 and normalized with a stable softmax over all
    experts. Experts are then selected iteratively in descending probability
    order, with exact ties resolved in favor of the lower expert ID. The
    selected probabilities are renormalized to sum to one, matching vLLM's
    ``topkGating`` production path.

    Returns fresh contiguous FP32 weights, int32 expert IDs, and int32 source
    indices. Source index ``(token, slot)`` is ``slot * T + token``.
    """
    num_tokens, num_experts = _validate(logits, top_k)

    logits_fp32 = logits.float()
    row_max = logits_fp32.max(dim=1, keepdim=True).values
    exponentials = torch.exp(logits_fp32 - row_max)
    probabilities = exponentials / exponentials.sum(dim=1, keepdim=True)

    weights = torch.empty((num_tokens, top_k), dtype=torch.float32)
    expert_ids = torch.empty((num_tokens, top_k), dtype=torch.int32)
    source_indices = torch.empty((num_tokens, top_k), dtype=torch.int32)

    for token in range(num_tokens):
        selected: set[int] = set()
        for slot in range(top_k):
            best_expert = -1
            best_probability = -1.0
            for expert in range(num_experts):
                if expert in selected:
                    continue
                probability = float(probabilities[token, expert])
                # Scanning IDs in ascending order and updating only for a
                # strict improvement makes lower IDs win exact ties.
                if probability > best_probability:
                    best_probability = probability
                    best_expert = expert
            selected.add(best_expert)
            weights[token, slot] = best_probability
            expert_ids[token, slot] = best_expert
            source_indices[token, slot] = slot * num_tokens + token

        weights[token].div_(weights[token].sum())

    return weights.contiguous(), expert_ids.contiguous(), source_indices.contiguous()


def _validate(logits: object, top_k: object) -> tuple[int, int]:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.layout is not torch.strided:
        raise ValueError("logits must have torch.strided layout")
    if logits.device.type == "meta":
        raise ValueError("meta tensors are not supported")
    if logits.device.type != "cpu":
        raise ValueError(f"semantic reference requires CPU logits, got {logits.device.type}")
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank 2, got rank {logits.ndim}")
    if logits.dtype is not torch.bfloat16:
        raise TypeError(f"logits dtype must be torch.bfloat16, got {logits.dtype}")

    num_tokens, num_experts = logits.shape
    if num_tokens <= 0 or num_experts <= 0:
        raise ValueError("logits token and expert dimensions must be positive")
    if type(top_k) is not int:
        raise TypeError("top_k must be an exact integer")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
    if num_tokens * top_k - 1 > _INT32_MAX:
        raise ValueError("source indices exceed the int32 range")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must contain only finite values")
    return num_tokens, num_experts
