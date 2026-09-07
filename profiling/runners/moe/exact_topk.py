"""Deterministically realize exact per-expert top-k degrees.

Shared by every whole-callable fused-MoE runner: the Rust sweep hands down a
per-expert batch histogram, and the runner has to turn it back into concrete
per-token expert ids whose degrees match that histogram exactly. Any drift here
silently changes which experts the measured kernel touches.
"""

from __future__ import annotations

import heapq


def exact_topk_ids(
    *, num_tokens: int, top_k: int, per_expert_batches: tuple[int, ...]
) -> list[list[int]]:
    """Realize expert degrees as distinct top-k ids for every token."""

    if len(per_expert_batches) < top_k:
        raise ValueError("num_experts must be at least top_k")
    if any(batch < 0 for batch in per_expert_batches):
        raise ValueError("per_expert_batches cannot contain negative counts")
    if any(batch > num_tokens for batch in per_expert_batches):
        raise ValueError("one expert cannot receive more than one row per token")
    expected = num_tokens * top_k
    if sum(per_expert_batches) != expected:
        raise ValueError(f"per_expert_batches must sum to num_tokens*top_k ({expected})")

    heap = [(-batch, expert) for expert, batch in enumerate(per_expert_batches) if batch]
    heapq.heapify(heap)
    rows: list[list[int]] = []
    for _ in range(num_tokens):
        if len(heap) < top_k:
            raise ValueError("expert counts cannot form distinct top-k rows")
        selected = [heapq.heappop(heap) for _ in range(top_k)]
        rows.append([expert for _negative_count, expert in selected])
        for negative_count, expert in selected:
            if negative_count + 1 < 0:
                heapq.heappush(heap, (negative_count + 1, expert))
    if heap:
        raise ValueError("expert counts were not exhausted by top-k construction")
    return rows


__all__ = ["exact_topk_ids"]
