"""Normalize cumulative serving counters into replay acceptance evidence."""

from __future__ import annotations


def replay_delta(before: dict, after: dict) -> dict:
    def delta(name: str, left: int, right: int) -> int:
        if any(type(value) is not int or value < 0 for value in (left, right)):
            raise ValueError(f"spec-decode {name} must contain nonnegative integer counters")
        if right < left:
            raise ValueError(f"spec-decode {name} decreased across replay")
        return right - left

    counts = {
        name: delta(name, before[name], after[name])
        for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    }
    left_positions = before["accepted_per_position"]
    right_positions = after["accepted_per_position"]
    if left_positions.keys() != right_positions.keys():
        raise ValueError("spec-decode position population changed across replay")
    positions = {
        position: delta(f"position {position}", left_positions[position], count)
        for position, count in right_positions.items()
    }
    if sorted(map(int, positions)) != list(range(len(positions))):
        raise ValueError("spec-decode positions must be contiguous starting at zero")
    drafts = counts["num_drafts"]
    draft_tokens = counts["num_draft_tokens"]
    accepted = counts["num_accepted_tokens"]
    prefix = [drafts, *(positions[str(pos)] for pos in range(len(positions)))]
    if accepted > draft_tokens or any(a < b for a, b in zip(prefix, prefix[1:])):
        raise ValueError("spec-decode accepted counters exceed their draft prefix")
    if sum(positions.values()) != accepted:
        raise ValueError("spec-decode position counters do not sum to accepted tokens")
    return {
        "schema_version": 1,
        "counter_scope": "replay_prometheus_delta",
        "num_drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "accepted_tokens_per_position": positions,
        "acceptance_rate": accepted / draft_tokens if draft_tokens else None,
        "acceptance_rate_percent": 100 * accepted / draft_tokens if draft_tokens else None,
        "mean_acceptance_length": 1 + accepted / drafts if drafts else None,
        "per_position_acceptance_rates": {
            position: count / drafts if drafts else None for position, count in positions.items()
        },
        "before": before,
        "after": after,
    }
