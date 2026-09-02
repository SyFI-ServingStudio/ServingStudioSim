"""Real Python used as a coding-agent text pool (not random tokens)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class SessionKV:
    session_id: str
    tokens: list[int]
    gpu: int
    resident: bool = True
    last_wait_ms: int = 0

    def prefix_len(self) -> int:
        return len(self.tokens)


class PrefixIndex:
    """Exact shared-prefix groups for fork/binpack experiments."""

    def __init__(self) -> None:
        self._by_stem: dict[tuple[int, ...], list[str]] = defaultdict(list)

    def add(self, session_id: str, stem: Iterable[int]) -> None:
        key = tuple(stem)
        self._by_stem[key].append(session_id)

    def colocation_groups(self) -> list[list[str]]:
        return [ids for ids in self._by_stem.values() if len(ids) > 1]


def pack_stems(groups: list[list[str]], n_gpus: int) -> dict[str, int]:
    """Place all forks of a stem on the same GPU (least-loaded GPU)."""
    load = [0] * n_gpus
    place: dict[str, int] = {}
    for group in sorted(groups, key=len, reverse=True):
        gpu = min(range(n_gpus), key=lambda g: load[g])
        for sid in group:
            place[sid] = gpu
            load[gpu] += 1
    return place


def evict_if_needed(sess: SessionKV, wait_ms: int, policy: str, threshold_ms: int) -> bool:
    if policy == "keep":
        return False
    if policy == "always_handoff":
        return True
    if policy == "jit":
        return wait_ms >= threshold_ms
    raise ValueError(policy)
