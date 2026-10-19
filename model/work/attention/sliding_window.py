"""Causal sliding-window GQA: every query sees only its last ``window`` keys.

A sliding-window layer is a :class:`~model.work.attention.gqa.GQA` layer (same
projections, the same asymmetric QK/V widths and optional softmax sinks) whose
query at position ``i`` attends keys ``max(0, i - window + 1) .. i``, its own key
included. Per request appending ``append`` queries after ``prefix`` cached tokens:

- attention math: ``flops_per_pair`` times the visible (query, key) pairs,
  ``capped_cumulative(prefix + append, window) - capped_cumulative(prefix, window)``;
- cached K/V read: the cached keys the request's first query can see,
  ``min(prefix, window - 1)`` — every later query sees a subset of those plus the
  chunk's own keys, which are a fused activation;
- KV-cache append: ``min(append, window - 1)`` tokens. A chunk's older tokens leave
  the window before any later query reads them, so writing them is not necessary.

The pair count is not linear in a batch's summed geometry, so the spec labels exact
per-request causal interactions (``needs_request_geometry``) and refuses a
collapsed aggregate rather than guessing a mean request.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from ..core import AttnInteraction, Workload
from .base import AttentionSemantic
from .gqa import GQA


def _interpolate(function: Callable[[int], float], x: float) -> float:
    """An integer-domain function at a real point, linearly interpolated: a mean
    request rebuilt from a collapsed aggregate may have fractional lengths."""
    low = math.floor(x)
    value = float(function(low))
    fraction = x - low
    if fraction:
        value += fraction * (function(low + 1) - value)
    return value


def capped_cumulative(n: float, cap: int | None) -> float:
    """``sum_{j=1..n} min(j, cap)``: keys (or distances) visible to the first
    ``n`` positions of a sequence. ``cap=None`` is the causal triangle."""
    if cap is None:
        return n * (n + 1) / 2.0

    def exact(m: int) -> int:
        if m <= cap:
            return m * (m + 1) // 2
        return cap * (cap + 1) // 2 + (m - cap) * cap

    return _interpolate(exact, n)


def _exact_request(interaction: AttnInteraction) -> tuple[int, int]:
    """``(append, prefix)`` of one exact causal request interaction."""
    query, key, cached = interaction.num_query, interaction.num_key, interaction.num_cached_key
    if (
        interaction.mask != "causal"
        or any(float(value) != int(value) for value in (query, key, cached))
        or int(key) != int(cached) + int(query)
        or int(query) < 1
        or int(cached) < 0
    ):
        raise ValueError(
            "sliding-window attention needs exact per-request causal geometry "
            f"(append, prefix + append, prefix); got {interaction!r}"
        )
    return int(query), int(cached)


@dataclass
class SlidingWindowGQA(GQA):
    #: Keys a query sees, its own included (vLLM/FlashAttention window ``(W - 1, 0)``).
    window: int = field(default=0, kw_only=True)

    needs_request_geometry = True

    def __post_init__(self) -> None:
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 1:
            raise ValueError(f"sliding window must be a positive integer, got {self.window!r}")

    def _pairs(self, append: int, prefix: int) -> float:
        return capped_cumulative(prefix + append, self.window) - capped_cumulative(
            prefix, self.window
        )

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        """One attention row per phase plus the persistent cache append. Every row is
        emitted even at zero work, so a row set never depends on batch composition."""
        rows: list[AttentionSemantic] = []
        appended = 0.0
        for phase, phase_workload in wl.attention_phases():
            pairs = cached = 0.0
            for interaction in phase_workload.attn:
                append, prefix = _exact_request(interaction)
                count = interaction.multiplicity
                pairs += count * self._pairs(append, prefix)
                cached += count * min(prefix, self.window - 1)
                appended += count * min(append, self.window - 1)
            rows.append(
                AttentionSemantic(
                    name="attn" if phase is None else f"attn.{phase}",
                    flops=self.flops_per_pair * pairs,
                    bytes=self.kv_entry_bytes * cached,
                    compute_dtype=self.phase_compute_dtype(phase),
                )
            )
        rows.append(
            AttentionSemantic(
                name="kv_cache_append", bucket="embedding", bytes=self.kv_entry_bytes * appended
            )
        )
        return rows

    # Historical AttentionSpec surface; Model.label uses semantic_segments.
    def internal_flops(self, wl: Workload) -> float:
        return sum(row.flops for row in self.semantic_segments(wl) if row.bucket == "attn_internal")

    def kv_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.bucket == "attn_internal")

    def cache_write_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.bucket == "embedding")
