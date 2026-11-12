"""Pooled top-k sparse attention geometry shared by GLM-5.3 kpool DSA and QSA.

Both mechanisms compress every ``pool`` consecutive tokens' indexer keys into one
entry, score a query at context ``k`` (itself included) against the
``floor(k / pool)`` complete entries, keep ``topk / pool`` of them expanded back
to tokens, and always attend the ``k mod pool`` tokens of the open group. A query
therefore attends ``min(k, topk + k mod pool)`` keys: all of them while at most
``topk / pool`` groups are complete (``k <= topk + pool - 1``).

Per-query work depends on each request's own context, so these specs need exact
causal interactions; a collapsed aggregate (one ``full`` interaction carrying
summed pairs) cannot recover the per-request caps and is rejected.

:class:`PooledTopkGeometry` owns that geometry and the batch evaluation; a spec
built on it supplies its projections (``matmul_groups``), learned weights
(``learned_weight_groups``), its pool and top-k (``pool``, ``topk``), and how one
phase's five work totals become rows (``_rows``).
"""

from __future__ import annotations

import numpy as np

from ..core import AttnInteraction, Workload
from .base import AttentionSemantic

# The closed forms take an int or an int64 array of contexts, so one request and
# a whole batch's requests share one definition. Every term stays below 2**63 for
# contexts up to ~2**31 tokens.


def _minimum(a, b):
    return np.minimum(a, b) if isinstance(a, np.ndarray) else min(a, b)


def _maximum(a, b):
    return np.maximum(a, b) if isinstance(a, np.ndarray) else max(a, b)


def pooled_prefix_sum(n, pool: int):
    """``sum_{k=1..n} floor(k / pool)`` in O(1)."""
    full, remainder = divmod(_maximum(n, 0), pool)
    return pool * full * (full - 1) // 2 + full * (remainder + 1)


def selected_prefix_sum(n, topk: int, pool: int):
    """``sum_{k=1..n} min(k, topk + k mod pool)`` in O(1).

    ``min(k, topk + k mod pool) == k`` exactly while ``k <= topk + pool - 1`` (at most
    ``topk / pool`` complete pools), and ``topk + k mod pool`` afterwards.
    """
    n = _maximum(n, 0)
    threshold = topk + pool - 1
    dense = _minimum(n, threshold)

    def remainder_sum(m):
        return m * (m + 1) // 2 - pool * pooled_prefix_sum(m, pool)

    past = _maximum(n, threshold)  # == threshold, so the tail is 0, when n <= threshold
    tail = topk * (past - threshold) + remainder_sum(past) - remainder_sum(threshold)
    return dense * (dense + 1) // 2 + tail


def selected_keys(k, topk: int, pool: int):
    # k <= 0 gives min(...) == k, so the outer max is the empty-context 0.
    return _maximum(_minimum(k, topk + k % pool), 0)


#: Below this many interactions, Python ints beat numpy's per-call overhead
#: (a locked decode shape carries a handful; an aggregate level ~0.5M).
_VECTORIZE_FROM = 64

#: A phase's work: (scored pooled entries, selected pairs, index-entry reads,
#: attended-key reads, completed groups).
PhaseWork = tuple[float, float, float, float, float]


class PooledTopkGeometry:
    """Exact per-request work of a pooled top-k sparse attention.

    Mixed into a dataclass spec that defines ``pool``, ``topk``, ``mechanism`` (for
    errors) and ``_rows``. The reads are the chunk's first query's: every cached
    pooled entry it scores, and its selected keys other than itself. Later queries
    may add keys, so the reads are a lower bound on distinct traffic.
    """

    split_attention_phases = True
    needs_request_geometry = True

    pool: int
    topk: int
    mechanism: str

    def _rows(
        self, wl: Workload, phase_work: list[tuple[str | None, PhaseWork]]
    ) -> list[AttentionSemantic]:
        raise NotImplementedError

    def _interaction_work(self, cached, last) -> tuple:
        """One causal interaction's :data:`PhaseWork`, for scalar or int64-array
        ``cached``/``last``."""
        pool, topk = self.pool, self.topk
        return (
            pooled_prefix_sum(last, pool) - pooled_prefix_sum(cached, pool),
            selected_prefix_sum(last, topk, pool) - selected_prefix_sum(cached, topk, pool),
            cached // pool,
            _maximum(selected_keys(cached + 1, topk, pool) - 1, 0),
            last // pool - cached // pool,
        )

    def _require_causal(self, interactions: list[AttnInteraction]) -> None:
        for interaction in interactions:
            if interaction.mask != "causal":
                raise ValueError(
                    f"{self.mechanism} needs exact per-request causal geometry; "
                    f"got a {interaction.mask!r} interaction (collapsed aggregate?)"
                )

    def _weighted_work(self, interactions: list[AttnInteraction]) -> PhaseWork:
        """:meth:`_interaction_work` summed over ``interactions`` by multiplicity."""
        if len(interactions) >= _VECTORIZE_FROM:
            return tuple(float(value) for value in self._weighted_work_many([interactions])[0])
        self._require_causal(interactions)
        totals = [0.0] * 5
        for interaction in interactions:
            cached = interaction.num_cached_key
            work = self._interaction_work(cached, cached + interaction.num_query)
            for index, value in enumerate(work):
                totals[index] += interaction.multiplicity * value
        return tuple(totals)

    def _weighted_work_many(self, partitions: list[list[AttnInteraction]]) -> np.ndarray:
        """``[partition, 5]``: :meth:`_weighted_work` of every partition, as one
        array evaluation over all of their interactions."""
        flat = [interaction for partition in partitions for interaction in partition]
        self._require_causal(flat)
        size = len(flat)
        cached = np.fromiter((i.num_cached_key for i in flat), np.int64, size)
        last = cached + np.fromiter((i.num_query for i in flat), np.int64, size)
        weight = np.fromiter((i.multiplicity for i in flat), np.float64, size)
        owner = np.repeat(np.arange(len(partitions)), [len(partition) for partition in partitions])
        return np.stack(
            [
                np.bincount(owner, weights=weight * work, minlength=len(partitions))
                for work in self._interaction_work(cached, last)
            ],
            axis=1,
        )

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        phases = wl.attention_phases()
        return self._rows(
            wl, [(phase, self._weighted_work(phase_wl.attn)) for phase, phase_wl in phases]
        )

    def semantic_segments_batch(self, wls: list[Workload]) -> list[list[AttentionSemantic]]:
        """:meth:`semantic_segments` of each workload, with every workload's
        interactions evaluated in one array pass (a locked group's shapes carry a
        handful of requests each, too few to vectorize one at a time)."""
        phases = [wl.attention_phases() for wl in wls]
        work = self._weighted_work_many(
            [phase_wl.attn for wl_phases in phases for _phase, phase_wl in wl_phases]
        )
        rows: list[list[AttentionSemantic]] = []
        start = 0
        for wl, wl_phases in zip(wls, phases, strict=True):
            phase_work = [
                (phase, tuple(float(value) for value in work[start + index]))
                for index, (phase, _phase_wl) in enumerate(wl_phases)
            ]
            rows.append(self._rows(wl, phase_work))
            start += len(wl_phases)
        return rows

    # The historical AttentionSpec surface; Model.label uses semantic_segments.
    def internal_flops(self, wl: Workload) -> float:
        return sum(row.flops for row in self.semantic_segments(wl) if row.bucket == "attn_internal")

    def kv_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.bucket == "attn_internal")

    def cache_write_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.bucket == "embedding")
