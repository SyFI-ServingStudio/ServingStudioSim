from __future__ import annotations

import pytest

from profiling.runners.moe.nvfp4_fused_moe import _exact_topk_ids


def test_exact_topk_ids_realize_distinct_expert_counts() -> None:
    batches = (4, 3, 3, 2)
    ids = _exact_topk_ids(num_tokens=6, top_k=2, per_expert_batches=batches)

    flattened = [expert for row in ids for expert in row]
    assert [flattened.count(expert) for expert in range(4)] == list(batches)
    assert all(len(row) == len(set(row)) == 2 for row in ids)


def test_exact_topk_ids_reject_impossible_expert_degree() -> None:
    with pytest.raises(ValueError, match="more than one row per token"):
        _exact_topk_ids(num_tokens=2, top_k=2, per_expert_batches=(3, 1))
