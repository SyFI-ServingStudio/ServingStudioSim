from __future__ import annotations

import pytest

from profiling.runners.moe.exact_topk import exact_topk_ids
from profiling.runners.moe.nvfp4_fused_moe import _logical_bytes


def test_exact_topk_ids_realize_distinct_expert_counts() -> None:
    batches = (4, 3, 3, 2)
    ids = exact_topk_ids(num_tokens=6, top_k=2, per_expert_batches=batches)

    flattened = [expert for row in ids for expert in row]
    assert [flattened.count(expert) for expert in range(4)] == list(batches)
    assert all(len(row) == len(set(row)) == 2 for row in ids)


def test_exact_topk_ids_reject_impossible_expert_degree() -> None:
    with pytest.raises(ValueError, match="more than one row per token"):
        exact_topk_ids(num_tokens=2, top_k=2, per_expert_batches=(3, 1))


def _args() -> dict:
    # One EP rank of four: 32 local experts of 128, half of them idle.
    per_expert_batches = tuple(8 if expert % 2 == 0 else 0 for expert in range(128))
    return {
        "num_tokens": 64,
        "hidden_size": 6144,
        "intermediate_size": 1536,
        "num_experts": 128,
        "num_local_experts": 32,
        "top_k": 8,
        "per_expert_batches": per_expert_batches,
    }


def test_logical_bytes_follow_the_finalize_mode_of_the_dispatch() -> None:
    """The two dispatches do not produce the same output.

    vLLM finalizes to one row per input token; SGLang defers and writes one
    unfinalized row per local expert assignment. Billing both at the finalized
    size would misreport bandwidth by ``2 * hidden * (tokens - local_rows)`` on
    every deferred row, in whichever direction the rank's share happens to fall.
    """
    args = _args()
    local_rows = sum(args["per_expert_batches"][: args["num_local_experts"]])
    assert local_rows == 128  # 16 active local experts x 8 rows

    finalized = _logical_bytes(args, do_finalize=True)
    deferred = _logical_bytes(args, do_finalize=False)

    hidden = args["hidden_size"]
    assert deferred - finalized == 2 * hidden * (local_rows - args["num_tokens"])
    assert finalized != deferred


def test_logical_bytes_charge_weights_only_for_active_local_experts() -> None:
    """Weight traffic dominates at small batch, so idle experts must not count.

    Every expert this rank owns holds the same weights; charging all 32 when only
    16 receive a row would roughly double the reported bytes at decode shapes.
    """
    args = _args()
    dense = dict(args, per_expert_batches=tuple(4 for _ in range(128)))

    hidden, intermediate = args["hidden_size"], args["intermediate_size"]
    per_expert_weights = (
        intermediate * hidden
        + intermediate * hidden // 8
        + hidden * intermediate // 2
        + hidden * intermediate // 16
    )
    # Same 128 local rows either way, so the whole difference is weight traffic
    # plus the three FP32 per-expert scale vectors.
    assert sum(dense["per_expert_batches"][: dense["num_local_experts"]]) == 128
    extra = _logical_bytes(dense, do_finalize=True) - _logical_bytes(args, do_finalize=True)
    assert extra == 16 * (per_expert_weights + 3 * 4)
