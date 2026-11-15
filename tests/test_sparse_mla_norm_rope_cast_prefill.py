"""sparse_mla_norm_rope_cast_prefill runner pieces that can break without a GPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from profiling.runners.attention import sparse_mla_norm_rope_cast_prefill as attn  # noqa: E402


def _shape(pairs, ratio=2, topk=8, window=8):
    return attn.validate(
        pairs, window, ratio, topk, 64, 512, 64, 8, True, "bf16", "bf16", "fp8_e4m3"
    )


def test_swa_index_matches_prepare_prompt_v4_pivots() -> None:
    """Two requests, window 8: history rows 0..15, this step's rows from 16.

    Request 0 (5 rows at 195..199) sees 7..3 history slots then its own rows;
    request 1 (3 rows, no history) sees only its own rows.
    """
    index, positions = attn.swa_indices(torch, _shape(((5, 200), (3, 3))), "cpu")
    assert positions.tolist() == [195, 196, 197, 198, 199, 0, 1, 2]
    assert index[0].tolist() == [1, 2, 3, 4, 5, 6, 7, 16]
    assert index[4].tolist() == [5, 6, 7, 16, 17, 18, 19, 20]
    assert index[5].tolist() == [21, -1, -1, -1, -1, -1, -1, -1]
    assert index[7].tolist() == [21, 22, 23, -1, -1, -1, -1, -1]


def test_topk_rows_stay_causal_inside_their_request() -> None:
    """Each row picks min((pos+1)//ratio, topk) distinct rows of its own request."""
    shape = _shape(((6, 40), (4, 9)), ratio=2, topk=8)
    _, positions = attn.swa_indices(torch, shape, "cpu")
    picks = attn.topk_indices(torch, shape, positions, torch.Generator().manual_seed(0))
    # Compressed part starts after 2 x 8 history rows and 10 step rows; request
    # 0 owns 20 compressed rows, request 1 the next 4.
    for row, (pos, request_base) in enumerate(zip(positions.tolist(), [26] * 6 + [46] * 4)):
        valid = [p for p in picks[row].tolist() if p >= 0]
        assert len(valid) == min((pos + 1) // 2, 8) == len(set(valid))
        assert valid == sorted(valid)
        assert all(request_base <= p < request_base + (pos + 1) // 2 for p in valid)


def test_reference_normalizes_q_per_head() -> None:
    """With the Q norm on, scaling the query leaves the output unchanged."""
    shape = _shape(((4, 16),), ratio=0, topk=0, window=8)
    work = attn.build(torch, shape, "cpu")
    rows = torch.arange(4)
    base = attn.reference(torch, shape, work, rows)
    work.q = work.q * 4
    scaled = attn.reference(torch, shape, work, rows)
    # Without the norm the 4x sharper softmax moves outputs by ~0.1; with it,
    # only the norm's eps and bf16 rounding differ.
    assert (scaled - base).abs().max().item() < 2e-3


def test_validate_rejects_unpaired_ratio_and_topk() -> None:
    with pytest.raises(ValueError, match="both 0"):
        _shape(((4, 16),), ratio=2, topk=0)
    with pytest.raises(ValueError, match="bf16"):
        attn.validate(((4, 16),), 8, 0, 0, 64, 512, 64, 8, True, "fp8_e4m3", "bf16", "fp8_e4m3")
