from __future__ import annotations

import math

import pytest
import torch

import profiling.runners.moe.moe_fused_topk_reference as reference_module
from profiling.runners.moe.moe_fused_topk_reference import moe_fused_topk_reference


def _scalar_oracle(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[float]] = []
    ids: list[list[int]] = []
    for row in logits.float().tolist():
        maximum = max(row)
        softmax = [math.exp(value - maximum) for value in row]
        denominator = sum(softmax)
        softmax = [value / denominator for value in softmax]
        selected = sorted(range(len(row)), key=lambda expert: (-softmax[expert], expert))[:top_k]
        selected_sum = sum(softmax[expert] for expert in selected)
        rows.append([softmax[expert] / selected_sum for expert in selected])
        ids.append(selected)
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(ids, dtype=torch.int32)


def test_unique_rankings_ties_and_source_indices() -> None:
    logits = torch.tensor(
        [[-2, 3, 1, 0], [4, 4, -1, 4]],
        dtype=torch.bfloat16,
    )
    weights, expert_ids, source_indices = moe_fused_topk_reference(logits, 3)
    expected_weights, expected_ids = _scalar_oracle(logits, 3)
    assert torch.equal(expert_ids, torch.tensor([[1, 2, 3], [0, 1, 3]], dtype=torch.int32))
    assert torch.equal(expert_ids, expected_ids)
    assert torch.equal(source_indices, torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.int32))
    torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-6)


def test_extreme_negative_equal_logits_and_selected_renormalization() -> None:
    logits = torch.tensor(
        [[1000, -1000, 0, -5], [-1000, -1001, -1002, -1003], [7, 7, 7, 7]],
        dtype=torch.bfloat16,
    )
    weights, expert_ids, _ = moe_fused_topk_reference(logits, 2)
    expected_weights, expected_ids = _scalar_oracle(logits, 2)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-6)
    assert torch.equal(expert_ids, expected_ids)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3), rtol=0, atol=1e-6)
    assert torch.equal(expert_ids[2], torch.tensor([0, 1], dtype=torch.int32))
    torch.testing.assert_close(weights[2], torch.tensor([0.5, 0.5]), rtol=0, atol=0)


@pytest.mark.parametrize(("num_experts", "top_k"), [(9, 1), (256, 8), (7, 7)])
def test_k_boundaries_and_qwen_representative(num_experts: int, top_k: int) -> None:
    logits = torch.arange(num_experts, dtype=torch.float32).repeat(3, 1).to(torch.bfloat16)
    weights, expert_ids, source_indices = moe_fused_topk_reference(logits, top_k)
    expected_weights, expected_ids = _scalar_oracle(logits, top_k)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-6)
    assert torch.equal(expert_ids, expected_ids)
    for token in range(3):
        assert torch.equal(
            source_indices[token],
            torch.tensor([slot * 3 + token for slot in range(top_k)], dtype=torch.int32),
        )


def test_output_storage_input_immutability_and_noncontiguous_input() -> None:
    backing = torch.tensor(
        [[1, 99, 2, 99, 3, 99], [3, 99, 2, 99, 1, 99]],
        dtype=torch.bfloat16,
    )
    logits = backing[:, ::2]
    assert not logits.is_contiguous()
    before = backing.clone()
    first = moe_fused_topk_reference(logits, 2)
    second = moe_fused_topk_reference(logits, 2)
    weights, expert_ids, source_indices = first
    assert weights.shape == expert_ids.shape == source_indices.shape == (2, 2)
    assert weights.dtype is torch.float32
    assert expert_ids.dtype is source_indices.dtype is torch.int32
    assert all(output.is_contiguous() for output in first)
    pointers = [output.data_ptr() for output in first]
    assert len(set(pointers)) == 3
    assert all(pointer != logits.data_ptr() for pointer in pointers)
    assert all(left.data_ptr() != right.data_ptr() for left, right in zip(first, second))
    assert torch.equal(backing, before)


@pytest.mark.parametrize("top_k", [True, False, 1.0, "1", None])
def test_rejects_noninteger_and_bool_top_k(top_k: object) -> None:
    with pytest.raises(TypeError):
        moe_fused_topk_reference(torch.zeros((1, 2), dtype=torch.bfloat16), top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1, 3])
def test_rejects_out_of_range_top_k(top_k: int) -> None:
    with pytest.raises(ValueError):
        moe_fused_topk_reference(torch.zeros((1, 2), dtype=torch.bfloat16), top_k)


def test_rejects_wrong_type_rank_dtype_empty_nonfinite_and_layout() -> None:
    with pytest.raises(TypeError):
        moe_fused_topk_reference(None, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        moe_fused_topk_reference(torch.zeros((2,), dtype=torch.bfloat16), 1)
    with pytest.raises(TypeError):
        moe_fused_topk_reference(torch.zeros((1, 2), dtype=torch.float32), 1)
    for shape in ((0, 2), (2, 0)):
        with pytest.raises(ValueError):
            moe_fused_topk_reference(torch.zeros(shape, dtype=torch.bfloat16), 1)
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            moe_fused_topk_reference(torch.tensor([[value]], dtype=torch.bfloat16), 1)
    with pytest.raises(ValueError):
        moe_fused_topk_reference(torch.zeros((1, 2), dtype=torch.bfloat16).to_sparse(), 1)
    with pytest.raises(ValueError):
        moe_fused_topk_reference(torch.empty((1, 2), dtype=torch.bfloat16, device="meta"), 1)


def test_rejects_source_index_overflow_without_large_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference_module, "_INT32_MAX", 2)
    with pytest.raises(ValueError, match="int32"):
        moe_fused_topk_reference(torch.zeros((2, 2), dtype=torch.bfloat16), 2)
