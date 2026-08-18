from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import profiling.runners.moe.moe_align_block_size_reference as reference_module
from profiling.runners.moe.moe_align_block_size_reference import (
    moe_align_block_size_reference,
)


def _assert_expert_multisets(
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    post_pad: int,
    block_size: int,
) -> None:
    sentinel = topk_ids.numel()
    flat = topk_ids.reshape(-1)
    for expert in range(int(topk_ids.max()) + 1):
        actual: list[int] = []
        for block, owner in enumerate(expert_ids[: post_pad // block_size].tolist()):
            if owner == expert:
                values = sorted_ids[block * block_size : (block + 1) * block_size]
                actual.extend(value for value in values.tolist() if value != sentinel)
        expected = torch.nonzero(flat == expert, as_tuple=False).flatten().tolist()
        assert actual == expected


def test_small_hand_example_has_exact_order_padding_and_inactive_capacity() -> None:
    topk_ids = torch.tensor([[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], dtype=torch.int32)
    sorted_ids, expert_ids, post_pad = moe_align_block_size_reference(topk_ids, 5, 4)
    assert sorted_ids.shape == (27,)
    assert expert_ids.shape == (7,)
    assert post_pad.tolist() == [16]
    assert sorted_ids[:16].tolist() == [
        3,
        6,
        9,
        12,
        0,
        4,
        10,
        12,
        1,
        7,
        11,
        12,
        2,
        5,
        8,
        12,
    ]
    assert torch.all(sorted_ids[16:] == 12)
    assert expert_ids.tolist() == [1, 2, 3, 4, -1, -1, -1]
    _assert_expert_multisets(topk_ids, sorted_ids, expert_ids, 16, 4)


@pytest.mark.parametrize(("modulus", "expected_post_pad"), [(256, 4096), (64, 1024), (8, 1024)])
def test_qwen_capacity_and_balanced_moderate_concentrated_padding(
    modulus: int, expected_post_pad: int
) -> None:
    assignment_ids = torch.arange(128 * 8, dtype=torch.int32).reshape(128, 8)
    topk_ids = assignment_ids.remainder(modulus)
    sorted_ids, expert_ids, post_pad = moe_align_block_size_reference(topk_ids, 256, 16)
    assert sorted_ids.shape == (4864,)
    assert expert_ids.shape == (304,)
    assert post_pad.tolist() == [expected_post_pad]
    assert torch.all(sorted_ids[expected_post_pad:] == 1024)
    assert torch.all(expert_ids[expected_post_pad // 16 :] == -1)
    _assert_expert_multisets(topk_ids, sorted_ids, expert_ids, expected_post_pad, 16)


def test_small_assignment_capacity_branch_zero_experts_and_slot_flattening() -> None:
    topk_ids = torch.tensor([[2, 7]], dtype=torch.int32)
    sorted_ids, expert_ids, post_pad = moe_align_block_size_reference(topk_ids, 8, 4)
    assert sorted_ids.shape == (8,)
    assert expert_ids.shape == (2,)
    assert post_pad.tolist() == [8]
    assert sorted_ids.tolist() == [0, 2, 2, 2, 1, 2, 2, 2]
    assert expert_ids.tolist() == [2, 7]


def test_deterministic_noncontiguous_input_storage_and_immutability() -> None:
    backing = torch.tensor([[2, 99, 0, 99], [1, 99, 3, 99]], dtype=torch.int32)
    topk_ids = backing[:, ::2]
    before = backing.clone()
    first = moe_align_block_size_reference(topk_ids, 4, 2)
    second = moe_align_block_size_reference(topk_ids, 4, 2)
    assert not topk_ids.is_contiguous()
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert [tensor.dtype for tensor in first] == [torch.int32, torch.int32, torch.int32]
    assert [tensor.shape for tensor in first] == [(8,), (4,), (1,)]
    assert all(tensor.is_contiguous() for tensor in first)
    pointers = [tensor.data_ptr() for tensor in first]
    assert len(set(pointers)) == 3
    assert all(pointer != topk_ids.data_ptr() for pointer in pointers)
    assert all(left.data_ptr() != right.data_ptr() for left, right in zip(first, second))
    assert torch.equal(backing, before)


@pytest.mark.parametrize("value", [True, False, 1.0, "2", None])
@pytest.mark.parametrize("field", ["num_experts", "block_size"])
def test_rejects_noninteger_scalar_arguments(field: str, value: object) -> None:
    kwargs = {"num_experts": 4, "block_size": 2, field: value}
    with pytest.raises(TypeError, match=field):
        moe_align_block_size_reference(torch.tensor([[0]], dtype=torch.int32), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(("num_experts", "block_size"), [(0, 2), (-1, 2), (4, 0), (4, -1)])
def test_rejects_nonpositive_scalar_arguments(num_experts: int, block_size: int) -> None:
    with pytest.raises(ValueError):
        moe_align_block_size_reference(
            torch.tensor([[0]], dtype=torch.int32), num_experts, block_size
        )


def test_rejects_type_rank_dtype_empty_range_duplicates_and_layouts() -> None:
    with pytest.raises(TypeError):
        moe_align_block_size_reference(None, 4, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank 2"):
        moe_align_block_size_reference(torch.tensor([0], dtype=torch.int32), 4, 2)
    with pytest.raises(TypeError, match="dtype"):
        moe_align_block_size_reference(torch.tensor([[0]], dtype=torch.int64), 4, 2)
    for shape in ((0, 1), (1, 0)):
        with pytest.raises(ValueError, match="positive"):
            moe_align_block_size_reference(torch.empty(shape, dtype=torch.int32), 4, 2)
    for ids in (
        torch.tensor([[-1]], dtype=torch.int32),
        torch.tensor([[4]], dtype=torch.int32),
    ):
        with pytest.raises(ValueError, match="expert IDs"):
            moe_align_block_size_reference(ids, 4, 2)
    with pytest.raises(ValueError, match="unique"):
        moe_align_block_size_reference(torch.tensor([[1, 1]], dtype=torch.int32), 4, 2)
    with pytest.raises(ValueError, match="strided"):
        moe_align_block_size_reference(torch.tensor([[0]], dtype=torch.int32).to_sparse(), 4, 2)
    with pytest.raises(ValueError, match="meta"):
        moe_align_block_size_reference(torch.empty((1, 1), dtype=torch.int32, device="meta"), 4, 2)


def test_rejects_non_cpu_tensor_without_allocating_cuda_storage() -> None:
    with FakeTensorMode():
        topk_ids = torch.empty((1, 1), dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="requires CPU"):
            moe_align_block_size_reference(topk_ids, 4, 2)


def test_rejects_checked_int32_overflow_before_output_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference_module, "_INT32_MAX", 7)
    with pytest.raises(ValueError, match="int32"):
        moe_align_block_size_reference(torch.tensor([[0, 1]], dtype=torch.int32), 4, 3)
