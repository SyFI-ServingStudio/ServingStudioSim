"""CPU tests for the ``dsa_pooled_index_cache_update`` operands and reference."""

from __future__ import annotations

import pytest
import torch

from profiling.runners.attention.dsa_pooled_index_cache_update import (
    build_operands,
    completed_pools,
    validate_args,
)
from profiling.runners.attention.dsa_pooled_index_cache_update_reference import (
    compress_pool,
    dsa_pooled_index_cache_update_reference,
    hadamard_matrix,
)
from profiling.runners.exceptions import ProfilerNotImplemented


def _shape(num_requests: int, next_n: int):
    return validate_args(num_requests, next_n, 4, 16, 128, 64, "bf16", "fp8_e4m3")


@pytest.mark.parametrize(("num_requests", "next_n"), [(1, 1), (5, 1), (3, 2), (16, 8), (7, 5)])
def test_operands_follow_the_fork_decode_metadata(num_requests: int, next_n: int) -> None:
    # Catches a slot mapping that writes a pool on the wrong token, two requests
    # sharing a cache slot or tail block, or a lone request that never completes
    # a pool (which would time only the cheap stash path).
    shape = _shape(num_requests, next_n)
    o = build_operands(torch, shape, torch.device("cpu"))
    last = o.positions % 4 == 3
    assert torch.equal(o.slot_mapping >= 0, last)
    slots = o.slot_mapping[last]
    assert slots.unique().numel() == slots.numel() == completed_pools(shape)
    assert bool(last[0, 0]), "request 0's first token must close a pool"
    blocks = o.tail_slot_mapping // 16
    assert bool((blocks == blocks[:, :1]).all())
    assert blocks[:, 0].unique().numel() == num_requests
    assert torch.equal(o.tail_slot_mapping % 16, o.positions % 16)


def test_compress_pool_equal_scores_is_the_rotated_mean() -> None:
    # Catches a softmax over the wrong axis or a non-orthonormal rotation.
    h = hadamard_matrix(128)
    torch.testing.assert_close(h @ h.T, torch.eye(128), atol=1e-5, rtol=0)
    keys = torch.randn(4, 128, generator=torch.Generator().manual_seed(0))
    keys = keys.to(torch.bfloat16).float()
    values, scale = compress_pool(keys, torch.zeros(4, 128), torch.zeros(4, 128))
    expected = h @ keys.mean(dim=0)
    torch.testing.assert_close(values * scale, expected, atol=0.08 * expected.abs().max(), rtol=0)
    assert float(torch.log2(scale)) == int(torch.log2(scale))


def test_reference_uses_tokens_stashed_earlier_in_the_same_step() -> None:
    # Catches a completion that reads stale ring slots instead of the tokens the
    # same request stashed earlier in this step (the fork's ordering contract).
    shape = validate_args(1, 4, 4, 16, 128, 64, "bf16", "fp8_e4m3")
    tail = torch.zeros(1, 2, 16, 128)
    key = torch.randn(1, 4, 128, generator=torch.Generator().manual_seed(1))
    score = torch.zeros(1, 4, 128)
    positions = torch.tensor([[16, 17, 18, 19]], dtype=torch.int32)
    tail_slots = positions % 16
    slot_mapping = torch.tensor([[-1, -1, -1, 5]], dtype=torch.int32)
    slots, values, scales = dsa_pooled_index_cache_update_reference(
        tail, tail_slots, key, score, torch.zeros(4, 128), slot_mapping, positions, 4
    )
    assert slots.tolist() == [5]
    values, scale = values[0], scales[0]
    expected = hadamard_matrix(128) @ key[0].mean(dim=0)
    torch.testing.assert_close(values * scale, expected, atol=0.08 * expected.abs().max(), rtol=0)
    torch.testing.assert_close(tail[0, 0, :4], key[0])
    assert shape.next_n == 4


def test_rejects_an_index_dim_the_rotation_cannot_take() -> None:
    with pytest.raises(ProfilerNotImplemented):
        validate_args(4, 1, 4, 16, 64, 64, "bf16", "fp8_e4m3")
    with pytest.raises(ValueError):
        validate_args(4, 1, 4, 6, 128, 64, "bf16", "fp8_e4m3")
