"""Behavioral tests for the MoE block-alignment input oracle."""

import sys
from collections import Counter

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.moe.moe_align_block_size_reference import (
    build_topk_ids,
    expected_block_owners,
    logical_bytes,
    validate_shape,
)


def test_route_builder_is_balanced_and_row_unique() -> None:
    shape = validate_shape(6, 8, 3, 16)
    routes = build_topk_ids(shape)

    assert all(len(row) == len(set(row)) == 3 for row in routes)
    observed = Counter(expert for row in routes for expert in row)
    assert tuple(observed[expert] for expert in range(8)) == (3, 3, 2, 2, 2, 2, 2, 2)


def test_padding_owners_and_logical_traffic_are_independently_derived() -> None:
    shape = validate_shape(3, 4, 2, 8)

    assert shape.padded_routes == 32
    assert expected_block_owners(shape.expert_counts, shape.block_size) == (0, 1, 2, 3)
    # 6 route IDs + 32 padded IDs + 4 owners + 1 count.
    assert logical_bytes(shape) == 4 * (6 + 32 + 4 + 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_k": 9},
        {"block_size": 0},
        {"num_tokens": 0},
        {"num_experts": 0},
    ],
)
def test_rejects_nonphysical_route_shapes(overrides: dict[str, object]) -> None:
    shape = {
        "num_tokens": 6,
        "num_experts": 8,
        "top_k": 3,
        "block_size": 16,
    }
    shape.update(overrides)

    with pytest.raises(ValueError):
        validate_shape(**shape)  # type: ignore[arg-type]


@pytest.mark.parametrize("block_size", [1, 7, 16, 128, 256])
def test_any_positive_block_size_pads_parametrically(block_size: int) -> None:
    shape = validate_shape(5, 4, 2, block_size)
    owners = expected_block_owners(shape.expert_counts, block_size)

    assert shape.padded_routes == len(owners) * block_size
    assert all(count <= block_size * owners.count(e) for e, count in enumerate(shape.expert_counts))


def test_vllm_runner_rejects_more_experts_than_its_block_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from profiling.runners.moe import moe_align_block_size_vllm_cuda as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ProfilerNotImplemented, match="at most 992 experts"):
        runner.profile_moe_align_block_size_vllm_cuda(4, 993, 2, 16)


def test_vllm_runner_needs_cuda_but_no_gpu_model() -> None:
    from types import SimpleNamespace

    from profiling.runners.moe import moe_align_block_size_vllm_cuda as runner

    def fake_torch(available: bool) -> SimpleNamespace:
        return SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: available,
                current_device=lambda: 0,
                get_device_name=lambda _: "NVIDIA B200",
            )
        )

    runner._require_cuda(fake_torch(True))
    with pytest.raises(ProfilerNotImplemented, match="requires CUDA"):
        runner._require_cuda(fake_torch(False))
