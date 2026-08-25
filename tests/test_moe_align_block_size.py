"""Behavioral tests for the MoE block-alignment input oracle."""

from collections import Counter

import pytest

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
        {"block_size": 7},
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
