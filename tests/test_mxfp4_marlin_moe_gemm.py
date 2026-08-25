"""Behavioral tests for the production MXFP4 Marlin MoE GEMM wrapper."""

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.moe.mxfp4_marlin_moe_gemm_vllm_marlin import (
    _logical_bytes,
    _validate_args,
)


def test_fc1_shape_preserves_exact_expert_distribution() -> None:
    batches = (2, 1, *(0 for _ in range(62)))
    shape = _validate_args(1, 4096, 4096, "bf16", 6, 8, False, batches)

    assert shape.per_group_batches == batches
    assert shape.capacity == 6


def test_logical_bytes_counts_only_active_expert_weights() -> None:
    shape = _validate_args(
        6,
        4096,
        2048,
        "bf16",
        1,
        8,
        True,
        (6, *(0 for _ in range(63))),
    )

    expected = 6 * 2048 * 2 + 4096 * 2048 // 2 + 4096 * (2048 // 32) + 6 * 4096 * 2 + 6 * 4
    assert _logical_bytes(shape) == expected


@pytest.mark.parametrize(
    "arguments",
    [
        (128, 4096, 4096, "bf16", 1, 8, False),
        (128, 4096, 2048, "bf16", 1, 8, False),
        (128, 4096, 4096, "fp16", 6, 8, False),
    ],
)
def test_rejects_non_deepseek_launch_contract(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(*arguments, (1, *(0 for _ in range(63))))
