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
    ("n", "k", "input_top_k", "mul_topk_weights", "num_local_experts"),
    [
        (4096, 4096, 6, False, 64),  # DeepSeek V4 FC1 at EP4
        (4096, 2048, 1, True, 64),  # DeepSeek V4 FC2 at EP4
        (4096, 4096, 6, False, 32),  # EP8
        (4096, 2048, 1, True, 256),  # EP1
        (3072, 7168, 8, False, 16),  # n % 128, k % 128
        (192, 128, 4, True, 3),  # n % 64 with k % 128
        (256, 64, 2, False, 5),  # n % 128 with k % 64
    ],
)
def test_accepts_any_marlin_tileable_shape_and_local_expert_count(
    n: int, k: int, input_top_k: int, mul_topk_weights: bool, num_local_experts: int
) -> None:
    batches = (1, *(0 for _ in range(num_local_experts - 1)))
    shape = _validate_args(4, n, k, "bf16", input_top_k, 16, mul_topk_weights, batches)

    assert shape.num_local_experts == num_local_experts
    assert (shape.n, shape.k, shape.input_top_k) == (n, k, input_top_k)


@pytest.mark.parametrize(
    ("arguments", "batches", "error"),
    [
        # Marlin thread tiles need (n % 64, k % 128) or (n % 128, k % 64).
        ((128, 192, 64, "bf16", 1, 8, False), (1,), ProfilerNotImplemented),
        ((128, 4096, 96, "bf16", 1, 8, False), (1,), ProfilerNotImplemented),
        ((128, 4096, 4096, "bf16", 1, 24, False), (1,), ProfilerNotImplemented),
        ((128, 4096, 4096, "fp16", 6, 8, False), (1,), ProfilerNotImplemented),
        ((128, 4096, 4096, "bf16", 1, 8, False), (), ValueError),
        (
            (128, 4096, 4096, "bf16", 1, 8, False),
            (1, *(0 for _ in range(991))),
            ProfilerNotImplemented,
        ),
    ],
)
def test_rejects_shapes_marlin_cannot_launch(
    arguments: tuple[object, ...], batches: tuple[int, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        _validate_args(*arguments, batches)
