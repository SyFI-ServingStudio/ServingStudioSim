"""Behavioral tests for the merged sqrt-softplus router kind."""

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.moe.moe_topk_softplus_sqrt_vllm_cuda import (
    _logical_bytes,
    _validate_args,
)


def test_modes_share_one_contract_but_keep_distinct_physical_inputs() -> None:
    learned = _validate_args("learned", 128, 256, 6, 0, "fp32")
    hashed = _validate_args("hash", 128, 256, 6, 129280, "fp32")

    assert learned.selection_mode == "learned"
    assert hashed.selection_mode == "hash"
    common_bytes = 4 * (128 * 256 + 2 * 128 * 6)
    assert _logical_bytes(learned) == common_bytes + 4 * (256 + 128 * 6)
    assert _logical_bytes(hashed) == common_bytes + 4 * (128 + 128 * 6)


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (("hash", 128, 256, 6, 0, "fp32"), ValueError),
        (("learned", 128, 256, 6, 129280, "fp32"), ValueError),
        (("learned", 128, 8, 9, 0, "fp32"), ValueError),
        (("learned", 128, 256, 6, 0, "bf16"), ProfilerNotImplemented),
        (("learned", 128, 100, 6, 0, "fp32"), ProfilerNotImplemented),
        (("hash", 128, 1000, 6, 4096, "fp32"), ProfilerNotImplemented),
    ],
)
def test_rejects_inconsistent_modes_and_kernel_less_expert_counts(
    arguments: tuple[object, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        _validate_args(*arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        ("learned", 128, 256, 8, 0, "fp32"),
        ("learned", 64, 384, 6, 0, "fp32"),
        ("learned", 64, 192, 4, 0, "fp32"),
        ("hash", 64, 384, 6, 129280, "fp32"),
        ("hash", 64, 128, 8, 32000, "fp32"),
        ("hash", 64, 576, 10, 1, "fp32"),
    ],
)
def test_accepts_any_dispatched_expert_count_top_k_and_hash_table(
    arguments: tuple[object, ...],
) -> None:
    shape = _validate_args(*arguments)
    assert (shape.num_experts, shape.top_k, shape.hash_vocab_size) == arguments[2:5]
    num_tokens, num_experts, top_k = arguments[1:4]
    mode_vector = num_experts if arguments[0] == "learned" else num_tokens
    assert _logical_bytes(shape) == 4 * (
        num_tokens * num_experts + 2 * num_tokens * top_k + mode_vector + num_tokens * top_k
    )
