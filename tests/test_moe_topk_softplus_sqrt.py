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
    "arguments",
    [
        ("hash", 128, 256, 6, 0, "fp32"),
        ("learned", 128, 256, 6, 129280, "fp32"),
        ("learned", 128, 256, 6, 0, "bf16"),
    ],
)
def test_rejects_non_production_mode_identity(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(*arguments)
