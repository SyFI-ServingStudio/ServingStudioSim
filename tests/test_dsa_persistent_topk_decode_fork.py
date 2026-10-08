import pytest
import torch

from profiling.runners.attention.dsa_persistent_topk_decode_fork import (
    _build_operands,
    _check_output,
    _max_ragged_lengths,
    _validate_args,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented


def test_production_identity_and_ragged_lengths_are_explicit():
    args = (4, 513, 1, 1024, 512, 1024, "fp32", "int32", "max_ragged")
    assert _validate_args(*args) == (513, 512, 511, 510)
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(*(*args[:-1], "uniform"))
    assert _max_ragged_lengths(3, 0) == (0, 0, 0)


def test_independent_topk_check_rejects_a_wrong_selected_set():
    operands = _build_operands(
        torch, lengths=(513, 512, 7), logits_row_stride=1024, device="cpu", top_k=512
    )
    for row, length in enumerate((513, 512, 7)):
        selected_count = min(length, 512)
        operands.indices[row, :selected_count] = torch.topk(
            operands.logits[row, :length], selected_count
        ).indices.to(torch.int32)
    _check_output(torch, operands)
    operands.indices[0, 0] = 0
    with pytest.raises((AssertionError, KernelLaunchFailed)):
        _check_output(torch, operands)


def test_unmeasured_but_launchable_shapes_are_accepted():
    base = (4, 513, 1, 1024, 512, 1024, "fp32", "int32", "max_ragged")
    assert len(_validate_args(300, *base[1:])) == 300
    assert _validate_args(*base[:3], 2_097_152, 2048, 2_097_152, *base[6:])[0] == 513
    assert _validate_args(*base[:4], 1024, *base[5:])[0] == 513
    # next_n rows per request end at that request's context.
    assert _validate_args(2, 513, 2, *base[3:]) == (512, 513, 511, 512)
    assert _max_ragged_lengths(2, 1, 3) == (0, 0, 1, 0, 0, 0)


def test_kernel_top_k_and_row_bounds_still_fail_closed():
    base = (4, 513, 1, 1024, 512, 1024, "fp32", "int32", "max_ragged")
    with pytest.raises(ProfilerNotImplemented, match="top_k in"):
        _validate_args(*base[:4], 256, *base[5:])
    with pytest.raises(ValueError, match="batch_size"):
        _validate_args(0, *base[1:])
    with pytest.raises(ValueError, match="next_n"):
        _validate_args(*base[:2], 0, *base[3:])
    with pytest.raises(ValueError, match="logits_row_stride"):
        _validate_args(*base[:5], 512, *base[6:])


def test_check_uses_the_operand_top_k():
    operands = _build_operands(
        torch, lengths=(1500, 900), logits_row_stride=2048, device="cpu", top_k=1024
    )
    assert operands.indices.shape == (2, 1024)
    for row, length in enumerate((1500, 900)):
        count = min(length, 1024)
        operands.indices[row, :count] = torch.topk(operands.logits[row, :length], count).indices.to(
            torch.int32
        )
    _check_output(torch, operands)
