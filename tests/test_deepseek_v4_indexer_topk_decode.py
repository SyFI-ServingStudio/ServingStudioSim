import pytest
import torch

from profiling.runners.attention.deepseek_v4_indexer_topk_decode_cuda import (
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
        torch, lengths=(513, 512, 7), logits_row_stride=1024, device="cpu"
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
