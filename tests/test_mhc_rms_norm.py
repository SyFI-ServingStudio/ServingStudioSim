"""Behavioral tests for the two production MHC boundaries."""

import math

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.mhc import mhc_fused_post_pre_rms_norm_vllm_tilelang as fused
from profiling.runners.mhc import mhc_pre_rms_norm_vllm_tilelang as pre
from profiling.runners.mhc import mhc_terminal_head_vllm_tilelang as head
from profiling.runners.mhc._common import validate_args


@pytest.mark.parametrize(
    "arguments",
    [(128, 2048, 4, "bf16"), (128, 4096, 2, "bf16"), (128, 4096, 4, "fp16")],
)
def test_rejects_non_production_identity(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ProfilerNotImplemented):
        validate_args("mhc", *arguments)


def _nbytes(*tensors: tuple[tuple[int, ...], int]) -> int:
    return sum(math.prod(shape) * itemsize for shape, itemsize in tensors)


@pytest.mark.parametrize("num_tokens", [1, 128])
def test_bandwidth_counts_each_external_tensor_once(num_tokens: int) -> None:
    t, h, m = num_tokens, 4096, 4
    residual = ((t, m, h), 2)
    hidden = ((t, h), 2)
    mixes = (((t, m, 1), 4), ((t, m, m), 4))
    pre_weights = (((m * (m + 2), m * h), 4), ((3,), 4), ((m * (m + 2),), 4), ((h,), 2))
    head_weights = (((m, m * h), 4), ((1,), 4), ((m,), 4), ((h,), 2))

    assert pre._logical_bytes(t) == _nbytes(residual, *pre_weights, *mixes, hidden)
    assert fused._logical_bytes(t) == _nbytes(
        hidden, residual, *mixes, *pre_weights, residual, *mixes, hidden
    )
    assert head._logical_bytes(t) == _nbytes(
        hidden, residual, *mixes, *head_weights, residual, hidden
    )
