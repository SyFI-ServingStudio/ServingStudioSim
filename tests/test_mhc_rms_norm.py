"""Behavioral tests for the two production MHC boundaries."""

import math

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.mhc import mhc_fused_post_pre_rms_norm_vllm_tilelang as fused
from profiling.runners.mhc import mhc_pre_rms_norm_vllm_tilelang as pre
from profiling.runners.mhc import mhc_terminal_head_vllm_tilelang as head
from profiling.runners.mhc._common import (
    Shape,
    validate_args,
)


@pytest.mark.parametrize(
    "arguments",
    [(128, 2048, 4, "bf16"), (128, 4096, 2, "bf16"), (128, 7168, 4, "bf16")],
)
def test_accepts_unmeasured_stream_geometry(arguments: tuple[object, ...]) -> None:
    # The TileLang kernels take hidden_size and hc_mult from the tensor shapes.
    shape = validate_args("mhc", *arguments)
    assert (shape.num_tokens, shape.hidden_size, shape.hc_mult) == arguments[:3]


@pytest.mark.parametrize(
    "arguments",
    [(0, 4096, 4, "bf16"), (128, 0, 4, "bf16"), (128, 4096, 0, "bf16"), (128, 4096, True, "bf16")],
)
def test_rejects_non_positive_geometry(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_args("mhc", *arguments)


def test_rejects_non_bf16_streams() -> None:
    # vLLM's TileLang wrappers assert bf16 residual streams.
    with pytest.raises(ProfilerNotImplemented, match="bf16"):
        validate_args("mhc", 128, 4096, 4, "fp16")


@pytest.mark.parametrize(
    "arguments",
    [
        (48, 4096, 4, "bf16"),
        (48, 5120, 2, "bf16"),
        (48, 5120, 4, "fp16"),
        ((1 << 20) + 1, 5120, 4, "bf16"),
    ],
)
def test_deepgemm_mega_rejects_shapes_outside_the_v41_dispatch(
    arguments: tuple[object, ...],
) -> None:
    # Catches the DeepSeek-V4.1 Mega mHC backend silently profiling the V4
    # hidden size, another hc_mult, or a batch the fork routes to TileLang.
    from profiling.runners.mhc.mhc_fused_post_pre_rms_norm_deepgemm_mega import (
        validate_args as validate_mega_args,
    )

    with pytest.raises(ProfilerNotImplemented):
        validate_mega_args(*arguments)


def _nbytes(*tensors: tuple[tuple[int, ...], int]) -> int:
    return sum(math.prod(shape) * itemsize for shape, itemsize in tensors)


@pytest.mark.parametrize(("num_tokens", "h", "m"), [(1, 4096, 4), (128, 4096, 4), (16, 2048, 2)])
def test_bandwidth_counts_each_external_tensor_once(num_tokens: int, h: int, m: int) -> None:
    t = num_tokens
    shape = Shape(t, h, m)
    residual = ((t, m, h), 2)
    hidden = ((t, h), 2)
    mixes = (((t, m, 1), 4), ((t, m, m), 4))
    pre_weights = (((m * (m + 2), m * h), 4), ((3,), 4), ((m * (m + 2),), 4), ((h,), 2))
    head_weights = (((m, m * h), 4), ((1,), 4), ((m,), 4), ((h,), 2))

    assert pre._logical_bytes(shape) == _nbytes(residual, *pre_weights, *mixes, hidden)
    assert fused._logical_bytes(shape) == _nbytes(
        hidden, residual, *mixes, *pre_weights, residual, *mixes, hidden
    )
    assert head._logical_bytes(shape) == _nbytes(
        hidden, residual, *mixes, *head_weights, residual, hidden
    )
