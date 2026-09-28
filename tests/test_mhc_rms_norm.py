"""Behavioral tests for the two production MHC boundaries."""

import math

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.mhc import mhc_fused_post_pre_rms_norm_vllm_tilelang as fused
from profiling.runners.mhc import mhc_pre_rms_norm_vllm_tilelang as pre
from profiling.runners.mhc import mhc_terminal_head_vllm_tilelang as head
from profiling.runners.mhc._common import (
    BOUNDARY_GPUS,
    TERMINAL_HEAD_GPUS,
    require_gpu,
    validate_args,
)


@pytest.mark.parametrize(
    "arguments",
    [(128, 2048, 4, "bf16"), (128, 4096, 2, "bf16"), (128, 4096, 4, "fp16")],
)
def test_rejects_non_production_identity(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ProfilerNotImplemented):
        validate_args("mhc", *arguments)


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
class _FakeCuda:
    def __init__(self, name: str) -> None:
        self._name = name

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def get_device_name(self, _device: int) -> str:
        return self._name


class _FakeTorch:
    def __init__(self, name: str) -> None:
        self.cuda = _FakeCuda(name)


def test_boundary_gate_admits_b200_but_terminal_head_stays_h200() -> None:
    # Catches the boundary runners rejecting the GLM-5.3-Flash B200 rows, or the
    # DSV4-only terminal head silently widening to an unverified GPU.
    require_gpu(_FakeTorch("NVIDIA B200"), "mhc", BOUNDARY_GPUS)
    require_gpu(_FakeTorch("NVIDIA H200"), "mhc", BOUNDARY_GPUS)
    with pytest.raises(ProfilerNotImplemented):
        require_gpu(_FakeTorch("NVIDIA B200"), "head", TERMINAL_HEAD_GPUS)
    with pytest.raises(ProfilerNotImplemented):
        require_gpu(_FakeTorch("NVIDIA H100 80GB HBM3"), "mhc", BOUNDARY_GPUS)


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
