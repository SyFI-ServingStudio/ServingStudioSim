"""Behavioral tests for the two production MHC boundaries."""

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.mhc._deepseek_v4 import validate_args


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
