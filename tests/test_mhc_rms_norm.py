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
