"""Behavioral tests for the two production MHC boundaries."""

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.mhc._deepseek_v4 import (
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
