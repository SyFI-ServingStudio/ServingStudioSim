"""Behavioral tests for the FP32-output GEMM boundary."""

from types import SimpleNamespace

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.gemm.gemm_fp32_output_torch_cublas import (
    _Launch,
    _logical_bytes,
    _validate_args,
)


def test_launch_preserves_production_mm_signature() -> None:
    calls: list[tuple[object, object, dict[str, object]]] = []
    fake_torch = SimpleNamespace(
        float32=object(),
        mm=lambda left, right, **kwargs: calls.append((left, right, kwargs)),
    )

    _Launch(fake_torch, "hidden", "weight.T").run()

    assert calls == [("hidden", "weight.T", {"out_dtype": fake_torch.float32})]


def test_logical_bytes_include_fp32_output() -> None:
    shape = _validate_args(8, 512, 4096, "bf16")

    assert _logical_bytes(shape) == 2 * 8 * 4096 + 2 * 512 * 4096 + 4 * 8 * 512


@pytest.mark.parametrize(
    "arguments",
    [(8, 128, 4096, "bf16"), (8, 512, 2048, "bf16"), (8, 512, 4096, "fp16")],
)
def test_rejects_non_production_identity(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(*arguments)
