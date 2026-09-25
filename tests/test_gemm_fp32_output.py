"""Behavioral tests for the FP32-output GEMM boundary."""

from types import SimpleNamespace

import pytest

from profiling.db.args import DType
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
    [
        (8, 128, 4096, "bf16"),
        (8, 512, 2048, "bf16"),
        (8, 512, 4096, "fp16"),
        (8, 288, 4096, "fp32"),
        (8, 32, 4096, "bf16"),
    ],
)
def test_rejects_non_production_identity(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(*arguments)


def test_fp32_launch_uses_the_cached_contiguous_weight_without_out_dtype() -> None:
    """GLM-5.3's indexer head-weights mm is fp32 x fp32; forcing out_dtype would change the op."""
    calls: list[tuple[object, object, dict[str, object]]] = []
    fake_torch = SimpleNamespace(
        float32=object(),
        mm=lambda left, right, **kwargs: calls.append((left, right, kwargs)),
    )

    _Launch(fake_torch, "hidden.float()", "wp_fp32", fp32_input=True).run()

    assert calls == [("hidden.float()", "wp_fp32", {})]


def _fork_args(*arguments: object):
    return _validate_args(
        *arguments,
        backend="gemm_fp32_output:torch_cublas_vllm_fork",
        supported_dtypes=frozenset({DType.BF16, DType.FP32}),
    )


def test_container_backend_rejects_fp32_input() -> None:
    """The container cuBLAS picks a non-production SGEMM for the fp32 form on B200."""
    with pytest.raises(ProfilerNotImplemented, match="input_dtype in"):
        _validate_args(8, 32, 4096, "fp32")


def test_fork_backend_accepts_both_input_dtypes() -> None:
    assert _fork_args(8, 32, 4096, "fp32").input_dtype is DType.FP32
    assert _fork_args(8, 288, 4096, "bf16").input_dtype is DType.BF16
    with pytest.raises(ProfilerNotImplemented):
        _fork_args(8, 288, 4096, "fp32")


def test_fork_backend_is_registered_on_the_serving_stack() -> None:
    from profiling.db.registry import find_kernel_profiler_spec as get_spec

    spec = get_spec("gemm_fp32_output", "torch_cublas_vllm_fork")
    assert spec.subprocess_env == "vllm_fork_env"
    assert spec.supports.compute == frozenset({DType.BF16, DType.FP32})
    assert get_spec("gemm_fp32_output", "torch_cublas").supports.compute == frozenset(
        {DType.BF16}
    )


def test_logical_bytes_use_four_byte_operands_for_fp32_input() -> None:
    shape = _fork_args(8, 32, 4096, "fp32")

    assert _logical_bytes(shape) == 4 * 8 * 4096 + 4 * 32 * 4096 + 4 * 8 * 32


def test_router_width_is_accepted() -> None:
    assert _validate_args(32, 288, 4096, "bf16").n == 288
