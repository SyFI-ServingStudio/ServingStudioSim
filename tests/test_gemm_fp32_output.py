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
    ("arguments", "error"),
    [
        ((8, 512, 4096, "fp16"), ProfilerNotImplemented),
        ((0, 512, 4096, "bf16"), ValueError),
        ((8, 0, 4096, "bf16"), ValueError),
        ((8, 512, -1, "fp32"), ValueError),
    ],
)
def test_rejects_unbuilt_dtype_and_empty_shapes(arguments: tuple[object, ...], error) -> None:
    with pytest.raises(error):
        _validate_args(*arguments)


@pytest.mark.parametrize(
    "arguments",
    [(8, 128, 4096, "bf16"), (8, 512, 2048, "bf16"), (8, 288, 4096, "fp32"), (8, 32, 7168, "bf16")],
)
def test_accepts_any_positive_shape(arguments: tuple[object, ...]) -> None:
    shape = _validate_args(*arguments)
    assert (shape.m, shape.n, shape.k) == arguments[:3]


def test_glm53_router_and_indexer_forms() -> None:
    assert _validate_args(8, 288, 4096, "bf16").n == 288
    shape = _validate_args(8, 32, 4096, "fp32")
    assert _logical_bytes(shape) == 4 * 8 * 4096 + 4 * 32 * 4096 + 4 * 8 * 32

    calls: list[tuple[object, object, dict[str, object]]] = []
    fake_torch = SimpleNamespace(mm=lambda left, right, **kw: calls.append((left, right, kw)))
    _Launch(fake_torch, "hidden.float()", "wp_fp32", fp32_input=True).run()
    assert calls == [("hidden.float()", "wp_fp32", {})]


def test_fp32_prepare_needs_highest_precision_and_caches_a_contiguous_weight() -> None:
    from profiling.runners.gemm.gemm_fp32_output_torch_cublas import _prepare

    class _Weight:
        @property
        def T(self):  # noqa: N802 - mirrors torch.Tensor.T
            return SimpleNamespace(contiguous=lambda: "weight.T.contiguous()")

    precision = "high"
    fake_torch = SimpleNamespace(
        float32="float32",
        bfloat16="bfloat16",
        device=lambda *args: args,
        cuda=SimpleNamespace(current_device=lambda: 0),
        Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: None),
        get_float32_matmul_precision=lambda: precision,
        randn=lambda shape, **kw: "input" if shape[1] == 4096 and shape[0] == 8 else _Weight(),
    )
    shape = _validate_args(8, 32, 4096, "fp32")

    with pytest.raises(ProfilerNotImplemented, match="highest"):
        _prepare(fake_torch, shape)

    precision = "highest"
    launch = _prepare(fake_torch, shape)
    assert launch.fp32_input
    assert (launch.input_tensor, launch.weight_transposed) == ("input", "weight.T.contiguous()")


def test_ll_bf16_backend_reuses_the_kind_schema_in_the_fork_env() -> None:
    # The vllm.models.inkling module exists only in the alignment fork; any other env would
    # fail every spec at import time.
    from profiling.db.registry import find_kernel_profiler_spec
    from profiling.kernels.gemm_fp32_output import GemmFp32OutputArgs

    spec = find_kernel_profiler_spec("gemm_fp32_output", "vllm_ll_bf16_auto")
    assert spec.args_schema is GemmFp32OutputArgs
    assert spec.subprocess_env == "vllm_upstream_fork_env"
    assert spec.supports.compute == frozenset({DType.BF16})


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ((8, 264, 4096, "fp32"), ProfilerNotImplemented),
        ((0, 264, 4096, "bf16"), ValueError),
        ((8, 264, 0, "bf16"), ValueError),
    ],
)
def test_ll_bf16_rejects_unbuilt_dtype_and_empty_shapes(
    arguments: tuple[object, ...], error
) -> None:
    # The production call casts nothing: an fp32 activation would silently take
    # the cuBLAS branch at every m and mislabel it as the ll_bf16 path.
    from profiling.runners.gemm.gemm_fp32_output_vllm_ll_bf16 import (
        profile_gemm_fp32_output_vllm_ll_bf16,
    )

    with pytest.raises(error):
        profile_gemm_fp32_output_vllm_ll_bf16(*arguments)


def test_ll_bf16_rejects_the_cute_dsl_tier_without_cute_dsl(monkeypatch) -> None:
    # The production call would run torch.mm at m <= 64 and the row would be a
    # cuBLAS time under the ll_bf16 label.
    import sys

    from profiling.runners.gemm.gemm_fp32_output_vllm_ll_bf16 import (
        profile_gemm_fp32_output_vllm_ll_bf16,
    )

    def unexpected_call(*_args):
        raise AssertionError("the GEMM must not run")

    fake_moe = SimpleNamespace(
        _INKLING_LL_BF16_MAX_TOKENS=64,
        _linear_with_fp32_out=unexpected_call,
        current_platform=SimpleNamespace(has_device_capability=lambda capability: True),
        ll_bf16=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "vllm.models.inkling.nvidia.moe", fake_moe)

    with pytest.raises(ProfilerNotImplemented, match="cute-DSL"):
        profile_gemm_fp32_output_vllm_ll_bf16(64, 264, 4096, "bf16")
