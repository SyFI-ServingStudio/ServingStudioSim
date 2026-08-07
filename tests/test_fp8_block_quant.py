"""Focused tests for the ``fp8_block_quant`` L1 profiling contract."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.fp8_block_quant import KIND, Fp8BlockQuantArgs


def test_args_field_order_and_dtype_coercion():
    field_names = [field.name for field in Fp8BlockQuantArgs.__dataclass_fields__.values()]
    assert field_names == ["num_tokens", "hidden_size", "num_problems", "input_dtype"]

    args = coerce_args(
        Fp8BlockQuantArgs,
        {"num_tokens": 64, "hidden_size": 7168, "num_problems": 32, "input_dtype": "bf16"},
    )
    assert args == Fp8BlockQuantArgs(64, 7168, 32, DType.BF16)


def test_registry_contract_and_backend_support():
    profiler_spec = find_kernel_profiler_spec(KIND, "flashinfer_trtllm")
    assert KIND == "fp8_block_quant"
    assert profiler_spec.table_name == KIND
    assert profiler_spec.args_schema is Fp8BlockQuantArgs
    assert profiler_spec.runner_ref.module_name == ("profiling.runners.elementwise.fp8_block_quant")
    assert profiler_spec.runner_ref.function_name == ("profile_fp8_block_quant_flashinfer_trtllm")
    assert profiler_spec.subprocess_env == "flashinfer_pip_env"
    assert profiler_spec.supports.allows(DType.BF16, gpu="NVIDIA H100")
    assert profiler_spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not profiler_spec.supports.allows(DType.FP16, gpu="NVIDIA H200")


def test_import_is_lazy_for_runner_torch_and_flashinfer():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.fp8_block_quant; "
            "print('profiling.runners.elementwise.fp8_block_quant' in sys.modules, "
            "'torch' in sys.modules, 'flashinfer' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False False False"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_tokens": 0}, "num_tokens must be > 0"),
        ({"hidden_size": 130}, "divisible by 128"),
        ({"num_problems": 0}, "num_problems must be > 0"),
        ({"input_dtype": DType.FP16}, "requires input_dtype=bf16"),
    ],
)
def test_runner_validation(kwargs, message):
    from profiling.runners.elementwise import fp8_block_quant as runner

    arguments = {
        "num_tokens": 64,
        "hidden_size": 7168,
        "num_problems": 32,
        "input_dtype": DType.BF16,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        runner._validate_args(**arguments)


def test_grouped_scale_layout_uses_deepgemm_padding():
    from profiling.runners.elementwise import fp8_block_quant as runner

    assert runner._grouped_scale_shape(1, 7168, 1) == (56, 32)
    assert runner._grouped_scale_shape(64, 7168, 1) == (56, 64)
    assert runner._grouped_scale_shape(65, 7168, 1) == (56, 96)
    assert runner._grouped_scale_shape(65, 7168, 32) == (56, 1056)


def test_uniform_problem_boundaries_end_at_public_num_tokens():
    from profiling.runners.elementwise import fp8_block_quant as runner

    assert runner._uniform_problem_boundaries(65, 1) == [0, 65]
    boundaries = runner._uniform_problem_boundaries(65, 32)
    assert len(boundaries) == 33
    assert boundaries[0] == 0
    assert boundaries[-1] == 65
    assert set(right - left for left, right in zip(boundaries[:-1], boundaries[1:])) == {
        2,
        3,
    }


def test_grouped_launch_policy_selects_binary_search_only_for_small_m():
    from profiling.runners.elementwise import fp8_block_quant as runner

    small_num_blocks, small_uses_binary_search = runner._grouped_launch_policy(
        num_tokens=1,
        hidden_size=4096,
        num_problems=32,
        num_device_sms=132,
    )
    large_num_blocks, large_uses_binary_search = runner._grouped_launch_policy(
        num_tokens=16384,
        hidden_size=4096,
        num_problems=32,
        num_device_sms=132,
    )

    assert small_num_blocks == 4
    assert small_uses_binary_search
    assert large_num_blocks == 132
    assert not large_uses_binary_search


def test_missing_flashinfer_jit_is_typed(monkeypatch):
    from profiling.runners.elementwise import fp8_block_quant as runner

    # Blanking the top-level package alone is not enough to simulate a wheel
    # without JIT support: `from flashinfer.jit import env` is served straight
    # out of `sys.modules["flashinfer.jit"]` when some earlier test in the
    # session already imported it, never consulting the `None` parent. Blank
    # every module the loader actually names, and drop the memoized success a
    # previous caller may have left behind, so the test states its condition
    # rather than depending on what ran before it.
    for module_name in ("flashinfer", "flashinfer.jit", "flashinfer.jit.core"):
        monkeypatch.setitem(sys.modules, module_name, None)
    runner._load_flashinfer_grouped_quantizer.cache_clear()
    try:
        with pytest.raises(runner.ProfilerNotImplemented, match="FlashInfer"):
            runner._load_flashinfer_grouped_quantizer()
    finally:
        runner._load_flashinfer_grouped_quantizer.cache_clear()


def test_profile_calls_exact_flashinfer_binding_and_reports_logical_bytes(monkeypatch):
    from profiling.runners.elementwise import fp8_block_quant as runner

    allocations = []
    quantizer_calls = []
    quantizer = SimpleNamespace(
        run_grouped_fp8_block_quant=lambda *arguments: quantizer_calls.append(arguments)
    )
    fake_torch = SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float8_e4m3fn="float8_e4m3fn",
        float32="float32",
        int64="int64",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_capability=lambda _device: (9, 0),
            get_device_name=lambda _device: "NVIDIA H200",
        ),
        randn=lambda shape, **kwargs: allocations.append(("randn", shape, kwargs)) or object(),
        empty=lambda shape, **kwargs: allocations.append(("empty", shape, kwargs)) or object(),
        tensor=lambda values, **kwargs: allocations.append(("tensor", values, kwargs)) or object(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runner, "_load_flashinfer_grouped_quantizer", lambda: quantizer)
    monkeypatch.setattr(
        runner.Timer,
        "cupti",
        lambda function, **kwargs: (function(), 2.0)[1],
    )
    monkeypatch.setattr(
        runner.Energy,
        "perf",
        lambda function, **kwargs: (function(), 0.25)[1],
    )

    metrics = runner.profile_fp8_block_quant_flashinfer_trtllm(65, 128, 32, "bf16")

    assert [allocation[1] for allocation in allocations] == [
        (65, 128),
        (65, 128),
        (1, 1056),
        runner._uniform_problem_boundaries(65, 32),
    ]
    assert len(quantizer_calls) == 2
    assert all(call[-1] == 32 for call in quantizer_calls)
    expected_bytes = 65 * 128 * 2 + 65 * 128 + 65 * 4
    assert metrics.time_ms == 2.0
    assert metrics.tflops == 0.0
    assert metrics.memory_bandwidth_gbps == pytest.approx((expected_bytes / 0.002) / 1e9)
    assert metrics.energy_j == 0.25


def test_generated_facade_symbols_exist():
    from profiling import perf_api

    assert hasattr(perf_api, "get_fp8_block_quant_times")
    assert hasattr(perf_api, "count_missing_fp8_block_quant")
