"""Focused tests for the dense vLLM per-token-group FP8 quant profiler."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.fp8_per_token_group_quant import (
    KIND,
    Fp8PerTokenGroupQuantArgs,
)


def test_args_field_order_and_dtype_coercion():
    field_names = [
        field.name for field in Fp8PerTokenGroupQuantArgs.__dataclass_fields__.values()
    ]
    assert field_names == [
        "num_tokens",
        "hidden_size",
        "group_size",
        "input_dtype",
        "scale_format",
    ]
    args = coerce_args(
        Fp8PerTokenGroupQuantArgs,
        {
            "num_tokens": 64,
            "hidden_size": 4096,
            "group_size": 128,
            "input_dtype": "bf16",
            "scale_format": "ue8m0_column_major",
        },
    )
    assert args == Fp8PerTokenGroupQuantArgs(
        64,
        4096,
        128,
        DType.BF16,
        "ue8m0_column_major",
    )


def test_registry_contract_and_backend_support():
    profiler_spec = find_kernel_profiler_spec(KIND, "vllm_cuda")
    assert KIND == "fp8_per_token_group_quant"
    assert profiler_spec.table_name == KIND
    assert profiler_spec.args_schema is Fp8PerTokenGroupQuantArgs
    assert profiler_spec.runner_ref.module_name == (
        "profiling.runners.elementwise.fp8_per_token_group_quant"
    )
    assert profiler_spec.runner_ref.function_name == (
        "profile_fp8_per_token_group_quant_vllm_cuda"
    )
    assert profiler_spec.subprocess_env == "vllm_env"
    assert profiler_spec.supports.allows(DType.BF16, gpu="NVIDIA H100")
    assert profiler_spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not profiler_spec.supports.allows(DType.FP16, gpu="NVIDIA H200")


def test_import_is_lazy_for_runner_torch_and_vllm():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.fp8_per_token_group_quant; "
            "print('profiling.runners.elementwise.fp8_per_token_group_quant' "
            "in sys.modules, 'torch' in sys.modules, 'vllm' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False False False"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_tokens": 0}, "num_tokens must be > 0"),
        ({"hidden_size": 0}, "hidden_size must be > 0"),
        ({"hidden_size": 4100}, "divide hidden_size"),
        ({"group_size": 64}, "requires group_size=128"),
        ({"input_dtype": DType.FP16}, "requires input_dtype=bf16"),
        ({"scale_format": "fp32_row_major"}, "ue8m0_column_major"),
    ],
)
def test_runner_validation(kwargs, message):
    from profiling.runners.elementwise import fp8_per_token_group_quant as runner

    arguments = {
        "num_tokens": 64,
        "hidden_size": 4096,
        "group_size": 128,
        "input_dtype": DType.BF16,
        "scale_format": "ue8m0_column_major",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        runner._validate_args(**arguments)


def test_profile_calls_exact_vllm_op_and_reports_logical_bytes(monkeypatch):
    from profiling.runners.elementwise import fp8_per_token_group_quant as runner

    allocations = []
    quant_calls = []

    class FakeScaleTensor:
        def permute(self, *dimensions):
            allocations.append(("permute", dimensions))
            return self

    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        float8_e4m3fn="float8_e4m3fn",
        float32="float32",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_capability=lambda _device: (9, 0),
            get_device_name=lambda _device: "NVIDIA H200",
        ),
        randn=lambda shape, **kwargs: allocations.append(("randn", shape, kwargs))
        or object(),
        empty=lambda shape, **kwargs: allocations.append(("empty", shape, kwargs))
        or FakeScaleTensor(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        runner,
        "_load_vllm_quant_op",
        lambda _torch, *_: lambda *arguments: quant_calls.append(arguments),
    )
    monkeypatch.setattr(runner.Timer, "cupti", lambda function, **kwargs: (function(), 2.0)[1])
    monkeypatch.setattr(
        runner.Energy,
        "perf",
        lambda function, **kwargs: (function(), 0.25)[1],
    )

    metrics = runner.profile_fp8_per_token_group_quant_vllm_cuda(
        64,
        4096,
        128,
        "bf16",
        "ue8m0_column_major",
    )

    assert [allocation[1] for allocation in allocations[:3]] == [
        (64, 4096),
        (64, 4096),
        (32, 64),
    ]
    assert allocations[3] == ("permute", (-1, -2))
    assert len(quant_calls) == 2
    assert all(call[3] == 128 for call in quant_calls)
    assert all(call[7:] == (True, True, False) for call in quant_calls)
    expected_bytes = 64 * 4096 * 2 + 64 * 4096 + 64 * 32 * 4
    assert metrics.time_ms == 2.0
    assert metrics.tflops == 0.0
    assert metrics.memory_bandwidth_gbps == pytest.approx((expected_bytes / 0.002) / 1e9)
    assert metrics.energy_j == 0.25


@pytest.mark.parametrize(
    ("scale_format", "capability", "gpu_name", "accepted"),
    [
        ("ue8m0_column_major", (9, 0), "NVIDIA H200", True),
        ("ue8m0_column_major", (10, 0), "NVIDIA B200", False),
        ("ue8m0_row_major", (10, 0), "NVIDIA B200", True),
        ("ue8m0_row_major", (9, 0), "NVIDIA H200", False),
        ("ue8m0_packed_int32", (10, 0), "NVIDIA B200", True),
        ("ue8m0_packed_int32", (9, 0), "NVIDIA H200", False),
    ],
)
def test_scale_format_is_paired_with_its_verified_gpu(scale_format, capability, gpu_name, accepted):
    """A layout profiled on the wrong architecture would time a path production never runs."""
    from profiling.runners.elementwise import fp8_per_token_group_quant as runner
    from profiling.runners.exceptions import ProfilerNotImplemented

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_capability=lambda _device: capability,
            get_device_name=lambda _device: gpu_name,
        )
    )
    if accepted:
        runner._validate_cuda_device(fake_torch, scale_format)
    else:
        with pytest.raises(ProfilerNotImplemented):
            runner._validate_cuda_device(fake_torch, scale_format)


def test_packed_format_allocates_deepgemm_tma_layout_and_calls_packed_op():
    """The packed writer takes no layout flags and an MN-major int32 scale tensor."""
    from profiling.runners.elementwise import fp8_per_token_group_quant as runner

    strided = []
    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        float8_e4m3fn="float8_e4m3fn",
        int32="int32",
        randn=lambda shape, **kwargs: object(),
        empty=lambda shape, **kwargs: object(),
        empty_strided=lambda shape, stride, **kwargs: strided.append((shape, stride, kwargs))
        or object(),
    )
    operands = runner._allocate_operands(
        fake_torch, num_tokens=33, hidden_size=4096, group_size=128, scale_format="ue8m0_packed_int32"
    )
    assert strided == [((33, 8), (1, 36), {"dtype": "int32", "device": "cuda"})]
    calls = []
    runner._launch_quant(
        lambda *arguments: calls.append(arguments), *operands, 128, "ue8m0_packed_int32"
    )
    assert len(calls[0]) == 7


def test_generated_facade_symbols_exist():
    from profiling import perf_api

    assert hasattr(perf_api, "get_fp8_per_token_group_quant_times")
    assert hasattr(perf_api, "count_missing_fp8_per_token_group_quant")


def test_exact_vllm_op_matches_torch_reference_on_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("vllm._C_stable_libtorch")

    from profiling.runners.elementwise import fp8_per_token_group_quant as runner

    runner._validate_cuda_device(torch)
    quant_op = runner._load_vllm_quant_op(torch)
    input_tensor, output_quantized, output_scales = runner._allocate_operands(
        torch,
        num_tokens=17,
        hidden_size=256,
        group_size=128,
    )
    runner._launch_quant(
        quant_op,
        input_tensor,
        output_quantized,
        output_scales,
        128,
    )
    torch.cuda.synchronize()

    grouped_input = input_tensor.float().view(17, 2, 128)
    absmax = grouped_input.abs().amax(dim=-1).clamp_min(runner._EPSILON)
    reference_scales = torch.exp2(torch.ceil(torch.log2(absmax / runner._FP8_E4M3_MAX)))
    reference_quantized = (
        (grouped_input / reference_scales.unsqueeze(-1))
        .clamp(runner._FP8_E4M3_MIN, runner._FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .view(17, 256)
    )

    torch.testing.assert_close(output_scales, reference_scales, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        output_quantized.float(),
        reference_quantized.float(),
        rtol=0.0,
        atol=0.0,
    )


def test_fork_backend_runs_the_blackwell_layouts_on_the_serving_stack() -> None:
    from profiling.db.registry import find_kernel_profiler_spec
    from profiling.runners.exceptions import ProfilerNotImplemented
    from profiling.runners.elementwise.fp8_per_token_group_quant import (
        profile_fp8_per_token_group_quant_vllm_fork_cuda,
    )

    spec = find_kernel_profiler_spec("fp8_per_token_group_quant", "vllm_fork_cuda")
    assert spec.subprocess_env == "vllm_fork_env"
    assert spec.supports.gpus == frozenset({"NVIDIA B200"})
    with pytest.raises(ProfilerNotImplemented, match="scale_format"):
        profile_fp8_per_token_group_quant_vllm_fork_cuda(
            32, 4096, 128, "bf16", "ue8m0_column_major"
        )
