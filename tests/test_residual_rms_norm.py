"""CPU registration tests for the ``residual_rms_norm`` L1 kind."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import (
    MetricFamily,
    find_kernel_profiler_spec,
    known_backends,
)
from profiling.kernels.residual_rms_norm import KIND, ResidualRmsNormArgs
from profiling.runners.exceptions import ProfilerNotImplemented


def test_args_field_order_and_dtype_coercion():
    assert [field.name for field in fields(ResidualRmsNormArgs)] == [
        "m",
        "hidden",
        "dtype",
    ]
    args = coerce_args(
        ResidualRmsNormArgs,
        {"m": 128, "hidden": 6144, "dtype": "bfloat16"},
    )
    assert args == ResidualRmsNormArgs(
        m=128,
        hidden=6144,
        dtype=DType.BF16,
    )
    with pytest.raises(Exception):
        args.m = 1


def test_kind_table_backend_and_runner_ref_contract():
    spec = find_kernel_profiler_spec(KIND, "torch")

    assert KIND == "residual_rms_norm"
    assert spec.kernel_kind == KIND
    assert spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.args_schema is ResidualRmsNormArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.runner_ref.module_name == ("profiling.runners.norm.residual_rms_norm_torch")
    assert spec.runner_ref.function_name == "profile_residual_rms_norm"


def test_torch_backend_support_is_16_bit_only():
    support = find_kernel_profiler_spec(KIND, "torch").supports

    assert support.allows(DType.BF16, gpu="NVIDIA H200")
    assert support.allows(DType.FP16, gpu="NVIDIA H200")
    assert not support.allows(DType.FP32, gpu="NVIDIA H200")
    assert not support.allows(DType.FP8_E4M3, gpu="NVIDIA H200")


def test_vllm_cuda_registration_reuses_kind_table_and_args():
    spec = find_kernel_profiler_spec(KIND, "vllm_cuda")

    assert known_backends(KIND) == ["torch", "vllm_cuda"]
    assert spec.kernel_kind == KIND
    assert spec.table_name == KIND
    assert spec.args_schema is ResidualRmsNormArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.subprocess_env == "vllm_env"
    assert spec.runner_ref.module_name == (
        "profiling.runners.norm.residual_rms_norm_vllm_cuda"
    )
    assert spec.runner_ref.function_name == "profile_residual_rms_norm_vllm_cuda"


def test_vllm_cuda_support_is_16_bit_h200_only():
    support = find_kernel_profiler_spec(KIND, "vllm_cuda").supports

    assert support.allows(DType.BF16, gpu="NVIDIA H200")
    assert support.allows(DType.FP16, gpu="NVIDIA H200")
    assert not support.allows(DType.FP32, gpu="NVIDIA H200")
    assert not support.allows(DType.FP8_E4M3, gpu="NVIDIA H200")
    assert not support.allows(DType.BF16, gpu="NVIDIA H100")
    assert not support.allows(DType.BF16, gpu="NVIDIA B200")


def test_registry_barrel_does_not_import_frameworks_or_runners():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('vllm' in sys.modules); "
                "print('profiling.runners.norm.residual_rms_norm_torch' "
                "in sys.modules); "
                "print('profiling.runners.norm.residual_rms_norm_vllm_cuda' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["False", "False", "False", "False"]


def test_runner_ref_resolves_to_callable():
    runner = find_kernel_profiler_spec(KIND, "torch").runner_ref.load()

    assert callable(runner)
    assert runner.__module__ == "profiling.runners.norm.residual_rms_norm_torch"
    assert runner.__name__ == "profile_residual_rms_norm"


def test_vllm_cuda_runner_ref_resolves_without_importing_frameworks():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from profiling.db.registry import find_kernel_profiler_spec; "
                "runner = find_kernel_profiler_spec("
                "'residual_rms_norm', 'vllm_cuda').runner_ref.load(); "
                "print(runner.__module__); "
                "print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('vllm' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.norm.residual_rms_norm_vllm_cuda",
        "profile_residual_rms_norm_vllm_cuda",
        "False",
        "False",
    ]


def test_runner_rejects_invalid_shapes_and_unadvertised_dtype_before_cuda():
    from profiling.runners.norm.residual_rms_norm_torch import (
        profile_residual_rms_norm,
    )

    with pytest.raises(ValueError, match="m and hidden must be > 0"):
        profile_residual_rms_norm(m=0, hidden=6144, dtype=DType.BF16)
    with pytest.raises(ValueError, match="m and hidden must be > 0"):
        profile_residual_rms_norm(m=128, hidden=0, dtype=DType.BF16)
    with pytest.raises(ValueError, match="only bf16 and fp16"):
        profile_residual_rms_norm(m=128, hidden=6144, dtype=DType.FP32)


def test_vllm_cuda_runner_rejects_invalid_args_before_importing_frameworks():
    from profiling.runners.norm.residual_rms_norm_vllm_cuda import (
        profile_residual_rms_norm_vllm_cuda,
    )

    with pytest.raises(ValueError, match="m and hidden must be > 0"):
        profile_residual_rms_norm_vllm_cuda(
            m=0,
            hidden=6144,
            dtype=DType.BF16,
        )
    with pytest.raises(ValueError, match="m and hidden must be > 0"):
        profile_residual_rms_norm_vllm_cuda(
            m=128,
            hidden=0,
            dtype=DType.BF16,
        )
    with pytest.raises(ValueError, match="only bf16 and fp16"):
        profile_residual_rms_norm_vllm_cuda(
            m=128,
            hidden=6144,
            dtype=DType.FP32,
        )


def test_vllm_cuda_runner_rejects_missing_cuda_and_unverified_gpu():
    from profiling.runners.norm.residual_rms_norm_vllm_cuda import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)

    h100 = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA H100",
        ),
    )
    with pytest.raises(
        ProfilerNotImplemented,
        match="verified only on NVIDIA H200, got NVIDIA H100",
    ):
        _validate_cuda_device(h100)


def test_generated_facades_are_available():
    assert hasattr(perf_api, "get_residual_rms_norm_times")
    assert hasattr(perf_api, "count_missing_residual_rms_norm")
