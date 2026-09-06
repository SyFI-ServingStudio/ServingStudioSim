"""Registration and Torch-runner tests for DSA paged-decode MQA logits."""

from __future__ import annotations

import builtins
import inspect
import subprocess
import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.kernels.dsa_paged_mqa_logits_decode import (
    KIND,
    DsaPagedMqaLogitsDecodeArgs,
)
from profiling.runners.attention.dsa_paged_mqa_logits_decode_reference import (
    dsa_paged_mqa_logits_decode_reference,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

_BACKEND = "torch"
_DEEPGEMM_BACKEND = "deepgemm_fp8"
_BASE_SPEC = {
    "batch_size": 2,
    "context_len": 65,
    "next_n": 1,
    "max_model_len": 128,
    "num_heads": 64,
    "head_dim": 128,
    "block_size": 64,
    "q_dtype": "fp8_e4m3",
    "cache_dtype": "fp8_e4m3",
    "scale_dtype": "fp32",
    "weight_dtype": "fp32",
    "output_dtype": "fp32",
    "context_mode": "uniform",
    "page_mapping": "unique_scattered",
    "cache_format": "page_planar_fp8_fp32_scale",
    "clean_logits": False,
}


def test_args_field_order_and_dtype_coercion():
    assert [field.name for field in fields(DsaPagedMqaLogitsDecodeArgs)] == [
        "batch_size",
        "context_len",
        "next_n",
        "max_model_len",
        "num_heads",
        "head_dim",
        "block_size",
        "q_dtype",
        "cache_dtype",
        "scale_dtype",
        "weight_dtype",
        "output_dtype",
        "context_mode",
        "page_mapping",
        "cache_format",
        "clean_logits",
    ]
    args = coerce_args(DsaPagedMqaLogitsDecodeArgs, _BASE_SPEC)
    assert args == DsaPagedMqaLogitsDecodeArgs(
        batch_size=2,
        context_len=65,
        next_n=1,
        max_model_len=128,
        num_heads=64,
        head_dim=128,
        block_size=64,
        q_dtype=DType.FP8_E4M3,
        cache_dtype=DType.FP8_E4M3,
        scale_dtype=DType.FP32,
        weight_dtype=DType.FP32,
        output_dtype=DType.FP32,
        context_mode="uniform",
        page_mapping="unique_scattered",
        cache_format="page_planar_fp8_fp32_scale",
        clean_logits=False,
    )


def test_registration_support_and_facades():
    spec = find_kernel_profiler_spec(KIND, _BACKEND)

    assert KIND == "dsa_paged_mqa_logits_decode"
    assert known_backends(KIND) == [_BACKEND, _DEEPGEMM_BACKEND]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.args_schema is DsaPagedMqaLogitsDecodeArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.dsa_paged_mqa_logits_decode"
    )
    assert spec.runner_ref.function_name == "profile_dsa_paged_mqa_logits_decode_torch"
    assert spec.supports.allows(
        DType.FP8_E4M3,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H200",
    )
    assert not spec.supports.allows(
        DType.BF16,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H200",
    )
    assert not spec.supports.allows(
        DType.FP8_E4M3,
        kv_dtype=DType.BF16,
        gpu="NVIDIA H200",
    )
    assert not spec.supports.allows(
        DType.FP8_E4M3,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H100",
    )
    assert not spec.supports.allows(
        DType.FP8_E4M3,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA B200",
    )
    assert hasattr(perf_api, "get_dsa_paged_mqa_logits_decode_times")
    assert hasattr(perf_api, "count_missing_dsa_paged_mqa_logits_decode")


def test_deepgemm_registration_reuses_kind_table_args_family_and_facades():
    torch_spec = find_kernel_profiler_spec(KIND, _BACKEND)
    deepgemm_spec = find_kernel_profiler_spec(KIND, _DEEPGEMM_BACKEND)

    assert deepgemm_spec.kernel_kind == torch_spec.kernel_kind == KIND
    assert deepgemm_spec.table_name == torch_spec.table_name == KIND
    assert deepgemm_spec.args_schema is torch_spec.args_schema is DsaPagedMqaLogitsDecodeArgs
    assert deepgemm_spec.metric_family is torch_spec.metric_family is MetricFamily.COMPUTE
    assert deepgemm_spec.batch_outlier_policy == BatchOutlierPolicy()
    assert deepgemm_spec.subprocess_env == "vllm_env"
    assert deepgemm_spec.runner_ref.module_name == (
        "profiling.runners.attention.dsa_paged_mqa_logits_decode"
    )
    assert deepgemm_spec.runner_ref.function_name == (
        "profile_dsa_paged_mqa_logits_decode_deepgemm_fp8"
    )
    assert deepgemm_spec.supports.allows(
        DType.FP8_E4M3,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H200",
    )
    assert deepgemm_spec.supports.allows(
        DType.FP8_E4M3,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA B200",
    )
    assert hasattr(perf_api, "get_dsa_paged_mqa_logits_decode_times")
    assert hasattr(perf_api, "count_missing_dsa_paged_mqa_logits_decode")


def test_neutral_deepgemm_entry_routes_through_the_pinned_loader(monkeypatch):
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    seen = {}

    def common(load_backend, **kwargs):
        seen["loader"] = load_backend
        seen["kwargs"] = kwargs
        return "metrics"

    monkeypatch.setattr(runner, "_profile_dsa_paged_mqa_logits_decode_deepgemm_fp8", common)
    assert runner.profile_dsa_paged_mqa_logits_decode_deepgemm_fp8(**_BASE_SPEC) == "metrics"
    assert seen == {"loader": runner._load_deepgemm_backend, "kwargs": _BASE_SPEC}


def test_registry_barrel_import_is_lazy():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('vllm' in sys.modules); "
                "print('deep_gemm' in sys.modules); "
                "print("
                "'profiling.runners.attention.dsa_paged_mqa_logits_decode' "
                "in sys.modules"
                ")"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["False", "False", "False", "False"]


def test_runner_refs_resolve_without_importing_frameworks_or_running_jit():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from profiling.db.registry import find_kernel_profiler_spec; "
                "runner = find_kernel_profiler_spec("
                "'dsa_paged_mqa_logits_decode', 'torch').runner_ref.load(); "
                "deepgemm_runner = find_kernel_profiler_spec("
                "'dsa_paged_mqa_logits_decode', 'deepgemm_fp8'"
                ").runner_ref.load(); "
                "print(runner.__module__); "
                "print(runner.__name__); "
                "print(deepgemm_runner.__module__); "
                "print(deepgemm_runner.__name__); "
                "print('torch' in sys.modules); "
                "print('vllm' in sys.modules); "
                "print('deep_gemm' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.dsa_paged_mqa_logits_decode",
        "profile_dsa_paged_mqa_logits_decode_torch",
        "profiling.runners.attention.dsa_paged_mqa_logits_decode",
        "profile_dsa_paged_mqa_logits_decode_deepgemm_fp8",
        "False",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"batch_size": 0}, "must be > 0"),
        ({"context_len": 0}, "must be > 0"),
        ({"max_model_len": 0}, "must be > 0"),
        ({"context_len": 129}, "must be <= max_model_len"),
        ({"next_n": 0}, "next_n > 0"),
        ({"num_heads": 16}, r"num_heads in \[32, 64\]"),
        ({"head_dim": 64}, r"head_dim == 128"),
        ({"block_size": 32}, r"block_size == 64"),
        ({"q_dtype": DType.BF16}, "q_dtype=cache_dtype=fp8_e4m3"),
        ({"cache_dtype": DType.BF16}, "q_dtype=cache_dtype=fp8_e4m3"),
        ({"scale_dtype": DType.BF16}, "scale_dtype=weight_dtype"),
        ({"weight_dtype": DType.BF16}, "scale_dtype=weight_dtype"),
        ({"output_dtype": DType.BF16}, "scale_dtype=weight_dtype"),
        ({"context_mode": "mixed"}, "context_mode='uniform'"),
        ({"page_mapping": "shared"}, "page_mapping='unique_scattered'"),
        ({"cache_format": "adjacent"}, "cache_format='page_planar"),
        ({"clean_logits": True}, "clean_logits=false"),
    ],
)
def test_rejects_unsupported_args_before_allocation(overrides, match):
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import _validate_args

    with pytest.raises(ValueError, match=match):
        _validate_args(**(_BASE_SPEC | overrides))


def test_accepts_sglang_native_32_head_instantiation():
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import _validate_args

    validated = _validate_args(**(_BASE_SPEC | {"num_heads": 32}))

    assert validated[4] == 32


def test_rejects_nonboolean_clean_logits():
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import _validate_args

    with pytest.raises(TypeError, match="clean_logits must be a bool"):
        _validate_args(**(_BASE_SPEC | {"clean_logits": 0}))


def test_profile_rejects_invalid_args_before_importing_torch(monkeypatch):
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    real_import = builtins.__import__
    imported_torch = False

    def track_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal imported_torch
        if name == "torch":
            imported_torch = True
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", track_import)
    with pytest.raises(ValueError, match="batch_size"):
        runner.profile_dsa_paged_mqa_logits_decode_torch(**(_BASE_SPEC | {"batch_size": 0}))
    assert not imported_torch


def test_missing_torch_is_typed(monkeypatch):
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    real_import = builtins.__import__

    def missing_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("synthetic missing torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_torch)
    with pytest.raises(ProfilerNotImplemented, match="torch is required"):
        runner.profile_dsa_paged_mqa_logits_decode_torch(**_BASE_SPEC)


def test_rejects_missing_cuda_and_unverified_gpu():
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)

    h100 = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA H100",
        )
    )
    with pytest.raises(ProfilerNotImplemented, match="verified only on NVIDIA H200"):
        _validate_cuda_device(h100)


def test_deepgemm_rejects_cuda_gpu_and_sm_count_mismatches():
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import (
        _validate_deepgemm_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_deepgemm_cuda_device(no_cuda)

    h100 = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA H100",
        )
    )
    with pytest.raises(
        ProfilerNotImplemented,
        match="verified only on NVIDIA H200 or NVIDIA B200",
    ):
        _validate_deepgemm_cuda_device(h100)

    wrong_sm_count = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA H200",
            get_device_properties=lambda _device: SimpleNamespace(multi_processor_count=130),
        )
    )
    with pytest.raises(ProfilerNotImplemented, match="132-SM NVIDIA H200"):
        _validate_deepgemm_cuda_device(wrong_sm_count)

    b200 = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA B200",
            get_device_properties=lambda _device: SimpleNamespace(multi_processor_count=148),
        )
    )
    assert _validate_deepgemm_cuda_device(b200) == 148


def test_deepgemm_cupti_filter_is_architecture_agnostic():
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import (
        _DEEPGEMM_KERNEL_NAME,
    )

    assert _DEEPGEMM_KERNEL_NAME == "fp8_paged_mqa_logits"


def test_deepgemm_entry_rejects_invalid_args_before_framework_loading(monkeypatch):
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    loaded = False

    def fail_if_loaded():
        nonlocal loaded
        loaded = True
        raise AssertionError("framework loader should not run")

    monkeypatch.setattr(runner, "_load_deepgemm_backend", fail_if_loaded)
    invalid_cases = [
        ({"batch_size": 0}, "must be > 0"),
        ({"context_len": 0}, "must be > 0"),
        ({"max_model_len": 0}, "must be > 0"),
        ({"context_len": 129}, "must be <= max_model_len"),
        ({"next_n": -1}, "next_n > 0"),
        ({"num_heads": 16}, r"num_heads in \[32, 64\]"),
        ({"head_dim": 64}, r"head_dim == 128"),
        ({"block_size": 32}, r"block_size == 64"),
        ({"q_dtype": DType.BF16}, "q_dtype=cache_dtype=fp8_e4m3"),
        ({"cache_dtype": DType.BF16}, "q_dtype=cache_dtype=fp8_e4m3"),
        ({"scale_dtype": DType.BF16}, "scale_dtype=weight_dtype"),
        ({"weight_dtype": DType.BF16}, "scale_dtype=weight_dtype"),
        ({"output_dtype": DType.BF16}, "scale_dtype=weight_dtype"),
        ({"context_mode": "mixed"}, "context_mode='uniform'"),
        ({"page_mapping": "shared"}, "page_mapping='unique_scattered'"),
        ({"cache_format": "adjacent"}, "cache_format='page_planar"),
        ({"clean_logits": True}, "clean_logits=false"),
    ]
    for overrides, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            runner.profile_dsa_paged_mqa_logits_decode_deepgemm_fp8(**(_BASE_SPEC | overrides))
    assert not loaded


def test_deepgemm_loader_rejects_missing_framework_support_and_apis(monkeypatch):
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    real_import = builtins.__import__

    def missing_vllm(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.utils":
            raise ImportError("synthetic missing vLLM")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_vllm)
    with pytest.raises(
        ProfilerNotImplemented,
        match="instrumented vLLM/DeepGEMM environment is required",
    ):
        runner._load_deepgemm_backend()

    monkeypatch.setattr(builtins, "__import__", real_import)
    fake_deepgemm = SimpleNamespace(
        is_deep_gemm_supported=lambda: False,
        get_paged_mqa_logits_metadata=lambda *args, **kwargs: None,
        fp8_paged_mqa_logits=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils",
        SimpleNamespace(deep_gemm=fake_deepgemm),
    )
    with pytest.raises(ProfilerNotImplemented, match="unavailable or unsupported"):
        runner._load_deepgemm_backend()


@pytest.mark.parametrize(
    "missing_api",
    [
        "is_deep_gemm_supported",
        "get_paged_mqa_logits_metadata",
    ],
)
def test_deepgemm_loader_rejects_each_missing_api(monkeypatch, missing_api):
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    fake_deepgemm = SimpleNamespace(
        is_deep_gemm_supported=lambda: True,
        get_paged_mqa_logits_metadata=lambda *args, **kwargs: None,
        fp8_paged_mqa_logits=lambda *args, **kwargs: None,
    )
    setattr(fake_deepgemm, missing_api, None)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils",
        SimpleNamespace(deep_gemm=fake_deepgemm),
    )
    with pytest.raises(ProfilerNotImplemented, match=f"{missing_api} is unavailable"):
        runner._load_deepgemm_backend()


@pytest.mark.parametrize(
    "paged_api",
    ["fp8_paged_mqa_logits", "fp8_fp4_paged_mqa_logits"],
)
def test_deepgemm_loader_accepts_either_paged_entry_point(monkeypatch, paged_api):
    """The fork renamed the paged entry point when it unified FP8 and MXFP4
    dispatch. Either name must load; only their joint absence is an error."""
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    fake_deepgemm = SimpleNamespace(
        is_deep_gemm_supported=lambda: True,
        get_paged_mqa_logits_metadata=lambda *args, **kwargs: None,
    )
    setattr(fake_deepgemm, paged_api, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils",
        SimpleNamespace(deep_gemm=fake_deepgemm),
    )
    _, loaded = runner._load_deepgemm_backend()
    assert runner._paged_mqa_logits_entry_point(loaded) is getattr(fake_deepgemm, paged_api)


def test_deepgemm_loader_rejects_both_paged_entry_points_missing(monkeypatch):
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    fake_deepgemm = SimpleNamespace(
        is_deep_gemm_supported=lambda: True,
        get_paged_mqa_logits_metadata=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils",
        SimpleNamespace(deep_gemm=fake_deepgemm),
    )
    with pytest.raises(ProfilerNotImplemented, match="neither fp8_paged_mqa_logits"):
        runner._load_deepgemm_backend()


@pytest.mark.parametrize("next_n", [1, 6])
def test_deepgemm_adapter_and_callable_forwarding(next_n):
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import (
        _build_operands,
        _prepare_deepgemm_call,
    )

    operands = _build_operands(
        torch,
        batch_size=2,
        context_len=65,
        next_n=next_n,
        max_model_len=128,
        num_heads=64,
        head_dim=128,
        block_size=64,
        device="cpu",
    )
    original_contexts = operands.context_lens.clone()
    calls = []
    metadata = object()

    class FakeDeepGemm:
        @staticmethod
        def get_paged_mqa_logits_metadata(context_lens, block_size, num_sms):
            calls.append(("metadata", context_lens, block_size, num_sms))
            return metadata

        @staticmethod
        def fp8_paged_mqa_logits(*args, **kwargs):
            calls.append(("kernel", args, kwargs))
            return "output"

    runnable_context_lens, actual_metadata, kernel = _prepare_deepgemm_call(
        FakeDeepGemm,
        operands,
        block_size=64,
        num_sms=132,
        max_model_len=128,
    )
    assert len(calls) == 1
    assert calls[0][0] == "metadata"
    assert runnable_context_lens.shape == (2,)
    assert runnable_context_lens.dtype is torch.int32
    assert runnable_context_lens.is_contiguous()
    assert torch.equal(runnable_context_lens, operands.context_lens[:, 0])
    assert actual_metadata is metadata
    assert operands.q.shape == (2, next_n, 64, 128)
    assert torch.equal(operands.context_lens, original_contexts)

    assert kernel() == "output"
    assert len(calls) == 2
    args, kwargs = calls[1][1:]
    assert args == (
        operands.q,
        operands.cache,
        operands.weights,
        runnable_context_lens,
        operands.block_table,
        metadata,
        128,
    )
    assert kwargs == {"clean_logits": False}


def test_spec5_public_runner_reaches_backend_loading(monkeypatch):
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    def backend_unavailable():
        raise ProfilerNotImplemented("test backend unavailable")

    monkeypatch.setattr(runner, "_load_deepgemm_backend", backend_unavailable)
    with pytest.raises(ProfilerNotImplemented, match="test backend unavailable"):
        runner.profile_dsa_paged_mqa_logits_decode_deepgemm_fp8(**(_BASE_SPEC | {"next_n": 6}))


def test_deepgemm_launch_failure_is_typed(monkeypatch):
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA H200",
            get_device_properties=lambda _device: SimpleNamespace(multi_processor_count=132),
        )
    )
    fake_deepgemm = SimpleNamespace()
    fake_operands = SimpleNamespace()
    monkeypatch.setattr(
        runner,
        "_load_deepgemm_backend",
        lambda: (fake_torch, fake_deepgemm),
    )
    monkeypatch.setattr(runner, "_build_operands", lambda *args, **kwargs: fake_operands)
    monkeypatch.setattr(
        runner,
        "_prepare_deepgemm_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic launch failure")),
    )
    with pytest.raises(KernelLaunchFailed, match="synthetic launch failure"):
        runner.profile_dsa_paged_mqa_logits_decode_deepgemm_fp8(**_BASE_SPEC)


def test_launch_failure_is_typed(monkeypatch):
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA H200",
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        runner,
        "_build_operands",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic launch failure")),
    )
    with pytest.raises(KernelLaunchFailed, match="synthetic launch failure"):
        runner.profile_dsa_paged_mqa_logits_decode_torch(**_BASE_SPEC)


@pytest.mark.parametrize("context_len", [1, 64, 65, 128])
def test_operand_construction_exact_layout_and_scattered_pages(context_len):
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import (
        _build_operands,
    )

    batch_size = 2
    operands = _build_operands(
        torch,
        batch_size=batch_size,
        context_len=context_len,
        next_n=1,
        max_model_len=128,
        num_heads=64,
        head_dim=128,
        block_size=64,
        device="cpu",
    )
    logical_pages = (context_len + 63) // 64

    assert operands.q.shape == (batch_size, 1, 64, 128)
    assert operands.q.stride() == (8192, 8192, 128, 1)
    assert operands.q.dtype is torch.float8_e4m3fn
    assert operands.q.is_contiguous()
    assert operands.weights.shape == (batch_size, 64)
    assert operands.weights.stride() == (64, 1)
    assert operands.weights.dtype is torch.float32
    assert operands.context_lens.shape == (batch_size, 1)
    assert operands.context_lens.dtype is torch.int32
    assert operands.context_lens.is_contiguous()
    assert operands.block_table.shape == (batch_size, logical_pages)
    assert operands.block_table.dtype is torch.int32
    assert operands.block_table.is_contiguous()
    assert operands.logical_pages == logical_pages
    assert operands.padded_context == logical_pages * 64

    page_ids = operands.block_table.flatten()
    assert page_ids.unique().numel() == page_ids.numel()
    assert bool(torch.all(page_ids >= 0))
    assert bool(torch.all(page_ids < operands.cache.shape[0]))
    assert torch.equal(
        page_ids,
        2 * torch.arange(page_ids.numel(), dtype=torch.int32) + 1,
    )

    assert operands.cache.shape == (operands.cache.shape[0], 64, 1, 132)
    assert operands.cache.stride() == (8448, 132, 132, 1)
    assert operands.cache.dtype is torch.uint8
    assert operands.cache.is_contiguous()
    assert operands.key_view.shape == (operands.cache.shape[0], 64, 128)
    assert operands.key_view.stride() == (8448, 128, 1)
    assert operands.key_view.dtype is torch.float8_e4m3fn
    assert operands.scale_view.shape == (operands.cache.shape[0], 64)
    assert operands.scale_view.stride() == (2112, 1)
    assert operands.scale_view.dtype is torch.float32
    assert operands.key_view.untyped_storage().data_ptr() == (
        operands.cache.untyped_storage().data_ptr()
    )
    assert operands.scale_view.untyped_storage().data_ptr() == (
        operands.cache.untyped_storage().data_ptr()
    )
    assert operands.key_view.storage_offset() == 0
    assert operands.scale_view.storage_offset() == 2048
    assert bool(torch.isfinite(operands.scale_view).all())
    assert bool(torch.all(operands.scale_view > 0))


def test_cache_initialization_uses_only_page_independent_templates(monkeypatch):
    from profiling.runners.attention import dsa_paged_mqa_logits_decode as runner

    stable_value_shapes = []
    real_stable_values = runner._stable_values

    def record_stable_values(torch_module, shape, *, phase, device):
        stable_value_shapes.append(shape)
        return real_stable_values(
            torch_module,
            shape,
            phase=phase,
            device=device,
        )

    monkeypatch.setattr(runner, "_stable_values", record_stable_values)
    operands = runner._build_operands(
        torch,
        batch_size=3,
        context_len=65,
        next_n=1,
        max_model_len=128,
        num_heads=64,
        head_dim=128,
        block_size=64,
        device="cpu",
    )

    num_pages = operands.cache.shape[0]
    assert num_pages == 256
    assert stable_value_shapes == [
        (3, 1, 64, 128),
        (3, 64),
        (64, 128),
        (64,),
    ]
    assert all(num_pages not in shape for shape in stable_value_shapes)

    for page in [1, 17, num_pages - 1]:
        assert torch.equal(operands.key_view[page], operands.key_view[0])
        assert torch.equal(operands.scale_view[page], operands.scale_view[0])
    assert bool(torch.any(operands.key_view[0].float() < 0))
    assert bool(torch.any(operands.key_view[0].float() > 0))
    assert bool(torch.isfinite(operands.scale_view).all())
    assert bool(torch.all(operands.scale_view > 0))
    assert operands.scale_view[0].unique().numel() > 1


def test_page_plane_raw_byte_offsets():
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import (
        _build_operands,
    )

    operands = _build_operands(
        torch,
        batch_size=1,
        context_len=1,
        next_n=1,
        max_model_len=1,
        num_heads=64,
        head_dim=128,
        block_size=64,
        device="cpu",
    )
    page = operands.cache[1].reshape(-1)
    assert torch.equal(
        page[:128],
        operands.key_view[1, 0].view(torch.uint8),
    )
    assert torch.equal(
        page[8192:8196],
        operands.scale_view[1, 0].reshape(1).view(torch.uint8),
    )
    assert 8192 == 64 * 128
    assert 8448 == 64 * (128 + 4)


@pytest.mark.parametrize(
    ("context_len", "expected_pages", "expected_cells"),
    [
        (1, 1, 64),
        (64, 1, 64),
        (65, 2, 128),
        (128, 2, 128),
    ],
)
def test_scheduled_work_block_rounding(context_len, expected_pages, expected_cells):
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import _scheduled_work

    assert _scheduled_work(
        batch_size=1,
        context_len=context_len,
        next_n=1,
        num_heads=64,
        head_dim=128,
        block_size=64,
    ) == (expected_pages, expected_cells, 2 * expected_cells * 64 * 128)


@pytest.mark.parametrize(
    ("batch_size", "context_len", "expected_flops", "expected_bytes"),
    [
        (1, 128, 2_097_152, 25_868),
        (16, 4096, 1_073_741_824, 9_052_224),
        (128, 128, 268_435_456, 3_311_104),
    ],
)
def test_representative_scheduled_accounting(
    batch_size,
    context_len,
    expected_flops,
    expected_bytes,
):
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import (
        _logical_scheduled_bytes,
        _scheduled_work,
    )

    assert (
        _scheduled_work(
            batch_size=batch_size,
            context_len=context_len,
            next_n=1,
            num_heads=64,
            head_dim=128,
            block_size=64,
        )[2]
        == expected_flops
    )
    assert (
        _logical_scheduled_bytes(
            batch_size=batch_size,
            context_len=context_len,
            next_n=1,
            num_heads=64,
            head_dim=128,
            block_size=64,
        )
        == expected_bytes
    )


def test_scheduled_accounting_ignores_max_model_len():
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import (
        _logical_scheduled_bytes,
        _scheduled_work,
    )

    assert "max_model_len" not in inspect.signature(_scheduled_work).parameters
    assert "max_model_len" not in inspect.signature(_logical_scheduled_bytes).parameters


def test_scheduled_work_rejects_nonpositive_dimensions():
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import _scheduled_work

    with pytest.raises(ValueError, match="must be > 0"):
        _scheduled_work(
            batch_size=0,
            context_len=1,
            next_n=1,
            num_heads=64,
            head_dim=128,
            block_size=64,
        )


def test_cpu_composite_matches_reference_and_preserves_every_operand():
    from profiling.runners.attention.dsa_paged_mqa_logits_decode import (
        _build_operands,
        _torch_composite,
    )

    operands = _build_operands(
        torch,
        batch_size=2,
        context_len=65,
        next_n=1,
        max_model_len=80,
        num_heads=64,
        head_dim=128,
        block_size=64,
        device="cpu",
    )
    tensors = (
        operands.q,
        operands.cache,
        operands.weights,
        operands.context_lens,
        operands.block_table,
    )
    snapshots = [tensor.clone() for tensor in tensors]

    actual = _torch_composite(operands, context_len=65, max_model_len=80)
    expected = dsa_paged_mqa_logits_decode_reference(
        operands.q,
        operands.cache,
        operands.weights,
        operands.context_lens,
        operands.block_table,
        block_size=64,
        max_model_len=80,
        clean_logits=False,
    )

    assert actual.shape == (2, 80)
    assert actual.stride() == (80, 1)
    assert actual.dtype is torch.float32
    assert torch.allclose(actual[:, :65], expected[:, :65], rtol=1e-5, atol=1e-5)
    assert bool(torch.isnan(actual[:, 65:]).all())
    assert all(
        actual.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
        for tensor in tensors
    )
    for tensor, snapshot in zip(tensors, snapshots, strict=True):
        assert torch.equal(tensor, snapshot)
