"""Registration and Torch-runner tests for DSA prefill MQA logits."""

from __future__ import annotations

import builtins
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
from profiling.kernels.dsa_mqa_logits_prefill import (
    KIND,
    DsaMqaLogitsPrefillArgs,
)
from profiling.runners.attention.dsa_mqa_logits_prefill_reference import (
    dsa_mqa_logits_prefill_reference,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

_BACKEND = "torch"
_DEEPGEMM_BACKEND = "vllm_deepgemm_fp8"
_BASE_SPEC = {
    "num_queries": 16,
    "num_keys": 4096,
    "num_sequences": 1,
    "num_heads": 64,
    "head_dim": 128,
    "q_dtype": "fp8_e4m3",
    "k_dtype": "fp8_e4m3",
    "k_scale_dtype": "fp32",
    "weight_dtype": "fp32",
    "output_dtype": "fp32",
    "span_mode": "single_causal_tail",
    "clean_logits": False,
}


def test_args_field_order_and_dtype_coercion():
    assert [field.name for field in fields(DsaMqaLogitsPrefillArgs)] == [
        "num_queries",
        "num_keys",
        "num_sequences",
        "num_heads",
        "head_dim",
        "q_dtype",
        "k_dtype",
        "k_scale_dtype",
        "weight_dtype",
        "output_dtype",
        "span_mode",
        "clean_logits",
    ]
    args = coerce_args(DsaMqaLogitsPrefillArgs, _BASE_SPEC)
    assert args == DsaMqaLogitsPrefillArgs(
        num_queries=16,
        num_keys=4096,
        num_sequences=1,
        num_heads=64,
        head_dim=128,
        q_dtype=DType.FP8_E4M3,
        k_dtype=DType.FP8_E4M3,
        k_scale_dtype=DType.FP32,
        weight_dtype=DType.FP32,
        output_dtype=DType.FP32,
        span_mode="single_causal_tail",
        clean_logits=False,
    )


def test_registration_support_and_facades():
    spec = find_kernel_profiler_spec(KIND, _BACKEND)

    assert KIND == "dsa_mqa_logits_prefill"
    assert known_backends(KIND) == [_BACKEND, _DEEPGEMM_BACKEND]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.args_schema is DsaMqaLogitsPrefillArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == ("profiling.runners.attention.dsa_mqa_logits_prefill")
    assert spec.runner_ref.function_name == "profile_dsa_mqa_logits_prefill_torch"
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
    assert hasattr(perf_api, "get_dsa_mqa_logits_prefill_times")
    assert hasattr(perf_api, "count_missing_dsa_mqa_logits_prefill")


def test_deepgemm_registration_reuses_kind_table_args_family_and_facades():
    torch_spec = find_kernel_profiler_spec(KIND, _BACKEND)
    deepgemm_spec = find_kernel_profiler_spec(KIND, _DEEPGEMM_BACKEND)

    assert deepgemm_spec.kernel_kind == torch_spec.kernel_kind == KIND
    assert deepgemm_spec.table_name == torch_spec.table_name == KIND
    assert deepgemm_spec.args_schema is torch_spec.args_schema is DsaMqaLogitsPrefillArgs
    assert deepgemm_spec.metric_family is torch_spec.metric_family is MetricFamily.COMPUTE
    assert deepgemm_spec.batch_outlier_policy == BatchOutlierPolicy()
    assert deepgemm_spec.subprocess_env == "vllm_env"
    assert deepgemm_spec.runner_ref.module_name == (
        "profiling.runners.attention.dsa_mqa_logits_prefill"
    )
    assert deepgemm_spec.runner_ref.function_name == (
        "profile_dsa_mqa_logits_prefill_vllm_deepgemm_fp8"
    )
    assert hasattr(perf_api, "get_dsa_mqa_logits_prefill_times")
    assert hasattr(perf_api, "count_missing_dsa_mqa_logits_prefill")


def test_deepgemm_support_is_fp8_e4m3_h200_only():
    support = find_kernel_profiler_spec(KIND, _DEEPGEMM_BACKEND).supports

    assert support.allows(
        DType.FP8_E4M3,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H200",
    )
    assert not support.allows(
        DType.BF16,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H200",
    )
    assert not support.allows(
        DType.FP8_E4M3,
        kv_dtype=DType.BF16,
        gpu="NVIDIA H200",
    )
    assert not support.allows(
        DType.FP8_E4M3,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H100",
    )


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
                "'profiling.runners.attention.dsa_mqa_logits_prefill' "
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
                "'dsa_mqa_logits_prefill', 'torch').runner_ref.load(); "
                "deepgemm_runner = find_kernel_profiler_spec("
                "'dsa_mqa_logits_prefill', 'vllm_deepgemm_fp8'"
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
        "profiling.runners.attention.dsa_mqa_logits_prefill",
        "profile_dsa_mqa_logits_prefill_torch",
        "profiling.runners.attention.dsa_mqa_logits_prefill",
        "profile_dsa_mqa_logits_prefill_vllm_deepgemm_fp8",
        "False",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    ("num_queries", "num_keys", "match"),
    [
        (0, 128, "must be > 0"),
        (1, 0, "must be > 0"),
        (129, 128, "must be <= num_keys"),
    ],
)
def test_rejects_invalid_query_key_dimensions_before_allocation(num_queries, num_keys, match):
    from profiling.runners.attention.dsa_mqa_logits_prefill import _validate_args

    kwargs = dict(_BASE_SPEC)
    kwargs.update(num_queries=num_queries, num_keys=num_keys)
    with pytest.raises(ValueError, match=match):
        _validate_args(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("num_sequences", 2, "num_sequences=1"),
        ("num_heads", 32, r"== \(64, 128\)"),
        ("head_dim", 64, r"== \(64, 128\)"),
        ("q_dtype", DType.BF16, "q_dtype=k_dtype=fp8_e4m3"),
        ("k_dtype", DType.FP16, "q_dtype=k_dtype=fp8_e4m3"),
        ("k_scale_dtype", DType.BF16, "k_scale_dtype=weight_dtype"),
        ("weight_dtype", DType.BF16, "k_scale_dtype=weight_dtype"),
        ("output_dtype", DType.BF16, "k_scale_dtype=weight_dtype"),
        ("span_mode", "ragged", "single_causal_tail"),
        ("clean_logits", True, "clean_logits=false"),
    ],
)
def test_rejects_unsupported_static_values_before_allocation(field, value, match):
    from profiling.runners.attention.dsa_mqa_logits_prefill import _validate_args

    kwargs = dict(_BASE_SPEC)
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        _validate_args(**kwargs)


def test_rejects_nonboolean_clean_logits():
    from profiling.runners.attention.dsa_mqa_logits_prefill import _validate_args

    kwargs = dict(_BASE_SPEC)
    kwargs["clean_logits"] = 0
    with pytest.raises(TypeError, match="clean_logits must be a bool"):
        _validate_args(**kwargs)


def test_rejects_missing_cuda_and_unverified_gpu():
    from profiling.runners.attention.dsa_mqa_logits_prefill import (
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


def test_deepgemm_rejects_missing_cuda_and_unverified_gpu():
    from profiling.runners.attention.dsa_mqa_logits_prefill import (
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
    with pytest.raises(ProfilerNotImplemented, match="verified only on NVIDIA H200"):
        _validate_deepgemm_cuda_device(h100)


def test_deepgemm_entry_rejects_invalid_args_before_framework_loading(monkeypatch):
    from profiling.runners.attention import dsa_mqa_logits_prefill as runner

    loaded = False

    def fail_if_loaded():
        nonlocal loaded
        loaded = True
        raise AssertionError("framework loader should not run")

    monkeypatch.setattr(runner, "_load_deepgemm_backend", fail_if_loaded)
    invalid_cases = [
        ({"num_queries": 0}, "must be > 0"),
        ({"num_keys": 0}, "must be > 0"),
        ({"num_queries": 129, "num_keys": 128}, "must be <= num_keys"),
        ({"num_sequences": 2}, "num_sequences=1"),
        ({"num_heads": 32}, r"== \(64, 128\)"),
        ({"head_dim": 64}, r"== \(64, 128\)"),
        ({"q_dtype": DType.BF16}, "q_dtype=k_dtype=fp8_e4m3"),
        ({"k_dtype": DType.FP16}, "q_dtype=k_dtype=fp8_e4m3"),
        ({"k_scale_dtype": DType.BF16}, "k_scale_dtype=weight_dtype"),
        ({"weight_dtype": DType.BF16}, "k_scale_dtype=weight_dtype"),
        ({"output_dtype": DType.BF16}, "k_scale_dtype=weight_dtype"),
        ({"span_mode": "ragged"}, "single_causal_tail"),
        ({"clean_logits": True}, "clean_logits=false"),
    ]
    for overrides, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            runner.profile_dsa_mqa_logits_prefill_vllm_deepgemm_fp8(**(_BASE_SPEC | overrides))
    assert not loaded


def test_deepgemm_loader_rejects_missing_support(monkeypatch):
    from profiling.runners.attention import dsa_mqa_logits_prefill as runner

    fake_deepgemm = SimpleNamespace(
        is_deep_gemm_supported=lambda: False,
        fp8_mqa_logits=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils",
        SimpleNamespace(deep_gemm=fake_deepgemm),
    )
    with pytest.raises(ProfilerNotImplemented, match="unavailable or unsupported"):
        runner._load_deepgemm_backend()


def test_deepgemm_loader_rejects_missing_framework(monkeypatch):
    from profiling.runners.attention import dsa_mqa_logits_prefill as runner

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


def test_deepgemm_launch_failure_is_typed(monkeypatch):
    from profiling.runners.attention import dsa_mqa_logits_prefill as runner

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA H200",
        )
    )
    fake_deepgemm = SimpleNamespace(
        fp8_mqa_logits=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic launch failure")
        )
    )
    fake_operands = SimpleNamespace(
        q=object(),
        k=object(),
        k_scale=object(),
        weights=object(),
        k_start=object(),
        k_end=object(),
    )
    monkeypatch.setattr(
        runner,
        "_load_deepgemm_backend",
        lambda: (fake_torch, fake_deepgemm),
    )
    monkeypatch.setattr(runner, "_build_operands", lambda *args, **kwargs: fake_operands)
    with pytest.raises(KernelLaunchFailed, match="synthetic launch failure"):
        runner.profile_dsa_mqa_logits_prefill_vllm_deepgemm_fp8(**_BASE_SPEC)


@pytest.mark.parametrize("num_queries", [1, 2, 3, 16])
def test_operand_construction_exact_layout_and_spans(num_queries):
    from profiling.runners.attention.dsa_mqa_logits_prefill import _build_operands

    num_keys = 128 if num_queries <= 3 else 4096
    operands = _build_operands(
        torch,
        num_queries=num_queries,
        num_keys=num_keys,
        num_heads=64,
        head_dim=128,
        device="cpu",
    )

    assert operands.q.shape == (num_queries, 64, 128)
    assert operands.q.stride() == (8192, 128, 1)
    assert operands.q.dtype is torch.float8_e4m3fn
    assert operands.q.is_contiguous()
    assert operands.k.shape == (num_keys, 128)
    assert operands.k.stride() == (128, 1)
    assert operands.k.dtype is torch.float8_e4m3fn
    assert operands.k.is_contiguous()
    assert operands.k_scale.shape == (num_keys,)
    assert operands.k_scale.dtype is torch.float32
    assert bool(torch.all(operands.k_scale > 0))
    assert operands.weights.shape == (num_queries, 64)
    assert operands.weights.dtype is torch.float32
    assert operands.k_start.dtype is operands.k_end.dtype is torch.int32
    assert torch.equal(operands.k_start, torch.zeros_like(operands.k_start))
    assert torch.equal(
        operands.k_end,
        torch.arange(
            num_keys - num_queries + 1,
            num_keys + 1,
            dtype=torch.int32,
        ),
    )
    assert operands.valid_mask.shape == (num_queries, num_keys)
    assert operands.valid_mask.dtype is torch.bool


@pytest.mark.parametrize(
    ("num_queries", "num_keys", "expected_sum_w", "expected_cells", "expected_flops"),
    [
        (1, 128, 256, 512, 8_388_608),
        (3, 512, 1_024, 2_048, 33_554_432),
        (16, 4096, 32_768, 65_536, 1_073_741_824),
        (128, 4096, 262_144, 524_288, 8_589_934_592),
    ],
)
def test_scheduled_work(num_queries, num_keys, expected_sum_w, expected_cells, expected_flops):
    from profiling.runners.attention.dsa_mqa_logits_prefill import _scheduled_work

    assert _scheduled_work(
        num_queries=num_queries,
        num_keys=num_keys,
        num_heads=64,
        head_dim=128,
    ) == (expected_sum_w, expected_cells, expected_flops)


@pytest.mark.parametrize(
    ("num_queries", "num_keys", "expected_bytes"),
    [
        (1, 128, 44_296),
        (16, 4096, 4_722_816),
        (128, 4096, 37_782_528),
    ],
)
def test_logical_scheduled_bytes(num_queries, num_keys, expected_bytes):
    from profiling.runners.attention.dsa_mqa_logits_prefill import (
        _logical_scheduled_bytes,
    )

    assert (
        _logical_scheduled_bytes(
            num_queries=num_queries,
            num_keys=num_keys,
            num_heads=64,
            head_dim=128,
        )
        == expected_bytes
    )


def test_scheduled_work_rejects_invalid_dimensions():
    from profiling.runners.attention.dsa_mqa_logits_prefill import _scheduled_work

    with pytest.raises(ValueError, match="must be > 0"):
        _scheduled_work(num_queries=0, num_keys=1, num_heads=64, head_dim=128)
    with pytest.raises(ValueError, match="must be <= num_keys"):
        _scheduled_work(num_queries=2, num_keys=1, num_heads=64, head_dim=128)


@pytest.mark.parametrize(("num_queries", "num_keys"), [(1, 8), (3, 8), (4, 11)])
def test_cpu_composite_matches_reference_and_preserves_inputs(num_queries, num_keys):
    from profiling.runners.attention.dsa_mqa_logits_prefill import (
        _build_operands,
        _torch_composite,
    )

    operands = _build_operands(
        torch,
        num_queries=num_queries,
        num_keys=num_keys,
        num_heads=64,
        head_dim=128,
        device="cpu",
    )
    snapshots = [
        tensor.clone()
        for tensor in (
            operands.q,
            operands.k,
            operands.k_scale,
            operands.weights,
            operands.k_start,
            operands.k_end,
            operands.valid_mask,
        )
    ]

    actual = _torch_composite(operands)
    expected = dsa_mqa_logits_prefill_reference(
        operands.q,
        operands.k,
        operands.k_scale,
        operands.weights,
        operands.k_start,
        operands.k_end,
        clean_logits=False,
    )

    assert actual.shape == (num_queries, num_keys)
    assert actual.dtype is torch.float32
    assert torch.allclose(
        actual[operands.valid_mask],
        expected[operands.valid_mask],
        rtol=1e-5,
        atol=1e-5,
    )
    assert bool(torch.isnan(actual[~operands.valid_mask]).all())
    for tensor, snapshot in zip(
        (
            operands.q,
            operands.k,
            operands.k_scale,
            operands.weights,
            operands.k_start,
            operands.k_end,
            operands.valid_mask,
        ),
        snapshots,
        strict=True,
    ):
        assert torch.equal(tensor, snapshot)
