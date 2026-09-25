"""Registration and Torch-runner tests for DSA prefill top-k selection."""

from __future__ import annotations

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
from profiling.kernels.dsa_topk_prefill import KIND, DsaTopkPrefillArgs
from profiling.runners.attention.dsa_topk_prefill_reference import (
    dsa_topk_prefill_reference,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

_BACKEND = "torch"
_VLLM_BACKEND = "vllm_cuda"
_SGLANG_BACKEND = "sglang_cuda"
_BASE_SPEC = {
    "num_queries": 128,
    "num_keys": 8192,
    "num_sequences": 1,
    "top_k": 2048,
    "logits_row_stride": 8448,
    "logits_dtype": "fp32",
    "index_dtype": "int32",
    "span_mode": "single_causal_tail",
}


def test_sglang_public_wrapper_receives_the_serving_arguments() -> None:
    from profiling.runners.attention.dsa_topk_prefill import _launch_sglang_topk

    recorded = {}

    def callable_(**kwargs):
        recorded.update(kwargs)
        return "output"

    operands = SimpleNamespace(logits="logits", row_starts="row_starts")
    extras = SimpleNamespace(
        lengths="lengths",
        src_page_table="page_table",
        cu_seqlens_q="cu_seqlens_q",
    )

    assert _launch_sglang_topk(callable_, operands, extras, top_k=2048) == "output"
    assert recorded == {
        "score": "logits",
        "lengths": "lengths",
        "page_table_size_1": "page_table",
        "cu_seqlens_q": "cu_seqlens_q",
        "topk": 2048,
        "row_starts": "row_starts",
    }


def test_args_field_order_and_dtype_coercion() -> None:
    assert [field.name for field in fields(DsaTopkPrefillArgs)] == [
        "num_queries",
        "num_keys",
        "num_sequences",
        "top_k",
        "logits_row_stride",
        "logits_dtype",
        "index_dtype",
        "span_mode",
    ]
    args = coerce_args(DsaTopkPrefillArgs, _BASE_SPEC)
    assert args == DsaTopkPrefillArgs(
        num_queries=128,
        num_keys=8192,
        num_sequences=1,
        top_k=2048,
        logits_row_stride=8448,
        logits_dtype=DType.FP32,
        index_dtype="int32",
        span_mode="single_causal_tail",
    )


def test_registration_support_and_facades() -> None:
    spec = find_kernel_profiler_spec(KIND, _BACKEND)

    assert KIND == "dsa_topk_prefill"
    assert known_backends(KIND) == [_BACKEND, _VLLM_BACKEND, _SGLANG_BACKEND, "vllm_fork_cuda"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.args_schema is DsaTopkPrefillArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.supports.kv is None
    assert spec.supports.allows(DType.FP32, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA H100")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA B200")
    assert spec.runner_ref.module_name == ("profiling.runners.attention.dsa_topk_prefill")
    assert spec.runner_ref.function_name == "profile_dsa_topk_prefill_torch"
    assert hasattr(perf_api, "get_dsa_topk_prefill_times")
    assert hasattr(perf_api, "count_missing_dsa_topk_prefill")


def test_vllm_registration_reuses_schema_table_family_support_and_facades() -> None:
    torch_spec = find_kernel_profiler_spec(KIND, _BACKEND)
    vllm_spec = find_kernel_profiler_spec(KIND, _VLLM_BACKEND)

    assert vllm_spec.kernel_kind == torch_spec.kernel_kind == KIND
    assert vllm_spec.table_name == torch_spec.table_name == KIND
    assert vllm_spec.args_schema is torch_spec.args_schema is DsaTopkPrefillArgs
    assert vllm_spec.metric_family is torch_spec.metric_family is MetricFamily.COMPUTE
    assert vllm_spec.batch_outlier_policy == torch_spec.batch_outlier_policy
    assert vllm_spec.batch_outlier_policy == BatchOutlierPolicy()
    assert torch_spec.subprocess_env is None
    assert vllm_spec.subprocess_env == "vllm_env"
    assert vllm_spec.supports.kv is None
    assert vllm_spec.supports.allows(DType.FP32, gpu="NVIDIA H200")
    assert vllm_spec.supports.allows(DType.FP32, gpu="NVIDIA B200")
    assert not vllm_spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not vllm_spec.supports.allows(DType.FP32, gpu="NVIDIA H100")
    assert vllm_spec.runner_ref.module_name == ("profiling.runners.attention.dsa_topk_prefill")
    assert vllm_spec.runner_ref.function_name == ("profile_dsa_topk_prefill_vllm_cuda")
    assert hasattr(perf_api, "get_dsa_topk_prefill_times")
    assert hasattr(perf_api, "count_missing_dsa_topk_prefill")


def test_registry_barrel_import_is_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('vllm' in sys.modules); "
                "print('profiling.runners.attention.dsa_topk_prefill' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["False", "False", "False"]


def test_runner_ref_resolves_without_importing_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from profiling.db.registry import "
                "find_kernel_profiler_spec; runner = find_kernel_profiler_spec("
                "'dsa_topk_prefill', 'torch').runner_ref.load(); "
                "vllm_runner = find_kernel_profiler_spec("
                "'dsa_topk_prefill', 'vllm_cuda').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print(vllm_runner.__module__); print(vllm_runner.__name__); "
                "print('torch' in sys.modules); print('vllm' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.dsa_topk_prefill",
        "profile_dsa_topk_prefill_torch",
        "profiling.runners.attention.dsa_topk_prefill",
        "profile_dsa_topk_prefill_vllm_cuda",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"num_queries": 0}, "must be > 0"),
        ({"num_keys": 0}, "must be > 0"),
        ({"num_queries": 129, "num_keys": 128}, "must be <= num_keys"),
        ({"num_sequences": 2}, "num_sequences=1"),
        ({"top_k": 1024}, "top_k=2048"),
        ({"logits_row_stride": 0}, "positive and >= num_keys"),
        ({"logits_row_stride": 4095}, "positive and >= num_keys"),
        ({"logits_dtype": DType.BF16}, "logits_dtype=fp32"),
        ({"index_dtype": "int64"}, "index_dtype='int32'"),
        ({"span_mode": "ragged"}, "single_causal_tail"),
    ],
)
def test_rejects_unsupported_args_before_framework_import(overrides, match) -> None:
    from profiling.runners.attention import dsa_topk_prefill as runner

    kwargs = dict(_BASE_SPEC)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        runner.profile_dsa_topk_prefill_torch(**kwargs)


def test_explicit_unpadded_and_padded_strides_are_valid() -> None:
    from profiling.runners.attention.dsa_topk_prefill import _validate_args

    for stride in (8192, 8448, 16384):
        validated = _validate_args(**(_BASE_SPEC | {"logits_row_stride": stride}))
        assert validated[4] == stride


def test_rejects_missing_cuda_and_unverified_gpu() -> None:
    from profiling.runners.attention.dsa_topk_prefill import _validate_cuda_device

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


def test_vllm_rejects_missing_cuda_and_unverified_gpu() -> None:
    from profiling.runners.attention.dsa_topk_prefill import (
        _validate_vllm_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_vllm_cuda_device(no_cuda)

    h100 = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA H100",
        )
    )
    with pytest.raises(ProfilerNotImplemented, match="verified only on NVIDIA H200"):
        _validate_vllm_cuda_device(h100)

    b200 = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA B200",
        )
    )
    _validate_vllm_cuda_device(b200)


def test_vllm_rejects_common_and_backend_specific_args_before_loading(
    monkeypatch,
) -> None:
    from profiling.runners.attention import dsa_topk_prefill as runner

    loaded = False

    def fail_if_loaded():
        nonlocal loaded
        loaded = True
        raise AssertionError("vLLM loader must not run")

    monkeypatch.setattr(runner, "_load_vllm_cuda_backend", fail_if_loaded)
    invalid_cases = [
        ({"num_queries": 0}, "must be > 0"),
        ({"num_keys": 0}, "must be > 0"),
        ({"num_queries": 129, "num_keys": 128}, "must be <= num_keys"),
        ({"num_sequences": 2}, "num_sequences=1"),
        ({"top_k": 1024}, "top_k=2048"),
        ({"logits_row_stride": 0}, "positive and >= num_keys"),
        ({"logits_row_stride": 4095}, "positive and >= num_keys"),
        ({"logits_dtype": DType.BF16}, "logits_dtype=fp32"),
        ({"index_dtype": "int64"}, "index_dtype='int32'"),
        ({"span_mode": "ragged"}, "single_causal_tail"),
    ]
    for overrides, match in invalid_cases:
        kwargs = dict(_BASE_SPEC)
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=match):
            runner.profile_dsa_topk_prefill_vllm_cuda(**kwargs)
    assert not loaded


def test_vllm_op_resolution_requires_registered_callable() -> None:
    from profiling.runners.attention.dsa_topk_prefill import (
        _resolve_vllm_prefill_op,
    )

    missing_namespace = SimpleNamespace(ops=SimpleNamespace())
    with pytest.raises(ProfilerNotImplemented, match="torch.ops._C is unavailable"):
        _resolve_vllm_prefill_op(missing_namespace)

    missing_op = SimpleNamespace(ops=SimpleNamespace(_C=SimpleNamespace()))
    with pytest.raises(
        ProfilerNotImplemented,
        match="top_k_per_row_prefill is unavailable",
    ):
        _resolve_vllm_prefill_op(missing_op)

    noncallable = SimpleNamespace(
        ops=SimpleNamespace(_C=SimpleNamespace(top_k_per_row_prefill=object()))
    )
    with pytest.raises(ProfilerNotImplemented, match="is not callable"):
        _resolve_vllm_prefill_op(noncallable)


def test_operand_layout_and_causal_tail_spans() -> None:
    from profiling.runners.attention.dsa_topk_prefill import _build_operands

    operands = _build_operands(
        torch,
        num_queries=8,
        num_keys=8,
        top_k=4,
        logits_row_stride=13,
        device="cpu",
    )

    assert operands.logits_backing.shape == (8, 13)
    assert operands.logits_backing.stride() == (13, 1)
    assert operands.logits.shape == (8, 8)
    assert operands.logits.stride() == (13, 1)
    assert not operands.logits.is_contiguous()
    assert operands.logits.untyped_storage().data_ptr() == (
        operands.logits_backing.untyped_storage().data_ptr()
    )
    assert operands.logits.storage_offset() == 0
    assert operands.row_starts.tolist() == [0] * 8
    assert operands.row_ends.tolist() == list(range(1, 9))
    assert operands.row_starts.dtype is torch.int32
    assert operands.row_ends.dtype is torch.int32
    assert operands.row_starts.is_contiguous()
    assert operands.row_ends.is_contiguous()
    assert operands.out.shape == (8, 4)
    assert operands.out.stride() == (4, 1)
    assert operands.out.dtype is torch.int32
    assert operands.valid_mask.shape == (8, 8)
    assert operands.long_row_indices.tolist() == [4, 5, 6, 7]
    assert bool(torch.isfinite(operands.logits).all())
    assert operands.logits.unique().numel() == 8
    assert bool(torch.any(operands.logits < 0))
    assert bool(torch.any(operands.logits > 0))


def _selected_value_map(logits, starts, ends, indices):
    selected = []
    for row in range(logits.shape[0]):
        length = int(ends[row] - starts[row])
        if length <= indices.shape[1]:
            selected.append(None)
            continue
        row_indices = indices[row].to(torch.int64)
        selected.append(
            sorted(logits[row, int(starts[row]) : int(ends[row])][row_indices].tolist())
        )
    return selected


def test_vectorized_composite_matches_reference_and_preserves_inputs() -> None:
    from profiling.runners.attention.dsa_topk_prefill import (
        _build_operands,
        _torch_composite,
    )

    operands = _build_operands(
        torch,
        num_queries=8,
        num_keys=8,
        top_k=4,
        logits_row_stride=11,
        device="cpu",
    )
    logits_before = operands.logits_backing.clone()
    starts_before = operands.row_starts.clone()
    ends_before = operands.row_ends.clone()
    expected = torch.full_like(operands.out, -99)
    dsa_topk_prefill_reference(
        operands.logits,
        operands.row_starts,
        operands.row_ends,
        expected,
        top_k=4,
    )
    out_ptr = operands.out.untyped_storage().data_ptr()

    actual = _torch_composite(operands)

    assert actual is operands.out
    assert actual.untyped_storage().data_ptr() == out_ptr
    assert torch.equal(actual[:4], expected[:4])
    assert _selected_value_map(
        operands.logits,
        operands.row_starts,
        operands.row_ends,
        actual,
    ) == _selected_value_map(
        operands.logits,
        operands.row_starts,
        operands.row_ends,
        expected,
    )
    assert actual[0].tolist() == [0, -1, -1, -1]
    assert actual[3].tolist() == [0, 1, 2, 3]
    assert all(0 <= index < 5 for index in actual[4].tolist())
    assert torch.equal(operands.logits_backing, logits_before)
    assert torch.equal(operands.row_starts, starts_before)
    assert torch.equal(operands.row_ends, ends_before)


def test_composite_long_rows_uses_local_indices_with_ties() -> None:
    from profiling.runners.attention.dsa_topk_prefill import (
        _build_operands,
        _torch_composite,
    )

    operands = _build_operands(
        torch,
        num_queries=1,
        num_keys=7,
        top_k=3,
        logits_row_stride=9,
        device="cpu",
    )
    operands.logits[0].copy_(torch.tensor([-2, 5, 5, 1, 5, 3, -4], dtype=torch.float32))
    actual = _torch_composite(operands)

    assert sorted(actual[0].tolist()) == [1, 2, 4]
    assert sorted(operands.logits[0, actual[0].long()].tolist()) == [5.0, 5.0, 5.0]


def test_logical_bytes_accounts_only_valid_logits_metadata_and_output() -> None:
    from profiling.runners.attention.dsa_topk_prefill import _logical_bytes

    # Causal lengths are 5, 6, 7, 8: 26 valid FP32 logits.
    assert _logical_bytes(num_queries=4, num_keys=8, top_k=3) == (4 * 26 + 8 * 4 + 4 * 4 * 3)
    with pytest.raises(ValueError, match="must be > 0"):
        _logical_bytes(num_queries=0, num_keys=8, top_k=3)
    with pytest.raises(ValueError, match="must be <= num_keys"):
        _logical_bytes(num_queries=9, num_keys=8, top_k=3)


def test_profile_translates_runtime_failure(monkeypatch) -> None:
    from profiling.runners.attention import dsa_topk_prefill as runner

    monkeypatch.setattr(runner, "_validate_cuda_device", lambda _torch: None)
    monkeypatch.setattr(
        runner,
        "_build_operands",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        runner.Timer,
        "cuda_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    with pytest.raises(KernelLaunchFailed, match="synthetic failure"):
        runner.profile_dsa_topk_prefill_torch(**_BASE_SPEC)


def test_vllm_profile_forwards_exact_operands_and_arguments(monkeypatch) -> None:
    from profiling.runners.attention import dsa_topk_prefill as runner

    operands = runner._build_operands(
        torch,
        num_queries=1,
        num_keys=2048,
        top_k=2048,
        logits_row_stride=2304,
        device="cpu",
    )
    calls = []

    def fake_op(*args):
        calls.append(args)
        return None

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(
        runner,
        "_load_vllm_cuda_backend",
        lambda: (fake_torch, fake_op),
    )
    monkeypatch.setattr(runner, "_validate_vllm_cuda_device", lambda _torch: None)
    monkeypatch.setattr(runner, "_build_operands", lambda *args, **kwargs: operands)

    def fake_cupti(kernel, *, kernel_name):
        assert kernel_name == "topKPerRowPrefill"
        assert kernel() is None
        return 0.5

    monkeypatch.setattr(runner.Timer, "cupti", fake_cupti)
    monkeypatch.setattr(runner.Energy, "perf", lambda *args, **kwargs: 0.25)

    metrics = runner.profile_dsa_topk_prefill_vllm_cuda(
        **(_BASE_SPEC | {"num_queries": 1, "num_keys": 2048, "logits_row_stride": 2304})
    )

    assert len(calls) == 2  # exact-shape warmup, then the timed callable
    for call in calls:
        assert call == (
            operands.logits,
            operands.row_starts,
            operands.row_ends,
            operands.out,
            1,
            2304,
            1,
            2048,
        )
    assert metrics.time_ms == 0.5
    assert metrics.energy_j == 0.25
    assert metrics.tflops == 0.0
    assert metrics.memory_bandwidth_gbps > 0


def test_vllm_profile_translates_runtime_failure(monkeypatch) -> None:
    from profiling.runners.attention import dsa_topk_prefill as runner

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))

    def fail_launch(*args):
        raise RuntimeError("synthetic vLLM launch failure")

    monkeypatch.setattr(
        runner,
        "_load_vllm_cuda_backend",
        lambda: (fake_torch, fail_launch),
    )
    monkeypatch.setattr(runner, "_validate_vllm_cuda_device", lambda _torch: None)
    monkeypatch.setattr(
        runner,
        "_build_operands",
        lambda *args, **kwargs: SimpleNamespace(
            logits=SimpleNamespace(stride=lambda dim: (8448, 1)[dim]),
            row_starts=object(),
            row_ends=object(),
            out=object(),
        ),
    )
    with pytest.raises(KernelLaunchFailed, match="synthetic vLLM launch failure"):
        runner.profile_dsa_topk_prefill_vllm_cuda(**_BASE_SPEC)


def test_fork_registration_runs_in_the_fork_env_on_b200_only() -> None:
    fork_spec = find_kernel_profiler_spec(KIND, "vllm_fork_cuda")

    assert fork_spec.table_name == KIND
    assert fork_spec.args_schema is DsaTopkPrefillArgs
    assert fork_spec.subprocess_env == "vllm_fork_env"
    assert fork_spec.supports.allows(DType.FP32, gpu="NVIDIA B200")
    assert not fork_spec.supports.allows(DType.FP32, gpu="NVIDIA H200")
    assert fork_spec.runner_ref.function_name == "profile_dsa_topk_prefill_vllm_fork_cuda"


def test_fork_accepts_the_kpool_top_k_and_vllm_cuda_does_not(monkeypatch) -> None:
    from profiling.runners.attention import dsa_topk_prefill as runner

    def fail_if_loaded():
        raise AssertionError("loader must not run")

    monkeypatch.setattr(runner, "_load_vllm_cuda_backend", fail_if_loaded)
    with pytest.raises(ValueError, match="top_k=2048, got 512"):
        runner.profile_dsa_topk_prefill_vllm_cuda(**(_BASE_SPEC | {"top_k": 512}))
    with pytest.raises(ValueError, match="top_k=512 or top_k=1024 or top_k=2048, got 256"):
        runner.profile_dsa_topk_prefill_vllm_fork_cuda(**(_BASE_SPEC | {"top_k": 256}))


def test_fork_profile_forwards_select_k_512_and_checks_its_gpus(monkeypatch) -> None:
    from profiling.runners.attention import dsa_topk_prefill as runner

    operands = runner._build_operands(
        torch, num_queries=4, num_keys=4096, top_k=512, logits_row_stride=4352, device="cpu"
    )
    calls, devices = [], []
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(
        runner, "_load_vllm_cuda_backend", lambda: (fake_torch, lambda *a: calls.append(a))
    )
    monkeypatch.setattr(
        runner, "_validate_vllm_cuda_device", lambda _torch, *rest: devices.append(rest)
    )
    monkeypatch.setattr(runner, "_build_operands", lambda *args, **kwargs: operands)
    monkeypatch.setattr(runner.Timer, "cupti", lambda kernel, *, kernel_name: kernel() or 0.5)
    monkeypatch.setattr(runner.Energy, "perf", lambda *args, **kwargs: 0.25)

    runner.profile_dsa_topk_prefill_vllm_fork_cuda(
        **(
            _BASE_SPEC
            | {"num_queries": 4, "num_keys": 4096, "top_k": 512, "logits_row_stride": 4352}
        )
    )

    assert devices == [("vllm_fork_cuda", ("NVIDIA B200",))]
    assert [call[-1] for call in calls] == [512, 512]


def test_fork_device_check_rejects_h200() -> None:
    from profiling.runners.attention.dsa_topk_prefill import _validate_vllm_cuda_device

    h200 = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA H200",
        )
    )
    with pytest.raises(
        ProfilerNotImplemented, match="vllm_fork_cuda is verified only on NVIDIA B200"
    ):
        _validate_vllm_cuda_device(h200, "vllm_fork_cuda", ("NVIDIA B200",))
