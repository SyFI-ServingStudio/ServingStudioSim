"""Registration and CPU runner tests for ``gdn_gated_rms_norm``."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.db.table import MissingEntry, ProfileRow, Table
from profiling.kernels.gdn_gated_rms_norm import (
    KIND,
    GdnGatedRmsNormArgs,
)
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SPEC = {"m": 32, "hidden": 128, "dtype": "bf16"}


def test_args_field_order_and_coercion() -> None:
    assert [field.name for field in fields(GdnGatedRmsNormArgs)] == [
        "m",
        "hidden",
        "dtype",
    ]
    args = coerce_args(
        GdnGatedRmsNormArgs,
        {"m": "32", "hidden": "128", "dtype": "torch.bfloat16"},
    )
    assert args == GdnGatedRmsNormArgs(m=32, hidden=128, dtype=DType.BF16)
    with pytest.raises(Exception):
        args.m = 1


def test_registration_table_kind_runner_and_support_contract() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")

    assert KIND == "gdn_gated_rms_norm"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.args_schema is GdnGatedRmsNormArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == ("profiling.runners.attention.gdn_gated_rms_norm_torch")
    assert spec.runner_ref.function_name == "profile_gdn_gated_rms_norm"

    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.supports.kv is None
    assert spec.supports.gpus is None
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA H200")


def test_vllm_triton_registration_reuses_schema_table_and_is_h200_only() -> None:
    spec = find_kernel_profiler_spec(KIND, "vllm_triton")

    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "vllm_triton"
    assert spec.args_schema is GdnGatedRmsNormArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env == "vllm_env"
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_gated_rms_norm_vllm_triton"
    )
    assert spec.runner_ref.function_name == "profile_gdn_gated_rms_norm_vllm_triton"

    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.supports.kv is None
    assert spec.supports.gpus == frozenset({"NVIDIA H200"})
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA H100")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")


def test_registry_barrel_and_runner_ref_are_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_gated_rms_norm_torch' "
                "in sys.modules); "
                "print('profiling.runners.attention."
                "gdn_gated_rms_norm_vllm_triton' in sys.modules); "
                "print('profiling.runners.attention.gdn_gated_rms_norm_reference' "
                "in sys.modules); print('vllm' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["False", "False", "False", "False", "False"]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from profiling.db.registry import "
                "find_kernel_profiler_spec; runner = find_kernel_profiler_spec("
                "'gdn_gated_rms_norm', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_gated_rms_norm_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_gated_rms_norm_torch",
        "profile_gdn_gated_rms_norm",
        "False",
        "False",
    ]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from profiling.db.registry import "
                "find_kernel_profiler_spec; runner = find_kernel_profiler_spec("
                "'gdn_gated_rms_norm', 'vllm_triton').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); print('vllm' in sys.modules); "
                "print('profiling.runners.attention.gdn_gated_rms_norm_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_gated_rms_norm_vllm_triton",
        "profile_gdn_gated_rms_norm_vllm_triton",
        "False",
        "False",
        "False",
    ]


@pytest.mark.parametrize(("name", "value"), [("m", 0), ("hidden", 0)])
def test_runner_rejects_nonpositive_dimensions_before_cuda(name: str, value: int) -> None:
    from profiling.runners.attention.gdn_gated_rms_norm_torch import _validate_args

    with pytest.raises(ValueError, match="must be > 0"):
        _validate_args(**(_SPEC | {name: value}))


@pytest.mark.parametrize("dtype", ["fp16", "fp32", "fp8_e4m3"])
def test_runner_rejects_unsupported_dtype_before_cuda(dtype: str) -> None:
    from profiling.runners.attention.gdn_gated_rms_norm_torch import _validate_args

    with pytest.raises(ValueError, match="requires dtype=bf16"):
        _validate_args(**(_SPEC | {"dtype": dtype}))


@pytest.mark.parametrize(("name", "value"), [("m", 0), ("hidden", 0)])
def test_vllm_runner_rejects_nonpositive_dimensions_before_cuda(name: str, value: int) -> None:
    from profiling.runners.attention.gdn_gated_rms_norm_vllm_triton import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="must be > 0"):
        _validate_args(**(_SPEC | {name: value}))


@pytest.mark.parametrize("dtype", ["fp16", "fp32", "fp8_e4m3"])
def test_vllm_runner_rejects_unsupported_dtype_before_cuda(dtype: str) -> None:
    from profiling.runners.attention.gdn_gated_rms_norm_vllm_triton import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="requires dtype=bf16"):
        _validate_args(**(_SPEC | {"dtype": dtype}))


def test_vllm_runner_rejects_hidden_above_fused_limit_before_cuda() -> None:
    from profiling.runners.attention.gdn_gated_rms_norm_vllm_triton import (
        _validate_args,
    )

    assert _validate_args(**(_SPEC | {"hidden": 32768})).hidden == 32768
    with pytest.raises(ValueError, match="hidden<=32768"):
        _validate_args(**(_SPEC | {"hidden": 32769}))


def test_runner_shapes_and_bounded_deterministic_cpu_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_gated_rms_norm_torch import (
        _build_operands,
        _operand_shapes,
        _validate_args,
    )

    args = _validate_args(m=3, hidden=8, dtype="bf16")
    shapes = _operand_shapes(args)
    assert shapes.x == shapes.z == shapes.output == (3, 8)
    assert shapes.weight == (8,)

    first = _build_operands(torch, args, device=torch.device("cpu"))
    second = _build_operands(torch, args, device=torch.device("cpu"))
    assert tuple(first.x.shape) == shapes.x
    assert tuple(first.z.shape) == shapes.z
    assert tuple(first.weight.shape) == shapes.weight
    assert first.x.dtype is first.z.dtype is first.weight.dtype is torch.bfloat16
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.z, second.z)
    assert torch.equal(first.weight, second.weight)
    assert torch.isfinite(first.x).all()
    assert torch.isfinite(first.z).all()
    assert torch.isfinite(first.weight).all()


def test_vllm_runner_shapes_contiguity_and_bounded_guard_geometry() -> None:
    import torch

    from profiling.runners.attention.gdn_gated_rms_norm_vllm_triton import (
        _build_operands,
        _correctness_guard_args,
        _operand_shapes,
        _validate_args,
    )

    args = _validate_args(m=3, hidden=8, dtype="bf16")
    shapes = _operand_shapes(args)
    assert shapes.x == shapes.z == shapes.output == (3, 8)
    assert shapes.weight == (8,)

    first = _build_operands(torch, args, device=torch.device("cpu"))
    second = _build_operands(torch, args, device=torch.device("cpu"))
    for operand in (first.x, first.z, first.weight):
        assert operand.dtype is torch.bfloat16
        assert operand.is_contiguous()
        assert operand.stride(-1) == 1
        assert torch.isfinite(operand).all()
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.z, second.z)
    assert torch.equal(first.weight, second.weight)

    assert _correctness_guard_args(_validate_args(m=32, hidden=128, dtype="bf16")) == (
        _validate_args(m=32, hidden=128, dtype="bf16")
    )
    bounded = _correctness_guard_args(_validate_args(m=4096, hidden=128, dtype="bf16"))
    assert bounded.m == 512
    assert bounded.hidden == 128


def test_vllm_correctness_guard_checks_semantics_and_immutability_on_cpu() -> None:
    import torch

    from profiling.runners.attention.gdn_gated_rms_norm_reference import (
        gdn_gated_rms_norm_reference,
    )
    from profiling.runners.attention.gdn_gated_rms_norm_vllm_triton import (
        _build_operands,
        _check_correctness,
        _validate_args,
    )

    args = _validate_args(m=3, hidden=8, dtype="bf16")
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    snapshots = (operands.x.clone(), operands.z.clone(), operands.weight.clone())

    def fake_fused(x, weight, bias, **kwargs):
        assert bias is None
        assert kwargs == {
            "z": operands.z,
            "eps": 1e-6,
            "group_size": None,
            "norm_before_gate": True,
            "activation": "silu",
        }
        return gdn_gated_rms_norm_reference(x, kwargs["z"], weight)

    _check_correctness(torch, fake_fused, operands, args, synchronize=lambda: None)
    assert torch.equal(operands.x, snapshots[0])
    assert torch.equal(operands.z, snapshots[1])
    assert torch.equal(operands.weight, snapshots[2])

    def fake_aliasing_fused(x, _weight, _bias, **_kwargs):
        return x

    with pytest.raises(AssertionError, match="fresh storage"):
        _check_correctness(
            torch,
            fake_aliasing_fused,
            operands,
            args,
            synchronize=lambda: None,
        )


def test_vllm_profile_timed_callable_is_one_fused_call(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_gated_rms_norm_vllm_triton as runner

    args = runner._validate_args(m=3, hidden=8, dtype="bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    calls = {"fused": 0, "guard": 0, "timer": 0, "energy": 0}

    def fake_fused(*positional, **keyword):
        calls["fused"] += 1
        assert positional == (operands.x, operands.weight, None)
        assert keyword == {
            "z": operands.z,
            "eps": 1e-6,
            "group_size": None,
            "norm_before_gate": True,
            "activation": "silu",
        }
        return operands.x.clone()

    def fake_guard(*_args, **_kwargs) -> None:
        calls["guard"] += 1

    def fake_timer(fn, *, kernel_name):
        calls["timer"] += 1
        before = calls["fused"]
        fn()
        assert calls["fused"] == before + 1
        assert kernel_name == "layer_norm_fwd_kernel"
        return 0.5

    def fake_energy(fn, *, warmup, per_iter_time_ms):
        calls["energy"] += 1
        before = calls["fused"]
        fn()
        assert calls["fused"] == before + 1
        assert warmup == 5
        assert per_iter_time_ms == 0.5
        return 0.25

    monkeypatch.setattr(runner, "_require_h200", lambda _torch: None)
    monkeypatch.setattr(runner, "_load_fused_callable", lambda: fake_fused)
    monkeypatch.setattr(runner, "_build_operands", lambda *_args, **_kwargs: operands)
    monkeypatch.setattr(runner, "_check_correctness", fake_guard)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    metrics = runner.profile_gdn_gated_rms_norm_vllm_triton(m=3, hidden=8, dtype="bf16")
    assert metrics.time_ms == 0.5
    assert metrics.energy_j == 0.25
    assert calls == {"fused": 2, "guard": 1, "timer": 1, "energy": 1}


def test_runner_reports_missing_cuda_as_typed_unsupported() -> None:
    from profiling.runners.attention.gdn_gated_rms_norm_torch import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)


def test_semantic_metrics_are_explicit_logical_counts() -> None:
    from profiling.runners.attention.gdn_gated_rms_norm_torch import (
        _logical_bytes,
        _semantic_flops,
    )

    assert _semantic_flops(m=2, hidden=3) == 46
    assert _logical_bytes(m=2, hidden=3, dtype=DType.BF16) == 42


def test_generated_facades_and_read_only_missing_query(tmp_path, monkeypatch) -> None:
    assert hasattr(perf_api, "get_gdn_gated_rms_norm_times")
    assert hasattr(perf_api, "count_missing_gdn_gated_rms_norm")
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")

    assert (
        perf_api.count_missing_gdn_gated_rms_norm([_SPEC], backend="torch", gpu_name="NVIDIA H200")
        == 1
    )
    result = perf_api.get_gdn_gated_rms_norm_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(result, MissingEntry)
    assert result.args == coerce_args(GdnGatedRmsNormArgs, _SPEC)
    assert (
        perf_api.count_missing_gdn_gated_rms_norm(
            [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
        )
        == 1
    )
    vllm_result = perf_api.get_gdn_gated_rms_norm_times(
        [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(vllm_result, MissingEntry)
    assert vllm_result.args == result.args
    assert not perf_api.DB_PATH.exists()

    table = Table(find_kernel_profiler_spec(KIND, "torch"), perf_api.DB_PATH)
    assert table.args_columns == ["m", "hidden", "dtype"]


def test_db_round_trip_uses_exact_schema_and_compute_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "profile.db"
    monkeypatch.setattr(perf_api, "DB_PATH", db_path)
    profiler_spec = find_kernel_profiler_spec(KIND, "torch")
    args = coerce_args(GdnGatedRmsNormArgs, _SPEC)
    metrics = ComputeMetrics(
        time_ms=0.125,
        tflops=0.25,
        memory_bandwidth_gbps=1.5,
        energy_j=0.01,
    )
    Table(profiler_spec, db_path).insert(
        [
            ProfileRow(
                args=args,
                metrics=metrics,
                gpu_name="NVIDIA H200",
                backend="torch",
            )
        ]
    )

    result = perf_api.get_gdn_gated_rms_norm_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert result == metrics
    assert (
        perf_api.count_missing_gdn_gated_rms_norm([_SPEC], backend="torch", gpu_name="NVIDIA H200")
        == 0
    )
