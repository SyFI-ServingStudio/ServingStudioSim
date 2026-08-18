"""Registration and CPU runner tests for ``gdn_causal_conv_decode``."""

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
from profiling.kernels.gdn_causal_conv_decode import (
    KIND,
    GdnCausalConvDecodeArgs,
)
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SPEC = {
    "batch_size": 1,
    "channels": 8192,
    "kernel_size": 4,
    "dtype": "bf16",
    "state_dtype": "bf16",
}


def test_args_field_order_and_coercion() -> None:
    assert [field.name for field in fields(GdnCausalConvDecodeArgs)] == [
        "batch_size",
        "channels",
        "kernel_size",
        "dtype",
        "state_dtype",
    ]
    args = coerce_args(
        GdnCausalConvDecodeArgs,
        _SPEC
        | {
            "batch_size": "1",
            "channels": "8192",
            "kernel_size": "4",
            "dtype": "bfloat16",
            "state_dtype": "torch.bfloat16",
        },
    )
    assert args == GdnCausalConvDecodeArgs(
        batch_size=1,
        channels=8192,
        kernel_size=4,
        dtype=DType.BF16,
        state_dtype=DType.BF16,
    )
    with pytest.raises(Exception):
        args.channels = 1


def test_registration_table_kind_runner_and_support_contract() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")

    assert KIND == "gdn_causal_conv_decode"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.args_schema is GdnCausalConvDecodeArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_causal_conv_decode_torch"
    )
    assert spec.runner_ref.function_name == "profile_gdn_causal_conv_decode"

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
    assert spec.args_schema is GdnCausalConvDecodeArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env == "vllm_env"
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_causal_conv_decode_vllm_triton"
    )
    assert spec.runner_ref.function_name == ("profile_gdn_causal_conv_decode_vllm_triton")

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
                "print('profiling.runners.attention.gdn_causal_conv_decode_torch' "
                "in sys.modules); "
                "print('profiling.runners.attention."
                "gdn_causal_conv_decode_vllm_triton' in sys.modules); "
                "print('profiling.runners.attention.gdn_causal_conv_decode_reference' "
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
                "'gdn_causal_conv_decode', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_causal_conv_decode_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_causal_conv_decode_torch",
        "profile_gdn_causal_conv_decode",
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
                "'gdn_causal_conv_decode', 'vllm_triton').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); print('vllm' in sys.modules); "
                "print('profiling.runners.attention.gdn_causal_conv_decode_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_causal_conv_decode_vllm_triton",
        "profile_gdn_causal_conv_decode_vllm_triton",
        "False",
        "False",
        "False",
    ]


@pytest.mark.parametrize(("name", "value"), [("batch_size", 0), ("channels", 0)])
def test_runner_rejects_nonpositive_dimensions_before_cuda(name: str, value: int) -> None:
    from profiling.runners.attention.gdn_causal_conv_decode_torch import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="must be > 0"):
        _validate_args(**(_SPEC | {name: value}))


@pytest.mark.parametrize("kernel_size", [1, 7])
def test_runner_rejects_unsupported_width_before_cuda(kernel_size: int) -> None:
    from profiling.runners.attention.gdn_causal_conv_decode_torch import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="production Triton kernel"):
        _validate_args(**(_SPEC | {"kernel_size": kernel_size}))


@pytest.mark.parametrize(
    ("dtype", "state_dtype"),
    [
        ("fp16", "bf16"),
        ("fp32", "bf16"),
        ("bf16", "fp16"),
        ("bf16", "fp32"),
    ],
)
def test_runner_rejects_unsupported_dtypes_before_cuda(
    dtype: str,
    state_dtype: str,
) -> None:
    from profiling.runners.attention.gdn_causal_conv_decode_torch import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="requires dtype=bf16 and state_dtype=bf16"):
        _validate_args(**(_SPEC | {"dtype": dtype, "state_dtype": state_dtype}))


@pytest.mark.parametrize(("name", "value"), [("batch_size", 0), ("channels", 0)])
def test_vllm_runner_rejects_nonpositive_dimensions_before_cuda(
    name: str,
    value: int,
) -> None:
    from profiling.runners.attention.gdn_causal_conv_decode_vllm_triton import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="must be > 0"):
        _validate_args(**(_SPEC | {name: value}))


@pytest.mark.parametrize("kernel_size", [1, 7])
def test_vllm_runner_rejects_unsupported_width_before_cuda(kernel_size: int) -> None:
    from profiling.runners.attention.gdn_causal_conv_decode_vllm_triton import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="production Triton kernel"):
        _validate_args(**(_SPEC | {"kernel_size": kernel_size}))


@pytest.mark.parametrize(
    ("dtype", "state_dtype"),
    [("fp16", "bf16"), ("fp32", "bf16"), ("bf16", "fp16"), ("bf16", "fp32")],
)
def test_vllm_runner_rejects_unsupported_dtypes_before_cuda(
    dtype: str,
    state_dtype: str,
) -> None:
    from profiling.runners.attention.gdn_causal_conv_decode_vllm_triton import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="requires dtype=bf16 and state_dtype=bf16"):
        _validate_args(**(_SPEC | {"dtype": dtype, "state_dtype": state_dtype}))


def test_runner_derived_shapes_slots_and_bounded_cpu_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_causal_conv_decode_torch import (
        _build_operands,
        _operand_shapes,
        _valid_slot_indices,
        _validate_args,
    )

    args = _validate_args(**(_SPEC | {"batch_size": 3, "channels": 8}))
    shapes = _operand_shapes(args)
    assert shapes.x == (3, 8)
    assert shapes.weight == (8, 4)
    assert shapes.state_storage == (4, 8, 3)
    assert shapes.slot_indices == (3,)
    assert shapes.output == (3, 8)
    assert _valid_slot_indices(3) == (1, 2, 3)

    operands = _build_operands(torch, args, device=torch.device("cpu"))
    assert tuple(operands.x.shape) == shapes.x
    assert tuple(operands.weight.shape) == shapes.weight
    assert tuple(operands.state.shape) == shapes.state_storage
    assert tuple(operands.slot_indices.shape) == shapes.slot_indices
    assert operands.slot_indices.dtype is torch.int32
    assert operands.slot_indices.tolist() == [1, 2, 3]
    assert torch.isfinite(operands.x).all()
    assert torch.isfinite(operands.weight).all()
    assert torch.isfinite(operands.state).all()


def test_vllm_runner_derived_shapes_slots_and_contiguous_cpu_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_causal_conv_decode_vllm_triton import (
        _build_operands,
        _operand_shapes,
        _valid_slot_indices,
        _validate_args,
    )

    args = _validate_args(**(_SPEC | {"batch_size": 3, "channels": 8}))
    shapes = _operand_shapes(args)
    assert shapes.x == (3, 8)
    assert shapes.weight == (8, 4)
    assert shapes.state_storage == (4, 8, 3)
    assert shapes.slot_indices == (3,)
    assert shapes.output == (3, 8)
    assert _valid_slot_indices(3) == (1, 2, 3)

    operands = _build_operands(torch, args, device=torch.device("cpu"))
    assert tuple(operands.x.shape) == shapes.x
    assert tuple(operands.weight.shape) == shapes.weight
    assert tuple(operands.state.shape) == shapes.state_storage
    assert tuple(operands.slot_indices.shape) == shapes.slot_indices
    assert operands.x.is_contiguous()
    assert operands.x.stride(1) == 1
    assert operands.weight.is_contiguous()
    assert operands.state.is_contiguous()
    assert operands.slot_indices.dtype is torch.int32
    assert operands.slot_indices.tolist() == [1, 2, 3]
    assert torch.isfinite(operands.x).all()
    assert torch.isfinite(operands.weight).all()
    assert torch.isfinite(operands.state).all()


def test_vllm_correctness_guard_checks_aliases_and_restores_cpu_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_causal_conv_decode_reference import (
        gdn_causal_conv_decode_reference,
    )
    from profiling.runners.attention.gdn_causal_conv_decode_vllm_triton import (
        _build_operands,
        _check_correctness,
        _validate_args,
    )

    args = _validate_args(
        batch_size=2,
        channels=8,
        kernel_size=4,
        dtype="bf16",
        state_dtype="bf16",
    )
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    x_before = operands.x.clone()
    state_before = operands.state.clone()
    weight_before = operands.weight.clone()
    indices_before = operands.slot_indices.clone()

    def fake_fused(
        x,
        state,
        weight,
        *,
        bias,
        activation,
        conv_state_indices,
        validate_data,
    ):
        assert bias is None
        assert activation == "silu"
        assert validate_data is False
        output, returned_state = gdn_causal_conv_decode_reference(
            x.clone(), weight, state, conv_state_indices
        )
        assert returned_state is state
        x.copy_(output)
        return x.unsqueeze(-1).squeeze(-1)

    _check_correctness(
        torch,
        fake_fused,
        operands,
        args,
        synchronize=lambda: None,
    )
    assert torch.equal(operands.x, x_before)
    assert torch.equal(operands.state, state_before)
    assert torch.equal(operands.weight, weight_before)
    assert torch.equal(operands.slot_indices, indices_before)


def test_runner_reports_missing_cuda_as_typed_unsupported() -> None:
    from profiling.runners.attention.gdn_causal_conv_decode_torch import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)


def test_vllm_profile_timed_callable_is_one_fused_call(
    monkeypatch,
) -> None:
    import torch

    import profiling.runners.attention.gdn_causal_conv_decode_vllm_triton as runner

    args = runner._validate_args(**(_SPEC | {"channels": 8}))
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    calls = {"fused": 0, "guard": 0, "timer": 0, "energy": 0}

    def fake_fused(*positional, **keyword):
        calls["fused"] += 1
        assert len(positional) == 3
        assert positional[0] is operands.x
        assert positional[1] is operands.state
        assert positional[2] is operands.weight
        assert keyword["bias"] is None
        assert keyword["activation"] == "silu"
        assert keyword["conv_state_indices"] is operands.slot_indices
        assert keyword["validate_data"] is False
        assert set(keyword) == {
            "bias",
            "activation",
            "conv_state_indices",
            "validate_data",
        }
        return operands.x.unsqueeze(-1).squeeze(-1)

    def fake_guard(*_args, **_kwargs) -> None:
        calls["guard"] += 1

    def fake_timer(fn, *, kernel_name):
        calls["timer"] += 1
        before = calls["fused"]
        fn()
        assert calls["fused"] == before + 1
        assert kernel_name == "_causal_conv1d_update_kernel"
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

    metrics = runner.profile_gdn_causal_conv_decode_vllm_triton(**_SPEC)
    assert metrics.time_ms == 0.5
    assert metrics.energy_j == 0.25
    assert calls == {"fused": 2, "guard": 1, "timer": 1, "energy": 1}


def test_semantic_metrics_are_explicit_logical_counts() -> None:
    from profiling.runners.attention.gdn_causal_conv_decode_torch import (
        _logical_bytes,
        _semantic_flops,
    )

    assert _semantic_flops(batch_size=2, channels=3, kernel_size=4) == 54
    assert (
        _logical_bytes(
            batch_size=2,
            channels=3,
            kernel_size=4,
            dtype=DType.BF16,
            state_dtype=DType.BF16,
        )
        == 128
    )


def test_generated_facades_and_read_only_missing_query(tmp_path, monkeypatch) -> None:
    assert hasattr(perf_api, "get_gdn_causal_conv_decode_times")
    assert hasattr(perf_api, "count_missing_gdn_causal_conv_decode")
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")

    assert (
        perf_api.count_missing_gdn_causal_conv_decode(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 1
    )
    result = perf_api.get_gdn_causal_conv_decode_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(result, MissingEntry)
    assert result.args == coerce_args(GdnCausalConvDecodeArgs, _SPEC)
    assert (
        perf_api.count_missing_gdn_causal_conv_decode(
            [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
        )
        == 1
    )
    vllm_result = perf_api.get_gdn_causal_conv_decode_times(
        [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(vllm_result, MissingEntry)
    assert vllm_result.args == result.args
    assert not perf_api.DB_PATH.exists()

    table = Table(find_kernel_profiler_spec(KIND, "torch"), perf_api.DB_PATH)
    assert table.args_columns == [
        "batch_size",
        "channels",
        "kernel_size",
        "dtype",
        "state_dtype",
    ]


def test_db_round_trip_uses_exact_schema_and_compute_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "profile.db"
    monkeypatch.setattr(perf_api, "DB_PATH", db_path)
    profiler_spec = find_kernel_profiler_spec(KIND, "torch")
    args = coerce_args(GdnCausalConvDecodeArgs, _SPEC)
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

    result = perf_api.get_gdn_causal_conv_decode_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert result == metrics
    assert (
        perf_api.count_missing_gdn_causal_conv_decode(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 0
    )
