"""Registration and CPU runner tests for ``gdn_causal_conv_prefill``."""

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
from profiling.kernels.gdn_causal_conv_prefill import (
    KIND,
    GdnCausalConvPrefillArgs,
)
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SPEC = {
    "batch_size": 1,
    "sequence_length": 128,
    "channels": 8192,
    "kernel_size": 4,
    "dtype": "bf16",
    "state_dtype": "bf16",
}


def _fake_metadata_helper(query_start_loc_cpu, *, device):
    import torch

    sequence_lengths = query_start_loc_cpu.diff()
    chunks = -(-sequence_lengths // 8)
    batch_ids = torch.repeat_interleave(torch.arange(len(chunks), dtype=torch.int64), chunks)
    offsets = torch.cat([torch.arange(int(count), dtype=torch.int32) for count in chunks])
    capacity = max(1024, len(batch_ids)) * 2
    batch_ptr = torch.full((capacity,), -1, dtype=torch.int32, device=device)
    token_chunk_offset_ptr = torch.full((capacity,), -1, dtype=torch.int32, device=device)
    batch_ptr[: len(batch_ids)].copy_(batch_ids.to(device=device, dtype=torch.int32))
    token_chunk_offset_ptr[: len(offsets)].copy_(offsets.to(device=device))
    nums_dict = {
        8: {
            "nums": chunks,
            "tot": int(chunks.sum()),
            "mlist": batch_ids,
            "mlist_len": len(batch_ids),
            "offsetlist": offsets,
            "batch_ptr": batch_ptr,
            "token_chunk_offset_ptr": token_chunk_offset_ptr,
        }
    }
    return nums_dict, batch_ptr, token_chunk_offset_ptr


def test_args_field_order_and_coercion() -> None:
    assert [field.name for field in fields(GdnCausalConvPrefillArgs)] == [
        "batch_size",
        "sequence_length",
        "channels",
        "kernel_size",
        "dtype",
        "state_dtype",
    ]
    args = coerce_args(
        GdnCausalConvPrefillArgs,
        _SPEC
        | {
            "batch_size": "1",
            "sequence_length": "128",
            "channels": "8192",
            "kernel_size": "4",
            "dtype": "bfloat16",
            "state_dtype": "torch.bfloat16",
        },
    )
    assert args == GdnCausalConvPrefillArgs(
        batch_size=1,
        sequence_length=128,
        channels=8192,
        kernel_size=4,
        dtype=DType.BF16,
        state_dtype=DType.BF16,
    )
    with pytest.raises(Exception):
        args.channels = 1


def test_registration_table_kind_runner_and_support_contract() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")

    assert KIND == "gdn_causal_conv_prefill"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.args_schema is GdnCausalConvPrefillArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_causal_conv_prefill_torch"
    )
    assert spec.runner_ref.function_name == "profile_gdn_causal_conv_prefill"

    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.supports.kv is None
    assert spec.supports.gpus is None
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA H200")


def test_vllm_registration_reuses_schema_table_and_is_h200_only() -> None:
    spec = find_kernel_profiler_spec(KIND, "vllm_triton")

    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "vllm_triton"
    assert spec.args_schema is GdnCausalConvPrefillArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env == "vllm_env"
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton"
    )
    assert spec.runner_ref.function_name == ("profile_gdn_causal_conv_prefill_vllm_triton")

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
                "print('profiling.runners.attention.gdn_causal_conv_prefill_torch' "
                "in sys.modules); "
                "print('profiling.runners.attention."
                "gdn_causal_conv_prefill_vllm_triton' in sys.modules); "
                "print('profiling.runners.attention.gdn_causal_conv_prefill_reference' "
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
                "'gdn_causal_conv_prefill', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_causal_conv_prefill_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_causal_conv_prefill_torch",
        "profile_gdn_causal_conv_prefill",
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
                "'gdn_causal_conv_prefill', 'vllm_triton').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); print('vllm' in sys.modules); "
                "print('profiling.runners.attention.gdn_causal_conv_prefill_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton",
        "profile_gdn_causal_conv_prefill_vllm_triton",
        "False",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    ("name", "value"),
    [("batch_size", 0), ("sequence_length", 0), ("channels", 0)],
)
def test_runner_rejects_nonpositive_dimensions_before_cuda(name: str, value: int) -> None:
    from profiling.runners.attention.gdn_causal_conv_prefill_torch import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="must be > 0"):
        _validate_args(**(_SPEC | {name: value}))


@pytest.mark.parametrize("kernel_size", [1, 5])
def test_runner_rejects_unsupported_width_before_cuda(kernel_size: int) -> None:
    from profiling.runners.attention.gdn_causal_conv_prefill_torch import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="production prefill Triton kernel"):
        _validate_args(**(_SPEC | {"kernel_size": kernel_size}))


@pytest.mark.parametrize("kernel_size", [2, 3, 4])
def test_runner_accepts_supported_widths(kernel_size: int) -> None:
    from profiling.runners.attention.gdn_causal_conv_prefill_torch import (
        _validate_args,
    )

    assert _validate_args(**(_SPEC | {"kernel_size": kernel_size})).kernel_size == (kernel_size)


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
    from profiling.runners.attention.gdn_causal_conv_prefill_torch import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="requires dtype=bf16 and state_dtype=bf16"):
        _validate_args(**(_SPEC | {"dtype": dtype, "state_dtype": state_dtype}))


@pytest.mark.parametrize(
    ("name", "value"),
    [("batch_size", 0), ("sequence_length", 0), ("channels", 0)],
)
def test_vllm_runner_rejects_nonpositive_dimensions_before_import(name: str, value: int) -> None:
    from profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="must be > 0"):
        _validate_args(**(_SPEC | {name: value}))


@pytest.mark.parametrize("kernel_size", [1, 5, 6])
def test_vllm_runner_rejects_unsupported_width_before_cuda(kernel_size: int) -> None:
    from profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton import (
        _validate_args,
    )

    message = "silently computes W5 incorrectly" if kernel_size == 5 else "supported"
    with pytest.raises(ValueError, match=message):
        _validate_args(**(_SPEC | {"kernel_size": kernel_size}))


@pytest.mark.parametrize("kernel_size", [2, 3, 4])
def test_vllm_runner_accepts_verified_widths(kernel_size: int) -> None:
    from profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton import (
        _validate_args,
    )

    assert _validate_args(**(_SPEC | {"kernel_size": kernel_size})).kernel_size == (kernel_size)


@pytest.mark.parametrize(
    ("dtype", "state_dtype"),
    [("fp16", "bf16"), ("fp32", "bf16"), ("bf16", "fp16"), ("bf16", "fp32")],
)
def test_vllm_runner_rejects_unsupported_dtypes_before_cuda(dtype: str, state_dtype: str) -> None:
    from profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="requires dtype=bf16 and state_dtype=bf16"):
        _validate_args(**(_SPEC | {"dtype": dtype, "state_dtype": state_dtype}))


def test_runner_derived_shapes_slots_and_bounded_cpu_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_causal_conv_prefill_torch import (
        _build_operands,
        _operand_shapes,
        _valid_slot_indices,
        _validate_args,
    )

    args = _validate_args(**(_SPEC | {"batch_size": 3, "sequence_length": 5, "channels": 8}))
    shapes = _operand_shapes(args)
    assert shapes.x == (3, 5, 8)
    assert shapes.weight == (8, 4)
    assert shapes.state_storage == (4, 8, 3)
    assert shapes.slot_indices == (3,)
    assert shapes.output == (3, 5, 8)
    assert _valid_slot_indices(3) == (1, 2, 3)

    operands = _build_operands(torch, args, device=torch.device("cpu"))
    assert tuple(operands.x.shape) == shapes.x
    assert tuple(operands.weight.shape) == shapes.weight
    assert tuple(operands.state.shape) == shapes.state_storage
    assert tuple(operands.slot_indices.shape) == shapes.slot_indices
    assert operands.x.dtype is torch.bfloat16
    assert operands.weight.dtype is torch.bfloat16
    assert operands.state.dtype is torch.bfloat16
    assert operands.slot_indices.dtype is torch.int32
    assert operands.slot_indices.tolist() == [1, 2, 3]
    assert torch.isfinite(operands.x).all()
    assert torch.isfinite(operands.weight).all()
    assert torch.isfinite(operands.state).all()


def test_vllm_derived_shapes_packing_metadata_and_chunk_mapping() -> None:
    import torch

    from profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton import (
        _build_operands,
        _expected_chunk_mapping,
        _guard_args,
        _operand_shapes,
        _validate_args,
    )

    args = _validate_args(**(_SPEC | {"batch_size": 3, "sequence_length": 17, "channels": 8}))
    shapes = _operand_shapes(args)
    assert shapes.semantic_x == (3, 17, 8)
    assert shapes.packed_x == (8, 51)
    assert shapes.weight == (8, 4)
    assert shapes.state_storage == (4, 8, 3)
    assert shapes.slot_indices == (3,)
    assert shapes.query_start_loc == (4,)
    assert shapes.has_initial_state == (3,)
    assert shapes.metadata_buffers == (2048,)
    assert shapes.packed_output == (8, 51)
    assert shapes.semantic_output == (3, 17, 8)
    assert shapes.program_count == 9
    assert _expected_chunk_mapping(3, 17) == (
        (0, 0, 0, 1, 1, 1, 2, 2, 2),
        (0, 1, 2, 0, 1, 2, 0, 1, 2),
    )

    operands = _build_operands(torch, args, _fake_metadata_helper, device=torch.device("cpu"))
    assert tuple(operands.semantic_x.shape) == shapes.semantic_x
    assert operands.semantic_x.is_contiguous()
    assert tuple(operands.packed_x.shape) == shapes.packed_x
    assert operands.packed_x.stride() == (1, 8)
    assert operands.packed_x.untyped_storage().data_ptr() == (
        operands.semantic_x.untyped_storage().data_ptr()
    )
    assert operands.weight.is_contiguous()
    assert operands.state.is_contiguous()
    assert operands.slot_indices.tolist() == [1, 2, 3]
    assert operands.query_start_loc_cpu.tolist() == [0, 17, 34, 51]
    assert operands.query_start_loc_cpu.dtype is torch.int32
    assert operands.query_start_loc_gpu.dtype is torch.int32
    assert not operands.has_initial_state.any()
    assert tuple(operands.metadata.batch_ptr.shape) == shapes.metadata_buffers
    assert tuple(operands.metadata.token_chunk_offset_ptr.shape) == (shapes.metadata_buffers)
    assert operands.metadata.batch_ptr[:9].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert operands.metadata.token_chunk_offset_ptr[:9].tolist() == [
        0,
        1,
        2,
        0,
        1,
        2,
        0,
        1,
        2,
    ]
    # Exact smoke geometry fits the full guard; only larger activations cap C.
    exact = _validate_args(**_SPEC)
    assert _guard_args(exact) == exact
    huge = _validate_args(**(_SPEC | {"channels": 16384}))
    assert _guard_args(huge).channels == 8192


def test_vllm_correctness_guard_restores_operands_and_checks_prior_independence() -> None:
    import torch

    from profiling.runners.attention.gdn_causal_conv_prefill_reference import (
        gdn_causal_conv_prefill_reference,
    )
    from profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton import (
        _build_operands,
        _check_correctness,
        _validate_args,
    )

    args = _validate_args(
        batch_size=2,
        sequence_length=9,
        channels=8,
        kernel_size=4,
        dtype="bf16",
        state_dtype="bf16",
    )
    operands = _build_operands(torch, args, _fake_metadata_helper, device=torch.device("cpu"))
    snapshots = {
        "x": operands.semantic_x.clone(),
        "weight": operands.weight.clone(),
        "state": operands.state.clone(),
        "slots": operands.slot_indices.clone(),
        "query_cpu": operands.query_start_loc_cpu.clone(),
        "query_gpu": operands.query_start_loc_gpu.clone(),
        "initial": operands.has_initial_state.clone(),
        "batch_ptr": operands.metadata.batch_ptr.clone(),
        "offset_ptr": operands.metadata.token_chunk_offset_ptr.clone(),
    }
    calls = 0

    def fake_fused(
        packed_x,
        weight,
        bias,
        state,
        query_start_loc,
        *,
        cache_indices,
        has_initial_state,
        activation,
        metadata,
        validate_data,
    ):
        nonlocal calls
        calls += 1
        assert bias is None
        assert activation == "silu"
        assert validate_data is True
        assert not has_initial_state.any()
        assert metadata is operands.metadata
        batch_size = len(query_start_loc) - 1
        sequence_length = int(query_start_loc[1])
        channels = packed_x.shape[0]
        semantic_x = packed_x.transpose(0, 1).reshape(batch_size, sequence_length, channels)
        output, returned_state = gdn_causal_conv_prefill_reference(
            semantic_x, weight, state, cache_indices
        )
        assert returned_state is state
        return output.reshape(batch_size * sequence_length, channels).transpose(0, 1)

    _check_correctness(
        torch,
        fake_fused,
        operands,
        args,
        synchronize=lambda: None,
    )
    assert calls == 2
    assert torch.equal(operands.semantic_x, snapshots["x"])
    assert torch.equal(operands.weight, snapshots["weight"])
    assert torch.equal(operands.state, snapshots["state"])
    assert torch.equal(operands.slot_indices, snapshots["slots"])
    assert torch.equal(operands.query_start_loc_cpu, snapshots["query_cpu"])
    assert torch.equal(operands.query_start_loc_gpu, snapshots["query_gpu"])
    assert torch.equal(operands.has_initial_state, snapshots["initial"])
    assert torch.equal(operands.metadata.batch_ptr, snapshots["batch_ptr"])
    assert torch.equal(operands.metadata.token_chunk_offset_ptr, snapshots["offset_ptr"])


def test_vllm_profile_timed_callable_is_one_fused_call(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton as runner

    args = runner._validate_args(**(_SPEC | {"sequence_length": 9, "channels": 8}))
    operands = runner._build_operands(
        torch, args, _fake_metadata_helper, device=torch.device("cpu")
    )
    calls = {"fused": 0, "build": 0, "guard": 0, "timer": 0, "energy": 0}

    def fake_fused(*positional, **keyword):
        calls["fused"] += 1
        assert len(positional) == 5
        assert positional[0] is operands.packed_x
        assert positional[1] is operands.weight
        assert positional[2] is None
        assert positional[3] is operands.state
        assert positional[4] is operands.query_start_loc_gpu
        assert keyword == {
            "cache_indices": operands.slot_indices,
            "has_initial_state": operands.has_initial_state,
            "activation": "silu",
            "metadata": operands.metadata,
            "validate_data": True,
        }
        if calls["fused"] == 1:
            assert not torch.all(operands.state[1:] == 1)
        else:
            assert torch.all(operands.state[1:] == 1)
        operands.state[1:].fill_(1)
        return torch.empty_like(operands.packed_x)

    def fake_build(*_args, **_kwargs):
        calls["build"] += 1
        return operands

    def fake_guard(*_args, **_kwargs):
        calls["guard"] += 1

    def fake_timer(fn, *, kernel_name):
        calls["timer"] += 1
        assert calls["build"] == 2
        fn()
        assert kernel_name == "_causal_conv1d_fwd_kernel"
        return 0.5

    def fake_energy(fn, *, warmup, per_iter_time_ms):
        calls["energy"] += 1
        assert calls["build"] == 2
        assert warmup == 5
        assert per_iter_time_ms == 0.5
        fn()
        return 0.25

    monkeypatch.setattr(runner, "_require_h200", lambda _torch: None)
    monkeypatch.setattr(runner, "_load_vllm_components", lambda: (fake_fused, object()))
    monkeypatch.setattr(runner, "_build_operands", fake_build)
    monkeypatch.setattr(runner, "_check_correctness", fake_guard)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    metrics = runner.profile_gdn_causal_conv_prefill_vllm_triton(
        **(_SPEC | {"sequence_length": 9, "channels": 8})
    )
    assert metrics.time_ms == 0.5
    assert metrics.energy_j == 0.25
    assert calls == {"fused": 2, "build": 2, "guard": 1, "timer": 1, "energy": 1}


def test_runner_reports_missing_cuda_as_typed_unsupported() -> None:
    from profiling.runners.attention.gdn_causal_conv_prefill_torch import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)


def test_profile_timed_calls_need_no_state_reset(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_causal_conv_prefill_reference as reference
    import profiling.runners.attention.gdn_causal_conv_prefill_torch as runner

    args = runner._validate_args(**(_SPEC | {"sequence_length": 5, "channels": 8}))
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    calls = {"reference": 0, "timer": 0, "energy": 0}

    def fake_reference(x, weight, state, slot_indices):
        calls["reference"] += 1
        selected = state.index_select(0, slot_indices.long())
        if calls["reference"] == 1:
            assert not torch.all(selected == 1)
        else:
            assert torch.all(selected == 1)
        state.index_fill_(0, slot_indices.long(), 1)
        return torch.empty_like(x), state

    def fake_timer(fn):
        calls["timer"] += 1
        fn()
        return 0.5

    def fake_energy(fn, *, warmup, per_iter_time_ms):
        calls["energy"] += 1
        assert warmup == 5
        assert per_iter_time_ms == 0.5
        fn()
        return 0.25

    monkeypatch.setattr(runner, "_validate_cuda_device", lambda _torch: None)
    monkeypatch.setattr(runner, "_build_operands", lambda *_args, **_kwargs: operands)
    monkeypatch.setattr(reference, "gdn_causal_conv_prefill_reference", fake_reference)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))

    metrics = runner.profile_gdn_causal_conv_prefill(
        **(_SPEC | {"sequence_length": 5, "channels": 8})
    )
    assert metrics.time_ms == 0.5
    assert metrics.energy_j == 0.25
    assert calls == {"reference": 2, "timer": 1, "energy": 1}


def test_semantic_metrics_are_explicit_logical_counts() -> None:
    from profiling.runners.attention.gdn_causal_conv_prefill_torch import (
        _logical_bytes,
        _semantic_flops,
    )

    assert (
        _semantic_flops(
            batch_size=2,
            sequence_length=5,
            channels=3,
            kernel_size=4,
        )
        == 270
    )
    assert (
        _logical_bytes(
            batch_size=2,
            sequence_length=5,
            channels=3,
            kernel_size=4,
            dtype=DType.BF16,
            state_dtype=DType.BF16,
        )
        == 188
    )


def test_generated_facades_and_read_only_missing_query(tmp_path, monkeypatch) -> None:
    assert hasattr(perf_api, "get_gdn_causal_conv_prefill_times")
    assert hasattr(perf_api, "count_missing_gdn_causal_conv_prefill")
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")

    assert (
        perf_api.count_missing_gdn_causal_conv_prefill(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 1
    )
    result = perf_api.get_gdn_causal_conv_prefill_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(result, MissingEntry)
    assert result.args == coerce_args(GdnCausalConvPrefillArgs, _SPEC)
    assert (
        perf_api.count_missing_gdn_causal_conv_prefill(
            [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
        )
        == 1
    )
    vllm_result = perf_api.get_gdn_causal_conv_prefill_times(
        [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(vllm_result, MissingEntry)
    assert vllm_result.args == result.args
    assert not perf_api.DB_PATH.exists()

    table = Table(find_kernel_profiler_spec(KIND, "torch"), perf_api.DB_PATH)
    assert table.args_columns == [
        "batch_size",
        "sequence_length",
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
    args = coerce_args(GdnCausalConvPrefillArgs, _SPEC)
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

    result = perf_api.get_gdn_causal_conv_prefill_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert result == metrics
    assert (
        perf_api.count_missing_gdn_causal_conv_prefill(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 0
    )
