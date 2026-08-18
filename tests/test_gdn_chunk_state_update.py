"""Registration and Torch runner tests for ``gdn_chunk_state_update``."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.db.table import MissingEntry, ProfileRow, Table
from profiling.kernels.gdn_chunk_state_update import KIND, GdnChunkStateUpdateArgs
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SPEC = {
    "num_tokens": 128,
    "num_chunks": 2,
    "num_sequences": 1,
    "max_chunks_per_sequence": 2,
    "num_key_heads": 16,
    "num_heads": 32,
    "key_head_dim": 128,
    "value_head_dim": 128,
    "dtype": "bf16",
}


def test_args_field_order_and_coercion() -> None:
    assert [field.name for field in fields(GdnChunkStateUpdateArgs)] == [
        "num_tokens",
        "num_chunks",
        "num_sequences",
        "max_chunks_per_sequence",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "value_head_dim",
        "dtype",
    ]
    args = coerce_args(
        GdnChunkStateUpdateArgs,
        {
            **{name: str(value) for name, value in _SPEC.items() if name != "dtype"},
            "dtype": "torch.bfloat16",
        },
    )
    assert args == GdnChunkStateUpdateArgs(
        num_tokens=128,
        num_chunks=2,
        num_sequences=1,
        max_chunks_per_sequence=2,
        num_key_heads=16,
        num_heads=32,
        key_head_dim=128,
        value_head_dim=128,
        dtype=DType.BF16,
    )
    with pytest.raises(Exception):
        args.num_tokens = 1


def test_registration_contract() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")
    assert KIND == "gdn_chunk_state_update"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.args_schema is GdnChunkStateUpdateArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env == "default_env"
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_chunk_state_update_torch"
    )
    assert spec.runner_ref.function_name == "profile_gdn_chunk_state_update"
    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.supports.kv is None and spec.supports.gpus is None
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")

    fused = find_kernel_profiler_spec(KIND, "vllm_triton")
    assert fused.kernel_kind == fused.table_name == KIND
    assert fused.backend == "vllm_triton"
    assert fused.args_schema is GdnChunkStateUpdateArgs
    assert fused.metric_family is MetricFamily.COMPUTE
    assert fused.batch_outlier_policy == BatchOutlierPolicy()
    assert fused.subprocess_env == "vllm_env"
    assert fused.runner_ref.module_name == (
        "profiling.runners.attention.gdn_chunk_state_update_vllm_triton"
    )
    assert fused.runner_ref.function_name == "profile_gdn_chunk_state_update_vllm_triton"
    assert fused.supports.compute == frozenset({DType.BF16})
    assert fused.supports.kv is None
    assert fused.supports.gpus == frozenset({"NVIDIA H200"})
    assert fused.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not fused.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not fused.supports.allows(DType.FP16, gpu="NVIDIA H200")


def test_registry_barrel_and_runner_reference_imports_are_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_state_update_torch' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_state_update_vllm_triton' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_state_update_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["False", "False", "False", "False"]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from profiling.db.registry import "
                "find_kernel_profiler_spec; runner = find_kernel_profiler_spec("
                "'gdn_chunk_state_update', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_state_update_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_chunk_state_update_torch",
        "profile_gdn_chunk_state_update",
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
                "'gdn_chunk_state_update', 'vllm_triton').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('vllm' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_state_update_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_chunk_state_update_vllm_triton",
        "profile_gdn_chunk_state_update_vllm_triton",
        "False",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "num_tokens",
        "num_chunks",
        "num_sequences",
        "max_chunks_per_sequence",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "value_head_dim",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_rejects_noninteger_values_before_torch_import(
    name: str, value: object, monkeypatch
) -> None:
    import profiling.runners.attention.gdn_chunk_state_update_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="exact integer"):
        runner.profile_gdn_chunk_state_update(**(_SPEC | {name: value}))


@pytest.mark.parametrize(
    "name",
    [
        "num_tokens",
        "num_chunks",
        "num_sequences",
        "max_chunks_per_sequence",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "value_head_dim",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_rejects_nonpositive_values_before_torch_import(name: str, value: int, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_state_update_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="must be > 0"):
        runner.profile_gdn_chunk_state_update(**(_SPEC | {name: value}))


@pytest.mark.parametrize("dtype", ["fp16", "fp32", "fp8_e4m3"])
def test_rejects_unsupported_dtype_before_torch_import(dtype: str, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_state_update_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="requires dtype=bf16"):
        runner.profile_gdn_chunk_state_update(**(_SPEC | {"dtype": dtype}))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"num_key_heads": 3}, "divisible"),
        ({"num_chunks": 1}, r"M\+N-1 <= C"),
        ({"num_chunks": 3}, r"C <= N\*M"),
        (
            {
                "num_tokens": 64,
                "num_chunks": 2,
                "num_sequences": 1,
                "max_chunks_per_sequence": 2,
            },
            r"64\*\(C-N\)\+N <= T",
        ),
        (
            {
                "num_tokens": 129,
                "num_chunks": 2,
                "num_sequences": 1,
                "max_chunks_per_sequence": 2,
            },
            r"T <= 64\*C",
        ),
    ],
)
def test_rejects_infeasible_domain_before_torch_import(
    updates: dict[str, object], message: str, monkeypatch
) -> None:
    import profiling.runners.attention.gdn_chunk_state_update_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=message):
        runner.profile_gdn_chunk_state_update(**(_SPEC | updates))


@pytest.mark.parametrize(
    ("tokens", "chunks", "sequences", "maximum", "counts", "lengths"),
    [
        (65, 2, 1, 2, (2,), (65,)),
        (128, 2, 1, 2, (2,), (128,)),
        (195, 6, 3, 4, (4, 1, 1), (193, 1, 1)),
        (384, 6, 3, 4, (4, 1, 1), (256, 64, 64)),
        (579, 12, 3, 4, (4, 4, 4), (193, 193, 193)),
        (768, 12, 3, 4, (4, 4, 4), (256, 256, 256)),
        (260, 8, 4, 2, (2, 2, 2, 2), (65, 65, 65, 65)),
        (448, 7, 4, 4, (4, 1, 1, 1), (256, 64, 64, 64)),
        (640, 10, 4, 4, (4, 2, 2, 2), (256, 128, 128, 128)),
    ],
)
def test_canonical_construction_and_metadata(
    tokens: int,
    chunks: int,
    sequences: int,
    maximum: int,
    counts: tuple[int, ...],
    lengths: tuple[int, ...],
) -> None:
    from profiling.runners.attention.gdn_chunk_state_update_torch import (
        _canonical_boundaries,
        _canonical_chunk_counts,
        _canonical_chunk_offsets,
        _canonical_index_pairs,
        _canonical_lengths,
    )

    actual_counts = _canonical_chunk_counts(chunks, sequences, maximum)
    actual_lengths = _canonical_lengths(tokens, chunks, sequences, maximum)
    assert actual_counts == counts
    assert actual_lengths == lengths
    assert len(actual_counts) == len(actual_lengths) == sequences
    assert all(count > 0 and count <= maximum for count in actual_counts)
    assert sum(actual_counts) == chunks and max(actual_counts) == maximum
    assert sum(actual_lengths) == tokens
    assert tuple(math.ceil(length / 64) for length in actual_lengths) == counts
    assert _canonical_boundaries(tokens, chunks, sequences, maximum) == (
        0,
        *tuple(accumulate_for_test(actual_lengths)),
    )
    assert _canonical_chunk_offsets(actual_counts) == (
        0,
        *tuple(accumulate_for_test(actual_counts)),
    )
    assert _canonical_index_pairs(actual_counts) == tuple(
        (sequence, local_chunk)
        for sequence, count in enumerate(actual_counts)
        for local_chunk in range(count)
    )


def accumulate_for_test(values: tuple[int, ...]):
    total = 0
    for value in values:
        total += value
        yield total


def test_operand_shapes_dtypes_and_exact_metadata() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_state_update_torch import (
        _build_operands,
        _operand_shapes,
        _validate_args,
    )

    args = _validate_args(130, 3, 2, 2, 2, 4, 3, 5, "bf16")
    shapes = _operand_shapes(args)
    assert shapes.k == (130, 2, 3)
    assert shapes.w == (130, 4, 3)
    assert shapes.u == (130, 4, 5)
    assert shapes.g_cumsum == (130, 4)
    assert shapes.initial_state == shapes.final_state == shapes.state == (2, 4, 5, 3)
    assert shapes.cu_seqlens == (3,)
    assert shapes.chunk_indices == (3, 2)
    assert shapes.chunk_offsets == (3,)
    assert shapes.h == (3, 4, 5, 3)
    assert shapes.v_new == (130, 4, 5)

    operands = _build_operands(torch, args, device=torch.device("cpu"))
    assert operands.chunk_counts == (2, 1)
    assert operands.lengths == (97, 33)
    assert operands.boundaries == (0, 97, 130)
    assert operands.cu_seqlens.tolist() == [0, 97, 130]
    assert operands.chunk_indices.tolist() == [[0, 0], [0, 1], [1, 0]]
    assert operands.chunk_offsets.tolist() == [0, 2, 3]
    assert operands.k.dtype is operands.w.dtype is operands.u.dtype is torch.bfloat16
    assert operands.h.dtype is operands.v_new.dtype is torch.bfloat16
    assert operands.g_cumsum.dtype is operands.initial_state.dtype is torch.float32
    assert operands.final_state.dtype is operands.workspaces.state.dtype is torch.float32
    assert operands.cu_seqlens.dtype is operands.chunk_indices.dtype is torch.int32
    assert operands.chunk_offsets.dtype is torch.int32
    for tensor in (
        operands.k,
        operands.w,
        operands.u,
        operands.g_cumsum,
        operands.initial_state,
        operands.cu_seqlens,
        operands.chunk_indices,
        operands.chunk_offsets,
        operands.h,
        operands.v_new,
        operands.final_state,
    ):
        assert tensor.is_contiguous()


def test_helper_matches_accepted_reference_with_signed_inputs_and_resets() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_state_update_reference import (
        gdn_chunk_state_update_reference,
    )
    from profiling.runners.attention.gdn_chunk_state_update_torch import (
        _build_operands,
        _state_update_into,
        _validate_args,
    )

    args = _validate_args(130, 3, 2, 2, 2, 4, 3, 5, "bf16")
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    actual = _state_update_into(torch, operands)
    expected = gdn_chunk_state_update_reference(
        operands.k,
        operands.w,
        operands.u,
        operands.g_cumsum,
        operands.initial_state,
        operands.cu_seqlens,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-2, atol=2e-5)


def test_helper_handles_ragged_3_65_2_and_global_chunk_order() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_state_update_reference import (
        gdn_chunk_state_update_reference,
    )
    from profiling.runners.attention.gdn_chunk_state_update_torch import (
        _build_operands,
        _state_update_into,
        _validate_args,
    )

    args = _validate_args(70, 4, 3, 2, 2, 4, 2, 2, "bf16")
    original = _build_operands(torch, args, device=torch.device("cpu"))
    boundaries = (0, 3, 68, 70)
    operands = replace(
        original,
        lengths=(3, 65, 2),
        chunk_counts=(1, 2, 1),
        boundaries=boundaries,
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32),
        chunk_indices=torch.tensor([[0, 0], [1, 0], [1, 1], [2, 0]], dtype=torch.int32),
        chunk_offsets=torch.tensor([0, 1, 3, 4], dtype=torch.int32),
    )
    actual = _state_update_into(torch, operands)
    expected = gdn_chunk_state_update_reference(
        operands.k,
        operands.w,
        operands.u,
        operands.g_cumsum,
        operands.initial_state,
        operands.cu_seqlens,
    )
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-2, atol=2e-5)
    assert actual[0].shape[0] == 4


def test_helper_preserves_three_distinct_bf16_rounding_sites() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_state_update_torch import (
        _build_operands,
        _state_update_into,
        _validate_args,
    )

    snapshot_args = _validate_args(1, 1, 1, 1, 1, 1, 1, 1, "bf16")
    snapshot = _build_operands(torch, snapshot_args, device=torch.device("cpu"))
    snapshot.k.zero_()
    snapshot.w.fill_(0.5)
    snapshot.u.fill_(0.25)
    snapshot.g_cumsum.zero_()
    snapshot.initial_state.fill_(0.99)
    h, v_new, _ = _state_update_into(torch, snapshot)
    assert h[0, 0, 0, 0] == torch.tensor(0.99).to(torch.bfloat16)
    assert v_new[0, 0, 0] == torch.tensor(-0.244140625, dtype=torch.bfloat16)
    wrong_snapshot = (snapshot.u.float() - snapshot.w.float() * 0.99).to(torch.bfloat16)
    assert v_new[0, 0, 0] != wrong_snapshot[0, 0, 0]

    factor_args = _validate_args(2, 1, 1, 1, 1, 1, 1, 1, "bf16")
    factor = _build_operands(torch, factor_args, device=torch.device("cpu"))
    factor.k.copy_(torch.tensor([[[1]], [[0]]], dtype=torch.bfloat16))
    factor.w.copy_(torch.tensor([[[-3.9375]], [[0]]], dtype=torch.bfloat16))
    factor.u.copy_(torch.tensor([[[-4]], [[0]]], dtype=torch.bfloat16))
    factor.g_cumsum.copy_(torch.tensor([[0.0], [1.2]], dtype=torch.float32))
    factor.initial_state.fill_(-3.9375)
    _, stored_v, final = _state_update_into(torch, factor)
    residual = factor.u[0, 0, 0].float() - factor.w[0, 0, 0].float() * -3.9375
    decay = torch.exp(factor.g_cumsum[1, 0] - factor.g_cumsum[0, 0])
    correct_factor = (residual * decay).to(torch.bfloat16).float()
    reused_v = (stored_v[0, 0, 0].float() * decay).to(torch.bfloat16).float()
    assert correct_factor != reused_v and correct_factor != residual * decay
    expected_final = -3.9375 * torch.exp(factor.g_cumsum[1, 0]) + correct_factor
    assert final[0, 0, 0, 0] == expected_final


def test_repeated_helper_calls_overwrite_outputs_state_and_workspaces() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_state_update_torch import (
        _build_operands,
        _state_update_into,
        _validate_args,
    )

    args = _validate_args(70, 4, 3, 2, 2, 4, 3, 2, "bf16")
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    inputs = (
        operands.k,
        operands.w,
        operands.u,
        operands.g_cumsum,
        operands.initial_state,
        operands.cu_seqlens,
        operands.chunk_indices,
        operands.chunk_offsets,
    )
    snapshots = tuple(tensor.clone() for tensor in inputs)
    first = tuple(tensor.clone() for tensor in _state_update_into(torch, operands))
    for tensor in (operands.h, operands.v_new, operands.final_state):
        tensor.fill_(11)
    for name, tensor in vars(operands.workspaces).items():
        if name != "head_to_key":
            tensor.fill_(13)
    second = _state_update_into(torch, operands)
    for result, expected in zip(second, first):
        assert torch.equal(result, expected)
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(tensor, snapshot)


def test_outputs_are_contiguous_fresh_and_non_aliasing() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_state_update_torch import (
        _build_operands,
        _state_update_into,
        _validate_args,
    )

    operands = _build_operands(
        torch,
        _validate_args(70, 4, 3, 2, 2, 4, 3, 2, "bf16"),
        device=torch.device("cpu"),
    )
    outputs = _state_update_into(torch, operands)
    assert [tensor.dtype for tensor in outputs] == [
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
    ]
    assert all(tensor.is_contiguous() and torch.isfinite(tensor).all() for tensor in outputs)
    output_pointers = {tensor.untyped_storage().data_ptr() for tensor in outputs}
    input_pointers = {
        tensor.untyped_storage().data_ptr()
        for tensor in (
            operands.k,
            operands.w,
            operands.u,
            operands.g_cumsum,
            operands.initial_state,
            operands.cu_seqlens,
            operands.chunk_indices,
            operands.chunk_offsets,
        )
    }
    assert len(output_pointers) == 3
    assert not output_pointers.intersection(input_pointers)


def test_runner_reports_missing_cuda_as_typed_unsupported() -> None:
    from profiling.runners.attention.gdn_chunk_state_update_torch import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)


def test_semantic_flops_and_logical_bytes_are_exact() -> None:
    from profiling.runners.attention.gdn_chunk_state_update_torch import (
        _logical_bytes,
        _semantic_flops,
    )

    small = _semantic_flops(
        num_tokens=3,
        num_chunks=1,
        num_sequences=1,
        max_chunks_per_sequence=1,
        num_heads=2,
        key_head_dim=2,
        value_head_dim=3,
    )
    assert small == 2 * (4 * 3 * 2 * 3 + 3 * 3 + 2 * 3 + 2 * 3 + 1)
    assert (
        _semantic_flops(
            num_tokens=128,
            num_chunks=2,
            num_sequences=1,
            max_chunks_per_sequence=2,
            num_heads=32,
            key_head_dim=128,
            value_head_dim=128,
        )
        == 270_016_576
    )
    assert (
        _logical_bytes(
            num_tokens=128,
            num_chunks=2,
            num_sequences=1,
            num_key_heads=16,
            num_heads=32,
            key_head_dim=128,
            value_head_dim=128,
        )
        == 9_977_856
    )


def test_profile_times_only_preallocated_semantics_with_internal_reset(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_state_update_torch as runner

    args = runner._validate_args(70, 4, 3, 2, 2, 4, 3, 2, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    calls = {"build": 0, "semantic": 0, "timer": 0, "energy": 0}
    original_semantic = runner._state_update_into

    def fake_build(*_args, **_kwargs):
        calls["build"] += 1
        return operands

    def counted_semantic(torch_module, selected_operands):
        calls["semantic"] += 1
        # This corruption proves the measured helper owns its reset copy.
        selected_operands.workspaces.state.fill_(99)
        return original_semantic(torch_module, selected_operands)

    def fake_timer(fn):
        calls["timer"] += 1
        result = fn()
        assert result == (operands.h, operands.v_new, operands.final_state)
        return 0.5

    def fake_energy(fn, *, warmup, per_iter_time_ms):
        calls["energy"] += 1
        first = tuple(
            tensor.clone() for tensor in (operands.h, operands.v_new, operands.final_state)
        )
        fn()
        assert all(
            torch.equal(left, right)
            for left, right in zip(first, (operands.h, operands.v_new, operands.final_state))
        )
        assert warmup == 5 and per_iter_time_ms == 0.5
        return 0.25

    monkeypatch.setattr(runner, "_validate_cuda_device", lambda _torch: None)
    monkeypatch.setattr(runner, "_build_operands", fake_build)
    monkeypatch.setattr(runner, "_state_update_into", counted_semantic)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))

    metrics = runner.profile_gdn_chunk_state_update(**_SPEC)
    assert metrics.time_ms == 0.5 and metrics.energy_j == 0.25
    assert metrics.tflops > 0 and metrics.memory_bandwidth_gbps > 0
    assert calls == {"build": 1, "semantic": 2, "timer": 1, "energy": 1}


def test_generated_facades_and_read_only_missing_query(tmp_path, monkeypatch) -> None:
    assert hasattr(perf_api, "get_gdn_chunk_state_update_times")
    assert hasattr(perf_api, "count_missing_gdn_chunk_state_update")
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")
    assert (
        perf_api.count_missing_gdn_chunk_state_update(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 1
    )
    result = perf_api.get_gdn_chunk_state_update_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(result, MissingEntry)
    assert result.args == coerce_args(GdnChunkStateUpdateArgs, _SPEC)
    assert not perf_api.DB_PATH.exists()

    table = Table(find_kernel_profiler_spec(KIND, "torch"), perf_api.DB_PATH)
    assert table.args_columns == [
        "num_tokens",
        "num_chunks",
        "num_sequences",
        "max_chunks_per_sequence",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "value_head_dim",
        "dtype",
    ]


def test_db_round_trip_uses_exact_schema_and_compute_metrics(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "profile.db"
    monkeypatch.setattr(perf_api, "DB_PATH", db_path)
    profiler_spec = find_kernel_profiler_spec(KIND, "torch")
    args = coerce_args(GdnChunkStateUpdateArgs, _SPEC)
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
    assert perf_api.get_gdn_chunk_state_update_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    ) == [metrics]
    assert (
        perf_api.count_missing_gdn_chunk_state_update(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 0
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"num_tokens": 0}, "must be > 0"),
        ({"num_chunks": 1}, r"M\+N-1 <= C"),
        ({"num_chunks": 3}, r"C <= N\*M"),
        ({"num_tokens": 64}, r"64\*\(C-N\)\+N <= T"),
        ({"num_tokens": 129}, r"T <= 64\*C"),
        ({"num_key_heads": 3}, "divisible"),
        ({"key_head_dim": 64}, "key_head_dim=value_head_dim=128"),
        ({"value_head_dim": 64}, "key_head_dim=value_head_dim=128"),
        ({"dtype": "fp16"}, "requires dtype=bf16"),
    ],
)
def test_vllm_rejects_invalid_args_before_import(
    updates: dict[str, object], message: str, monkeypatch
) -> None:
    import profiling.runners.attention.gdn_chunk_state_update_vllm_triton as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=message):
        runner.profile_gdn_chunk_state_update_vllm_triton(**(_SPEC | updates))


@pytest.mark.parametrize(
    "name",
    [
        "num_tokens",
        "num_chunks",
        "num_sequences",
        "max_chunks_per_sequence",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "value_head_dim",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_vllm_rejects_noninteger_args_before_import(name: str, value: object, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_state_update_vllm_triton as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="exact integer"):
        runner.profile_gdn_chunk_state_update_vllm_triton(**(_SPEC | {name: value}))


@pytest.mark.parametrize("name", ["FLA_USE_FAST_OPS", "FLA_USE_CUDA_GRAPH"])
@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_vllm_rejects_truthy_environment_before_import(name: str, value: str, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_state_update_vllm_triton as runner

    monkeypatch.setenv(name, value)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=name):
        runner.profile_gdn_chunk_state_update_vllm_triton(**_SPEC)


@pytest.mark.parametrize("value", [None, "", "0", "false", "False"])
def test_vllm_accepts_safe_environment(value: str | None, monkeypatch) -> None:
    from profiling.runners.attention.gdn_chunk_state_update_vllm_triton import (
        _validate_args,
    )

    for name in ("FLA_USE_FAST_OPS", "FLA_USE_CUDA_GRAPH"):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert _validate_args(**_SPEC).num_tokens == 128


def test_vllm_shapes_packed_metadata_and_deterministic_operands() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_state_update_vllm_triton as runner

    args = runner._validate_args(130, 3, 2, 2, 2, 4, 128, 128, "bf16")
    shapes = runner._operand_shapes(args)
    assert shapes.k == (1, 130, 2, 128)
    assert shapes.w == shapes.u == (1, 130, 4, 128)
    assert shapes.g_cumsum == (1, 130, 4)
    assert shapes.initial_state == shapes.final_state == (2, 4, 128, 128)
    assert shapes.cu_seqlens == shapes.chunk_offsets == (3,)
    assert shapes.chunk_indices == (3, 2)
    assert shapes.h == (1, 3, 4, 128, 128)
    assert shapes.v_new == (1, 130, 4, 128)

    first = runner._build_operands(torch, args, device=torch.device("cpu"))
    second = runner._build_operands(torch, args, device=torch.device("cpu"))
    runner._validate_operands(torch, first, args, require_cuda=False)
    assert first.chunk_counts == (2, 1)
    assert first.lengths == (97, 33)
    assert first.boundaries == (0, 97, 130)
    assert first.cu_seqlens.tolist() == [0, 97, 130]
    assert first.chunk_indices.tolist() == [[0, 0], [0, 1], [1, 0]]
    assert first.chunk_offsets.tolist() == [0, 2, 3]
    for name in (
        "k",
        "w",
        "u",
        "g_cumsum",
        "initial_state",
        "cu_seqlens",
        "chunk_indices",
        "chunk_offsets",
    ):
        tensor = getattr(first, name)
        assert tensor.is_contiguous()
        assert tensor.stride() == runner._packed_stride(tuple(tensor.shape))
        assert torch.equal(tensor, getattr(second, name))


def test_vllm_guard_geometry_preserves_qwen_uses_static_witness_and_rejects() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_state_update_vllm_triton as runner

    qwen = runner._validate_args(**_SPEC)
    assert runner._guard_args(qwen) is qwen
    assert runner._guard_elements(qwen) == 3_936_256

    large = runner._validate_args(4096, 64, 8, 16, 16, 32, 128, 128, "bf16")
    guard = runner._guard_args(large)
    assert guard == replace(
        large,
        num_tokens=128,
        num_chunks=2,
        num_sequences=1,
        max_chunks_per_sequence=2,
    )
    assert (
        guard.num_key_heads,
        guard.num_heads,
        guard.key_head_dim,
        guard.value_head_dim,
        guard.dtype,
    ) == (
        large.num_key_heads,
        large.num_heads,
        large.key_head_dim,
        large.value_head_dim,
        large.dtype,
    )
    assert runner._guard_elements(guard) <= runner._MAX_GUARD_ELEMENTS
    assert runner._canonical_lengths(128, 2, 1, 2) == (128,)
    witness = runner._build_operands(torch, guard, device=torch.device("cpu"))
    runner._validate_operands(torch, witness, guard, require_cuda=False)
    assert witness.chunk_counts == (2,)
    assert witness.lengths == (128,)
    assert witness.boundaries == (0, 128)
    assert witness.chunk_indices.tolist() == [[0, 0], [0, 1]]
    assert witness.chunk_offsets.tolist() == [0, 2]
    assert guard.num_heads // guard.num_key_heads == 2
    for tensor in (witness.k, witness.w, witness.u, witness.initial_state):
        assert torch.count_nonzero(tensor) > 0

    measured_counts = runner._canonical_chunk_counts(64, 8, 16)
    measured_lengths = runner._canonical_lengths(4096, 64, 8, 16)
    assert measured_counts == (16, 7, 7, 7, 7, 7, 7, 6)
    assert measured_lengths == tuple(64 * count for count in measured_counts)
    assert runner._canonical_chunk_offsets(measured_counts) == (0, 16, 23, 30, 37, 44, 51, 58, 64)
    assert len(runner._canonical_index_pairs(measured_counts)) == 64

    oversized_static = runner._validate_args(128, 2, 1, 2, 32, 64, 128, 128, "bf16")
    with pytest.raises(ValueError, match="static Hg, H, K, V, and dtype"):
        runner._guard_args(oversized_static)


def test_vllm_operand_validation_rejects_layout_metadata_and_finiteness() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_state_update_vllm_triton as runner

    args = runner._validate_args(3, 1, 1, 1, 2, 4, 128, 128, "bf16")

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.cu_seqlens[-1] = 2
    with pytest.raises(ValueError, match="canonical boundaries"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.chunk_indices[0, 1] = 1
    with pytest.raises(ValueError, match="canonical mapping"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.chunk_offsets[-1] = 0
    with pytest.raises(ValueError, match="canonical offsets"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.g_cumsum[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    object.__setattr__(operands, "w", operands.w.float())
    with pytest.raises(ValueError, match="dtype"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    backing = torch.empty((1, 3, 2, 256), dtype=torch.bfloat16)
    object.__setattr__(operands, "k", backing[..., ::2])
    with pytest.raises(ValueError, match="packed contiguous"):
        runner._validate_operands(torch, operands, args, require_cuda=False)


def test_vllm_correctness_guard_checks_three_outputs_and_immutability() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_state_update_vllm_triton as runner
    from profiling.runners.attention.gdn_chunk_state_update_reference import (
        gdn_chunk_state_update_reference,
    )

    args = runner._validate_args(3, 1, 1, 1, 2, 4, 128, 128, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    names = (
        "k",
        "w",
        "u",
        "g_cumsum",
        "initial_state",
        "cu_seqlens",
        "chunk_indices",
        "chunk_offsets",
    )
    snapshots = {name: getattr(operands, name).clone() for name in names}
    calls = 0

    def fake_fused(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["gk"] is None
        assert kwargs["output_final_state"] and kwargs["save_new_value"]
        assert not kwargs["use_exp2"] and kwargs["chunk_size"] == 64
        expected = gdn_chunk_state_update_reference(
            kwargs["k"].squeeze(0),
            kwargs["w"].squeeze(0),
            kwargs["u"].squeeze(0),
            kwargs["g"].squeeze(0),
            kwargs["initial_state"],
            kwargs["cu_seqlens"],
        )
        return expected[0].unsqueeze(0), expected[1].unsqueeze(0), expected[2]

    runner._check_correctness(torch, fake_fused, operands, args, synchronize=lambda: None)
    assert calls == 1
    for name, snapshot in snapshots.items():
        assert torch.equal(getattr(operands, name), snapshot)


def test_vllm_correctness_guard_rejects_wrong_output_properties() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_state_update_vllm_triton as runner

    args = runner._validate_args(3, 1, 1, 1, 2, 4, 128, 128, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))

    def wrong_dtype(**_kwargs):
        return (
            torch.empty((1, 1, 4, 128, 128), dtype=torch.float32),
            torch.empty((1, 3, 4, 128), dtype=torch.bfloat16),
            torch.empty((1, 4, 128, 128), dtype=torch.float32),
        )

    with pytest.raises(AssertionError, match="unexpected h dtype"):
        runner._check_correctness(
            torch,
            wrong_dtype,
            operands,
            args,
            synchronize=lambda: None,
        )


def test_vllm_times_one_wrapper_and_reuses_torch_metrics(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_state_update_vllm_triton as runner
    from profiling.runners.attention.gdn_chunk_state_update_torch import (
        _logical_bytes,
        _semantic_flops,
    )

    args = runner._validate_args(3, 1, 1, 1, 2, 4, 128, 128, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    calls = {"build": 0, "guard": 0, "fused": 0, "timer": 0, "energy": 0}
    built_args = []

    def fake_build(_torch, selected_args, **_kwargs):
        calls["build"] += 1
        built_args.append(selected_args)
        return operands

    def fake_fused(**_kwargs):
        calls["fused"] += 1
        return (
            torch.empty((1, 1, 4, 128, 128), dtype=torch.bfloat16),
            torch.empty((1, 3, 4, 128), dtype=torch.bfloat16),
            torch.empty((1, 4, 128, 128), dtype=torch.float32),
        )

    def fake_timer(fn, *, kernel_name):
        calls["timer"] += 1
        assert kernel_name == "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
        fn()
        return 0.5

    def fake_energy(fn, *, warmup, per_iter_time_ms):
        calls["energy"] += 1
        assert warmup == 5 and per_iter_time_ms == 0.5
        fn()
        return 0.25

    monkeypatch.setattr(runner, "_require_h200", lambda _torch: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(runner, "_load_fused_callable", lambda: fake_fused)
    monkeypatch.setattr(runner, "_build_operands", fake_build)
    monkeypatch.setattr(runner, "_validate_operands", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner, "_check_correctness", lambda *_args, **_kwargs: calls.__setitem__("guard", 1)
    )
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))

    metrics = runner.profile_gdn_chunk_state_update_vllm_triton(3, 1, 1, 1, 2, 4, 128, 128, "bf16")
    assert calls == {"build": 2, "guard": 1, "fused": 2, "timer": 1, "energy": 1}
    assert (built_args[0].num_tokens, built_args[0].num_chunks) == (3, 1)
    assert (
        built_args[1].num_tokens,
        built_args[1].num_chunks,
        built_args[1].num_sequences,
        built_args[1].max_chunks_per_sequence,
    ) == (128, 2, 1, 2)
    assert built_args[1].num_key_heads == built_args[0].num_key_heads == 2
    assert built_args[1].num_heads == built_args[0].num_heads == 4
    assert metrics.time_ms == 0.5 and metrics.energy_j == 0.25
    expected_flops = _semantic_flops(
        num_tokens=3,
        num_chunks=1,
        num_sequences=1,
        max_chunks_per_sequence=1,
        num_heads=4,
        key_head_dim=128,
        value_head_dim=128,
    )
    expected_bytes = _logical_bytes(
        num_tokens=3,
        num_chunks=1,
        num_sequences=1,
        num_key_heads=2,
        num_heads=4,
        key_head_dim=128,
        value_head_dim=128,
    )
    assert metrics.tflops == expected_flops / 0.0005 / 1e12
    assert metrics.memory_bandwidth_gbps == expected_bytes / 0.0005 / 1e9


def test_generated_facades_preserve_both_backend_queries(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")
    for backend in ("torch", "vllm_triton"):
        assert (
            perf_api.count_missing_gdn_chunk_state_update(
                [_SPEC], backend=backend, gpu_name="NVIDIA H200"
            )
            == 1
        )
        result = perf_api.get_gdn_chunk_state_update_times(
            [_SPEC], backend=backend, gpu_name="NVIDIA H200"
        )[0]
        assert isinstance(result, MissingEntry)
    assert not perf_api.DB_PATH.exists()
