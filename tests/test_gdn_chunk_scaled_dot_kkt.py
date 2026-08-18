"""Registration and Torch runner tests for ``gdn_chunk_scaled_dot_kkt``."""

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
from profiling.kernels.gdn_chunk_scaled_dot_kkt import (
    KIND,
    GdnChunkScaledDotKktArgs,
)
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SPEC = {
    "num_tokens": 128,
    "num_chunks": 2,
    "num_key_heads": 16,
    "num_heads": 32,
    "key_head_dim": 128,
    "dtype": "bf16",
}


def test_args_field_order_and_coercion() -> None:
    assert [field.name for field in fields(GdnChunkScaledDotKktArgs)] == [
        "num_tokens",
        "num_chunks",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "dtype",
    ]
    args = coerce_args(
        GdnChunkScaledDotKktArgs,
        {
            "num_tokens": "128",
            "num_chunks": "2",
            "num_key_heads": "16",
            "num_heads": "32",
            "key_head_dim": "128",
            "dtype": "torch.bfloat16",
        },
    )
    assert args == GdnChunkScaledDotKktArgs(
        num_tokens=128,
        num_chunks=2,
        num_key_heads=16,
        num_heads=32,
        key_head_dim=128,
        dtype=DType.BF16,
    )
    with pytest.raises(Exception):
        args.num_tokens = 1


def test_registration_table_kind_runner_and_support_contract() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")

    assert KIND == "gdn_chunk_scaled_dot_kkt"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.args_schema is GdnChunkScaledDotKktArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env == "default_env"
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch"
    )
    assert spec.runner_ref.function_name == "profile_gdn_chunk_scaled_dot_kkt"

    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.supports.kv is None
    assert spec.supports.gpus is None
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA H200")

    fused = find_kernel_profiler_spec(KIND, "vllm_triton")
    assert fused.kernel_kind == fused.table_name == KIND
    assert fused.args_schema is GdnChunkScaledDotKktArgs
    assert fused.metric_family is MetricFamily.COMPUTE
    assert fused.batch_outlier_policy == BatchOutlierPolicy()
    assert fused.subprocess_env == "vllm_env"
    assert fused.runner_ref.module_name == (
        "profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton"
    )
    assert fused.runner_ref.function_name == ("profile_gdn_chunk_scaled_dot_kkt_vllm_triton")
    assert fused.supports.compute == frozenset({DType.BF16})
    assert fused.supports.kv is None
    assert fused.supports.gpus == frozenset({"NVIDIA H200"})
    assert fused.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not fused.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not fused.supports.allows(DType.FP16, gpu="NVIDIA H200")


def test_registry_barrel_and_runner_ref_are_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch' "
                "in sys.modules); "
                "print('profiling.runners.attention."
                "gdn_chunk_scaled_dot_kkt_vllm_triton' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_scaled_dot_kkt_reference' "
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
                "'gdn_chunk_scaled_dot_kkt', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_scaled_dot_kkt_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch",
        "profile_gdn_chunk_scaled_dot_kkt",
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
                "'gdn_chunk_scaled_dot_kkt', 'vllm_triton').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('vllm' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_scaled_dot_kkt_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton",
        "profile_gdn_chunk_scaled_dot_kkt_vllm_triton",
        "False",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    "name",
    ["num_tokens", "num_chunks", "num_key_heads", "num_heads", "key_head_dim"],
)
@pytest.mark.parametrize("value", [0, -1])
def test_runner_rejects_nonpositive_values_before_import(
    name: str,
    value: int,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="must be > 0"):
        runner.profile_gdn_chunk_scaled_dot_kkt(**(_SPEC | {name: value}))


@pytest.mark.parametrize("dtype", ["fp16", "fp32", "fp8_e4m3"])
def test_runner_rejects_unsupported_dtype_before_import(dtype: str, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="requires dtype=bf16"):
        runner.profile_gdn_chunk_scaled_dot_kkt(**(_SPEC | {"dtype": dtype}))


@pytest.mark.parametrize(
    ("num_tokens", "num_chunks"),
    [(65, 1), (128, 1), (129, 2), (4, 5)],
)
def test_runner_rejects_infeasible_chunk_counts_before_import(
    num_tokens: int,
    num_chunks: int,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=r"ceil\(num_tokens/64\)"):
        runner.profile_gdn_chunk_scaled_dot_kkt(
            **(_SPEC | {"num_tokens": num_tokens, "num_chunks": num_chunks})
        )


def test_runner_rejects_incompatible_head_grouping_before_import(monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="divisible by num_key_heads"):
        runner.profile_gdn_chunk_scaled_dot_kkt(**(_SPEC | {"num_key_heads": 3}))


@pytest.mark.parametrize(
    "name",
    ["num_tokens", "num_chunks", "num_key_heads", "num_heads", "key_head_dim"],
)
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_runner_rejects_non_integer_values_before_import(
    name: str,
    value: object,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="exact integer"):
        runner.profile_gdn_chunk_scaled_dot_kkt(**(_SPEC | {name: value}))


@pytest.mark.parametrize(
    ("num_tokens", "num_chunks", "expected"),
    [
        (1, 1, (1,)),
        (64, 1, (64,)),
        (65, 2, (33, 32)),
        (128, 2, (64, 64)),
        (82, 3, (28, 27, 27)),
        (5, 5, (1, 1, 1, 1, 1)),
    ],
)
def test_canonical_partition_invariants(
    num_tokens: int,
    num_chunks: int,
    expected: tuple[int, ...],
) -> None:
    from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch import (
        _canonical_boundaries,
        _canonical_lengths,
    )

    lengths = _canonical_lengths(num_tokens, num_chunks)
    boundaries = _canonical_boundaries(num_tokens, num_chunks)
    assert lengths == expected
    assert len(lengths) == num_chunks
    assert sum(lengths) == num_tokens
    assert all(1 <= length <= 64 for length in lengths)
    assert boundaries[0] == 0 and boundaries[-1] == num_tokens
    assert tuple(right - left for left, right in zip(boundaries, boundaries[1:])) == lengths
    assert sum((length + 63) // 64 for length in lengths) == num_chunks


def test_derived_shapes_dtypes_metadata_and_deterministic_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch import (
        _build_operands,
        _operand_shapes,
        _validate_args,
    )

    args = _validate_args(82, 3, 2, 4, 8, "bf16")
    shapes = _operand_shapes(args)
    assert shapes.k == (82, 2, 8)
    assert shapes.beta == shapes.g_cumsum == (82, 4)
    assert shapes.cu_seqlens == (4,)
    assert shapes.output == (82, 4, 64)
    assert shapes.expanded_key == shapes.scaled_key == (28, 4, 8)
    assert shapes.square == (4, 28, 28)

    first = _build_operands(torch, args, device=torch.device("cpu"))
    second = _build_operands(torch, args, device=torch.device("cpu"))
    assert first.k.dtype is torch.bfloat16
    assert first.beta.dtype is first.g_cumsum.dtype is torch.float32
    assert first.cu_seqlens.dtype is torch.int32
    assert first.output.dtype is torch.float32
    assert first.boundaries == (0, 28, 55, 82)
    assert tuple(first.cu_seqlens.tolist()) == first.boundaries
    assert first.workspaces.head_to_key.tolist() == [0, 0, 1, 1]
    assert torch.equal(first.k, second.k)
    assert torch.equal(first.beta, second.beta)
    assert torch.equal(first.g_cumsum, second.g_cumsum)
    assert torch.isfinite(first.k.float()).all()
    assert torch.isfinite(first.beta).all()
    assert torch.isfinite(first.g_cumsum).all()


def test_device_generic_helper_agrees_with_accepted_reference() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_reference import (
        gdn_chunk_scaled_dot_kkt_reference,
    )
    from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch import (
        _build_operands,
        _scaled_dot_kkt_into,
        _validate_args,
    )

    args = _validate_args(70, 2, 2, 4, 5, "bf16")
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    snapshots = (
        operands.k.clone(),
        operands.beta.clone(),
        operands.g_cumsum.clone(),
        operands.cu_seqlens.clone(),
    )
    expected = gdn_chunk_scaled_dot_kkt_reference(
        operands.k,
        operands.beta,
        operands.g_cumsum,
        operands.cu_seqlens,
    )

    actual = _scaled_dot_kkt_into(
        torch,
        operands.k,
        operands.beta,
        operands.g_cumsum,
        operands.output,
        operands.boundaries,
        operands.workspaces,
    )

    assert actual is operands.output
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(actual[0]).item() == 0
    assert torch.count_nonzero(actual[35]).item() == 0
    assert torch.count_nonzero(actual[1, :, 0]).item() == 4
    for original, snapshot in zip(
        (operands.k, operands.beta, operands.g_cumsum, operands.cu_seqlens),
        snapshots,
        strict=True,
    ):
        assert torch.equal(original, snapshot)


def test_helper_grouped_heads_row_beta_positive_sign_and_reset_placement() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch import (
        _build_operands,
        _scaled_dot_kkt_into,
        _validate_args,
    )

    args = _validate_args(4, 2, 2, 4, 1, "bf16")
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    operands.k.zero_()
    operands.k[0:2, 0, 0] = 1
    operands.k[2:4, 1, 0] = 1
    operands.beta.copy_(
        torch.tensor(
            [[1, 1, 1, 1], [2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]],
            dtype=torch.float32,
        )
    )
    operands.g_cumsum.zero_()

    actual = _scaled_dot_kkt_into(
        torch,
        operands.k,
        operands.beta,
        operands.g_cumsum,
        operands.output,
        operands.boundaries,
        operands.workspaces,
    )

    assert actual[1, 0, 0].item() == 2
    assert actual[1, 1, 0].item() == 3
    assert actual[1, 2, 0].item() == 0
    assert torch.count_nonzero(actual[2]).item() == 0
    assert actual[3, 2, 0].item() == 12
    assert actual[3, 3, 0].item() == 13
    assert actual[3, 0, 0].item() == 0


def test_runner_reports_missing_cuda_as_typed_unsupported() -> None:
    from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)


def test_semantic_metrics_are_exact_logical_counts() -> None:
    from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch import (
        _logical_bytes,
        _semantic_flops,
    )

    assert (
        _semantic_flops(
            num_tokens=3,
            num_chunks=1,
            num_heads=2,
            key_head_dim=4,
        )
        == 84
    )
    assert (
        _semantic_flops(
            num_tokens=128,
            num_chunks=2,
            num_heads=32,
            key_head_dim=128,
        )
        == 33_812_480
    )
    assert (
        _logical_bytes(
            num_tokens=128,
            num_key_heads=16,
            num_heads=32,
            key_head_dim=128,
        )
        == 1_605_632
    )


def test_profile_times_only_preallocated_semantics_and_needs_no_reset(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch as runner

    args = runner._validate_args(6, 2, 2, 4, 3, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    input_snapshots = (
        operands.k.clone(),
        operands.beta.clone(),
        operands.g_cumsum.clone(),
        operands.cu_seqlens.clone(),
    )
    output_ptr = operands.output.data_ptr()
    calls = {"build": 0, "semantic": 0, "timer": 0, "energy": 0}
    original_semantic = runner._scaled_dot_kkt_into

    def fake_build(*_args, **_kwargs):
        calls["build"] += 1
        return operands

    def counted_semantic(*positional, **keyword):
        calls["semantic"] += 1
        return original_semantic(*positional, **keyword)

    def fake_timer(fn):
        calls["timer"] += 1
        assert fn() is operands.output
        return 0.5

    def fake_energy(fn, *, warmup, per_iter_time_ms):
        calls["energy"] += 1
        first = operands.output.clone()
        assert fn() is operands.output
        assert torch.equal(operands.output, first)
        assert warmup == 5 and per_iter_time_ms == 0.5
        return 0.25

    monkeypatch.setattr(runner, "_validate_cuda_device", lambda _torch: None)
    monkeypatch.setattr(runner, "_build_operands", fake_build)
    monkeypatch.setattr(runner, "_scaled_dot_kkt_into", counted_semantic)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))

    metrics = runner.profile_gdn_chunk_scaled_dot_kkt(6, 2, 2, 4, 3, "bf16")
    assert metrics.time_ms == 0.5 and metrics.energy_j == 0.25
    assert calls == {"build": 1, "semantic": 2, "timer": 1, "energy": 1}
    assert operands.output.data_ptr() == output_ptr
    for original, snapshot in zip(
        (operands.k, operands.beta, operands.g_cumsum, operands.cu_seqlens),
        input_snapshots,
        strict=True,
    ):
        assert torch.equal(original, snapshot)


def test_generated_facades_and_read_only_missing_query(tmp_path, monkeypatch) -> None:
    assert hasattr(perf_api, "get_gdn_chunk_scaled_dot_kkt_times")
    assert hasattr(perf_api, "count_missing_gdn_chunk_scaled_dot_kkt")
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")

    assert (
        perf_api.count_missing_gdn_chunk_scaled_dot_kkt(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 1
    )
    result = perf_api.get_gdn_chunk_scaled_dot_kkt_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(result, MissingEntry)
    assert result.args == coerce_args(GdnChunkScaledDotKktArgs, _SPEC)
    assert not perf_api.DB_PATH.exists()

    table = Table(find_kernel_profiler_spec(KIND, "torch"), perf_api.DB_PATH)
    assert table.args_columns == [
        "num_tokens",
        "num_chunks",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "dtype",
    ]


def test_db_round_trip_uses_exact_schema_and_compute_metrics(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "profile.db"
    monkeypatch.setattr(perf_api, "DB_PATH", db_path)
    profiler_spec = find_kernel_profiler_spec(KIND, "torch")
    args = coerce_args(GdnChunkScaledDotKktArgs, _SPEC)
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

    result = perf_api.get_gdn_chunk_scaled_dot_kkt_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert result == metrics
    assert (
        perf_api.count_missing_gdn_chunk_scaled_dot_kkt(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 0
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"num_tokens": 0}, "must be > 0"),
        ({"num_chunks": 1}, r"ceil\(num_tokens/64\)"),
        ({"num_key_heads": 3}, "divisible by num_key_heads"),
        ({"key_head_dim": 32}, "key_head_dim"),
        ({"key_head_dim": 256}, "key_head_dim"),
        ({"dtype": "fp16"}, "requires dtype=bf16"),
    ],
)
def test_vllm_runner_rejects_invalid_args_before_torch_import(
    updates: dict[str, object],
    message: str,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=message):
        runner.profile_gdn_chunk_scaled_dot_kkt_vllm_triton(**(_SPEC | updates))


@pytest.mark.parametrize("value", ["1", "true", "false", "yes"])
def test_vllm_runner_rejects_truthy_fast_ops_before_import(value: str, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton as runner

    monkeypatch.setenv("FLA_USE_FAST_OPS", value)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="FLA_USE_FAST_OPS"):
        runner.profile_gdn_chunk_scaled_dot_kkt_vllm_triton(**_SPEC)


@pytest.mark.parametrize("value", [None, "", "0"])
def test_vllm_runner_accepts_frozen_fast_ops_settings(value: str | None, monkeypatch) -> None:
    from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton import (
        _validate_args,
    )

    if value is None:
        monkeypatch.delenv("FLA_USE_FAST_OPS", raising=False)
    else:
        monkeypatch.setenv("FLA_USE_FAST_OPS", value)
    assert _validate_args(128, 2, 16, 32, 128, "bf16").key_head_dim == 128


def test_vllm_derived_shapes_metadata_and_deterministic_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton import (
        _build_operands,
        _canonical_index_pairs,
        _operand_shapes,
        _validate_args,
        _validate_operands,
    )

    args = _validate_args(82, 3, 4, 8, 64, "bf16")
    shapes = _operand_shapes(args)
    assert shapes.k == (1, 82, 4, 64)
    assert shapes.beta == shapes.g_cumsum == (1, 82, 8)
    assert shapes.cu_seqlens == (4,)
    assert shapes.chunk_indices == (3, 2)
    assert shapes.output == (1, 82, 8, 64)
    assert _canonical_index_pairs(3) == ((0, 0), (1, 0), (2, 0))

    first = _build_operands(torch, args, device=torch.device("cpu"))
    second = _build_operands(torch, args, device=torch.device("cpu"))
    _validate_operands(torch, first, args, require_cuda=False)
    assert first.boundaries == (0, 28, 55, 82)
    assert first.cu_seqlens.tolist() == [0, 28, 55, 82]
    assert first.chunk_indices.tolist() == [[0, 0], [1, 0], [2, 0]]
    assert first.k.dtype is torch.bfloat16 and first.k.is_contiguous()
    assert first.beta.dtype is first.g_cumsum.dtype is torch.float32
    assert first.cu_seqlens.dtype is first.chunk_indices.dtype is torch.int32
    assert torch.equal(first.k, second.k)
    assert torch.equal(first.beta, second.beta)
    assert torch.equal(first.g_cumsum, second.g_cumsum)


def test_vllm_operand_validation_rejects_malformed_metadata() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton as runner

    args = runner._validate_args(65, 2, 2, 4, 64, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.cu_seqlens[1] = 34
    with pytest.raises(ValueError, match="canonical boundaries"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.chunk_indices[1, 1] = 1
    with pytest.raises(ValueError, match="canonical sequence/chunk mapping"):
        runner._validate_operands(torch, operands, args, require_cuda=False)


def test_vllm_guard_geometry_preserves_qwen_and_bounds_large_shape() -> None:
    from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton import (
        _guard_args,
        _validate_args,
    )

    qwen = _validate_args(128, 2, 16, 32, 128, "bf16")
    assert _guard_args(qwen) is qwen

    large = _validate_args(262_144, 4096, 16, 32, 128, "bf16")
    guard = _guard_args(large)
    assert guard.num_tokens == 252
    assert guard.num_chunks == 252
    assert guard.num_key_heads == 16
    assert guard.num_heads == 32
    assert guard.key_head_dim == 128
    assert guard.num_tokens * (guard.num_key_heads * guard.key_head_dim + 66 * guard.num_heads) <= (
        1_048_576
    )
    assert (guard.num_tokens + 63) // 64 <= guard.num_chunks <= guard.num_tokens


def test_vllm_correctness_guard_checks_reference_masks_and_immutability() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton as runner
    from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_reference import (
        gdn_chunk_scaled_dot_kkt_reference,
    )

    args = runner._validate_args(5, 2, 2, 4, 64, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    snapshots = tuple(
        tensor.clone()
        for tensor in (
            operands.k,
            operands.beta,
            operands.g_cumsum,
            operands.cu_seqlens,
            operands.chunk_indices,
        )
    )
    calls = 0

    def fake_fused(k, *, g, beta, cu_seqlens, chunk_indices, chunk_size, output_dtype):
        nonlocal calls
        calls += 1
        assert chunk_size == 64 and output_dtype is torch.float32
        assert chunk_indices.tolist() == [[0, 0], [1, 0]]
        return gdn_chunk_scaled_dot_kkt_reference(
            k.squeeze(0), beta.squeeze(0), g.squeeze(0), cu_seqlens
        ).unsqueeze(0)

    runner._check_correctness(
        torch,
        fake_fused,
        operands,
        args,
        synchronize=lambda: None,
    )
    assert calls == 1
    for original, snapshot in zip(
        (
            operands.k,
            operands.beta,
            operands.g_cumsum,
            operands.cu_seqlens,
            operands.chunk_indices,
        ),
        snapshots,
        strict=True,
    ):
        assert torch.equal(original, snapshot)


def test_vllm_profile_times_only_one_wrapper_call_and_reuses_metrics(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton as runner

    args = runner._validate_args(5, 2, 1, 2, 64, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    calls = {"build": 0, "guard": 0, "fused": 0, "timer": 0, "energy": 0}

    def fake_build(*_args, **_kwargs):
        calls["build"] += 1
        return operands

    def fake_fused(*_args, **_kwargs):
        calls["fused"] += 1
        return torch.empty((1, 5, 2, 64), dtype=torch.float32)

    def fake_guard(*_args, **_kwargs):
        calls["guard"] += 1

    def fake_timer(fn, *, kernel_name):
        calls["timer"] += 1
        assert kernel_name == "chunk_scaled_dot_kkt_fwd_kernel"
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
    monkeypatch.setattr(runner, "_check_correctness", fake_guard)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))

    metrics = runner.profile_gdn_chunk_scaled_dot_kkt_vllm_triton(5, 2, 1, 2, 64, "bf16")
    assert metrics.time_ms == 0.5 and metrics.energy_j == 0.25
    assert calls == {"build": 1, "guard": 1, "fused": 2, "timer": 1, "energy": 1}
    assert metrics.tflops > 0 and metrics.memory_bandwidth_gbps > 0


def test_generated_facades_preserve_both_backend_queries(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")
    for backend in ("torch", "vllm_triton"):
        assert (
            perf_api.count_missing_gdn_chunk_scaled_dot_kkt(
                [_SPEC], backend=backend, gpu_name="NVIDIA H200"
            )
            == 1
        )
        result = perf_api.get_gdn_chunk_scaled_dot_kkt_times(
            [_SPEC], backend=backend, gpu_name="NVIDIA H200"
        )[0]
        assert isinstance(result, MissingEntry)
    assert not perf_api.DB_PATH.exists()
