"""Registration and Torch runner tests for ``gdn_chunk_solve_tril``."""

from __future__ import annotations

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
from profiling.kernels.gdn_chunk_solve_tril import KIND, GdnChunkSolveTrilArgs
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SPEC = {
    "num_tokens": 128,
    "num_chunks": 2,
    "max_chunk_tokens": 64,
    "num_heads": 32,
    "dtype": "bf16",
}


def test_args_field_order_and_coercion() -> None:
    assert [field.name for field in fields(GdnChunkSolveTrilArgs)] == [
        "num_tokens",
        "num_chunks",
        "max_chunk_tokens",
        "num_heads",
        "dtype",
    ]
    args = coerce_args(
        GdnChunkSolveTrilArgs,
        {
            "num_tokens": "128",
            "num_chunks": "2",
            "max_chunk_tokens": "64",
            "num_heads": "32",
            "dtype": "torch.bfloat16",
        },
    )
    assert args == GdnChunkSolveTrilArgs(
        num_tokens=128,
        num_chunks=2,
        max_chunk_tokens=64,
        num_heads=32,
        dtype=DType.BF16,
    )
    with pytest.raises(Exception):
        args.num_tokens = 1


def test_registration_table_kind_runner_and_support_contract() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")

    assert KIND == "gdn_chunk_solve_tril"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.args_schema is GdnChunkSolveTrilArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env == "default_env"
    assert spec.runner_ref.module_name == ("profiling.runners.attention.gdn_chunk_solve_tril_torch")
    assert spec.runner_ref.function_name == "profile_gdn_chunk_solve_tril"
    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.supports.kv is None
    assert spec.supports.gpus is None
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA H200")

    fused = find_kernel_profiler_spec(KIND, "vllm_triton")
    assert fused.kernel_kind == fused.table_name == KIND
    assert fused.backend == "vllm_triton"
    assert fused.args_schema is GdnChunkSolveTrilArgs
    assert fused.metric_family is MetricFamily.COMPUTE
    assert fused.batch_outlier_policy == BatchOutlierPolicy()
    assert fused.subprocess_env == "vllm_env"
    assert fused.runner_ref.module_name == (
        "profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton"
    )
    assert fused.runner_ref.function_name == "profile_gdn_chunk_solve_tril_vllm_triton"
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
                "print('profiling.runners.attention.gdn_chunk_solve_tril_torch' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_solve_tril_reference' "
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
                "'gdn_chunk_solve_tril', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_solve_tril_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_chunk_solve_tril_torch",
        "profile_gdn_chunk_solve_tril",
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
                "'gdn_chunk_solve_tril', 'vllm_triton').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('vllm' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_solve_tril_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton",
        "profile_gdn_chunk_solve_tril_vllm_triton",
        "False",
        "False",
        "False",
    ]


@pytest.mark.parametrize("name", ["num_tokens", "num_chunks", "max_chunk_tokens", "num_heads"])
@pytest.mark.parametrize("value", [0, -1])
def test_runner_rejects_nonpositive_values_before_import(
    name: str,
    value: int,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_solve_tril_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="must be > 0"):
        runner.profile_gdn_chunk_solve_tril(**(_SPEC | {name: value}))


@pytest.mark.parametrize("dtype", ["fp16", "fp32", "fp8_e4m3"])
def test_runner_rejects_unsupported_dtype_before_import(dtype: str, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_solve_tril_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="requires dtype=bf16"):
        runner.profile_gdn_chunk_solve_tril(**(_SPEC | {"dtype": dtype}))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"max_chunk_tokens": 65}, "must be <= 64"),
        ({"num_tokens": 64}, r"M\+C-1 <= T"),
        ({"num_tokens": 129}, r"T <= C\*M"),
    ],
)
def test_runner_rejects_infeasible_domain_before_import(
    updates: dict[str, object],
    message: str,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_solve_tril_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=message):
        runner.profile_gdn_chunk_solve_tril(**(_SPEC | updates))


@pytest.mark.parametrize("name", ["num_tokens", "num_chunks", "max_chunk_tokens", "num_heads"])
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_runner_rejects_non_integer_values_before_import(
    name: str,
    value: object,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_solve_tril_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="exact integer"):
        runner.profile_gdn_chunk_solve_tril(**(_SPEC | {name: value}))


@pytest.mark.parametrize(
    ("num_tokens", "num_chunks", "max_chunk_tokens", "expected"),
    [
        (1, 1, 1, (1,)),
        (64, 1, 64, (64,)),
        (65, 2, 64, (64, 1)),
        (128, 2, 64, (64, 64)),
        (49, 3, 33, (33, 8, 8)),
        (97, 3, 49, (49, 24, 24)),
        (65, 3, 32, (32, 17, 16)),
        (8, 8, 1, (1, 1, 1, 1, 1, 1, 1, 1)),
    ],
)
def test_canonical_partition_invariants(
    num_tokens: int,
    num_chunks: int,
    max_chunk_tokens: int,
    expected: tuple[int, ...],
) -> None:
    from profiling.runners.attention.gdn_chunk_solve_tril_torch import (
        _canonical_boundaries,
        _canonical_lengths,
    )

    lengths = _canonical_lengths(num_tokens, num_chunks, max_chunk_tokens)
    boundaries = _canonical_boundaries(num_tokens, num_chunks, max_chunk_tokens)
    assert lengths == expected
    assert len(lengths) == num_chunks
    assert sum(lengths) == num_tokens
    assert min(lengths) >= 1
    assert max(lengths) == max_chunk_tokens <= 64
    assert boundaries[0] == 0 and boundaries[-1] == num_tokens
    assert tuple(right - left for left, right in zip(boundaries, boundaries[1:])) == lengths
    assert sum((length + 63) // 64 for length in lengths) == num_chunks


def test_derived_shapes_dtypes_metadata_and_deterministic_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_solve_tril_torch import (
        _build_operands,
        _operand_shapes,
        _validate_args,
    )

    args = _validate_args(49, 3, 33, 4, "bf16")
    shapes = _operand_shapes(args)
    assert shapes.A == shapes.output == (49, 4, 64)
    assert shapes.cu_seqlens == (4,)
    assert shapes.square == (4, 33, 33)
    assert shapes.row == (4, 33)
    assert shapes.row_product == (4, 1, 33)
    assert shapes.identity == (33, 33)

    first = _build_operands(torch, args, device=torch.device("cpu"))
    second = _build_operands(torch, args, device=torch.device("cpu"))
    assert first.A.dtype is torch.float32
    assert first.cu_seqlens.dtype is torch.int32
    assert first.output.dtype is torch.bfloat16
    assert first.boundaries == (0, 33, 41, 49)
    assert tuple(first.cu_seqlens.tolist()) == first.boundaries
    assert first.workspaces.strict_inverse.dtype is torch.float32
    assert first.workspaces.inverse.dtype is torch.float32
    assert first.workspaces.identity.dtype is torch.float32
    assert torch.equal(first.A, second.A)
    assert torch.isfinite(first.A).all()
    for start, end in zip(first.boundaries, first.boundaries[1:]):
        length = end - start
        block = first.A[start:end, :, :length].permute(1, 0, 2)
        assert torch.count_nonzero(torch.triu(block)).item() == 0
        assert torch.count_nonzero(first.A[start:end, :, length:]).item() == 0


def test_device_generic_helper_agrees_with_accepted_reference() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_solve_tril_reference import (
        gdn_chunk_solve_tril_reference,
    )
    from profiling.runners.attention.gdn_chunk_solve_tril_torch import (
        _build_operands,
        _solve_tril_into,
        _validate_args,
    )

    args = _validate_args(49, 3, 33, 3, "bf16")
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    A_snapshot = operands.A.clone()
    metadata_snapshot = operands.cu_seqlens.clone()
    expected = gdn_chunk_solve_tril_reference(operands.A, operands.cu_seqlens)

    actual = _solve_tril_into(
        torch,
        operands.A,
        operands.output,
        operands.boundaries,
        operands.workspaces,
    )

    assert actual is operands.output
    assert torch.equal(actual, expected)
    assert torch.equal(operands.A, A_snapshot)
    assert torch.equal(operands.cu_seqlens, metadata_snapshot)
    for start, end in zip(operands.boundaries, operands.boundaries[1:]):
        length = end - start
        block = actual[start:end, :, :length].permute(1, 0, 2)
        assert torch.equal(
            torch.diagonal(block, dim1=-2, dim2=-1),
            torch.ones((args.num_heads, length), dtype=torch.bfloat16),
        )
        assert torch.count_nonzero(torch.triu(block, diagonal=1)).item() == 0
        assert torch.count_nonzero(actual[start:end, :, length:]).item() == 0


def test_helper_signed_recurrence_resets_and_repeated_overwrite() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_solve_tril_torch import (
        _build_operands,
        _solve_tril_into,
        _validate_args,
    )

    args = _validate_args(6, 2, 3, 2, "bf16")
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    operands.A.zero_()
    operands.A[1, 0, 0] = 0.5
    operands.A[2, 0, :2] = torch.tensor([-0.25, 2.0])
    operands.A[4, 1, 0] = -0.5
    operands.A[5, 1, :2] = torch.tensor([0.25, -2.0])

    first = _solve_tril_into(
        torch,
        operands.A,
        operands.output,
        operands.boundaries,
        operands.workspaces,
    ).clone()
    operands.output.fill_(7)
    operands.workspaces.strict_inverse.fill_(11)
    operands.workspaces.inverse.fill_(13)
    operands.workspaces.negative_input_row.fill_(17)
    operands.workspaces.solved_row.fill_(19)
    operands.workspaces.row_product.fill_(23)
    second = _solve_tril_into(
        torch,
        operands.A,
        operands.output,
        operands.boundaries,
        operands.workspaces,
    )

    assert torch.equal(first, second)
    assert second[1, 0, 0].item() == -0.5
    assert second[2, 0, 0].item() == 1.25
    assert second[3, 0, 0].item() == 1.0
    assert torch.count_nonzero(second[3, :, 1:]).item() == 0


def test_runner_reports_missing_cuda_as_typed_unsupported() -> None:
    from profiling.runners.attention.gdn_chunk_solve_tril_torch import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)


def test_semantic_metrics_are_exact_logical_counts() -> None:
    from profiling.runners.attention.gdn_chunk_solve_tril_torch import (
        _logical_bytes,
        _semantic_flops,
    )

    assert (
        _semantic_flops(
            num_tokens=3,
            num_chunks=1,
            max_chunk_tokens=3,
            num_heads=2,
        )
        == 10
    )
    assert (
        _semantic_flops(
            num_tokens=128,
            num_chunks=2,
            max_chunk_tokens=64,
            num_heads=32,
        )
        == 5_462_016
    )
    assert _logical_bytes(num_tokens=128, num_heads=32) == 1_572_864


def test_profile_times_only_preallocated_semantics_and_needs_no_reset(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_solve_tril_torch as runner

    args = runner._validate_args(6, 2, 3, 2, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    A_snapshot = operands.A.clone()
    metadata_snapshot = operands.cu_seqlens.clone()
    output_ptr = operands.output.data_ptr()
    calls = {"build": 0, "semantic": 0, "timer": 0, "energy": 0}
    original_semantic = runner._solve_tril_into

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
    monkeypatch.setattr(runner, "_solve_tril_into", counted_semantic)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))

    metrics = runner.profile_gdn_chunk_solve_tril(6, 2, 3, 2, "bf16")
    assert metrics.time_ms == 0.5 and metrics.energy_j == 0.25
    assert calls == {"build": 1, "semantic": 2, "timer": 1, "energy": 1}
    assert operands.output.data_ptr() == output_ptr
    assert torch.equal(operands.A, A_snapshot)
    assert torch.equal(operands.cu_seqlens, metadata_snapshot)


def test_generated_facades_and_read_only_missing_query(tmp_path, monkeypatch) -> None:
    assert hasattr(perf_api, "get_gdn_chunk_solve_tril_times")
    assert hasattr(perf_api, "count_missing_gdn_chunk_solve_tril")
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")

    for backend in ("torch", "vllm_triton"):
        assert (
            perf_api.count_missing_gdn_chunk_solve_tril(
                [_SPEC], backend=backend, gpu_name="NVIDIA H200"
            )
            == 1
        )
        result = perf_api.get_gdn_chunk_solve_tril_times(
            [_SPEC], backend=backend, gpu_name="NVIDIA H200"
        )[0]
        assert isinstance(result, MissingEntry)
        assert result.args == coerce_args(GdnChunkSolveTrilArgs, _SPEC)
    assert not perf_api.DB_PATH.exists()

    table = Table(find_kernel_profiler_spec(KIND, "torch"), perf_api.DB_PATH)
    assert table.args_columns == [
        "num_tokens",
        "num_chunks",
        "max_chunk_tokens",
        "num_heads",
        "dtype",
    ]


def test_db_round_trip_uses_exact_schema_and_compute_metrics(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "profile.db"
    monkeypatch.setattr(perf_api, "DB_PATH", db_path)
    profiler_spec = find_kernel_profiler_spec(KIND, "torch")
    args = coerce_args(GdnChunkSolveTrilArgs, _SPEC)
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

    result = perf_api.get_gdn_chunk_solve_tril_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert result == metrics
    assert (
        perf_api.count_missing_gdn_chunk_solve_tril(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 0
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"num_tokens": 0}, "must be > 0"),
        ({"max_chunk_tokens": 65}, "must be <= 64"),
        ({"num_tokens": 64}, r"M\+C-1 <= T"),
        ({"num_tokens": 129}, r"T <= C\*M"),
        ({"dtype": "fp32"}, "requires dtype=bf16"),
    ],
)
def test_vllm_runner_rejects_invalid_args_before_torch_import(
    updates: dict[str, object],
    message: str,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=message):
        runner.profile_gdn_chunk_solve_tril_vllm_triton(**(_SPEC | updates))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FLA_TRIL_PRECISION", "tf32"),
        ("FLA_USE_TMA", "1"),
        ("FLA_USE_FAST_OPS", "true"),
    ],
)
def test_vllm_runner_rejects_incompatible_environment_before_import(
    name: str,
    value: str,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton as runner

    monkeypatch.setenv(name, value)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=name):
        runner.profile_gdn_chunk_solve_tril_vllm_triton(**_SPEC)


@pytest.mark.parametrize(
    ("precision", "tma", "fast_ops"),
    [(None, None, None), ("", "", ""), ("ieee", "0", "0")],
)
def test_vllm_runner_accepts_frozen_environment_settings(
    precision: str | None,
    tma: str | None,
    fast_ops: str | None,
    monkeypatch,
) -> None:
    from profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton import (
        _validate_args,
    )

    for name, value in (
        ("FLA_TRIL_PRECISION", precision),
        ("FLA_USE_TMA", tma),
        ("FLA_USE_FAST_OPS", fast_ops),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert _validate_args(**_SPEC).max_chunk_tokens == 64


def test_vllm_derived_shapes_metadata_and_deterministic_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton import (
        _build_operands,
        _canonical_index_pairs,
        _operand_shapes,
        _validate_args,
        _validate_operands,
    )

    args = _validate_args(49, 3, 33, 4, "bf16")
    shapes = _operand_shapes(args)
    assert shapes.A == shapes.output == (1, 49, 4, 64)
    assert shapes.cu_seqlens == (4,)
    assert shapes.chunk_indices == (3, 2)
    assert _canonical_index_pairs(3) == ((0, 0), (1, 0), (2, 0))

    first = _build_operands(torch, args, device=torch.device("cpu"))
    second = _build_operands(torch, args, device=torch.device("cpu"))
    _validate_operands(torch, first, args, require_cuda=False)
    assert first.boundaries == (0, 33, 41, 49)
    assert first.cu_seqlens.tolist() == [0, 33, 41, 49]
    assert first.chunk_indices.tolist() == [[0, 0], [1, 0], [2, 0]]
    assert first.A.dtype is torch.float32 and first.A.is_contiguous()
    assert first.A.stride(-1) == 1
    assert first.cu_seqlens.dtype is first.chunk_indices.dtype is torch.int32
    assert torch.equal(first.A, second.A)


def test_vllm_operand_validation_closes_structure_dtype_layout_and_metadata_gaps() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton as runner

    args = runner._validate_args(6, 2, 3, 2, "bf16")

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands = replace(operands, A=operands.A.to(torch.float16))
    with pytest.raises(ValueError, match="A must have dtype"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands = replace(
        operands,
        A=torch.empty((1, 6, 2, 128), dtype=torch.float32)[..., ::2],
    )
    with pytest.raises(ValueError, match="contiguous"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.cu_seqlens[1] = 2
    with pytest.raises(ValueError, match="canonical boundaries"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.chunk_indices[1, 1] = 1
    with pytest.raises(ValueError, match="canonical sequence/chunk mapping"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.A[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.A[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="strict-lower"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands = replace(
        operands,
        chunk_indices=torch.empty((2, 2), dtype=torch.int32, device="meta"),
    )
    with pytest.raises(ValueError, match="one device"):
        runner._validate_operands(torch, operands, args, require_cuda=False)


def test_vllm_guard_geometry_preserves_qwen_and_bounds_large_shape() -> None:
    from profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton import (
        _MAX_GUARD_ELEMENTS,
        _guard_args,
        _validate_args,
    )

    qwen = _validate_args(128, 2, 64, 32, "bf16")
    assert _guard_args(qwen) is qwen

    large = _validate_args(262_144, 4096, 64, 32, "bf16")
    guard = _guard_args(large)
    assert guard == type(large)(256, 193, 64, 32, DType.BF16)
    assert 2 * 64 * guard.num_tokens * guard.num_heads <= _MAX_GUARD_ELEMENTS
    assert guard.max_chunk_tokens + guard.num_chunks - 1 <= guard.num_tokens
    assert guard.num_tokens <= guard.num_chunks * guard.max_chunk_tokens


def test_vllm_correctness_guard_checks_reference_structure_and_immutability() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton as runner
    from profiling.runners.attention.gdn_chunk_solve_tril_reference import (
        gdn_chunk_solve_tril_reference,
    )

    args = runner._validate_args(6, 2, 3, 2, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    snapshots = (operands.A.clone(), operands.cu_seqlens.clone(), operands.chunk_indices.clone())
    calls = 0

    def fake_fused(*, A, cu_seqlens, chunk_indices, output_dtype):
        nonlocal calls
        calls += 1
        assert output_dtype is torch.bfloat16
        assert chunk_indices.tolist() == [[0, 0], [1, 0]]
        return gdn_chunk_solve_tril_reference(A.squeeze(0), cu_seqlens).unsqueeze(0)

    runner._check_correctness(torch, fake_fused, operands, args, synchronize=lambda: None)
    assert calls == 1
    for original, snapshot in zip(
        (operands.A, operands.cu_seqlens, operands.chunk_indices), snapshots, strict=True
    ):
        assert torch.equal(original, snapshot)


def test_vllm_correctness_guard_rejects_wrong_output_properties() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton as runner

    args = runner._validate_args(6, 2, 3, 2, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))

    def wrong_dtype(**_kwargs):
        return torch.empty((1, 6, 2, 64), dtype=torch.float32)

    with pytest.raises(AssertionError, match="output dtype"):
        runner._check_correctness(torch, wrong_dtype, operands, args, synchronize=lambda: None)


def test_vllm_profile_times_one_wrapper_call_with_exact_selector_and_no_reset(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton as runner

    args = runner._validate_args(6, 2, 3, 2, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    calls = {"build": 0, "guard": 0, "fused": 0, "timer": 0, "energy": 0}

    def fake_build(*_args, **_kwargs):
        calls["build"] += 1
        return operands

    def fake_fused(*_args, **_kwargs):
        calls["fused"] += 1
        return torch.empty((1, 6, 2, 64), dtype=torch.bfloat16)

    def fake_guard(*_args, **_kwargs):
        calls["guard"] += 1

    def fake_timer(fn, *, kernel_name):
        calls["timer"] += 1
        assert kernel_name == "merge_16x16_to_64x64_inverse_kernel"
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

    metrics = runner.profile_gdn_chunk_solve_tril_vllm_triton(6, 2, 3, 2, "bf16")
    assert metrics.time_ms == 0.5 and metrics.energy_j == 0.25
    assert calls == {"build": 1, "guard": 1, "fused": 2, "timer": 1, "energy": 1}
    assert metrics.tflops > 0 and metrics.memory_bandwidth_gbps > 0
