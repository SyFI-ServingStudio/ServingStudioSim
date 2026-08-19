"""Registration and Torch runner tests for ``gdn_chunk_local_cumsum``."""

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
from profiling.kernels.gdn_chunk_local_cumsum import (
    KIND,
    GdnChunkLocalCumsumArgs,
)
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SPEC = {
    "num_tokens": 128,
    "num_chunks": 2,
    "num_heads": 32,
    "dtype": "fp32",
}


def test_args_field_order_and_coercion() -> None:
    assert [field.name for field in fields(GdnChunkLocalCumsumArgs)] == [
        "num_tokens",
        "num_chunks",
        "num_heads",
        "dtype",
    ]
    args = coerce_args(
        GdnChunkLocalCumsumArgs,
        {
            "num_tokens": "128",
            "num_chunks": "2",
            "num_heads": "32",
            "dtype": "torch.float32",
        },
    )
    assert args == GdnChunkLocalCumsumArgs(
        num_tokens=128,
        num_chunks=2,
        num_heads=32,
        dtype=DType.FP32,
    )
    with pytest.raises(Exception):
        args.num_tokens = 1


def test_registration_table_kind_runner_and_support_contract() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")

    assert KIND == "gdn_chunk_local_cumsum"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.args_schema is GdnChunkLocalCumsumArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_chunk_local_cumsum_torch"
    )
    assert spec.runner_ref.function_name == "profile_gdn_chunk_local_cumsum"

    assert spec.supports.compute == frozenset({DType.FP32})
    assert spec.supports.kv is None
    assert spec.supports.gpus is None
    assert spec.supports.allows(DType.FP32, gpu="NVIDIA H200")
    assert spec.supports.allows(DType.FP32, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA H200")


def test_vllm_registration_reuses_schema_table_and_is_h200_only() -> None:
    spec = find_kernel_profiler_spec(KIND, "vllm_triton")

    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "vllm_triton"
    assert spec.args_schema is GdnChunkLocalCumsumArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env == "vllm_env"
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton"
    )
    assert spec.runner_ref.function_name == "profile_gdn_chunk_local_cumsum_vllm_triton"

    assert spec.supports.compute == frozenset({DType.FP32})
    assert spec.supports.kv is None
    assert spec.supports.gpus == frozenset({"NVIDIA H200"})
    assert spec.supports.allows(DType.FP32, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA H100")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA H200")


def test_registry_barrel_and_runner_ref_are_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_local_cumsum_torch' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_local_cumsum_reference' "
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
                "'gdn_chunk_local_cumsum', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_local_cumsum_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_chunk_local_cumsum_torch",
        "profile_gdn_chunk_local_cumsum",
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
                "'gdn_chunk_local_cumsum', 'vllm_triton').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); print('vllm' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_local_cumsum_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton",
        "profile_gdn_chunk_local_cumsum_vllm_triton",
        "False",
        "False",
        "False",
    ]


@pytest.mark.parametrize("name", ["num_tokens", "num_chunks", "num_heads"])
@pytest.mark.parametrize("value", [0, -1])
def test_runner_rejects_nonpositive_values_before_import(
    name: str,
    value: int,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_local_cumsum_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="must be > 0"):
        runner.profile_gdn_chunk_local_cumsum(**(_SPEC | {name: value}))


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp8_e4m3"])
def test_runner_rejects_unsupported_dtype_before_import(dtype: str, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_local_cumsum_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="requires dtype=fp32"):
        runner.profile_gdn_chunk_local_cumsum(**(_SPEC | {"dtype": dtype}))


@pytest.mark.parametrize(
    ("num_tokens", "num_chunks"),
    [(65, 1), (128, 1), (129, 2), (4, 5)],
)
def test_runner_rejects_infeasible_chunk_counts_before_import(
    num_tokens: int,
    num_chunks: int,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_local_cumsum_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=r"ceil\(num_tokens/64\)"):
        runner.profile_gdn_chunk_local_cumsum(
            **(_SPEC | {"num_tokens": num_tokens, "num_chunks": num_chunks})
        )


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
    from profiling.runners.attention.gdn_chunk_local_cumsum_torch import (
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


def test_runner_shapes_dtypes_and_deterministic_cpu_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_local_cumsum_torch import (
        _build_operands,
        _operand_shapes,
        _validate_args,
    )

    args = _validate_args(82, 3, 4, "fp32")
    shapes = _operand_shapes(args)
    assert shapes.g == shapes.output == (82, 4)
    assert shapes.cu_seqlens == (4,)

    first = _build_operands(torch, args, device=torch.device("cpu"))
    second = _build_operands(torch, args, device=torch.device("cpu"))
    assert first.g.dtype is first.output.dtype is torch.float32
    assert first.cu_seqlens.dtype is torch.int32
    assert first.boundaries == (0, 28, 55, 82)
    assert tuple(first.cu_seqlens.tolist()) == first.boundaries
    assert torch.equal(first.g, second.g)
    assert torch.isfinite(first.g).all()
    assert first.g.min() >= -0.125 and first.g.max() <= 0.0


def test_device_generic_helper_agrees_with_accepted_reference() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_local_cumsum_reference import (
        gdn_chunk_local_cumsum_reference,
    )
    from profiling.runners.attention.gdn_chunk_local_cumsum_torch import (
        _chunk_local_cumsum_into,
    )

    generator = torch.Generator().manual_seed(7)
    g = torch.randn(70, 3, generator=generator, dtype=torch.float32)
    boundaries = (0, 65, 70)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)
    expected = gdn_chunk_local_cumsum_reference(g, cu_seqlens)
    output = torch.full_like(g, float("nan"))

    actual = _chunk_local_cumsum_into(torch, g, output, boundaries)
    assert actual is output
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.equal(actual[64], g[64])
    assert torch.equal(actual[65], g[65])


def test_runner_reports_missing_cuda_as_typed_unsupported() -> None:
    from profiling.runners.attention.gdn_chunk_local_cumsum_torch import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)


def test_semantic_metrics_are_explicit_logical_counts() -> None:
    from profiling.runners.attention.gdn_chunk_local_cumsum_torch import (
        _logical_bytes,
        _semantic_flops,
    )

    assert _semantic_flops(num_tokens=128, num_chunks=2, num_heads=32) == 4032
    assert _semantic_flops(num_tokens=5, num_chunks=5, num_heads=7) == 0
    assert _logical_bytes(num_tokens=128, num_heads=32) == 32768


def test_profile_times_only_preallocated_semantic_work_and_needs_no_reset(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_local_cumsum_torch as runner

    args = runner._validate_args(70, 2, 3, "fp32")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    input_snapshot = operands.g.clone()
    calls = {"build": 0, "semantic": 0, "timer": 0, "energy": 0}
    original_semantic = runner._chunk_local_cumsum_into

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
        assert fn() is operands.output
        assert warmup == 5 and per_iter_time_ms == 0.5
        return 0.25

    monkeypatch.setattr(runner, "_validate_cuda_device", lambda _torch: None)
    monkeypatch.setattr(runner, "_build_operands", fake_build)
    monkeypatch.setattr(runner, "_chunk_local_cumsum_into", counted_semantic)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))

    metrics = runner.profile_gdn_chunk_local_cumsum(70, 2, 3, "fp32")
    assert metrics.time_ms == 0.5 and metrics.energy_j == 0.25
    assert calls == {"build": 1, "semantic": 2, "timer": 1, "energy": 1}
    assert torch.equal(operands.g, input_snapshot)
    first_output = operands.output.clone()
    original_semantic(torch, operands.g, operands.output, operands.boundaries)
    assert torch.equal(operands.output, first_output)


@pytest.mark.parametrize("name", ["num_tokens", "num_chunks", "num_heads"])
@pytest.mark.parametrize("value", [0, -1])
def test_vllm_runner_rejects_nonpositive_values_before_import(
    name: str,
    value: int,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="must be > 0"):
        runner.profile_gdn_chunk_local_cumsum_vllm_triton(**(_SPEC | {name: value}))


@pytest.mark.parametrize("name", ["num_tokens", "num_chunks", "num_heads"])
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_vllm_runner_rejects_non_integer_values_before_import(
    name: str,
    value: object,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="exact integer"):
        runner.profile_gdn_chunk_local_cumsum_vllm_triton(**(_SPEC | {name: value}))


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp8_e4m3"])
def test_vllm_runner_rejects_unsupported_dtype_before_import(dtype: str, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="requires dtype=fp32"):
        runner.profile_gdn_chunk_local_cumsum_vllm_triton(**(_SPEC | {"dtype": dtype}))


@pytest.mark.parametrize(
    ("num_tokens", "num_chunks"),
    [(65, 1), (128, 1), (129, 2), (4, 5)],
)
def test_vllm_runner_rejects_infeasible_chunks_before_import(
    num_tokens: int,
    num_chunks: int,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=r"ceil\(num_tokens/64\)"):
        runner.profile_gdn_chunk_local_cumsum_vllm_triton(
            **(_SPEC | {"num_tokens": num_tokens, "num_chunks": num_chunks})
        )


def test_vllm_canonical_shapes_metadata_and_deterministic_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton import (
        _build_operands,
        _canonical_boundaries,
        _canonical_index_pairs,
        _canonical_lengths,
        _operand_shapes,
        _validate_args,
        _validate_operands,
    )

    args = _validate_args(82, 3, 4, "fp32")
    assert _canonical_lengths(82, 3) == (28, 27, 27)
    assert _canonical_boundaries(82, 3) == (0, 28, 55, 82)
    assert _canonical_index_pairs(3) == ((0, 0), (1, 0), (2, 0))
    shapes = _operand_shapes(args)
    assert shapes.g == shapes.output == (1, 82, 4)
    assert shapes.cu_seqlens == (4,)
    assert shapes.chunk_indices == (3, 2)

    first = _build_operands(torch, args, device=torch.device("cpu"))
    second = _build_operands(torch, args, device=torch.device("cpu"))
    _validate_operands(torch, first, args, require_cuda=False)
    assert first.g.dtype is torch.float32 and first.g.is_contiguous()
    assert first.cu_seqlens.dtype is torch.int32 and first.cu_seqlens.is_contiguous()
    assert first.chunk_indices.dtype is torch.int32 and first.chunk_indices.is_contiguous()
    assert tuple(first.cu_seqlens.tolist()) == first.boundaries
    assert tuple(map(tuple, first.chunk_indices.tolist())) == _canonical_index_pairs(3)
    assert torch.equal(first.g, second.g)
    assert torch.isfinite(first.g).all()
    assert first.g.min() >= -0.125 and first.g.max() <= -0.001


def test_vllm_operand_validation_rejects_layout_and_malformed_metadata() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton as runner

    args = runner._validate_args(82, 3, 4, "fp32")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))

    base = torch.empty(1, 82, 8, dtype=torch.float32)
    noncontiguous = runner._Operands(**{**operands.__dict__, "g": base[..., ::2]})
    with pytest.raises(ValueError, match="g must be contiguous"):
        runner._validate_operands(torch, noncontiguous, args, require_cuda=False)

    bad_boundaries = runner._Operands(
        **{
            **operands.__dict__,
            "cu_seqlens": torch.tensor([0, 28, 28, 82], dtype=torch.int32),
        }
    )
    with pytest.raises(ValueError, match="canonical boundaries"):
        runner._validate_operands(torch, bad_boundaries, args, require_cuda=False)

    bad_indices = runner._Operands(
        **{
            **operands.__dict__,
            "chunk_indices": torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.int32),
        }
    )
    with pytest.raises(ValueError, match="canonical sequence/chunk mapping"):
        runner._validate_operands(torch, bad_indices, args, require_cuda=False)


def test_vllm_guard_geometry_bounds_elements_and_preserves_feasibility() -> None:
    from profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton import (
        _MAX_GUARD_ELEMENTS,
        _guard_args,
        _validate_args,
    )

    qwen = _validate_args(128, 2, 32, "fp32")
    assert _guard_args(qwen) == qwen

    large = _validate_args(100_000, 2_000, 32, "fp32")
    bounded = _guard_args(large)
    assert bounded.num_heads == large.num_heads
    assert bounded.num_tokens == _MAX_GUARD_ELEMENTS // 32
    assert bounded.num_chunks == 2_000
    assert bounded.num_tokens * bounded.num_heads <= _MAX_GUARD_ELEMENTS
    assert (bounded.num_tokens + 63) // 64 <= bounded.num_chunks <= bounded.num_tokens

    huge_heads = _validate_args(128, 2, _MAX_GUARD_ELEMENTS + 1, "fp32")
    tiny = _guard_args(huge_heads)
    assert (tiny.num_tokens, tiny.num_chunks) == (1, 1)


def test_vllm_correctness_guard_matches_reference_and_checks_storage() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton as runner
    from profiling.runners.attention.gdn_chunk_local_cumsum_reference import (
        gdn_chunk_local_cumsum_reference,
    )

    args = runner._validate_args(70, 2, 3, "fp32")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    snapshots = {
        name: getattr(operands, name).clone() for name in ("g", "cu_seqlens", "chunk_indices")
    }

    def fake_fused(g, **kwargs):
        assert kwargs == {
            "chunk_size": 64,
            "reverse": False,
            "cu_seqlens": operands.cu_seqlens,
            "chunk_indices": operands.chunk_indices,
            "head_first": False,
            "output_dtype": torch.float32,
        }
        return gdn_chunk_local_cumsum_reference(g.squeeze(0), operands.cu_seqlens).unsqueeze(0)

    runner._check_correctness(
        torch,
        fake_fused,
        operands,
        args,
        synchronize=lambda: None,
    )
    for name, snapshot in snapshots.items():
        assert torch.equal(getattr(operands, name), snapshot)

    with pytest.raises(AssertionError, match="fresh non-aliased storage"):
        runner._check_correctness(
            torch,
            lambda g, **_kwargs: g,
            operands,
            args,
            synchronize=lambda: None,
        )


def test_vllm_profile_times_only_one_wrapper_call_with_selector(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton as runner

    args = runner._validate_args(70, 2, 3, "fp32")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    input_snapshots = {
        name: getattr(operands, name).clone() for name in ("g", "cu_seqlens", "chunk_indices")
    }
    calls = {"build": 0, "guard": 0, "fused": 0, "timer": 0, "energy": 0}

    def fake_build(*_args, **_kwargs):
        calls["build"] += 1
        return operands

    def fake_fused(g, **_kwargs):
        calls["fused"] += 1
        return g.clone()

    def fake_guard(_torch, fused_callable, guard_operands, _guard_args):
        calls["guard"] += 1
        fused_callable(guard_operands.g)

    def fake_timer(fn, *, kernel_name):
        calls["timer"] += 1
        before = calls["fused"]
        fn()
        assert calls["fused"] == before + 1
        assert kernel_name == "chunk_local_cumsum_scalar_kernel"
        return 0.5

    def fake_energy(fn, *, warmup, per_iter_time_ms):
        calls["energy"] += 1
        before = calls["fused"]
        fn()
        assert calls["fused"] == before + 1
        assert warmup == 5 and per_iter_time_ms == 0.5
        return 0.25

    monkeypatch.setattr(runner, "_require_h200", lambda _torch: None)
    monkeypatch.setattr(runner, "_load_fused_callable", lambda: fake_fused)
    monkeypatch.setattr(runner, "_build_operands", fake_build)
    monkeypatch.setattr(runner, "_validate_operands", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_check_correctness", fake_guard)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    metrics = runner.profile_gdn_chunk_local_cumsum_vllm_triton(70, 2, 3, "fp32")
    assert metrics.time_ms == 0.5 and metrics.energy_j == 0.25
    assert calls == {"build": 1, "guard": 1, "fused": 3, "timer": 1, "energy": 1}
    for name, snapshot in input_snapshots.items():
        assert torch.equal(getattr(operands, name), snapshot)


def test_vllm_reuses_torch_semantic_metric_helpers() -> None:
    import profiling.runners.attention.gdn_chunk_local_cumsum_torch as torch_runner
    import profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton as vllm_runner

    assert vllm_runner._semantic_flops is torch_runner._semantic_flops
    assert vllm_runner._logical_bytes is torch_runner._logical_bytes


def test_generated_facades_and_read_only_missing_query(tmp_path, monkeypatch) -> None:
    assert hasattr(perf_api, "get_gdn_chunk_local_cumsum_times")
    assert hasattr(perf_api, "count_missing_gdn_chunk_local_cumsum")
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")

    assert (
        perf_api.count_missing_gdn_chunk_local_cumsum(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 1
    )
    result = perf_api.get_gdn_chunk_local_cumsum_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(result, MissingEntry)
    assert result.args == coerce_args(GdnChunkLocalCumsumArgs, _SPEC)
    assert (
        perf_api.count_missing_gdn_chunk_local_cumsum(
            [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
        )
        == 1
    )
    vllm_result = perf_api.get_gdn_chunk_local_cumsum_times(
        [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(vllm_result, MissingEntry)
    assert vllm_result.args == result.args
    assert not perf_api.DB_PATH.exists()

    table = Table(find_kernel_profiler_spec(KIND, "torch"), perf_api.DB_PATH)
    assert table.args_columns == ["num_tokens", "num_chunks", "num_heads", "dtype"]


def test_db_round_trip_uses_exact_schema_and_compute_metrics(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "profile.db"
    monkeypatch.setattr(perf_api, "DB_PATH", db_path)
    profiler_spec = find_kernel_profiler_spec(KIND, "torch")
    args = coerce_args(GdnChunkLocalCumsumArgs, _SPEC)
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

    result = perf_api.get_gdn_chunk_local_cumsum_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert result == metrics
    assert (
        perf_api.count_missing_gdn_chunk_local_cumsum(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 0
    )
