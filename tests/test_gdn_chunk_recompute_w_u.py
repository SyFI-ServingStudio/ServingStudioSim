"""Registration and Torch runner tests for ``gdn_chunk_recompute_w_u``."""

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
from profiling.kernels.gdn_chunk_recompute_w_u import KIND, GdnChunkRecomputeWUArgs
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SPEC = {
    "num_tokens": 128,
    "num_chunks": 2,
    "num_key_heads": 16,
    "num_heads": 32,
    "key_head_dim": 128,
    "value_head_dim": 128,
    "dtype": "bf16",
}


def test_args_field_order_and_coercion() -> None:
    assert [field.name for field in fields(GdnChunkRecomputeWUArgs)] == [
        "num_tokens",
        "num_chunks",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "value_head_dim",
        "dtype",
    ]
    args = coerce_args(
        GdnChunkRecomputeWUArgs,
        {
            "num_tokens": "128",
            "num_chunks": "2",
            "num_key_heads": "16",
            "num_heads": "32",
            "key_head_dim": "128",
            "value_head_dim": "128",
            "dtype": "torch.bfloat16",
        },
    )
    assert args == GdnChunkRecomputeWUArgs(
        num_tokens=128,
        num_chunks=2,
        num_key_heads=16,
        num_heads=32,
        key_head_dim=128,
        value_head_dim=128,
        dtype=DType.BF16,
    )
    with pytest.raises(Exception):
        args.num_tokens = 1


def test_registration_table_kind_runner_and_support_contract() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")

    assert KIND == "gdn_chunk_recompute_w_u"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.args_schema is GdnChunkRecomputeWUArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env == "default_env"
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_chunk_recompute_w_u_torch"
    )
    assert spec.runner_ref.function_name == "profile_gdn_chunk_recompute_w_u"
    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.supports.kv is None
    assert spec.supports.gpus is None
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA H200")

    fused = find_kernel_profiler_spec(KIND, "vllm_triton")
    assert fused.kernel_kind == fused.table_name == KIND
    assert fused.args_schema is GdnChunkRecomputeWUArgs
    assert fused.metric_family is MetricFamily.COMPUTE
    assert fused.batch_outlier_policy == BatchOutlierPolicy()
    assert fused.subprocess_env == "vllm_env"
    assert fused.runner_ref.module_name == (
        "profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton"
    )
    assert fused.runner_ref.function_name == ("profile_gdn_chunk_recompute_w_u_vllm_triton")
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
                "print('profiling.runners.attention.gdn_chunk_recompute_w_u_torch' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_recompute_w_u_reference' "
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
                "'gdn_chunk_recompute_w_u', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_recompute_w_u_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_chunk_recompute_w_u_torch",
        "profile_gdn_chunk_recompute_w_u",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "num_tokens",
        "num_chunks",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "value_head_dim",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_runner_rejects_nonpositive_values_before_import(
    name: str,
    value: int,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_recompute_w_u_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="must be > 0"):
        runner.profile_gdn_chunk_recompute_w_u(**(_SPEC | {name: value}))


@pytest.mark.parametrize("dtype", ["fp16", "fp32", "fp8_e4m3"])
def test_runner_rejects_unsupported_dtype_before_import(dtype: str, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_recompute_w_u_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="requires dtype=bf16"):
        runner.profile_gdn_chunk_recompute_w_u(**(_SPEC | {"dtype": dtype}))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"num_chunks": 1}, "ceil"),
        ({"num_chunks": 129}, "ceil"),
        ({"num_key_heads": 3}, "divisible"),
    ],
)
def test_runner_rejects_infeasible_domain_before_import(
    updates: dict[str, object],
    message: str,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_recompute_w_u_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=message):
        runner.profile_gdn_chunk_recompute_w_u(**(_SPEC | updates))


@pytest.mark.parametrize(
    "name",
    [
        "num_tokens",
        "num_chunks",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "value_head_dim",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_runner_rejects_non_integer_values_before_import(
    name: str,
    value: object,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_chunk_recompute_w_u_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="exact integer"):
        runner.profile_gdn_chunk_recompute_w_u(**(_SPEC | {name: value}))


@pytest.mark.parametrize(
    ("num_tokens", "num_chunks", "expected"),
    [
        (1, 1, (1,)),
        (64, 1, (64,)),
        (65, 2, (33, 32)),
        (128, 2, (64, 64)),
        (82, 3, (28, 27, 27)),
        (128, 128, (1,) * 128),
    ],
)
def test_canonical_partition_invariants(
    num_tokens: int,
    num_chunks: int,
    expected: tuple[int, ...],
) -> None:
    from profiling.runners.attention.gdn_chunk_recompute_w_u_torch import (
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

    from profiling.runners.attention.gdn_chunk_recompute_w_u_torch import (
        _build_operands,
        _operand_shapes,
        _validate_args,
    )

    args = _validate_args(82, 3, 2, 4, 8, 10, "bf16")
    shapes = _operand_shapes(args)
    assert shapes.k == (82, 2, 8)
    assert shapes.v == (82, 4, 10)
    assert shapes.beta == shapes.g_cumsum == (82, 4)
    assert shapes.A == (82, 4, 64)
    assert shapes.cu_seqlens == (4,)
    assert shapes.w == (82, 4, 8)
    assert shapes.u == (82, 4, 10)
    assert shapes.solved_fp32 == (4, 28, 28)
    assert shapes.k_factor == (4, 28, 8)
    assert shapes.v_factor == (4, 28, 10)

    first = _build_operands(torch, args, device=torch.device("cpu"))
    second = _build_operands(torch, args, device=torch.device("cpu"))
    assert first.k.dtype is first.v.dtype is first.A.dtype is torch.bfloat16
    assert first.beta.dtype is first.g_cumsum.dtype is torch.float32
    assert first.cu_seqlens.dtype is torch.int32
    assert first.w.dtype is first.u.dtype is torch.bfloat16
    assert first.boundaries == (0, 28, 55, 82)
    assert tuple(first.cu_seqlens.tolist()) == first.boundaries
    assert first.workspaces.head_to_key.tolist() == [0, 0, 1, 1]
    assert first.workspaces.k_factor_bf16.dtype is torch.bfloat16
    assert first.workspaces.k_factor_fp32.dtype is torch.float32
    assert first.workspaces.v_factor_bf16.dtype is torch.bfloat16
    assert first.workspaces.v_factor_fp32.dtype is torch.float32
    assert torch.equal(first.k, second.k)
    assert torch.equal(first.v, second.v)
    assert torch.equal(first.beta, second.beta)
    assert torch.equal(first.g_cumsum, second.g_cumsum)
    assert torch.equal(first.A, second.A)


def test_device_generic_helper_agrees_with_accepted_reference_and_is_immutable() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_recompute_w_u_reference import (
        gdn_chunk_recompute_w_u_reference,
    )
    from profiling.runners.attention.gdn_chunk_recompute_w_u_torch import (
        _build_operands,
        _recompute_w_u_into,
        _validate_args,
    )

    args = _validate_args(70, 2, 2, 4, 5, 7, "bf16")
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    snapshots = tuple(
        tensor.clone()
        for tensor in (
            operands.k,
            operands.v,
            operands.beta,
            operands.g_cumsum,
            operands.A,
            operands.cu_seqlens,
        )
    )
    expected = gdn_chunk_recompute_w_u_reference(
        operands.k,
        operands.v,
        operands.beta,
        operands.g_cumsum,
        operands.A,
        operands.cu_seqlens,
    )

    actual = _recompute_w_u_into(
        torch,
        operands.k,
        operands.v,
        operands.beta,
        operands.g_cumsum,
        operands.A,
        operands.w,
        operands.u,
        operands.boundaries,
        operands.workspaces,
    )

    assert actual == (operands.w, operands.u)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    for original, snapshot in zip(
        (
            operands.k,
            operands.v,
            operands.beta,
            operands.g_cumsum,
            operands.A,
            operands.cu_seqlens,
        ),
        snapshots,
        strict=True,
    ):
        assert torch.equal(original, snapshot)


def test_helper_grouping_sign_gate_resets_and_repeated_overwrite() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_recompute_w_u_torch import (
        _build_operands,
        _recompute_w_u_into,
        _validate_args,
    )

    args = _validate_args(6, 2, 2, 4, 1, 1, "bf16")
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    operands.k.zero_()
    operands.k[:3, 0, 0] = 1
    operands.k[3:, 1, 0] = 2
    operands.v.fill_(1)
    operands.beta.copy_(
        torch.tensor(
            [[1, -1, 0, 2], [2, 3, 4, 5], [1, 1, 1, 1]] * 2,
            dtype=torch.float32,
        )
    )
    operands.g_cumsum.zero_()
    operands.A.zero_()
    for start, end in zip(operands.boundaries, operands.boundaries[1:]):
        length = end - start
        operands.A[torch.arange(start, end), :, torch.arange(length)] = 1
    operands.A[1, :, 0] = -1

    def call():
        return _recompute_w_u_into(
            torch,
            operands.k,
            operands.v,
            operands.beta,
            operands.g_cumsum,
            operands.A,
            operands.w,
            operands.u,
            operands.boundaries,
            operands.workspaces,
        )

    first = tuple(tensor.clone() for tensor in call())
    assert first[0][1, 0, 0].item() == 1
    assert first[0][1, 1, 0].item() == 4
    assert first[0][1, 2, 0].item() == 0
    assert first[0][3, 0, 0].item() == 0
    assert first[0][3, 2, 0].item() == 0

    operands.w.fill_(7)
    operands.u.fill_(7)
    for value in vars(operands.workspaces).values():
        if hasattr(value, "fill_") and value is not operands.workspaces.head_to_key:
            value.fill_(11)
    second = call()
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_factor_rounding_occurs_before_fp32_matrix_product() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_recompute_w_u_torch import (
        _build_operands,
        _recompute_w_u_into,
        _validate_args,
    )

    args = _validate_args(3, 1, 1, 1, 1, 1, "bf16")
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    operands.k.fill_(1)
    operands.v.fill_(1)
    operands.beta.fill_(0.502)
    operands.g_cumsum.zero_()
    operands.A.zero_()
    operands.A[0, 0, 0] = operands.A[1, 0, 1] = operands.A[2, 0, 2] = 1
    operands.A[2, 0, :2] = 1

    w, u = _recompute_w_u_into(
        torch,
        operands.k,
        operands.v,
        operands.beta,
        operands.g_cumsum,
        operands.A,
        operands.w,
        operands.u,
        operands.boundaries,
        operands.workspaces,
    )
    all_fp32 = (operands.beta[:, 0].sum()).to(torch.bfloat16)
    assert w[2, 0, 0] != all_fp32
    assert u[2, 0, 0] != all_fp32


def test_runner_reports_missing_cuda_as_typed_unsupported() -> None:
    from profiling.runners.attention.gdn_chunk_recompute_w_u_torch import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)


def test_semantic_metrics_are_exact_logical_counts() -> None:
    from profiling.runners.attention.gdn_chunk_recompute_w_u_torch import (
        _logical_bytes,
        _semantic_flops,
    )

    assert (
        _semantic_flops(
            num_tokens=3,
            num_chunks=1,
            num_heads=2,
            key_head_dim=2,
            value_head_dim=3,
        )
        == 138
    )
    assert (
        _semantic_flops(
            num_tokens=128,
            num_chunks=2,
            num_heads=32,
            key_head_dim=128,
            value_head_dim=128,
        )
        == 68_685_824
    )
    assert (
        _logical_bytes(
            num_tokens=128,
            num_key_heads=16,
            num_heads=32,
            key_head_dim=128,
            value_head_dim=128,
        )
        == 4_227_072
    )


def test_profile_times_only_preallocated_semantics_and_needs_no_reset(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_recompute_w_u_torch as runner

    args = runner._validate_args(6, 2, 2, 4, 2, 3, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    snapshots = tuple(
        tensor.clone()
        for tensor in (
            operands.k,
            operands.v,
            operands.beta,
            operands.g_cumsum,
            operands.A,
            operands.cu_seqlens,
        )
    )
    pointers = (operands.w.data_ptr(), operands.u.data_ptr())
    calls = {"build": 0, "semantic": 0, "timer": 0, "energy": 0}
    original_semantic = runner._recompute_w_u_into

    def fake_build(*_args, **_kwargs):
        calls["build"] += 1
        return operands

    def counted_semantic(*positional, **keyword):
        calls["semantic"] += 1
        return original_semantic(*positional, **keyword)

    def fake_timer(fn):
        calls["timer"] += 1
        assert fn() == (operands.w, operands.u)
        return 0.5

    def fake_energy(fn, *, warmup, per_iter_time_ms):
        calls["energy"] += 1
        first = tuple(tensor.clone() for tensor in (operands.w, operands.u))
        assert fn() == (operands.w, operands.u)
        assert all(torch.equal(left, right) for left, right in zip(first, (operands.w, operands.u)))
        assert warmup == 5 and per_iter_time_ms == 0.5
        return 0.25

    monkeypatch.setattr(runner, "_validate_cuda_device", lambda _torch: None)
    monkeypatch.setattr(runner, "_build_operands", fake_build)
    monkeypatch.setattr(runner, "_recompute_w_u_into", counted_semantic)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))

    metrics = runner.profile_gdn_chunk_recompute_w_u(6, 2, 2, 4, 2, 3, "bf16")
    assert metrics.time_ms == 0.5 and metrics.energy_j == 0.25
    assert calls == {"build": 1, "semantic": 2, "timer": 1, "energy": 1}
    assert (operands.w.data_ptr(), operands.u.data_ptr()) == pointers
    for original, snapshot in zip(
        (
            operands.k,
            operands.v,
            operands.beta,
            operands.g_cumsum,
            operands.A,
            operands.cu_seqlens,
        ),
        snapshots,
        strict=True,
    ):
        assert torch.equal(original, snapshot)


def test_generated_facades_and_read_only_missing_query(tmp_path, monkeypatch) -> None:
    assert hasattr(perf_api, "get_gdn_chunk_recompute_w_u_times")
    assert hasattr(perf_api, "count_missing_gdn_chunk_recompute_w_u")
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")

    assert (
        perf_api.count_missing_gdn_chunk_recompute_w_u(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 1
    )
    result = perf_api.get_gdn_chunk_recompute_w_u_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(result, MissingEntry)
    assert result.args == coerce_args(GdnChunkRecomputeWUArgs, _SPEC)
    assert not perf_api.DB_PATH.exists()

    table = Table(find_kernel_profiler_spec(KIND, "torch"), perf_api.DB_PATH)
    assert table.args_columns == [
        "num_tokens",
        "num_chunks",
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
    args = coerce_args(GdnChunkRecomputeWUArgs, _SPEC)
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

    result = perf_api.get_gdn_chunk_recompute_w_u_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert result == metrics
    assert (
        perf_api.count_missing_gdn_chunk_recompute_w_u(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 0
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"num_tokens": 0}, "must be > 0"),
        ({"num_chunks": 0}, "must be > 0"),
        ({"num_chunks": 1}, "ceil"),
        ({"num_chunks": 129}, "ceil"),
        ({"num_key_heads": 3}, "divisible"),
        ({"key_head_dim": 64}, "key_head_dim=value_head_dim=128"),
        ({"value_head_dim": 64}, "key_head_dim=value_head_dim=128"),
        ({"dtype": "fp16"}, "requires dtype=bf16"),
    ],
)
def test_vllm_rejects_invalid_args_before_import(
    updates: dict[str, object], message: str, monkeypatch
) -> None:
    import profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=message):
        runner.profile_gdn_chunk_recompute_w_u_vllm_triton(**(_SPEC | updates))


@pytest.mark.parametrize(
    "name",
    [
        "num_tokens",
        "num_chunks",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "value_head_dim",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_vllm_rejects_noninteger_args_before_import(name: str, value: object, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="exact integer"):
        runner.profile_gdn_chunk_recompute_w_u_vllm_triton(**(_SPEC | {name: value}))


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_vllm_rejects_truthy_fast_ops_before_import(value: str, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton as runner

    monkeypatch.setenv("FLA_USE_FAST_OPS", value)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="FLA_USE_FAST_OPS"):
        runner.profile_gdn_chunk_recompute_w_u_vllm_triton(**_SPEC)


@pytest.mark.parametrize("value", [None, "", "0", "false", "False"])
def test_vllm_accepts_safe_fast_ops_settings(value: str | None, monkeypatch) -> None:
    from profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton import (
        _validate_args,
    )

    if value is None:
        monkeypatch.delenv("FLA_USE_FAST_OPS", raising=False)
    else:
        monkeypatch.setenv("FLA_USE_FAST_OPS", value)
    assert _validate_args(**_SPEC).num_tokens == 128


def test_vllm_shapes_metadata_strides_and_deterministic_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton import (
        _build_operands,
        _canonical_index_pairs,
        _operand_shapes,
        _validate_args,
        _validate_operands,
    )

    args = _validate_args(82, 3, 4, 8, 128, 128, "bf16")
    shapes = _operand_shapes(args)
    assert shapes.k == (1, 82, 4, 128)
    assert shapes.v == (1, 82, 8, 128)
    assert shapes.beta == shapes.g_cumsum == (1, 82, 8)
    assert shapes.A == (1, 82, 8, 64)
    assert shapes.cu_seqlens == (4,)
    assert shapes.chunk_indices == (3, 2)
    assert shapes.w == shapes.u == (1, 82, 8, 128)
    assert _canonical_index_pairs(3) == ((0, 0), (1, 0), (2, 0))

    first = _build_operands(torch, args, device=torch.device("cpu"))
    second = _build_operands(torch, args, device=torch.device("cpu"))
    _validate_operands(torch, first, args, require_cuda=False)
    assert first.boundaries == (0, 28, 55, 82)
    assert first.cu_seqlens.tolist() == [0, 28, 55, 82]
    assert first.chunk_indices.tolist() == [[0, 0], [1, 0], [2, 0]]
    for tensor in (
        first.k,
        first.v,
        first.beta,
        first.g_cumsum,
        first.A,
        first.cu_seqlens,
        first.chunk_indices,
    ):
        assert tensor.is_contiguous() and tensor.stride(-1) == 1
    assert first.k.dtype is first.v.dtype is first.A.dtype is torch.bfloat16
    assert first.beta.dtype is first.g_cumsum.dtype is torch.float32
    assert first.cu_seqlens.dtype is first.chunk_indices.dtype is torch.int32
    for name in ("k", "v", "beta", "g_cumsum", "A", "cu_seqlens", "chunk_indices"):
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_vllm_guard_geometry_bounds_storage_and_preserves_qwen() -> None:
    import profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton as runner

    qwen = runner._validate_args(**_SPEC)
    assert runner._guard_args(qwen) is qwen
    assert qwen.num_tokens * runner._guard_elements_per_token(qwen) <= 2_500_000

    large = runner._validate_args(262_144, 4096, 16, 32, 128, 128, "bf16")
    guard = runner._guard_args(large)
    assert guard.num_tokens == 151
    assert guard.num_chunks == 150
    assert guard.num_key_heads == 16 and guard.num_heads == 32
    assert guard.key_head_dim == guard.value_head_dim == 128
    assert guard.num_tokens * runner._guard_elements_per_token(guard) <= 2_500_000
    assert max(runner._canonical_lengths(guard.num_tokens, guard.num_chunks)) == 2
    assert (guard.num_tokens + 63) // 64 <= guard.num_chunks <= guard.num_tokens

    impossible = runner._validate_args(2, 1, 1, 8192, 128, 128, "bf16")
    with pytest.raises(ValueError, match="minimum nontrivial correctness guard"):
        runner._guard_args(impossible)


def test_vllm_operand_validation_rejects_layout_metadata_finiteness_and_dirty_A() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton as runner

    args = runner._validate_args(5, 2, 2, 4, 128, 128, "bf16")

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.cu_seqlens[1] = 4
    with pytest.raises(ValueError, match="canonical boundaries"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.chunk_indices[1, 1] = 1
    with pytest.raises(ValueError, match="canonical sequence/chunk mapping"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.beta[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.A[0, 0, 0, 0] = 0
    with pytest.raises(ValueError, match="unit diagonal"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.A[0, 0, 0, 1] = 1
    with pytest.raises(ValueError, match="upper triangle"):
        runner._validate_operands(torch, operands, args, require_cuda=False)

    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    backing = torch.empty((1, 5, 2, 256), dtype=torch.bfloat16)
    object.__setattr__(operands, "k", backing[..., ::2])
    with pytest.raises(ValueError, match="contiguous"):
        runner._validate_operands(torch, operands, args, require_cuda=False)


def test_vllm_correctness_guard_checks_both_outputs_and_immutability() -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton as runner
    from profiling.runners.attention.gdn_chunk_recompute_w_u_reference import (
        gdn_chunk_recompute_w_u_reference,
    )

    args = runner._validate_args(5, 2, 2, 4, 128, 128, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    snapshots = {
        name: getattr(operands, name).clone()
        for name in ("k", "v", "beta", "g_cumsum", "A", "cu_seqlens", "chunk_indices")
    }
    calls = 0

    def fake_fused(k, v, beta, g, A, cu_seqlens, chunk_indices):
        nonlocal calls
        calls += 1
        assert chunk_indices.tolist() == [[0, 0], [1, 0]]
        w, u = gdn_chunk_recompute_w_u_reference(
            k.squeeze(0),
            v.squeeze(0),
            beta.squeeze(0),
            g.squeeze(0),
            A.squeeze(0),
            cu_seqlens,
        )
        return w.unsqueeze(0), u.unsqueeze(0)

    runner._check_correctness(torch, fake_fused, operands, args, synchronize=lambda: None)
    assert calls == 1
    for name, snapshot in snapshots.items():
        assert torch.equal(getattr(operands, name), snapshot)


def test_vllm_profiles_one_wrapper_call_and_reuses_semantic_metrics(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton as runner

    args = runner._validate_args(5, 2, 1, 2, 128, 128, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    calls = {"build": 0, "guard": 0, "fused": 0, "timer": 0, "energy": 0}

    def fake_build(*_args, **_kwargs):
        calls["build"] += 1
        return operands

    def fake_fused(*_args, **_kwargs):
        calls["fused"] += 1
        return (
            torch.empty((1, 5, 2, 128), dtype=torch.bfloat16),
            torch.empty((1, 5, 2, 128), dtype=torch.bfloat16),
        )

    def fake_guard(*_args, **_kwargs):
        calls["guard"] += 1

    def fake_timer(fn, *, kernel_name):
        calls["timer"] += 1
        assert kernel_name == "recompute_w_u_fwd_kernel"
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

    metrics = runner.profile_gdn_chunk_recompute_w_u_vllm_triton(5, 2, 1, 2, 128, 128, "bf16")
    assert metrics.time_ms == 0.5 and metrics.energy_j == 0.25
    assert calls == {"build": 1, "guard": 1, "fused": 2, "timer": 1, "energy": 1}
    assert metrics.tflops > 0 and metrics.memory_bandwidth_gbps > 0


def test_generated_facades_preserve_both_backend_queries(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")
    for backend in ("torch", "vllm_triton"):
        assert (
            perf_api.count_missing_gdn_chunk_recompute_w_u(
                [_SPEC], backend=backend, gpu_name="NVIDIA H200"
            )
            == 1
        )
        result = perf_api.get_gdn_chunk_recompute_w_u_times(
            [_SPEC], backend=backend, gpu_name="NVIDIA H200"
        )[0]
        assert isinstance(result, MissingEntry)
    assert not perf_api.DB_PATH.exists()
