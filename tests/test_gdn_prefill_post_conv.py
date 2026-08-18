"""Registration and CPU runner tests for ``gdn_prefill_post_conv``."""

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
from profiling.kernels.gdn_prefill_post_conv import (
    KIND,
    GdnPrefillPostConvArgs,
)
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SPEC = {
    "num_tokens": 128,
    "num_qk_heads": 16,
    "num_value_heads": 32,
    "key_head_dim": 128,
    "value_head_dim": 128,
    "dtype": "bf16",
}


def test_args_field_order_and_coercion() -> None:
    assert [field.name for field in fields(GdnPrefillPostConvArgs)] == [
        "num_tokens",
        "num_qk_heads",
        "num_value_heads",
        "key_head_dim",
        "value_head_dim",
        "dtype",
    ]
    args = coerce_args(
        GdnPrefillPostConvArgs,
        {
            "num_tokens": "128",
            "num_qk_heads": "16",
            "num_value_heads": "32",
            "key_head_dim": "128",
            "value_head_dim": "128",
            "dtype": "torch.bfloat16",
        },
    )
    assert args == GdnPrefillPostConvArgs(
        num_tokens=128,
        num_qk_heads=16,
        num_value_heads=32,
        key_head_dim=128,
        value_head_dim=128,
        dtype=DType.BF16,
    )
    with pytest.raises(Exception):
        args.num_tokens = 1


def test_registration_table_kind_runner_and_support_contract() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")

    assert KIND == "gdn_prefill_post_conv"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.args_schema is GdnPrefillPostConvArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_prefill_post_conv_torch"
    )
    assert spec.runner_ref.function_name == "profile_gdn_prefill_post_conv"

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
    assert spec.args_schema is GdnPrefillPostConvArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env == "vllm_env"
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_prefill_post_conv_vllm_triton"
    )
    assert spec.runner_ref.function_name == "profile_gdn_prefill_post_conv_vllm_triton"

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
                "print('profiling.runners.attention.gdn_prefill_post_conv_torch' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_prefill_post_conv_vllm_triton' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_prefill_post_conv_reference' "
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
                "'gdn_prefill_post_conv', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_prefill_post_conv_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_prefill_post_conv_torch",
        "profile_gdn_prefill_post_conv",
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
                "'gdn_prefill_post_conv', 'vllm_triton').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); print('vllm' in sys.modules); "
                "print('profiling.runners.attention.gdn_prefill_post_conv_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_prefill_post_conv_vllm_triton",
        "profile_gdn_prefill_post_conv_vllm_triton",
        "False",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "num_tokens",
        "num_qk_heads",
        "num_value_heads",
        "key_head_dim",
        "value_head_dim",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_runner_rejects_nonpositive_dimensions_before_import(
    name: str,
    value: int,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_prefill_post_conv_torch as runner

    def forbidden_build(*_args, **_kwargs):
        raise AssertionError("CUDA allocation must not be reached")

    monkeypatch.setattr(runner, "_build_operands", forbidden_build)
    with pytest.raises(ValueError, match="must be > 0"):
        runner.profile_gdn_prefill_post_conv(**(_SPEC | {name: value}))


@pytest.mark.parametrize("dtype", ["fp16", "fp32", "fp8_e4m3"])
def test_runner_rejects_unsupported_dtype_before_import(dtype: str, monkeypatch) -> None:
    import profiling.runners.attention.gdn_prefill_post_conv_torch as runner

    def forbidden_build(*_args, **_kwargs):
        raise AssertionError("CUDA allocation must not be reached")

    monkeypatch.setattr(runner, "_build_operands", forbidden_build)
    with pytest.raises(ValueError, match="requires dtype=bf16"):
        runner.profile_gdn_prefill_post_conv(**(_SPEC | {"dtype": dtype}))


def test_runner_shapes_and_bounded_deterministic_cpu_operands() -> None:
    import torch

    from profiling.runners.attention.gdn_prefill_post_conv_torch import (
        _build_operands,
        _operand_shapes,
        _validate_args,
    )

    args = _validate_args(
        num_tokens=3,
        num_qk_heads=2,
        num_value_heads=3,
        key_head_dim=4,
        value_head_dim=5,
        dtype="bf16",
    )
    shapes = _operand_shapes(args)
    assert shapes.conv_output == (3, 31)
    assert shapes.a == shapes.b == shapes.g == shapes.beta == (3, 3)
    assert shapes.A_log == shapes.dt_bias == (3,)
    assert shapes.q == shapes.k == (3, 2, 4)
    assert shapes.v == (3, 3, 5)

    first = _build_operands(torch, args, device=torch.device("cpu"))
    second = _build_operands(torch, args, device=torch.device("cpu"))
    for operand in (first.conv_output, first.a, first.b):
        assert operand.dtype is torch.bfloat16
        assert torch.isfinite(operand).all()
    for parameter in (first.A_log, first.dt_bias):
        assert parameter.dtype is torch.float32
        assert torch.isfinite(parameter).all()
    for name in ("conv_output", "a", "b", "A_log", "dt_bias"):
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_runner_reports_missing_cuda_as_typed_unsupported() -> None:
    from profiling.runners.attention.gdn_prefill_post_conv_torch import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)


def test_semantic_metrics_are_explicit_logical_counts() -> None:
    from profiling.runners.attention.gdn_prefill_post_conv_torch import (
        _logical_bytes,
        _semantic_flops,
    )

    assert (
        _semantic_flops(
            num_tokens=2,
            num_qk_heads=2,
            num_value_heads=3,
            key_head_dim=4,
        )
        == 131
    )
    assert (
        _logical_bytes(
            num_tokens=2,
            num_qk_heads=2,
            num_value_heads=3,
            key_head_dim=4,
            value_head_dim=5,
            dtype=DType.BF16,
        )
        == 344
    )


@pytest.mark.parametrize(
    "name",
    [
        "num_tokens",
        "num_qk_heads",
        "num_value_heads",
        "key_head_dim",
        "value_head_dim",
    ],
)
def test_vllm_runner_rejects_nonpositive_dimensions_before_import(
    name: str,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_prefill_post_conv_vllm_triton as runner

    monkeypatch.setattr(
        runner,
        "_load_fused_callable",
        lambda: (_ for _ in ()).throw(AssertionError("vLLM import must not be reached")),
    )
    with pytest.raises(ValueError, match="must be > 0"):
        runner.profile_gdn_prefill_post_conv_vllm_triton(**(_SPEC | {name: 0}))


@pytest.mark.parametrize("dtype", ["fp16", "fp32", "fp8_e4m3"])
def test_vllm_runner_rejects_unsupported_dtype_before_import(dtype: str, monkeypatch) -> None:
    import profiling.runners.attention.gdn_prefill_post_conv_vllm_triton as runner

    monkeypatch.setattr(
        runner,
        "_load_fused_callable",
        lambda: (_ for _ in ()).throw(AssertionError("vLLM import must not be reached")),
    )
    with pytest.raises(ValueError, match="requires dtype=bf16"):
        runner.profile_gdn_prefill_post_conv_vllm_triton(**(_SPEC | {"dtype": dtype}))


@pytest.mark.parametrize(
    ("key_head_dim", "value_head_dim"),
    [(7, 5), (64, 128), (128, 64), (32, 32), (256, 256)],
)
def test_vllm_runner_rejects_unestablished_head_dimensions_before_import(
    key_head_dim: int,
    value_head_dim: int,
    monkeypatch,
) -> None:
    import profiling.runners.attention.gdn_prefill_post_conv_vllm_triton as runner

    monkeypatch.setattr(
        runner,
        "_load_fused_callable",
        lambda: (_ for _ in ()).throw(AssertionError("vLLM import must not be reached")),
    )
    with pytest.raises(ValueError, match="established"):
        runner.profile_gdn_prefill_post_conv_vllm_triton(
            **(
                _SPEC
                | {
                    "key_head_dim": key_head_dim,
                    "value_head_dim": value_head_dim,
                }
            )
        )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_vllm_runner_accepts_established_head_dimension_pairs(head_dim: int) -> None:
    from profiling.runners.attention.gdn_prefill_post_conv_vllm_triton import (
        _validate_args,
    )

    args = _validate_args(**(_SPEC | {"key_head_dim": head_dim, "value_head_dim": head_dim}))
    assert (args.key_head_dim, args.value_head_dim) == (head_dim, head_dim)


def test_vllm_runner_shapes_dtypes_contiguity_and_guard_bound() -> None:
    import torch

    from profiling.runners.attention.gdn_prefill_post_conv_vllm_triton import (
        _build_operands,
        _guard_args,
        _operand_shapes,
        _validate_args,
        _validate_operands,
    )

    qwen = _validate_args(**_SPEC)
    shapes = _operand_shapes(qwen)
    assert shapes.conv_output == (128, 8192)
    assert shapes.a == shapes.b == shapes.g == shapes.beta == (128, 32)
    assert shapes.A_log == shapes.dt_bias == (32,)
    assert shapes.q == shapes.k == (128, 16, 128)
    assert shapes.v == (128, 32, 128)
    assert _guard_args(qwen) == qwen
    assert _guard_args(_validate_args(**(_SPEC | {"num_tokens": 129}))).num_tokens == 128

    small = _validate_args(
        num_tokens=3,
        num_qk_heads=2,
        num_value_heads=3,
        key_head_dim=64,
        value_head_dim=64,
        dtype="bf16",
    )
    first = _build_operands(torch, small, device=torch.device("cpu"))
    second = _build_operands(torch, small, device=torch.device("cpu"))
    _validate_operands(torch, first, small, require_cuda=False)
    for name in ("conv_output", "a", "b"):
        tensor = getattr(first, name)
        assert tensor.dtype is torch.bfloat16
        assert tensor.is_contiguous() and tensor.stride(-1) == 1
        assert torch.equal(tensor, getattr(second, name))
    for name in ("A_log", "dt_bias"):
        tensor = getattr(first, name)
        assert tensor.dtype is torch.float32
        assert tensor.is_contiguous()
        assert torch.equal(tensor, getattr(second, name))


def test_vllm_operand_validation_closes_shape_dtype_and_layout_gaps() -> None:
    import torch

    import profiling.runners.attention.gdn_prefill_post_conv_vllm_triton as runner

    args = runner._validate_args(
        num_tokens=3,
        num_qk_heads=2,
        num_value_heads=3,
        key_head_dim=64,
        value_head_dim=64,
        dtype="bf16",
    )
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))

    wrong_shape = runner._Operands(
        **{
            **operands.__dict__,
            "a": torch.empty(3, 4, dtype=torch.bfloat16),
        }
    )
    with pytest.raises(ValueError, match="a must have shape"):
        runner._validate_operands(torch, wrong_shape, args, require_cuda=False)

    wrong_dtype = runner._Operands(
        **{
            **operands.__dict__,
            "A_log": operands.A_log.to(torch.bfloat16),
        }
    )
    with pytest.raises(ValueError, match="A_log must have dtype"):
        runner._validate_operands(torch, wrong_dtype, args, require_cuda=False)

    base = torch.empty(
        operands.conv_output.shape[0],
        operands.conv_output.shape[1] * 2,
        dtype=torch.bfloat16,
    )
    noncontiguous = runner._Operands(
        **{
            **operands.__dict__,
            "conv_output": base[:, ::2],
        }
    )
    with pytest.raises(ValueError, match="unit feature stride"):
        runner._validate_operands(torch, noncontiguous, args, require_cuda=False)


def test_vllm_correctness_guard_checks_all_outputs_and_immutability_on_cpu() -> None:
    import torch

    import profiling.runners.attention.gdn_prefill_post_conv_vllm_triton as runner
    from profiling.runners.attention.gdn_prefill_post_conv_reference import (
        gdn_prefill_post_conv_reference,
    )

    args = runner._validate_args(
        num_tokens=3,
        num_qk_heads=2,
        num_value_heads=3,
        key_head_dim=64,
        value_head_dim=64,
        dtype="bf16",
    )
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    snapshots = {
        name: getattr(operands, name).clone()
        for name in ("conv_output", "a", "b", "A_log", "dt_bias")
    }

    def fake_fused(conv_output, a, b, A_log, dt_bias, **kwargs):
        assert kwargs == {
            "num_k_heads": 2,
            "head_k_dim": 64,
            "head_v_dim": 64,
            "apply_l2norm": True,
            "output_g_exp": False,
        }
        return gdn_prefill_post_conv_reference(
            conv_output,
            a,
            b,
            A_log,
            dt_bias,
            num_qk_heads=2,
            num_value_heads=3,
            key_head_dim=64,
            value_head_dim=64,
        )

    runner._check_correctness(
        torch,
        fake_fused,
        operands,
        args,
        synchronize=lambda: None,
    )
    for name, snapshot in snapshots.items():
        assert torch.equal(getattr(operands, name), snapshot)

    def aliasing_fused(conv_output, *_args, **_kwargs):
        expected = fake_fused(
            operands.conv_output,
            operands.a,
            operands.b,
            operands.A_log,
            operands.dt_bias,
            num_k_heads=2,
            head_k_dim=64,
            head_v_dim=64,
            apply_l2norm=True,
            output_g_exp=False,
        )
        return (conv_output.view(-1)[:384].view(3, 2, 64), *expected[1:])

    with pytest.raises(AssertionError, match="fresh storage"):
        runner._check_correctness(
            torch,
            aliasing_fused,
            operands,
            args,
            synchronize=lambda: None,
        )


def test_vllm_profile_times_only_one_fused_call_and_needs_no_reset(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_prefill_post_conv_vllm_triton as runner

    args = runner._validate_args(
        num_tokens=3,
        num_qk_heads=2,
        num_value_heads=3,
        key_head_dim=64,
        value_head_dim=64,
        dtype="bf16",
    )
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    snapshots = tuple(
        getattr(operands, name).clone() for name in ("conv_output", "a", "b", "A_log", "dt_bias")
    )
    calls = {"build": 0, "guard": 0, "fused": 0, "timer": 0, "energy": 0}

    def fake_build(*_args, **_kwargs):
        calls["build"] += 1
        return operands

    def fake_guard(*_args, **_kwargs):
        calls["guard"] += 1

    def fake_fused(*_args, **_kwargs):
        calls["fused"] += 1
        return ()

    def fake_timer(fn, *, kernel_name):
        calls["timer"] += 1
        before = calls["fused"]
        fn()
        assert calls["fused"] == before + 1
        assert kernel_name == "_fused_post_conv_kernel"
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
    monkeypatch.setattr(runner, "_build_operands", fake_build)
    monkeypatch.setattr(runner, "_validate_operands", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_check_correctness", fake_guard)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    metrics = runner.profile_gdn_prefill_post_conv_vllm_triton(
        num_tokens=3,
        num_qk_heads=2,
        num_value_heads=3,
        key_head_dim=64,
        value_head_dim=64,
        dtype="bf16",
    )
    assert metrics.time_ms == 0.5
    assert metrics.energy_j == 0.25
    assert calls == {"build": 1, "guard": 1, "fused": 2, "timer": 1, "energy": 1}
    for name, snapshot in zip(
        ("conv_output", "a", "b", "A_log", "dt_bias"),
        snapshots,
        strict=True,
    ):
        assert torch.equal(getattr(operands, name), snapshot)


def test_profile_times_only_semantic_call_and_needs_no_reset(monkeypatch) -> None:
    import torch

    import profiling.runners.attention.gdn_prefill_post_conv_reference as reference
    import profiling.runners.attention.gdn_prefill_post_conv_torch as runner

    args = runner._validate_args(
        num_tokens=3,
        num_qk_heads=2,
        num_value_heads=3,
        key_head_dim=4,
        value_head_dim=5,
        dtype="bf16",
    )
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    snapshots = tuple(
        getattr(operands, name).clone() for name in ("conv_output", "a", "b", "A_log", "dt_bias")
    )
    original_reference = reference.gdn_prefill_post_conv_reference
    calls = {"build": 0, "reference": 0, "timer": 0, "energy": 0}

    def fake_build(*_args, **_kwargs):
        calls["build"] += 1
        return operands

    def counted_reference(*positional, **keyword):
        calls["reference"] += 1
        return original_reference(*positional, **keyword)

    def fake_timer(fn):
        calls["timer"] += 1
        outputs = fn()
        assert [tuple(output.shape) for output in outputs] == [
            (3, 2, 4),
            (3, 2, 4),
            (3, 3, 5),
            (3, 3),
            (3, 3),
        ]
        return 0.5

    def fake_energy(fn, *, warmup, per_iter_time_ms):
        calls["energy"] += 1
        fn()
        assert warmup == 5
        assert per_iter_time_ms == 0.5
        return 0.25

    monkeypatch.setattr(runner, "_validate_cuda_device", lambda _torch: None)
    monkeypatch.setattr(runner, "_build_operands", fake_build)
    monkeypatch.setattr(reference, "gdn_prefill_post_conv_reference", counted_reference)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(fake_energy))

    metrics = runner.profile_gdn_prefill_post_conv(
        num_tokens=3,
        num_qk_heads=2,
        num_value_heads=3,
        key_head_dim=4,
        value_head_dim=5,
        dtype="bf16",
    )
    assert metrics.time_ms == 0.5
    assert metrics.energy_j == 0.25
    assert calls == {"build": 1, "reference": 2, "timer": 1, "energy": 1}
    for name, snapshot in zip(
        ("conv_output", "a", "b", "A_log", "dt_bias"),
        snapshots,
        strict=True,
    ):
        assert torch.equal(getattr(operands, name), snapshot)


def test_generated_facades_and_read_only_missing_query(tmp_path, monkeypatch) -> None:
    assert hasattr(perf_api, "get_gdn_prefill_post_conv_times")
    assert hasattr(perf_api, "count_missing_gdn_prefill_post_conv")
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")

    assert (
        perf_api.count_missing_gdn_prefill_post_conv(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 1
    )
    result = perf_api.get_gdn_prefill_post_conv_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(result, MissingEntry)
    assert result.args == coerce_args(GdnPrefillPostConvArgs, _SPEC)
    assert (
        perf_api.count_missing_gdn_prefill_post_conv(
            [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
        )
        == 1
    )
    vllm_result = perf_api.get_gdn_prefill_post_conv_times(
        [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(vllm_result, MissingEntry)
    assert vllm_result.args == result.args
    assert not perf_api.DB_PATH.exists()

    table = Table(find_kernel_profiler_spec(KIND, "torch"), perf_api.DB_PATH)
    assert table.args_columns == [
        "num_tokens",
        "num_qk_heads",
        "num_value_heads",
        "key_head_dim",
        "value_head_dim",
        "dtype",
    ]


def test_db_round_trip_uses_exact_schema_and_compute_metrics(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "profile.db"
    monkeypatch.setattr(perf_api, "DB_PATH", db_path)
    profiler_spec = find_kernel_profiler_spec(KIND, "torch")
    args = coerce_args(GdnPrefillPostConvArgs, _SPEC)
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

    result = perf_api.get_gdn_prefill_post_conv_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert result == metrics
    assert (
        perf_api.count_missing_gdn_prefill_post_conv(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 0
    )
