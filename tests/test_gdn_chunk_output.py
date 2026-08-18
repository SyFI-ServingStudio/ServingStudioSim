"""Registration and Torch runner tests for ``gdn_chunk_output``."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.kernels.gdn_chunk_output import KIND, GdnChunkOutputArgs
from profiling.runners.attention.gdn_chunk_output_reference import gdn_chunk_output_reference

_SPEC = dict(
    num_tokens=128,
    num_chunks=2,
    num_key_heads=16,
    num_heads=32,
    key_head_dim=128,
    value_head_dim=128,
    dtype="bf16",
)


def test_args_registration_support_and_facades() -> None:
    assert [field.name for field in fields(GdnChunkOutputArgs)] == [
        "num_tokens",
        "num_chunks",
        "num_key_heads",
        "num_heads",
        "key_head_dim",
        "value_head_dim",
        "dtype",
    ]
    args = coerce_args(GdnChunkOutputArgs, {k: str(v) for k, v in _SPEC.items()})
    assert args.dtype is DType.BF16 and args.num_tokens == 128
    spec = find_kernel_profiler_spec(KIND, "torch")
    vllm = find_kernel_profiler_spec(KIND, "vllm_triton")
    assert KIND == spec.kernel_kind == spec.table_name == "gdn_chunk_output"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.args_schema is GdnChunkOutputArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.subprocess_env == "default_env"
    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.runner_ref.module_name.endswith("gdn_chunk_output_torch")
    assert spec.runner_ref.function_name == "profile_gdn_chunk_output"
    assert vllm.args_schema is GdnChunkOutputArgs
    assert vllm.table_name == KIND
    assert vllm.metric_family is MetricFamily.COMPUTE
    assert vllm.subprocess_env == "vllm_env"
    assert vllm.supports.compute == frozenset({DType.BF16})
    assert vllm.supports.gpus == frozenset({"NVIDIA H200"})
    assert vllm.runner_ref.module_name.endswith("gdn_chunk_output_vllm_triton")
    assert vllm.runner_ref.function_name == "profile_gdn_chunk_output_vllm_triton"
    assert hasattr(perf_api, "get_gdn_chunk_output_times")
    assert hasattr(perf_api, "count_missing_gdn_chunk_output")


def test_registry_import_is_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_output_torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_output_vllm_triton' in sys.modules); "
                "print('vllm' in sys.modules); "
                "print('profiling.runners.attention.gdn_chunk_output_reference' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["False", "False", "False", "False", "False"]


@pytest.mark.parametrize(
    "name",
    ["num_tokens", "num_chunks", "num_key_heads", "num_heads", "key_head_dim", "value_head_dim"],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_validation_precedes_torch_import(name: str, value: object, monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_output_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError):
        runner.profile_gdn_chunk_output(**(_SPEC | {name: value}))


@pytest.mark.parametrize(
    "updates", [{"num_chunks": 1}, {"num_chunks": 129}, {"num_key_heads": 3}, {"dtype": "fp16"}]
)
def test_domain_validation_precedes_torch_import(updates: dict[str, object], monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_output_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError):
        runner.profile_gdn_chunk_output(**(_SPEC | updates))


@pytest.mark.parametrize(
    "updates",
    [
        {"num_tokens": 0},
        {"num_chunks": 1},
        {"num_key_heads": 3},
        {"key_head_dim": 64},
        {"value_head_dim": 64},
        {"dtype": "fp16"},
    ],
)
def test_vllm_validation_precedes_torch_import(updates: dict[str, object], monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_output_vllm_triton as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError):
        runner.profile_gdn_chunk_output_vllm_triton(**(_SPEC | updates))


def test_vllm_fast_ops_validation_precedes_import(monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_output_vllm_triton as runner

    monkeypatch.setenv("FLA_USE_FAST_OPS", "1")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="FLA_USE_FAST_OPS"):
        runner.profile_gdn_chunk_output_vllm_triton(**_SPEC)


def test_canonical_geometry_preserves_qwen_n1_and_general_invariants() -> None:
    import profiling.runners.attention.gdn_chunk_output_torch as runner

    assert runner._canonical_lengths(128, 2) == (128,)
    assert runner._canonical_boundaries(128, 2) == (0, 128)
    for tokens, chunks in [(1, 1), (65, 2), (128, 3), (128, 128), (4096, 64)]:
        lengths = runner._canonical_lengths(tokens, chunks)
        assert all(length > 0 for length in lengths)
        assert sum(lengths) == tokens
        assert sum((length + 63) // 64 for length in lengths) == chunks


def test_shapes_and_metric_goldens() -> None:
    import profiling.runners.attention.gdn_chunk_output_torch as runner

    args = runner._validate_args(**_SPEC)
    shapes = runner._operand_shapes(args)
    assert shapes["q"] == (128, 16, 128)
    assert shapes["v_new"] == shapes["output"] == (128, 32, 128)
    assert shapes["h"] == (2, 32, 128, 128)
    assert shapes["cu_seqlens"] == (2,)
    p = 64
    pairs = p * (p + 1) // 2
    per_chunk_head = (
        2 * p * 128 * 128
        + 2 * p * p * 128
        + p
        + p * 128
        + 2 * pairs
        + 2 * pairs * 128
        + 2 * p * 128
    )
    assert (
        runner._semantic_flops(
            num_tokens=128, num_chunks=2, num_heads=32, key_head_dim=128, value_head_dim=128
        )
        == 2 * 32 * per_chunk_head
    )
    expected_bytes = (
        4 * 128 * 16 * 128
        + 2 * 128 * 32 * 128
        + 2 * 2 * 32 * 128 * 128
        + 4 * 128 * 32
        + 2 * 128 * 32 * 128
    )
    assert (
        runner._logical_bytes(
            num_tokens=128,
            num_chunks=2,
            num_key_heads=16,
            num_heads=32,
            key_head_dim=128,
            value_head_dim=128,
        )
        == expected_bytes
    )


def test_vllm_qwen_shapes_metadata_guard_and_metric_reuse() -> None:
    import profiling.runners.attention.gdn_chunk_output_torch as torch_runner
    import profiling.runners.attention.gdn_chunk_output_vllm_triton as runner

    args = runner._validate_args(**_SPEC)
    assert runner._operand_shapes(args) == {
        "q": (1, 128, 16, 128),
        "k": (1, 128, 16, 128),
        "v_new": (1, 128, 32, 128),
        "h": (1, 2, 32, 128, 128),
        "g_cumsum": (1, 128, 32),
        "cu_seqlens": (2,),
        "chunk_indices": (2, 2),
        "output": (1, 128, 32, 128),
    }
    assert runner._boundaries(args) == (0, 128)
    assert runner._index_pairs(args) == ((0, 0), (0, 1))
    assert runner._guard_args(args) == args
    large = runner._validate_args(4096, 64, 16, 32, 128, 128, "bf16")
    witness = runner._guard_args(large)
    assert (witness.num_tokens, witness.num_chunks) == (2, 1)
    assert (witness.num_key_heads, witness.num_heads) == (16, 32)
    assert runner._semantic_flops is torch_runner._semantic_flops
    assert runner._logical_bytes is torch_runner._logical_bytes


def test_vllm_correctness_guard_and_operand_validation_on_cpu() -> None:
    import profiling.runners.attention.gdn_chunk_output_vllm_triton as runner

    args = runner._validate_args(3, 1, 2, 4, 128, 128, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))

    def semantic_callable(**kwargs):
        return gdn_chunk_output_reference(
            kwargs["q"].squeeze(0),
            kwargs["k"].squeeze(0),
            kwargs["v"].squeeze(0),
            kwargs["h"].squeeze(0),
            kwargs["g"].squeeze(0),
            kwargs["cu_seqlens"],
        ).unsqueeze(0)

    snapshots = {
        name: getattr(operands, name).clone()
        for name in operands.__dataclass_fields__
        if hasattr(getattr(operands, name), "clone")
    }
    runner._check_correctness(torch, semantic_callable, operands, args, synchronize=lambda: None)
    for name, snapshot in snapshots.items():
        assert torch.equal(getattr(operands, name), snapshot)

    dirty_h = operands.h.transpose(-1, -2)
    assert dirty_h.shape == operands.h.shape and not dirty_h.is_contiguous()
    with pytest.raises(ValueError, match="packed contiguous"):
        runner._validate_operands(torch, replace(operands, h=dirty_h), args, require_cuda=False)
    dirty_indices = operands.chunk_indices.clone()
    dirty_indices[0, 0] = 1
    with pytest.raises(ValueError, match="chunk_indices"):
        runner._validate_operands(
            torch,
            replace(operands, chunk_indices=dirty_indices),
            args,
            require_cuda=False,
        )


def test_vllm_timed_call_is_one_wrapper_with_exact_selector(monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_output_vllm_triton as runner

    events: list[object] = []
    fake_operands = SimpleNamespace(q=SimpleNamespace(shape=(1, 128, 16, 128)))
    fake_torch = SimpleNamespace(
        OutOfMemoryError=RuntimeError,
        cuda=SimpleNamespace(current_device=lambda: 0),
        device=lambda *args: "cuda:0",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runner, "_require_h200", lambda torch_: events.append("device"))
    monkeypatch.setattr(runner, "_load_callable", lambda: object())
    monkeypatch.setattr(
        runner, "_build_operands", lambda *a, **k: events.append("build") or fake_operands
    )
    monkeypatch.setattr(runner, "_validate_operands", lambda *a, **k: events.append("validate"))
    monkeypatch.setattr(runner, "_guard_args", lambda args: args)
    monkeypatch.setattr(runner, "_check_correctness", lambda *a, **k: events.append("guard"))
    monkeypatch.setattr(runner, "_invoke", lambda *a, **k: events.append("wrapper"))

    def timer(fn, *, kernel_name):
        events.append(("selector", kernel_name))
        fn()
        return 1.0

    monkeypatch.setattr(runner.Timer, "cupti", timer)
    monkeypatch.setattr(runner.Energy, "perf", lambda fn, **k: (fn(), 0.1)[1])
    metrics = runner.profile_gdn_chunk_output_vllm_triton(**_SPEC)
    assert metrics.time_ms == 1.0
    assert events == [
        "device",
        "build",
        "validate",
        "guard",
        ("selector", "chunk_fwd_kernel_o"),
        "wrapper",
        "wrapper",
    ]


def test_helper_matches_reference_grouping_rounding_partial_and_resets() -> None:
    import profiling.runners.attention.gdn_chunk_output_torch as runner

    args = runner._validate_args(70, 4, 2, 4, 3, 2, "bf16")
    # Override the canonical construction to exercise the accepted ragged reset case.
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    boundaries = (0, 3, 68, 70)
    h = operands.h.clone()
    expected = gdn_chunk_output_reference(
        operands.q,
        operands.k,
        operands.v_new,
        h,
        operands.g_cumsum,
        torch.tensor(boundaries, dtype=torch.int32),
    )
    actual = runner._chunk_output_into(
        torch,
        operands.q,
        operands.k,
        operands.v_new,
        h,
        operands.g_cumsum,
        operands.output,
        boundaries,
        operands.workspaces,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_helper_overwrites_output_and_preserves_inputs() -> None:
    import profiling.runners.attention.gdn_chunk_output_torch as runner

    args = runner._validate_args(3, 1, 1, 2, 2, 2, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    snapshots = [
        x.clone() for x in (operands.q, operands.k, operands.v_new, operands.h, operands.g_cumsum)
    ]
    first = runner._chunk_output_into(
        torch,
        operands.q,
        operands.k,
        operands.v_new,
        operands.h,
        operands.g_cumsum,
        operands.output,
        operands.boundaries,
        operands.workspaces,
    ).clone()
    operands.output.fill_(float("nan"))
    second = runner._chunk_output_into(
        torch,
        operands.q,
        operands.k,
        operands.v_new,
        operands.h,
        operands.g_cumsum,
        operands.output,
        operands.boundaries,
        operands.workspaces,
    )
    assert torch.equal(first, second)
    for actual, snapshot in zip(
        (operands.q, operands.k, operands.v_new, operands.h, operands.g_cumsum), snapshots
    ):
        assert torch.equal(actual, snapshot)


def test_profile_timing_isolation(monkeypatch) -> None:
    import profiling.runners.attention.gdn_chunk_output_torch as runner

    events: list[str] = []
    fake_operands = SimpleNamespace(
        q=None,
        k=None,
        v_new=None,
        h=None,
        g_cumsum=None,
        output=None,
        boundaries=(),
        workspaces=None,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        device=lambda name: name,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        runner, "_build_operands", lambda *a, **k: events.append("build") or fake_operands
    )
    monkeypatch.setattr(runner, "_chunk_output_into", lambda *a, **k: events.append("semantic"))
    monkeypatch.setattr(runner.Timer, "cupti", lambda fn: (events.append("timer"), fn(), 1.0)[-1])
    monkeypatch.setattr(
        runner.Energy, "perf", lambda fn, **k: (events.append("energy"), fn(), 0.1)[-1]
    )
    metrics = runner.profile_gdn_chunk_output(**_SPEC)
    assert metrics.time_ms == 1.0
    assert events == ["build", "timer", "semantic", "energy", "semantic"]
