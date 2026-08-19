"""Registration and backend runner tests for ``moe_fused_topk``."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.db.table import MissingEntry
from profiling.kernels.moe_fused_topk import KIND, MoeFusedTopkArgs
from profiling.runners.moe.moe_fused_topk_reference import moe_fused_topk_reference

_SPEC = {"num_tokens": 128, "num_experts": 256, "top_k": 8, "dtype": "bf16"}


def test_args_registry_support_environment_and_facades() -> None:
    assert [field.name for field in fields(MoeFusedTopkArgs)] == [
        "num_tokens",
        "num_experts",
        "top_k",
        "dtype",
    ]
    args = coerce_args(MoeFusedTopkArgs, {key: str(value) for key, value in _SPEC.items()})
    assert args == MoeFusedTopkArgs(128, 256, 8, DType.BF16)
    torch_spec = find_kernel_profiler_spec(KIND, "torch")
    cuda_spec = find_kernel_profiler_spec(KIND, "vllm_cuda")
    assert KIND == torch_spec.kernel_kind == torch_spec.table_name == "moe_fused_topk"
    assert known_backends(KIND) == ["torch", "vllm_cuda"]
    for spec in (torch_spec, cuda_spec):
        assert spec.args_schema is MoeFusedTopkArgs
        assert spec.table_name == KIND
        assert spec.metric_family is MetricFamily.COMPUTE
        assert spec.supports.compute == frozenset({DType.BF16})
    assert torch_spec.subprocess_env == "default_env"
    assert torch_spec.runner_ref.module_name.endswith("moe_fused_topk_torch")
    assert torch_spec.runner_ref.function_name == "profile_moe_fused_topk"
    assert cuda_spec.subprocess_env == "vllm_env"
    assert cuda_spec.supports.gpus == frozenset({"NVIDIA H200"})
    assert cuda_spec.runner_ref.module_name.endswith("moe_fused_topk_vllm_cuda")
    assert cuda_spec.runner_ref.function_name == "profile_moe_fused_topk_vllm_cuda"
    assert hasattr(perf_api, "get_moe_fused_topk_times")
    assert hasattr(perf_api, "count_missing_moe_fused_topk")


def test_registry_import_is_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('profiling.runners.moe.moe_fused_topk_torch' in sys.modules); "
                "print('profiling.runners.moe.moe_fused_topk_vllm_cuda' in sys.modules); "
                "print('profiling.runners.moe.moe_fused_topk_reference' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["False", "False", "False", "False"]


@pytest.mark.parametrize(
    "updates",
    [
        {"num_tokens": 0},
        {"num_tokens": True},
        {"num_tokens": 1.5},
        {"num_experts": 128},
        {"top_k": 4},
        {"dtype": "fp16"},
    ],
)
def test_vllm_validation_precedes_torch_vllm_import(
    updates: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    import profiling.runners.moe.moe_fused_topk_vllm_cuda as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ValueError):
        runner.profile_moe_fused_topk_vllm_cuda(**(_SPEC | updates))


def test_vllm_shapes_guard_metrics_and_packed_operands() -> None:
    import profiling.runners.moe.moe_fused_topk_torch as torch_runner
    import profiling.runners.moe.moe_fused_topk_vllm_cuda as runner

    args = runner._validate_args(**_SPEC)
    assert runner._guard_args(args) == args
    small = runner._validate_args(4096, 256, 8, "bf16")
    assert runner._guard_args(small).num_tokens == 8
    shapes = runner._operand_shapes(args)
    assert shapes == {
        "logits": (128, 256),
        "hidden_states": (128, 1),
        "weights": (128, 8),
        "expert_ids": (128, 8),
        "source_indices": (128, 8),
    }
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    runner._validate_operands(torch, operands, args, require_cuda=False)
    assert operands.logits.dtype is torch.bfloat16 and operands.logits.stride() == (256, 1)
    assert operands.hidden_states.dtype is torch.bfloat16
    assert runner._semantic_flops(num_tokens=128, num_experts=256, top_k=8) == (
        torch_runner._semantic_flops(num_tokens=128, num_experts=256, top_k=8)
    )
    assert runner._logical_bytes(num_tokens=128, num_experts=256, top_k=8) == (
        torch_runner._logical_bytes(num_tokens=128, num_experts=256, top_k=8)
    )


def test_vllm_guard_matches_reference_and_checks_fresh_outputs() -> None:
    import profiling.runners.moe.moe_fused_topk_vllm_cuda as runner

    args = runner._validate_args(3, 256, 8, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    before = (operands.logits.clone(), operands.hidden_states.clone())

    def callable_(**kwargs):
        weights, expert_ids, source = moe_fused_topk_reference(kwargs["gating_output"], 8)
        return weights.clone(), expert_ids.clone(), source.clone()

    runner._check_correctness(torch, callable_, operands, args, synchronize=lambda: None)
    assert torch.equal(operands.logits, before[0])
    assert torch.equal(operands.hidden_states, before[1])


def test_vllm_rejects_nonpacked_and_bad_outputs() -> None:
    import profiling.runners.moe.moe_fused_topk_vllm_cuda as runner

    args = runner._validate_args(3, 256, 8, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    backing = torch.empty((3, 512), dtype=torch.bfloat16)
    nonpacked = backing[:, ::2]
    assert nonpacked.shape == operands.logits.shape and not nonpacked.is_contiguous()
    bad = runner._Operands(logits=nonpacked, hidden_states=operands.hidden_states)
    with pytest.raises(ValueError):
        runner._validate_operands(torch, bad, args, require_cuda=False)
    outputs = (
        torch.ones((3, 8), dtype=torch.float32),
        torch.zeros((3, 8), dtype=torch.int32),
        torch.zeros((3, 8), dtype=torch.int32),
    )
    with pytest.raises(AssertionError):
        runner._validate_outputs(torch, outputs, operands, args)


def test_vllm_timing_isolation_and_exact_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    import profiling.runners.moe.moe_fused_topk_vllm_cuda as runner

    events: list[str] = []
    fake_operands = SimpleNamespace(logits=object(), hidden_states=object())
    fake_torch = SimpleNamespace(
        OutOfMemoryError=type("FakeOOM", (Exception,), {}),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _: "NVIDIA H200",
        ),
        device=lambda *args: "cuda",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runner, "_load_callable", lambda: events.append("load") or object())
    monkeypatch.setattr(
        runner, "_build_operands", lambda *args, **kwargs: events.append("build") or fake_operands
    )
    monkeypatch.setattr(
        runner, "_validate_operands", lambda *args, **kwargs: events.append("validate")
    )
    monkeypatch.setattr(runner, "_check_correctness", lambda *args: events.append("guard"))
    monkeypatch.setattr(runner, "_invoke", lambda *args: events.append("wrapper") or (None,) * 3)

    def timer(fn, *, kernel_name):
        events.extend(("timer", kernel_name))
        fn()
        return 1.0

    monkeypatch.setattr(runner.Timer, "cupti", timer)
    monkeypatch.setattr(
        runner.Energy,
        "perf",
        lambda fn, **kwargs: (events.append("energy"), fn(), 0.1)[2],
    )
    metrics = runner.profile_moe_fused_topk_vllm_cuda(**_SPEC)
    assert metrics.time_ms == 1.0
    assert "topkGating" in runner._KERNEL_NAME and "ScoringFuncE0" in runner._KERNEL_NAME
    assert events == [
        "load",
        "build",
        "validate",
        "guard",
        "timer",
        runner._KERNEL_NAME,
        "wrapper",
        "energy",
        "wrapper",
    ]


@pytest.mark.parametrize("name", ["num_tokens", "num_experts", "top_k"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_scalar_validation_precedes_torch_import(
    name: str, value: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import profiling.runners.moe.moe_fused_topk_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError):
        runner.profile_moe_fused_topk(**(_SPEC | {name: value}))


@pytest.mark.parametrize("updates", [{"top_k": 257}, {"dtype": "fp16"}])
def test_domain_validation_precedes_torch_import(
    updates: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    import profiling.runners.moe.moe_fused_topk_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError):
        runner.profile_moe_fused_topk(**(_SPEC | updates))


def test_qwen_shapes_metrics_and_independent_goldens() -> None:
    import profiling.runners.moe.moe_fused_topk_torch as runner

    args = runner._validate_args(**_SPEC)
    shapes = runner._operand_shapes(args)
    assert shapes["logits"] == shapes["probabilities"] == (128, 256)
    for name in ("weights", "expert_ids", "source_indices", "selected_long"):
        assert shapes[name] == (128, 8)
    comparisons = 8 * 255 - 8 * 7 // 2
    assert runner._semantic_flops(num_tokens=128, num_experts=256, top_k=8) == 128 * (
        (5 * 256 - 2) + comparisons + 15
    )
    assert runner._logical_bytes(num_tokens=128, num_experts=256, top_k=8) == (
        2 * 128 * 256 + 12 * 128 * 8
    )


def test_helper_matches_reference_ties_qwen_and_overwrites() -> None:
    import profiling.runners.moe.moe_fused_topk_torch as runner

    args = runner._validate_args(2, 256, 8, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.logits[0].zero_()
    logits_before = operands.logits.clone()
    expected = moe_fused_topk_reference(operands.logits, 8)
    first = runner._fused_topk_into(
        torch,
        operands.logits,
        operands.weights,
        operands.expert_ids,
        operands.source_indices,
        operands.workspaces,
    )
    torch.testing.assert_close(first[0], expected[0], rtol=1e-6, atol=1e-6)
    assert torch.equal(first[1], expected[1])
    assert torch.equal(first[2], expected[2])
    assert torch.equal(first[1][0], torch.arange(8, dtype=torch.int32))
    assert torch.equal(operands.logits, logits_before)
    pointers = {output.data_ptr() for output in first}
    assert len(pointers) == 3 and operands.logits.data_ptr() not in pointers

    operands.weights.fill_(float("nan"))
    operands.expert_ids.fill_(-1)
    operands.source_indices.fill_(-1)
    operands.workspaces.probabilities.fill_(float("nan"))
    second = runner._fused_topk_into(
        torch,
        operands.logits,
        operands.weights,
        operands.expert_ids,
        operands.source_indices,
        operands.workspaces,
    )
    torch.testing.assert_close(second[0], expected[0], rtol=1e-6, atol=1e-6)
    assert torch.equal(second[1], expected[1]) and torch.equal(second[2], expected[2])


def test_correctness_guard_preserves_input() -> None:
    import profiling.runners.moe.moe_fused_topk_torch as runner

    args = runner._validate_args(3, 7, 3, "bf16")
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    before = operands.logits.clone()
    runner._check_correctness(torch, operands)
    assert torch.equal(operands.logits, before)


def test_timing_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    import profiling.runners.moe.moe_fused_topk_torch as runner

    events: list[str] = []
    fake_operands = SimpleNamespace(
        logits=object(),
        weights=object(),
        expert_ids=object(),
        source_indices=object(),
        workspaces=object(),
    )
    fake_torch = SimpleNamespace(
        OutOfMemoryError=RuntimeError,
        cuda=SimpleNamespace(is_available=lambda: True),
        device=lambda _: "cuda",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        runner, "_build_operands", lambda *args, **kwargs: events.append("build") or fake_operands
    )
    monkeypatch.setattr(runner, "_check_correctness", lambda *args: events.append("guard"))
    monkeypatch.setattr(
        runner, "_fused_topk_into", lambda *args: events.append("semantic") or (None, None, None)
    )

    def timer(fn):
        events.append("timer")
        fn()
        return 1.0

    monkeypatch.setattr(runner.Timer, "cupti", timer)
    monkeypatch.setattr(
        runner.Energy,
        "perf",
        lambda fn, **kwargs: (events.append("energy"), fn(), 0.1)[2],
    )
    metrics = runner.profile_moe_fused_topk(**_SPEC)
    assert metrics.time_ms == 1.0
    assert events == ["build", "guard", "timer", "semantic", "energy", "semantic"]


def test_generated_facades_and_task_local_missing_query(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")
    assert (
        perf_api.count_missing_moe_fused_topk([_SPEC], backend="torch", gpu_name="NVIDIA H200") == 1
    )
    result = perf_api.get_moe_fused_topk_times([_SPEC], backend="torch", gpu_name="NVIDIA H200")[0]
    assert isinstance(result, MissingEntry)
    assert result.args == coerce_args(MoeFusedTopkArgs, _SPEC)
    assert not perf_api.DB_PATH.exists()
