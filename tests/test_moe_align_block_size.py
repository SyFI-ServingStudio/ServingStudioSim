"""Registration and Torch-runner tests for ``moe_align_block_size``."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from profiling import perf_api
from profiling.db.batch import coerce_args
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.db.table import MissingEntry
from profiling.kernels.moe_align_block_size import KIND, MoeAlignBlockSizeArgs
from profiling.runners.moe.moe_align_block_size_reference import (
    moe_align_block_size_reference,
)

_SPEC = {"num_tokens": 128, "num_experts": 256, "top_k": 8, "block_size": 16}


def test_args_registry_support_environment_and_facades() -> None:
    assert [field.name for field in fields(MoeAlignBlockSizeArgs)] == list(_SPEC)
    args = coerce_args(MoeAlignBlockSizeArgs, {key: str(value) for key, value in _SPEC.items()})
    assert args == MoeAlignBlockSizeArgs(128, 256, 8, 16)
    torch_spec = find_kernel_profiler_spec(KIND, "torch")
    cuda_spec = find_kernel_profiler_spec(KIND, "vllm_cuda")
    assert KIND == torch_spec.kernel_kind == torch_spec.table_name == "moe_align_block_size"
    assert known_backends(KIND) == ["torch", "vllm_cuda"]
    for spec in (torch_spec, cuda_spec):
        assert spec.args_schema is MoeAlignBlockSizeArgs
        assert spec.table_name == KIND
        assert spec.metric_family is MetricFamily.COMPUTE
        assert spec.supports.compute is None
    assert torch_spec.subprocess_env == "default_env"
    assert torch_spec.runner_ref.module_name.endswith("moe_align_block_size_torch")
    assert torch_spec.runner_ref.function_name == "profile_moe_align_block_size"
    assert cuda_spec.subprocess_env == "vllm_env"
    assert cuda_spec.supports.gpus == frozenset({"NVIDIA H200"})
    assert cuda_spec.runner_ref.module_name.endswith("moe_align_block_size_vllm_cuda")
    assert cuda_spec.runner_ref.function_name == "profile_moe_align_block_size_vllm_cuda"
    assert hasattr(perf_api, "get_moe_align_block_size_times")
    assert hasattr(perf_api, "count_missing_moe_align_block_size")


def test_registry_import_is_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('profiling.runners.moe.moe_align_block_size_torch' in sys.modules); "
                "print('profiling.runners.moe.moe_align_block_size_vllm_cuda' in sys.modules); "
                "print('profiling.runners.moe.moe_align_block_size_reference' in sys.modules)"
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
        {"num_tokens": 8},
        {"num_tokens": True},
        {"num_tokens": 1.5},
        {"num_experts": 64},
        {"top_k": 4},
        {"block_size": 8},
    ],
)
def test_vllm_validation_precedes_torch_vllm_import(
    updates: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    import profiling.runners.moe.moe_align_block_size_vllm_cuda as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ValueError):
        runner.profile_moe_align_block_size_vllm_cuda(**(_SPEC | updates))


def test_vllm_packed_operands_guard_geometry_and_metric_reuse() -> None:
    import profiling.runners.moe.moe_align_block_size_torch as torch_runner
    import profiling.runners.moe.moe_align_block_size_vllm_cuda as runner

    args = runner._validate_args(**_SPEC)
    assert runner._guard_args(args) == args
    large = runner._validate_args(4096, 256, 8, 16)
    assert runner._guard_args(large).num_tokens == 9
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    runner._validate_operands(torch, operands, args, require_cuda=False)
    assert operands.topk_ids.shape == (128, 8)
    assert operands.topk_ids.dtype is torch.int32
    assert operands.topk_ids.stride() == (8, 1)
    assert all(row.unique().numel() == 8 for row in operands.topk_ids)
    assert runner._semantic_ops(**_SPEC) == torch_runner._semantic_ops(**_SPEC)
    assert runner._logical_bytes(**_SPEC) == torch_runner._logical_bytes(**_SPEC)


def test_vllm_correctness_guard_uses_semantic_multisets_and_fresh_outputs() -> None:
    import profiling.runners.moe.moe_align_block_size_vllm_cuda as runner

    args = runner._validate_args(**_SPEC)
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    before = operands.topk_ids.clone()

    def callable_(ids, block_size, num_experts, expert_map, **flags):
        outputs = moe_align_block_size_reference(ids, num_experts, block_size)
        return tuple(output.clone() for output in outputs)

    runner._check_correctness(torch, callable_, operands, args, synchronize=lambda: None)
    assert torch.equal(operands.topk_ids, before)


def test_vllm_rejects_nonpacked_inputs_and_bad_outputs() -> None:
    import profiling.runners.moe.moe_align_block_size_vllm_cuda as runner

    args = runner._validate_args(**_SPEC)
    backing = torch.empty((128, 16), dtype=torch.int32)
    nonpacked = runner._Operands(topk_ids=backing[:, ::2])
    with pytest.raises(ValueError, match="packed"):
        runner._validate_operands(torch, nonpacked, args, require_cuda=False)
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    outputs = (
        torch.empty((4864,), dtype=torch.int64),
        torch.empty((304,), dtype=torch.int32),
        torch.empty((1,), dtype=torch.int32),
    )
    with pytest.raises(AssertionError):
        runner._validate_outputs(torch, outputs, operands, args)


def test_vllm_timing_isolation_sums_complete_two_launch_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import profiling.runners.moe.moe_align_block_size_vllm_cuda as runner

    events: list[object] = []
    fake_operands = SimpleNamespace(topk_ids=object())
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
    metrics = runner.profile_moe_align_block_size_vllm_cuda(**_SPEC)
    assert metrics.time_ms == 1.0
    assert runner._KERNEL_NAME is None
    assert runner._EXPECTED_KERNEL_SUBSTRINGS == (
        "moe_align_block_size_kernel",
        "count_and_sort_expert_tokens_kernel",
    )
    assert events == [
        "load",
        "build",
        "validate",
        "guard",
        "timer",
        None,
        "wrapper",
        "energy",
        "wrapper",
    ]


@pytest.mark.parametrize("name", list(_SPEC))
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_scalar_validation_precedes_torch_import(
    name: str, value: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import profiling.runners.moe.moe_align_block_size_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError):
        runner.profile_moe_align_block_size(**(_SPEC | {name: value}))


def test_domain_and_overflow_validation_precedes_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import profiling.runners.moe.moe_align_block_size_torch as runner

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="top_k"):
        runner.profile_moe_align_block_size(1, 4, 5, 2)
    monkeypatch.setattr(runner, "_INT32_MAX", 7)
    with pytest.raises(ValueError, match="int32"):
        runner.profile_moe_align_block_size(2, 4, 2, 3)


def test_qwen_shapes_canonical_routing_and_capacities() -> None:
    import profiling.runners.moe.moe_align_block_size_torch as runner

    args = runner._validate_args(**_SPEC)
    assert runner._operand_shapes(args) == {
        "topk_ids": (128, 8),
        "sorted_token_ids": (4864,),
        "expert_ids": (304,),
        "num_tokens_post_pad": (1,),
    }
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    assert operands.topk_ids.dtype is torch.int32 and operands.topk_ids.is_contiguous()
    assert torch.equal(operands.topk_ids, torch.arange(1024).reshape(128, 8) % 256)
    assert all(row.unique().numel() == 8 for row in operands.topk_ids)


@pytest.mark.parametrize(("modulus", "expected_post_pad"), [(256, 4096), (64, 1024), (8, 1024)])
def test_helper_matches_balanced_moderate_and_concentrated_reference(
    modulus: int, expected_post_pad: int
) -> None:
    import profiling.runners.moe.moe_align_block_size_torch as runner

    args = runner._validate_args(**_SPEC)
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.topk_ids.copy_(torch.arange(1024, dtype=torch.int32).reshape(128, 8) % modulus)
    before = operands.topk_ids.clone()
    actual = runner._align_into(
        torch,
        operands.topk_ids,
        operands.sorted_token_ids,
        operands.expert_ids,
        operands.num_tokens_post_pad,
        operands.workspaces,
        num_experts=args.num_experts,
        block_size=args.block_size,
    )
    expected = moe_align_block_size_reference(operands.topk_ids, 256, 16)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))
    assert actual[2].item() == expected_post_pad
    assert torch.all(actual[0][expected_post_pad:] == 1024)
    assert torch.all(actual[1][expected_post_pad // 16 :] == -1)
    assert torch.equal(operands.topk_ids, before)


@pytest.mark.parametrize(
    "topk_ids,num_experts,block_size",
    [
        (torch.tensor([[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]]), 5, 4),
        (torch.tensor([[2, 7]]), 8, 4),
        (torch.tensor([[0, 3], [2, 3], [1, 3]]), 5, 2),
    ],
)
def test_helper_padding_zero_experts_small_capacity_and_multisets(
    topk_ids: torch.Tensor, num_experts: int, block_size: int
) -> None:
    import profiling.runners.moe.moe_align_block_size_torch as runner

    topk_ids = topk_ids.to(torch.int32)
    args = runner._validate_args(topk_ids.shape[0], num_experts, topk_ids.shape[1], block_size)
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    operands.topk_ids.copy_(topk_ids)
    actual = runner._align_into(
        torch,
        operands.topk_ids,
        operands.sorted_token_ids,
        operands.expert_ids,
        operands.num_tokens_post_pad,
        operands.workspaces,
        num_experts=num_experts,
        block_size=block_size,
    )
    expected = moe_align_block_size_reference(topk_ids, num_experts, block_size)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_helper_overwrites_outputs_and_uses_disjoint_storage() -> None:
    import profiling.runners.moe.moe_align_block_size_torch as runner

    args = runner._validate_args(4, 5, 3, 4)
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    expected = moe_align_block_size_reference(operands.topk_ids, 5, 4)
    for tensor in (
        operands.sorted_token_ids,
        operands.expert_ids,
        operands.num_tokens_post_pad,
    ):
        tensor.fill_(-123)
    actual = runner._align_into(
        torch,
        operands.topk_ids,
        operands.sorted_token_ids,
        operands.expert_ids,
        operands.num_tokens_post_pad,
        operands.workspaces,
        num_experts=5,
        block_size=4,
    )
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))
    pointers = {tensor.data_ptr() for tensor in actual}
    assert len(pointers) == 3 and operands.topk_ids.data_ptr() not in pointers
    for tensor in actual:
        tensor.fill_(-7)
    repeated = runner._align_into(
        torch,
        operands.topk_ids,
        operands.sorted_token_ids,
        operands.expert_ids,
        operands.num_tokens_post_pad,
        operands.workspaces,
        num_experts=5,
        block_size=4,
    )
    assert all(torch.equal(left, right) for left, right in zip(repeated, expected))


def test_metrics_have_independent_qwen_goldens() -> None:
    import profiling.runners.moe.moe_align_block_size_torch as runner

    assignments = 128 * 8
    capacity = assignments + 256 * 15
    blocks = (capacity + 15) // 16
    post_pad = 256 * 16
    expected_ops = (
        assignments + 3 * 256 + 2 * 255 + assignments * 10 + assignments + post_pad + blocks * 9
    )
    assert runner._semantic_ops(**_SPEC) == expected_ops
    assert runner._logical_bytes(**_SPEC) == 4 * (assignments + capacity + blocks + 1)


def test_correctness_guard_preserves_input() -> None:
    import profiling.runners.moe.moe_align_block_size_torch as runner

    args = runner._validate_args(3, 7, 3, 2)
    operands = runner._build_operands(torch, args, device=torch.device("cpu"))
    before = operands.topk_ids.clone()
    runner._check_correctness(torch, operands, args)
    assert torch.equal(operands.topk_ids, before)


def test_timing_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    import profiling.runners.moe.moe_align_block_size_torch as runner

    events: list[str] = []
    fake_operands = SimpleNamespace(
        topk_ids=object(),
        sorted_token_ids=object(),
        expert_ids=object(),
        num_tokens_post_pad=object(),
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
        runner, "_align_into", lambda *args, **kwargs: events.append("semantic") or (None,) * 3
    )
    monkeypatch.setattr(runner.Timer, "cupti", lambda fn: (events.append("timer"), fn(), 1.0)[2])
    monkeypatch.setattr(
        runner.Energy,
        "perf",
        lambda fn, **kwargs: (events.append("energy"), fn(), 0.1)[2],
    )
    metrics = runner.profile_moe_align_block_size(**_SPEC)
    assert metrics.time_ms == 1.0
    assert events == ["build", "guard", "timer", "semantic", "energy", "semantic"]


def test_generated_facades_and_task_local_missing_query(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")
    assert (
        perf_api.count_missing_moe_align_block_size(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 1
    )
    result = perf_api.get_moe_align_block_size_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(result, MissingEntry)
    assert result.args == coerce_args(MoeAlignBlockSizeArgs, _SPEC)
    assert not perf_api.DB_PATH.exists()
