"""Focused tests for the direct FlashInfer/TensorRT-LLM grouped GEMM helper."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from profiling.db.args import DType
from profiling.runners.gemm import flashinfer_trtllm_blockscale as runner


def _valid_args() -> dict:
    return {
        "n": 3072,
        "k": 4096,
        "dtype": DType.FP8_E4M3,
        "num_local_experts": 4,
        "per_group_batches": (2, 0, 3, 1),
        "num_input_tokens": 4,
        "experts_per_token": 2,
    }


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"n": 129}, "n must be > 0 and divisible by 128"),
        ({"k": 127}, "k must be > 0 and divisible by 128"),
        ({"dtype": DType.FP8_E5M2}, "requires dtype=fp8_e4m3"),
        ({"num_local_experts": 0, "per_group_batches": ()}, "num_local_experts must be > 0"),
        ({"per_group_batches": (1, 2)}, "expected num_local_experts=4"),
        ({"per_group_batches": (1, -1, 2, 3)}, "non-negative"),
        ({"per_group_batches": (0, 0, 0, 0)}, "no local rows to profile"),
        ({"num_input_tokens": 0}, "num_input_tokens must be > 0"),
        ({"experts_per_token": 0}, "experts_per_token must be > 0"),
        (
            {"num_input_tokens": 1, "experts_per_token": 1},
            "local routed rows 6 exceed global routed capacity",
        ),
    ],
)
def test_direct_contract_validation(changed, message):
    arguments = _valid_args()
    arguments.update(changed)
    with pytest.raises(ValueError, match=message):
        runner._validate_args(**arguments)


def test_production_capacity_uses_global_tokens_times_topk_not_local_sum():
    assert runner._production_capacity(64, 8, 32) == (512, 1504)
    assert runner._production_capacity(8192, 8, 32) == (65536, 66528)


def test_problem_offsets_are_raw_counts_and_fail_closed():
    boundaries = runner._problem_m_boundaries((2, 0, 3, 1))
    assert boundaries == (0, 2, 2, 5, 6)
    runner._validate_problem_boundaries(
        boundaries,
        num_local_experts=4,
        total_local_rows=6,
        max_shape_m=8,
    )
    with pytest.raises(ValueError, match="monotonic"):
        runner._validate_problem_boundaries(
            (0, 2, 1, 5, 6),
            num_local_experts=4,
            total_local_rows=6,
            max_shape_m=8,
        )
    with pytest.raises(ValueError, match=r"sum\(per_group_batches\)"):
        runner._validate_problem_boundaries(
            (0, 2, 2, 5, 7),
            num_local_experts=4,
            total_local_rows=6,
            max_shape_m=8,
        )


def test_prepare_uses_production_physical_layouts(monkeypatch):
    allocations = []

    def allocate(kind, shape, **kwargs):
        tensor = SimpleNamespace(kind=kind, shape=shape, kwargs=kwargs)
        allocations.append(tensor)
        return tensor

    fake_torch = SimpleNamespace(
        float8_e4m3fn="fp8",
        float32="fp32",
        bfloat16="bf16",
        int64="int64",
        empty=lambda shape, **kwargs: allocate("empty", shape, **kwargs),
        ones=lambda shape, **kwargs: allocate("ones", shape, **kwargs),
        tensor=lambda values, **kwargs: allocate("tensor", tuple(values), **kwargs),
    )
    launch = runner.prepare_fp8_blockscale_grouped_gemm_launch(
        fake_torch,
        **_valid_args(),
    )

    assert launch.activation.shape == (8, 4096)
    assert launch.activation_scales.shape == (32, 128)
    assert launch.weight.shape == (4, 3072, 4096)
    assert launch.weight_scales.shape == (4, 24, 32)
    assert launch.output.shape == (8, 3072)
    assert launch.problem_m_offsets.shape == (0, 2, 2, 5, 6)
    assert launch.total_local_rows == 6


def test_recipe_filter_is_shape_exact_and_scheduler_validated():
    recipe = (
        "void deep_gemm::fp8_gemm_kernel<(unsigned int)3072, "
        "(unsigned int)4096, (unsigned int)64, "
        "deep_gemm::GroupedWithOffsetScheduler<(unsigned int)3072, "
        "(unsigned int)64>, deep_gemm::GroupedWithOffsetSchedulerInput>(...)"
    )
    assert runner._kernel_name_filter(3072, 4096, swapped_ab=False) == (
        "_ZN9deep_gemm15fp8_gemm_kernelILj3072ELj4096E"
    )
    assert runner._validate_recipe_names([recipe], n=3072, k=4096, swapped_ab=False) == recipe
    with pytest.raises(RuntimeError, match="not GroupedWithOffset"):
        runner._validate_recipe_names(
            [recipe.replace("GroupedWithOffsetScheduler", "NormalScheduler")],
            n=3072,
            k=4096,
            swapped_ab=False,
        )
    with pytest.raises(RuntimeError, match="does not match"):
        runner._validate_recipe_names([recipe], n=4096, k=1536, swapped_ab=False)

    mangled_recipe = (
        "_ZN9deep_gemm15fp8_gemm_kernelILj3072ELj4096ELj64E"
        "NS_26GroupedWithOffsetSchedulerILj3072ELj64EEE"
    )
    assert (
        runner._validate_recipe_names([mangled_recipe], n=3072, k=4096, swapped_ab=False)
        == mangled_recipe
    )


def test_wrapper_source_calls_vendored_dispatch_without_synthetic_routing():
    source_path = (
        Path(runner.__file__).resolve().parent
        / "csrc"
        / "flashinfer_trtllm_blockscale_grouped_gemm.cu"
    )
    source = source_path.read_text()
    assert "blockscale::grouped_gemm_dispatch(" in source
    assert "fp8_gemm_kernel" not in source
    assert "token_selected_experts" not in source
    assert "topk_ids" not in source


def _gpu_skip_reason() -> str | None:
    try:
        import torch
    except ImportError:
        return "torch is unavailable"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if tuple(torch.cuda.get_device_capability()) != (9, 0):
        return "direct FP8 block-scale GEMM requires SM90"
    if runner._parse_cuda_version(torch.version.cuda) < (12, 8):
        return "direct FP8 block-scale GEMM requires CUDA >= 12.8"
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return "FlashInfer is unavailable"
    return None


def test_sm90_direct_grouped_gemm_matches_per_expert_torch(monkeypatch):
    skip_reason = _gpu_skip_reason()
    if skip_reason is not None:
        pytest.skip(skip_reason)

    import torch

    temporary_root = Path(os.environ["TMPDIR"])
    monkeypatch.setenv(
        "FLASHINFER_WORKSPACE_BASE",
        str(temporary_root / "vibesim_flashinfer_grouped_gemm_test"),
    )
    monkeypatch.setenv(
        "TRTLLM_DG_CACHE_DIR",
        str(temporary_root / "vibesim_trtllm_deepgemm_test"),
    )
    launch = runner.prepare_fp8_blockscale_grouped_gemm_launch(
        torch,
        n=128,
        k=128,
        dtype=DType.FP8_E4M3,
        num_local_experts=2,
        per_group_batches=(2, 1),
        num_input_tokens=32,
        experts_per_token=1,
    )
    activation_values = torch.randint(
        -2,
        3,
        launch.activation.shape,
        dtype=torch.int8,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    weight_values = torch.randint(
        -2,
        3,
        launch.weight.shape,
        dtype=torch.int8,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    launch.activation.copy_(activation_values)
    launch.weight.copy_(weight_values)
    launch.activation_scales.fill_(1.0)
    launch.weight_scales.fill_(1.0)

    launch.run_once()
    torch.cuda.synchronize()

    boundaries = (0, 2, 3)
    references = []
    for expert_index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        references.append(
            launch.activation[left:right].float()
            @ launch.weight[expert_index].float().transpose(0, 1)
        )
    expected = torch.cat(references, dim=0)
    actual = launch.output[:3].float()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=1.0)

    summary, recipe_name = runner.capture_fp8_blockscale_grouped_gemm_recipe(
        launch,
        num_device_sms=torch.cuda.get_device_properties(0).multi_processor_count,
        num_warmup=0,
        num_iter=1,
        clear_l2=False,
    )
    assert summary.matched_kernel_count_per_run == [1]
    assert "GroupedWithOffsetScheduler" in recipe_name
