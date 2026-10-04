"""Behavioral tests for the DeepSeek routed-expert sum."""

from types import SimpleNamespace

import pytest
import torch

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.moe.moe_sum_reference import moe_sum_reference
from profiling.runners.moe.moe_sum_vllm_cuda import _require_cuda, _validate_args


def test_reference_reduces_top_k_in_fp32_before_bf16_conversion() -> None:
    input_tensor = torch.tensor(
        [[[1000.0], [0.5], [0.5], [0.5], [0.5], [0.5]]], dtype=torch.bfloat16
    )

    result = moe_sum_reference(torch, input_tensor)

    # BF16 spacing around 1000 is four. Accumulating the five halves in BF16
    # would repeatedly round back to 1000; FP32 accumulation rounds once to 1004.
    assert result.item() == 1004.0


@pytest.mark.parametrize(
    ("top_k", "hidden_dim", "dtype", "error"),
    [
        (6, 4096, "fp16", ProfilerNotImplemented),
        (0, 4096, "bf16", ValueError),
        (6, 0, "bf16", ValueError),
    ],
)
def test_runner_rejects_unbuilt_dtype_and_empty_shapes(
    top_k: int, hidden_dim: int, dtype: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _validate_args(128, top_k, hidden_dim, dtype)


@pytest.mark.parametrize(("top_k", "hidden_dim"), [(6, 4096), (4, 4096), (8, 2048), (10, 7168)])
def test_runner_accepts_any_positive_top_k_and_hidden_dim(top_k: int, hidden_dim: int) -> None:
    assert _validate_args(128, top_k, hidden_dim, "bf16") == (128, top_k, hidden_dim)


@pytest.mark.parametrize("gpu_name", ["NVIDIA H200", "NVIDIA B200", "NVIDIA A100-SXM4-80GB"])
def test_runner_needs_cuda_but_no_particular_gpu(gpu_name: str) -> None:
    def fake_torch(available: bool) -> SimpleNamespace:
        return SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: available,
                current_device=lambda: 0,
                get_device_name=lambda _: gpu_name,
            )
        )

    _require_cuda(fake_torch(True))
    with pytest.raises(ProfilerNotImplemented, match="requires CUDA"):
        _require_cuda(fake_torch(False))
