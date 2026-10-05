"""Behavioral tests for the DeepSeek routed-expert sum."""

import pytest
import torch

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.moe.moe_sum_reference import moe_sum_reference
from profiling.runners.moe.moe_sum_vllm_cuda import _validate_args


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
