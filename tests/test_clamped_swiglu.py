"""Behavioral tests for DeepSeek's clamped routed-expert activation."""

import pytest
import torch

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.moe.clamped_swiglu_reference import clamped_swiglu_reference
from profiling.runners.moe.clamped_swiglu_vllm_inductor import _validate_args


def test_reference_applies_the_asymmetric_gate_and_up_clamps() -> None:
    input_tensor = torch.tensor([[20.0, -20.0, 20.0, -20.0]], dtype=torch.bfloat16)

    result = clamped_swiglu_reference(torch, input_tensor)

    expected = torch.tensor([[100.0, 0.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(result, expected, atol=0.1, rtol=0.01)


@pytest.mark.parametrize(("hidden_dim", "dtype"), [(1024, "bf16"), (2048, "fp16")])
def test_runner_rejects_shapes_outside_the_measured_deepseek_path(
    hidden_dim: int, dtype: str
) -> None:
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(128, hidden_dim, dtype)
