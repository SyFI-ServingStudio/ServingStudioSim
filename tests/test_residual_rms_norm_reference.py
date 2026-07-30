"""CPU contract tests for the residual-add RMSNorm Torch reference."""

from __future__ import annotations

import pytest
import torch

from profiling.runners.norm.residual_rms_norm_reference import (
    residual_rms_norm_reference,
)


def _manual_oracle(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed_fp32 = x.to(torch.float32) + residual.to(torch.float32)
    residual_out = summed_fp32.to(dtype=x.dtype)
    squared_sum = torch.sum(summed_fp32 * summed_fp32, dim=-1, keepdim=True)
    variance_fp32 = squared_sum / x.shape[-1]
    normalized = summed_fp32 * torch.reciprocal(torch.sqrt(variance_fp32 + eps))
    normalized_out = normalized.to(dtype=x.dtype) * weight
    return normalized_out, residual_out


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-6, 1e-6),
        (torch.bfloat16, 8e-3, 8e-3),
        (torch.float16, 1e-3, 1e-3),
    ],
)
def test_matches_manual_oracle_and_preserves_contract(dtype, rtol, atol):
    x = torch.tensor(
        [
            [0.25, -1.5, 2.75, 0.125, -0.5],
            [4.0, -3.0, 0.75, 1.25, -2.0],
        ],
        dtype=dtype,
    )
    residual = torch.tensor(
        [
            [-0.5, 0.25, 1.0, -0.75, 2.0],
            [0.5, 1.5, -2.25, 0.125, 3.0],
        ],
        dtype=dtype,
    )
    weight = torch.tensor([0.5, -1.25, 0.75, 1.5, 2.0], dtype=dtype)
    x_before = x.clone()
    residual_before = residual.clone()
    weight_before = weight.clone()

    actual, actual_residual = residual_rms_norm_reference(x, residual, weight, eps=1e-5)
    expected, expected_residual = _manual_oracle(x, residual, weight, 1e-5)

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=rtol, atol=atol)
    assert actual.shape == x.shape
    assert actual_residual.shape == x.shape
    assert actual.dtype == dtype
    assert actual_residual.dtype == dtype
    assert torch.equal(x, x_before)
    assert torch.equal(residual, residual_before)
    assert torch.equal(weight, weight_before)


def test_default_epsilon_is_glm_value():
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    residual = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
    weight = torch.tensor([0.75, 1.25], dtype=torch.float32)

    default_output = residual_rms_norm_reference(x, residual, weight)
    explicit_output = residual_rms_norm_reference(x, residual, weight, eps=1e-5)

    torch.testing.assert_close(default_output[0], explicit_output[0])
    torch.testing.assert_close(default_output[1], explicit_output[1])


def test_fp32_residual_sum_is_used_for_normalization_before_cast():
    x = torch.tensor([[0.1, 0.2, 0.3, 100.0]], dtype=torch.bfloat16)
    residual = torch.flip(x, dims=[-1]) * torch.tensor(-0.5, dtype=torch.bfloat16)
    weight = torch.linspace(0.25, 1.75, x.shape[-1], dtype=torch.bfloat16)

    actual, _ = residual_rms_norm_reference(x, residual, weight)
    expected, _ = _manual_oracle(x, residual, weight, 1e-5)

    summed_low_precision = x + residual
    variance_low_precision = summed_low_precision.square().mean(dim=-1, keepdim=True)
    low_precision_output = (
        summed_low_precision
        * torch.rsqrt(variance_low_precision + torch.tensor(1e-5, dtype=torch.bfloat16))
    ) * weight

    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=8e-3)
    assert not torch.equal(actual, low_precision_output)


def _valid_inputs(dtype=torch.float32):
    return (
        torch.ones((2, 4), dtype=dtype),
        torch.full((2, 4), 0.5, dtype=dtype),
        torch.linspace(0.5, 1.5, 4, dtype=dtype),
    )


def test_rejects_mismatched_input_shapes():
    x, _, weight = _valid_inputs()
    residual = torch.ones((1, 4), dtype=x.dtype)

    with pytest.raises(ValueError, match="same shape"):
        residual_rms_norm_reference(x, residual, weight)


def test_rejects_mismatched_input_and_weight_dtypes():
    x, residual, weight = _valid_inputs()

    with pytest.raises(TypeError, match="same dtype"):
        residual_rms_norm_reference(x, residual.to(torch.float16), weight)
    with pytest.raises(TypeError, match="weight dtype"):
        residual_rms_norm_reference(x, residual, weight.to(torch.float16))


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
def test_rejects_unsupported_input_dtype(dtype):
    x, residual, weight = _valid_inputs(dtype)

    with pytest.raises(TypeError, match="x dtype"):
        residual_rms_norm_reference(x, residual, weight)


@pytest.mark.parametrize(
    "weight",
    [
        torch.ones((4, 1), dtype=torch.float32),
        torch.ones((3,), dtype=torch.float32),
    ],
)
def test_rejects_invalid_weight_shape(weight):
    x, residual, _ = _valid_inputs()

    with pytest.raises(ValueError, match=r"weight must have shape \(4,\)"):
        residual_rms_norm_reference(x, residual, weight)


def test_rejects_non_2d_inputs():
    weight = torch.ones((4,), dtype=torch.float32)

    with pytest.raises(ValueError, match="x must be 2-D"):
        residual_rms_norm_reference(
            torch.ones((4,), dtype=torch.float32),
            torch.ones((4,), dtype=torch.float32),
            weight,
        )
    with pytest.raises(ValueError, match="residual must be 2-D"):
        residual_rms_norm_reference(
            torch.ones((2, 4), dtype=torch.float32),
            torch.ones((2, 2, 4), dtype=torch.float32),
            weight,
        )


@pytest.mark.parametrize("shape", [(0, 4), (2, 0)])
def test_rejects_nonpositive_dimensions(shape):
    x = torch.empty(shape, dtype=torch.float32)
    residual = torch.empty(shape, dtype=torch.float32)
    weight = torch.empty((shape[-1],), dtype=torch.float32)

    with pytest.raises(ValueError, match="dimensions must be positive"):
        residual_rms_norm_reference(x, residual, weight)


@pytest.mark.parametrize("eps", [0.0, -1e-5, float("nan"), float("inf")])
def test_rejects_invalid_epsilon(eps):
    x, residual, weight = _valid_inputs()

    with pytest.raises(ValueError, match="positive and finite"):
        residual_rms_norm_reference(x, residual, weight, eps=eps)
