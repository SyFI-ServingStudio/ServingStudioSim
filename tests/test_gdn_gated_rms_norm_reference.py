"""CPU tests for the Qwen GDN gated RMS normalization reference."""

from __future__ import annotations

import pytest
import torch

from profiling.runners.attention.gdn_gated_rms_norm_reference import (
    gdn_gated_rms_norm_reference,
)

_EPSILON = 1e-6


def _inputs(m: int = 3, hidden: int = 4) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260810)
    x = torch.randn(m, hidden, generator=generator, dtype=torch.bfloat16)
    z = torch.randn(m, hidden, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(hidden, generator=generator, dtype=torch.bfloat16)
    return x, z, weight


def _manual_equation(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    x_fp32 = x.float()
    z_fp32 = z.float()
    rstd = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + _EPSILON)
    normalized = x_fp32 * rstd * weight.float()
    gate = z_fp32 * torch.sigmoid(z_fp32)
    return (normalized * gate).to(torch.bfloat16)


def test_shape_dtype_manual_equation_immutability_and_fresh_output() -> None:
    x, z, weight = _inputs()
    originals = tuple(tensor.clone() for tensor in (x, z, weight))

    output = gdn_gated_rms_norm_reference(x, z, weight)

    assert output.shape == (3, 4)
    assert output.dtype is torch.bfloat16
    torch.testing.assert_close(output, _manual_equation(x, z, weight), rtol=0, atol=0)
    assert all(torch.equal(actual, before) for actual, before in zip((x, z, weight), originals))
    input_storage = {tensor.untyped_storage().data_ptr() for tensor in (x, z, weight)}
    assert output.untyped_storage().data_ptr() not in input_storage


def test_normalization_happens_before_gate() -> None:
    x = torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16)
    z = torch.tensor([[-2.0, 2.0]], dtype=torch.bfloat16)
    weight = torch.tensor([0.5, 1.5], dtype=torch.bfloat16)

    output = gdn_gated_rms_norm_reference(x, z, weight)
    x_fp32 = x.float()
    z_fp32 = z.float()
    activated = z_fp32 * torch.sigmoid(z_fp32)
    norm_after_gate_input = x_fp32 * activated
    wrong_rstd = torch.rsqrt(norm_after_gate_input.square().mean(dim=-1, keepdim=True) + _EPSILON)
    wrong_order = (norm_after_gate_input * wrong_rstd * weight.float()).to(torch.bfloat16)

    assert torch.equal(output, _manual_equation(x, z, weight))
    assert not torch.equal(output, wrong_order)


def test_rows_are_normalized_independently() -> None:
    x, z, weight = _inputs(m=2, hidden=3)
    combined = gdn_gated_rms_norm_reference(x, z, weight)

    row_outputs = torch.cat(
        [
            gdn_gated_rms_norm_reference(x[index : index + 1], z[index : index + 1], weight)
            for index in range(2)
        ]
    )

    assert torch.equal(combined, row_outputs)


def test_weight_broadcasts_across_rows() -> None:
    x = torch.ones((3, 3), dtype=torch.bfloat16)
    z = torch.ones_like(x)
    weight = torch.tensor([0.5, 1.0, 2.0], dtype=torch.bfloat16)

    output = gdn_gated_rms_norm_reference(x, z, weight)

    assert torch.equal(output[0], output[1])
    assert torch.equal(output[1], output[2])
    assert torch.equal(output, _manual_equation(x, z, weight))


@pytest.mark.parametrize("zero_operand", ["x", "z"])
def test_zero_inputs(zero_operand: str) -> None:
    x, z, weight = _inputs(m=2, hidden=3)
    if zero_operand == "x":
        x = torch.zeros_like(x)
    else:
        z = torch.zeros_like(z)

    output = gdn_gated_rms_norm_reference(x, z, weight)

    assert torch.count_nonzero(output) == 0
    assert torch.equal(output, _manual_equation(x, z, weight))


def test_unit_weight() -> None:
    x, z, _ = _inputs(m=2, hidden=3)
    weight = torch.ones(3, dtype=torch.bfloat16)

    output = gdn_gated_rms_norm_reference(x, z, weight)

    assert torch.equal(output, _manual_equation(x, z, weight))


@pytest.mark.parametrize(
    ("x_values", "z_values"),
    [([1e-3, -2e-3, 3e-3], [1e-3, -1e-3, 2e-3]), ([64.0, -32.0, 16.0], [8.0, -8.0, 4.0])],
)
def test_bounded_small_and_large_values(
    x_values: list[float],
    z_values: list[float],
) -> None:
    x = torch.tensor([x_values], dtype=torch.bfloat16)
    z = torch.tensor([z_values], dtype=torch.bfloat16)
    weight = torch.tensor([0.5, -1.0, 2.0], dtype=torch.bfloat16)

    output = gdn_gated_rms_norm_reference(x, z, weight)

    assert torch.isfinite(output).all()
    assert torch.equal(output, _manual_equation(x, z, weight))


def test_noncontiguous_strided_inputs_are_semantically_supported() -> None:
    x_base, z_base, _ = _inputs(m=3, hidden=8)
    weight_base = torch.arange(8, dtype=torch.bfloat16)
    x = x_base[:, ::2]
    z = z_base[:, ::2]
    weight = weight_base[::2]
    assert not x.is_contiguous() and not z.is_contiguous() and not weight.is_contiguous()

    output = gdn_gated_rms_norm_reference(x, z, weight)

    assert torch.equal(output, _manual_equation(x, z, weight))


@pytest.mark.parametrize("name", ["x", "z", "weight"])
def test_rejects_non_bf16_input(name: str) -> None:
    inputs = list(_inputs())
    index = ("x", "z", "weight").index(name)
    inputs[index] = inputs[index].float()

    with pytest.raises(TypeError, match=rf"{name} dtype must be torch.bfloat16"):
        gdn_gated_rms_norm_reference(*inputs)


@pytest.mark.parametrize(
    ("index", "replacement", "message"),
    [
        (0, torch.ones((2, 3, 1), dtype=torch.bfloat16), "x must be rank 2"),
        (1, torch.ones(4, dtype=torch.bfloat16), "z must be rank 2"),
        (2, torch.ones((1, 4), dtype=torch.bfloat16), "weight must be rank 1"),
        (1, torch.ones((3, 5), dtype=torch.bfloat16), "z shape must match x shape"),
        (2, torch.ones(5, dtype=torch.bfloat16), "weight must have shape"),
    ],
)
def test_rejects_invalid_ranks_or_shapes(
    index: int,
    replacement: torch.Tensor,
    message: str,
) -> None:
    inputs = list(_inputs())
    inputs[index] = replacement

    with pytest.raises(ValueError, match=message):
        gdn_gated_rms_norm_reference(*inputs)


@pytest.mark.parametrize(
    "inputs",
    [
        (
            torch.empty((0, 4), dtype=torch.bfloat16),
            torch.empty((0, 4), dtype=torch.bfloat16),
            torch.ones(4, dtype=torch.bfloat16),
        ),
        (
            torch.empty((2, 0), dtype=torch.bfloat16),
            torch.empty((2, 0), dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.bfloat16),
        ),
    ],
)
def test_rejects_nonpositive_dimensions(inputs: tuple[torch.Tensor, ...]) -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        gdn_gated_rms_norm_reference(*inputs)


def test_rejects_non_tensor_input() -> None:
    x, z, _ = _inputs()

    with pytest.raises(TypeError, match="weight must be a torch.Tensor"):
        gdn_gated_rms_norm_reference(x, z, object())  # type: ignore[arg-type]
