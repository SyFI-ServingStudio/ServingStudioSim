"""CPU tests for the Qwen GDN causal-convolution decode reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from profiling.runners.attention.gdn_causal_conv_decode_reference import (
    gdn_causal_conv_decode_reference,
)


def _inputs(
    *,
    batch_size: int = 2,
    channels: int = 3,
    kernel_size: int = 4,
    num_slots: int = 5,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260810)
    x = torch.randn(batch_size, channels, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(channels, kernel_size, generator=generator, dtype=torch.bfloat16)
    state = torch.randn(
        num_slots,
        channels,
        kernel_size - 1,
        generator=generator,
        dtype=torch.bfloat16,
    )
    slot_indices = torch.arange(1, batch_size + 1, dtype=torch.int32)
    return x, weight, state, slot_indices


def _manual_equation(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    slot_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_state = state.clone()
    selected = state.index_select(0, slot_indices.long())
    window = torch.cat((selected, x.unsqueeze(-1)), dim=-1)
    accumulator = (window.float() * weight.float().unsqueeze(0)).sum(dim=-1)
    output = F.silu(accumulator).to(torch.bfloat16)
    expected_state.index_copy_(0, slot_indices.long(), window[..., 1:])
    return output, expected_state


def test_shapes_dtypes_manual_equation_and_inplace_state_update() -> None:
    x, weight, state, slot_indices = _inputs()
    state_before = state.clone()
    x_before = x.clone()
    weight_before = weight.clone()
    expected_output, expected_state = _manual_equation(x, weight, state_before, slot_indices)
    storage = state.untyped_storage().data_ptr()

    output, returned_state = gdn_causal_conv_decode_reference(x, weight, state, slot_indices)

    assert output.shape == x.shape == (2, 3)
    assert output.dtype is torch.bfloat16
    assert returned_state.shape == (5, 3, 3)
    assert returned_state.dtype is torch.bfloat16
    assert returned_state is state
    assert returned_state.untyped_storage().data_ptr() == storage
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(returned_state, expected_state, rtol=0, atol=0)
    assert torch.equal(x, x_before)
    assert torch.equal(weight, weight_before)


def test_state_shift_drops_oldest_and_appends_newest_x() -> None:
    x = torch.tensor([[10.0, 20.0]], dtype=torch.bfloat16)
    weight = torch.ones((2, 4), dtype=torch.bfloat16)
    state = torch.zeros((3, 2, 3), dtype=torch.bfloat16)
    state[2] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    _, returned_state = gdn_causal_conv_decode_reference(
        x, weight, state, torch.tensor([2], dtype=torch.int32)
    )

    expected = torch.tensor([[2.0, 3.0, 10.0], [5.0, 6.0, 20.0]])
    assert torch.equal(returned_state[2], expected.to(torch.bfloat16))


def test_permuted_slots_are_independent_and_unselected_slots_are_preserved() -> None:
    x, weight, state, _ = _inputs(batch_size=2, channels=2, num_slots=5)
    slot_indices = torch.tensor([3, 1], dtype=torch.int32)
    state_before = state.clone()
    expected_output, expected_state = _manual_equation(x, weight, state_before, slot_indices)

    output, _ = gdn_causal_conv_decode_reference(x, weight, state, slot_indices)

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    assert torch.equal(state, expected_state)
    assert torch.equal(state[0], state_before[0])  # reserved slot
    assert torch.equal(state[2], state_before[2])
    assert torch.equal(state[4], state_before[4])


@pytest.mark.parametrize("zero_operand", ["x", "state", "weight"])
def test_zero_operand_edge_cases(zero_operand: str) -> None:
    inputs = list(_inputs(batch_size=1, channels=2))
    operand_index = {"x": 0, "weight": 1, "state": 2}[zero_operand]
    inputs[operand_index] = torch.zeros_like(inputs[operand_index])
    expected_output, expected_state = _manual_equation(*inputs)

    output, returned_state = gdn_causal_conv_decode_reference(*inputs)

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(returned_state, expected_state, rtol=0, atol=0)
    if zero_operand == "weight":
        assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("kernel_size", [2, 3, 4, 5, 6])
def test_supported_width_parameterization(kernel_size: int) -> None:
    inputs = _inputs(batch_size=1, channels=2, kernel_size=kernel_size)
    expected_output, expected_state = _manual_equation(*inputs)

    output, returned_state = gdn_causal_conv_decode_reference(*inputs)

    assert returned_state.shape[-1] == kernel_size - 1
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(returned_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["x", "weight", "state"])
def test_rejects_non_bf16_data(name: str) -> None:
    inputs = list(_inputs())
    index = ("x", "weight", "state").index(name)
    inputs[index] = inputs[index].float()

    with pytest.raises(TypeError, match=rf"{name} dtype must be torch.bfloat16"):
        gdn_causal_conv_decode_reference(*inputs)


def test_rejects_non_int32_slot_indices() -> None:
    inputs = list(_inputs())
    inputs[3] = inputs[3].to(torch.int64)

    with pytest.raises(TypeError, match="slot_indices dtype must be torch.int32"):
        gdn_causal_conv_decode_reference(*inputs)


@pytest.mark.parametrize(
    ("index", "replacement", "message"),
    [
        (0, torch.ones((2, 3, 1), dtype=torch.bfloat16), "x must be rank 2"),
        (1, torch.ones((4, 4), dtype=torch.bfloat16), "weight must have 3 channels"),
        (2, torch.ones((5, 3, 4), dtype=torch.bfloat16), "state must use dim-first"),
        (3, torch.ones((2, 1), dtype=torch.int32), "slot_indices must be rank 1"),
        (3, torch.ones(1, dtype=torch.int32), "slot_indices must have shape"),
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
        gdn_causal_conv_decode_reference(*inputs)


def test_rejects_non_dim_first_state_layout() -> None:
    inputs = list(_inputs(channels=2, kernel_size=4))
    inputs[2] = inputs[2].transpose(-1, -2)

    with pytest.raises(ValueError, match="state must use dim-first"):
        gdn_causal_conv_decode_reference(*inputs)


@pytest.mark.parametrize(
    ("slot_indices", "message"),
    [
        (torch.tensor([0, 2], dtype=torch.int32), "reserved slot zero"),
        (torch.tensor([-1, 2], dtype=torch.int32), "negative slots"),
        (torch.tensor([1, 5], dtype=torch.int32), "less than slot count 5"),
        (torch.tensor([2, 2], dtype=torch.int32), "must be unique"),
    ],
)
def test_rejects_invalid_or_duplicate_slots_before_mutation(
    slot_indices: torch.Tensor,
    message: str,
) -> None:
    inputs = list(_inputs())
    inputs[3] = slot_indices
    state_before = inputs[2].clone()

    with pytest.raises(ValueError, match=message):
        gdn_causal_conv_decode_reference(*inputs)

    assert torch.equal(inputs[2], state_before)


@pytest.mark.parametrize("kernel_size", [1, 7])
def test_rejects_widths_unsupported_by_production_triton(kernel_size: int) -> None:
    inputs = list(_inputs(batch_size=1, kernel_size=4))
    inputs[1] = torch.ones((3, kernel_size), dtype=torch.bfloat16)
    inputs[2] = torch.ones((5, 3, max(kernel_size - 1, 1)), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="kernel_size must be supported"):
        gdn_causal_conv_decode_reference(*inputs)


def test_rejects_non_tensor_input() -> None:
    inputs = list(_inputs())
    inputs[1] = object()

    with pytest.raises(TypeError, match="weight must be a torch.Tensor"):
        gdn_causal_conv_decode_reference(*inputs)  # type: ignore[arg-type]
