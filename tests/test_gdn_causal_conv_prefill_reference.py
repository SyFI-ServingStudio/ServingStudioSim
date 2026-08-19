from __future__ import annotations

import pytest
import torch

from profiling.runners.attention.gdn_causal_conv_prefill_reference import (
    gdn_causal_conv_prefill_reference,
)


def _inputs(
    *,
    batch_size: int = 2,
    sequence_length: int = 5,
    channels: int = 3,
    kernel_size: int = 4,
    slots: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(23)
    x = (torch.randn(batch_size, sequence_length, channels, generator=generator) * 0.2).to(
        torch.bfloat16
    )
    weight = (torch.randn(channels, kernel_size, generator=generator) * 0.3).to(torch.bfloat16)
    state = torch.randn(slots, channels, kernel_size - 1, generator=generator, dtype=torch.bfloat16)
    slot_indices = torch.arange(1, batch_size + 1, dtype=torch.int32)
    return x, weight, state, slot_indices


def _manual(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, channels = x.shape
    kernel_size = weight.shape[1]
    state_length = kernel_size - 1
    accumulator = torch.zeros(batch_size, sequence_length, channels, dtype=torch.float32)
    for batch in range(batch_size):
        for token in range(sequence_length):
            for channel in range(channels):
                for tap in range(kernel_size):
                    input_token = token + tap - state_length
                    if input_token >= 0:
                        accumulator[batch, token, channel] += (
                            x[batch, input_token, channel].float() * weight[channel, tap].float()
                        )
    output = (accumulator * torch.sigmoid(accumulator)).to(torch.bfloat16)
    updated_state = torch.zeros(batch_size, channels, state_length, dtype=torch.bfloat16)
    tail_length = min(sequence_length, state_length)
    updated_state[..., -tail_length:] = x[:, -tail_length:, :].permute(0, 2, 1)
    return output, updated_state


def test_manual_equation_shapes_dtypes_identity_and_immutability() -> None:
    x, weight, state, slot_indices = _inputs()
    x_before = x.clone()
    weight_before = weight.clone()
    state_before = state.clone()
    expected_output, expected_state = _manual(x, weight)

    output, returned_state = gdn_causal_conv_prefill_reference(x, weight, state, slot_indices)

    assert output.shape == x.shape
    assert output.dtype is torch.bfloat16
    assert returned_state is state
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(state.index_select(0, slot_indices.long()), expected_state)
    torch.testing.assert_close(state[0], state_before[0])
    torch.testing.assert_close(x, x_before)
    torch.testing.assert_close(weight, weight_before)
    assert output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    assert output.untyped_storage().data_ptr() != state.untyped_storage().data_ptr()


def test_causal_ordering_does_not_read_future_tokens() -> None:
    x, weight, state, slot_indices = _inputs(batch_size=1, sequence_length=5, channels=1)
    changed = x.clone()
    changed[:, 3:, :] += 8

    output, _ = gdn_causal_conv_prefill_reference(x, weight, state.clone(), slot_indices)
    changed_output, _ = gdn_causal_conv_prefill_reference(
        changed, weight, state.clone(), slot_indices
    )

    torch.testing.assert_close(output[:, :3], changed_output[:, :3], rtol=0, atol=0)
    assert not torch.equal(output[:, 3:], changed_output[:, 3:])


def test_sequences_and_channels_are_independent() -> None:
    x, weight, state, slot_indices = _inputs()
    changed = x.clone()
    changed[0, :, 1] += 1

    output, _ = gdn_causal_conv_prefill_reference(x, weight, state.clone(), slot_indices)
    changed_output, _ = gdn_causal_conv_prefill_reference(
        changed, weight, state.clone(), slot_indices
    )

    torch.testing.assert_close(output[1], changed_output[1], rtol=0, atol=0)
    torch.testing.assert_close(output[0, :, [0, 2]], changed_output[0, :, [0, 2]])
    assert not torch.equal(output[0, :, 1], changed_output[0, :, 1])


@pytest.mark.parametrize("sequence_length", [2, 5])
def test_selected_state_is_overwritten_without_reading_prior_state(
    sequence_length: int,
) -> None:
    x, weight, state, slot_indices = _inputs(batch_size=1, sequence_length=sequence_length)
    alternate_state = state.clone()
    alternate_state[1].fill_(17)
    expected_output, expected_state = _manual(x, weight)

    output, returned_state = gdn_causal_conv_prefill_reference(
        x, weight, alternate_state, slot_indices
    )

    assert returned_state is alternate_state
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(alternate_state[1], expected_state[0], rtol=0, atol=0)


def test_permuted_slots_and_unselected_slots_are_preserved() -> None:
    x, weight, state, _ = _inputs(batch_size=3, slots=7)
    slot_indices = torch.tensor([3, 1, 5], dtype=torch.int32)
    state_before = state.clone()
    _, expected_state = _manual(x, weight)

    _, returned_state = gdn_causal_conv_prefill_reference(x, weight, state, slot_indices)

    assert returned_state is state
    for batch, slot in enumerate(slot_indices.tolist()):
        torch.testing.assert_close(state[slot], expected_state[batch], rtol=0, atol=0)
    for slot in {0, 2, 4, 6}:
        torch.testing.assert_close(state[slot], state_before[slot], rtol=0, atol=0)


@pytest.mark.parametrize("kernel_size", [2, 3, 4])
def test_supported_widths_match_manual_equation(kernel_size: int) -> None:
    x, weight, state, slot_indices = _inputs(
        batch_size=2,
        sequence_length=3,
        channels=2,
        kernel_size=kernel_size,
    )
    expected_output, expected_state = _manual(x, weight)

    output, _ = gdn_causal_conv_prefill_reference(x, weight, state, slot_indices)

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(state.index_select(0, slot_indices.long()), expected_state)


@pytest.mark.parametrize("zero_operand", ["x", "weight"])
def test_zero_operands(zero_operand: str) -> None:
    x, weight, state, slot_indices = _inputs()
    if zero_operand == "x":
        x.zero_()
    else:
        weight.zero_()
    expected_output, expected_state = _manual(x, weight)

    output, _ = gdn_causal_conv_prefill_reference(x, weight, state, slot_indices)

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    torch.testing.assert_close(state.index_select(0, slot_indices.long()), expected_state)


def test_noncontiguous_semantic_inputs_are_supported() -> None:
    x, weight, state, slot_indices = _inputs()
    x = x.transpose(1, 2).contiguous().transpose(1, 2)
    weight = weight.transpose(0, 1).contiguous().transpose(0, 1)
    assert not x.is_contiguous()
    assert not weight.is_contiguous()
    expected_output, expected_state = _manual(x, weight)

    output, _ = gdn_causal_conv_prefill_reference(x, weight, state, slot_indices)

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(state.index_select(0, slot_indices.long()), expected_state)


@pytest.mark.parametrize("name", ["x", "weight", "state", "slot_indices"])
def test_rejects_non_tensor_inputs(name: str) -> None:
    values = list(_inputs())
    index = ("x", "weight", "state", "slot_indices").index(name)
    values[index] = None
    with pytest.raises(TypeError, match=rf"{name} must be a torch.Tensor"):
        gdn_causal_conv_prefill_reference(*values)


@pytest.mark.parametrize(
    ("name", "replacement", "rank"),
    [
        ("x", torch.zeros(2, 3, dtype=torch.bfloat16), 3),
        ("weight", torch.zeros(3, dtype=torch.bfloat16), 2),
        ("state", torch.zeros(6, 3, dtype=torch.bfloat16), 3),
        ("slot_indices", torch.ones(2, 1, dtype=torch.int32), 1),
    ],
)
def test_rejects_invalid_ranks(name: str, replacement: torch.Tensor, rank: int) -> None:
    values = list(_inputs())
    index = ("x", "weight", "state", "slot_indices").index(name)
    values[index] = replacement
    with pytest.raises(ValueError, match=rf"{name} must be rank {rank}"):
        gdn_causal_conv_prefill_reference(*values)


@pytest.mark.parametrize("name", ["x", "weight", "state"])
def test_rejects_non_bf16_data(name: str) -> None:
    values = list(_inputs())
    index = ("x", "weight", "state", "slot_indices").index(name)
    values[index] = values[index].float()
    with pytest.raises(TypeError, match=rf"{name} dtype must be torch.bfloat16"):
        gdn_causal_conv_prefill_reference(*values)


def test_rejects_non_int32_slots() -> None:
    x, weight, state, slot_indices = _inputs()
    with pytest.raises(TypeError, match="slot_indices dtype must be torch.int32"):
        gdn_causal_conv_prefill_reference(x, weight, state, slot_indices.long())


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (torch.empty(0, 5, 3, dtype=torch.bfloat16), "x dimensions must be positive"),
        (torch.empty(2, 0, 3, dtype=torch.bfloat16), "x dimensions must be positive"),
        (torch.empty(2, 5, 0, dtype=torch.bfloat16), "x dimensions must be positive"),
    ],
)
def test_rejects_nonpositive_semantic_dimensions(replacement: torch.Tensor, message: str) -> None:
    _, weight, state, slot_indices = _inputs()
    with pytest.raises(ValueError, match=message):
        gdn_causal_conv_prefill_reference(replacement, weight, state, slot_indices)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda values: values.__setitem__(1, torch.zeros(4, 4, dtype=torch.bfloat16)),
            "weight must have 3 channels",
        ),
        (
            lambda values: values.__setitem__(2, torch.zeros(6, 4, 3, dtype=torch.bfloat16)),
            "state must use dim-first shape",
        ),
        (
            lambda values: values.__setitem__(2, torch.zeros(6, 3, 2, dtype=torch.bfloat16)),
            "state must use dim-first shape",
        ),
        (
            lambda values: values.__setitem__(3, torch.tensor([1], dtype=torch.int32)),
            "slot_indices must have shape",
        ),
    ],
)
def test_rejects_incompatible_shapes(mutate: object, message: str) -> None:
    values = list(_inputs())
    mutate(values)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        gdn_causal_conv_prefill_reference(*values)


@pytest.mark.parametrize("kernel_size", [1, 5])
def test_rejects_unsupported_widths(kernel_size: int) -> None:
    x, _, _, slot_indices = _inputs()
    weight = torch.zeros(x.shape[-1], kernel_size, dtype=torch.bfloat16)
    state = torch.zeros(6, x.shape[-1], max(kernel_size - 1, 1), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="kernel_size must be supported"):
        gdn_causal_conv_prefill_reference(x, weight, state, slot_indices)


@pytest.mark.parametrize(
    ("slot_indices", "message"),
    [
        (torch.tensor([0, 2], dtype=torch.int32), "reserved slot zero"),
        (torch.tensor([-1, 2], dtype=torch.int32), "reserved slot zero"),
        (torch.tensor([1, 6], dtype=torch.int32), "less than slot count"),
        (torch.tensor([2, 2], dtype=torch.int32), "must be unique"),
    ],
)
def test_rejects_invalid_slots_without_mutation(slot_indices: torch.Tensor, message: str) -> None:
    x, weight, state, _ = _inputs()
    state_before = state.clone()
    with pytest.raises(ValueError, match=message):
        gdn_causal_conv_prefill_reference(x, weight, state, slot_indices)
    torch.testing.assert_close(state, state_before)
