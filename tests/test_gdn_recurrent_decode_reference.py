"""CPU tests for the Qwen Gated DeltaNet recurrent-decode reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from profiling.runners.attention.gdn_recurrent_decode_reference import (
    gdn_recurrent_decode_reference,
)


def _inputs(
    *,
    batch: int = 2,
    query_heads: int = 2,
    value_heads: int = 4,
    key_dim: int = 3,
    value_dim: int = 2,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260810)
    query = torch.randn(batch, query_heads, key_dim, generator=generator, dtype=torch.bfloat16)
    key = torch.randn(batch, query_heads, key_dim, generator=generator, dtype=torch.bfloat16)
    value = torch.randn(batch, value_heads, value_dim, generator=generator, dtype=torch.bfloat16)
    a = torch.randn(batch, value_heads, generator=generator, dtype=torch.bfloat16)
    b = torch.randn(batch, value_heads, generator=generator, dtype=torch.bfloat16)
    A_log = torch.randn(value_heads, generator=generator, dtype=torch.float32)
    dt_bias = torch.randn(value_heads, generator=generator, dtype=torch.float32)
    state = torch.randn(
        batch,
        value_heads,
        key_dim,
        value_dim,
        generator=generator,
        dtype=torch.float32,
    )
    return query, key, value, a, b, A_log, dt_bias, state


def _manual_equation(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    query, key, value, a, b, A_log, dt_bias, state = inputs
    repeats = value.shape[1] // query.shape[1]

    q = query.float()
    k = key.float()
    q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(repeats, dim=1) * (query.shape[-1] ** -0.5)
    k = k.repeat_interleave(repeats, dim=1)

    g = -torch.exp(A_log) * F.softplus(a.float() + dt_bias)
    beta = torch.sigmoid(b.float()).to(torch.bfloat16).float()
    decayed = state * torch.exp(g)[..., None, None]
    memory = torch.sum(decayed * k[..., None], dim=-2)
    delta = (value.float() - memory) * beta[..., None]
    updated = decayed + k[..., None] * delta[..., None, :]
    output = torch.sum(updated * q[..., None], dim=-2).to(torch.bfloat16)
    return output, updated


def test_shapes_dtypes_manual_equation_and_inplace_state_update() -> None:
    inputs = _inputs()
    state = inputs[-1]
    state_before = state.clone()
    expected_output, expected_state = _manual_equation((*inputs[:-1], state_before))
    storage = state.untyped_storage().data_ptr()

    output, returned_state = gdn_recurrent_decode_reference(*inputs)

    assert output.shape == (2, 4, 2)
    assert output.dtype is torch.bfloat16
    assert returned_state.shape == (2, 4, 3, 2)
    assert returned_state.dtype is torch.float32
    assert returned_state is state
    assert returned_state.untyped_storage().data_ptr() == storage
    assert not torch.equal(returned_state, state_before)
    torch.testing.assert_close(output, expected_output, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(returned_state, expected_state, rtol=1e-6, atol=1e-6)


def test_query_key_heads_expand_consecutively_to_value_heads() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.bfloat16)
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.bfloat16)
    value = torch.zeros((1, 4, 1), dtype=torch.bfloat16)
    a = torch.zeros((1, 4), dtype=torch.bfloat16)
    b = torch.full((1, 4), -100.0, dtype=torch.bfloat16)
    A_log = torch.zeros(4, dtype=torch.float32)
    dt_bias = torch.zeros(4, dtype=torch.float32)
    state = torch.zeros((1, 4, 2, 1), dtype=torch.float32)
    state[0, 0, 0, 0] = 1.0
    state[0, 1, 0, 0] = 2.0
    state[0, 2, 1, 0] = 3.0
    state[0, 3, 1, 0] = 4.0

    output, _ = gdn_recurrent_decode_reference(query, key, value, a, b, A_log, dt_bias, state)

    decay = torch.exp(-F.softplus(torch.tensor(0.0)))
    expected = (decay * torch.tensor([1.0, 2.0, 3.0, 4.0]) / (2.0**0.5)).to(torch.bfloat16)
    torch.testing.assert_close(output[0, :, 0], expected, rtol=8e-3, atol=8e-3)


def test_zero_beta_performs_decay_and_readout_without_delta_update() -> None:
    inputs = list(_inputs(batch=1, query_heads=1, value_heads=2))
    inputs[4] = torch.full((1, 2), -100.0, dtype=torch.bfloat16)
    state_before = inputs[-1].clone()
    query, key, _, a, _, A_log, dt_bias, _ = inputs

    output, updated_state = gdn_recurrent_decode_reference(*inputs)

    decay = torch.exp(-torch.exp(A_log) * F.softplus(a.float() + dt_bias))
    expected_state = state_before * decay[..., None, None]
    q = query.float()
    q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(2, dim=1) * (query.shape[-1] ** -0.5)
    expected_output = torch.sum(expected_state * q[..., None], dim=-2).to(torch.bfloat16)

    torch.testing.assert_close(updated_state, expected_state, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(output, expected_output, rtol=8e-3, atol=8e-3)


@pytest.mark.parametrize("name", ["query", "key", "value", "a", "b"])
def test_rejects_non_bf16_activation(name: str) -> None:
    inputs = list(_inputs())
    index = ("query", "key", "value", "a", "b").index(name)
    inputs[index] = inputs[index].float()

    with pytest.raises(TypeError, match=rf"{name} dtype must be torch.bfloat16"):
        gdn_recurrent_decode_reference(*inputs)


@pytest.mark.parametrize("name", ["A_log", "dt_bias", "state"])
def test_rejects_non_fp32_parameters_or_state(name: str) -> None:
    inputs = list(_inputs())
    index = ("A_log", "dt_bias", "state").index(name) + 5
    inputs[index] = inputs[index].to(torch.bfloat16)

    with pytest.raises(TypeError, match=rf"{name} dtype must be torch.float32"):
        gdn_recurrent_decode_reference(*inputs)


def test_rejects_incompatible_head_divisibility_before_state_mutation() -> None:
    inputs = list(_inputs(query_heads=3, value_heads=4))
    state_before = inputs[-1].clone()

    with pytest.raises(ValueError, match="value head count must be divisible"):
        gdn_recurrent_decode_reference(*inputs)

    assert torch.equal(inputs[-1], state_before)


@pytest.mark.parametrize(
    ("index", "replacement", "message"),
    [
        (0, torch.ones((2, 2, 1, 3), dtype=torch.bfloat16), "query must be rank 3"),
        (1, torch.ones((2, 2, 4), dtype=torch.bfloat16), "key shape must match"),
        (2, torch.ones((1, 4, 2), dtype=torch.bfloat16), "value batch dimension"),
        (3, torch.ones((2, 3), dtype=torch.bfloat16), "a must have shape"),
        (5, torch.ones(3, dtype=torch.float32), "A_log must have shape"),
        (7, torch.ones((2, 4, 2, 2), dtype=torch.float32), "state must have shape"),
    ],
)
def test_rejects_invalid_shapes(index: int, replacement: torch.Tensor, message: str) -> None:
    inputs = list(_inputs())
    inputs[index] = replacement

    with pytest.raises(ValueError, match=message):
        gdn_recurrent_decode_reference(*inputs)


def test_rejects_non_tensor_input() -> None:
    inputs = list(_inputs())
    inputs[4] = object()

    with pytest.raises(TypeError, match="b must be a torch.Tensor"):
        gdn_recurrent_decode_reference(*inputs)  # type: ignore[arg-type]
