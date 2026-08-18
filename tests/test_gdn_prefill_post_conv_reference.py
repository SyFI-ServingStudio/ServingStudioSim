"""CPU tests for Qwen GDN prefill post-convolution preparation semantics."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from profiling.runners.attention.gdn_prefill_post_conv_reference import (
    gdn_prefill_post_conv_reference,
)

_EPSILON = 1e-6


def _inputs(
    *,
    num_tokens: int = 3,
    num_qk_heads: int = 2,
    num_value_heads: int = 3,
    key_head_dim: int = 4,
    value_head_dim: int = 2,
) -> tuple[tuple[torch.Tensor, ...], dict[str, int]]:
    generator = torch.Generator().manual_seed(20260810)
    packed_width = 2 * num_qk_heads * key_head_dim + num_value_heads * value_head_dim
    conv_output = (torch.randn(num_tokens, packed_width, generator=generator) * 0.25).to(
        torch.bfloat16
    )
    a = (torch.randn(num_tokens, num_value_heads, generator=generator) * 0.5).to(torch.bfloat16)
    b = (torch.randn(num_tokens, num_value_heads, generator=generator) * 0.5).to(torch.bfloat16)
    A_log = torch.randn(num_value_heads, generator=generator, dtype=torch.float32) - 2
    dt_bias = torch.randn(num_value_heads, generator=generator, dtype=torch.float32) * 0.1
    kwargs = {
        "num_qk_heads": num_qk_heads,
        "num_value_heads": num_value_heads,
        "key_head_dim": key_head_dim,
        "value_head_dim": value_head_dim,
    }
    return (conv_output, a, b, A_log, dt_bias), kwargs


def _manual(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> tuple[torch.Tensor, ...]:
    num_tokens = conv_output.shape[0]
    q_width = num_qk_heads * key_head_dim
    q_raw = conv_output[:, :q_width].reshape(num_tokens, num_qk_heads, key_head_dim)
    k_raw = conv_output[:, q_width : 2 * q_width].reshape(num_tokens, num_qk_heads, key_head_dim)
    v = conv_output[:, 2 * q_width :].reshape(num_tokens, num_value_heads, value_head_dim)

    def normalize(raw: torch.Tensor) -> torch.Tensor:
        raw_fp32 = raw.float()
        denominator = torch.sqrt(raw_fp32.square().sum(dim=-1, keepdim=True) + _EPSILON)
        return (raw_fp32 / denominator).to(torch.bfloat16)

    gate_input = a.float() + dt_bias.float()
    below_threshold = torch.logaddexp(gate_input, torch.zeros_like(gate_input))
    softplus = torch.where(gate_input > 20.0, gate_input, below_threshold)
    g = -torch.exp(A_log.float()) * softplus
    beta = torch.sigmoid(b.float())
    return normalize(q_raw), normalize(k_raw), v.clone(), g, beta


def _call(inputs: tuple[torch.Tensor, ...], kwargs: dict[str, int]) -> tuple[torch.Tensor, ...]:
    return gdn_prefill_post_conv_reference(*inputs, **kwargs)


def test_manual_equations_shapes_dtypes_contiguity_freshness_and_immutability() -> None:
    inputs, kwargs = _inputs()
    snapshots = tuple(tensor.clone() for tensor in inputs)

    outputs = _call(inputs, kwargs)
    expected = _manual(*inputs, **kwargs)

    expected_shapes = [(3, 2, 4), (3, 2, 4), (3, 3, 2), (3, 3), (3, 3)]
    expected_dtypes = [torch.bfloat16] * 3 + [torch.float32] * 2
    assert [tuple(output.shape) for output in outputs] == expected_shapes
    assert [output.dtype for output in outputs] == expected_dtypes
    assert all(output.is_contiguous() for output in outputs)
    for actual, wanted in zip(outputs[:3], expected[:3]):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    for actual, wanted in zip(outputs[3:], expected[3:]):
        torch.testing.assert_close(actual, wanted, rtol=1e-6, atol=1e-6)
    assert all(torch.equal(actual, before) for actual, before in zip(inputs, snapshots))

    input_ptrs = {tensor.untyped_storage().data_ptr() for tensor in inputs}
    output_ptrs = [tensor.untyped_storage().data_ptr() for tensor in outputs]
    assert not input_ptrs.intersection(output_ptrs)
    assert len(set(output_ptrs)) == len(output_ptrs)


def test_packed_q_k_v_split_boundaries() -> None:
    num_tokens, h, hv, k, vdim = 2, 2, 3, 2, 1
    q_width = h * k
    q_raw = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.bfloat16)
    k_raw = torch.tensor([[5, 6, 7, 8], [8, 7, 6, 5]], dtype=torch.bfloat16)
    v_raw = torch.tensor([[9, 10, 11], [12, 13, 14]], dtype=torch.bfloat16)
    conv_output = torch.cat((q_raw, k_raw, v_raw), dim=-1)
    a = torch.zeros(num_tokens, hv, dtype=torch.bfloat16)
    b = torch.zeros_like(a)
    params = torch.zeros(hv, dtype=torch.float32)

    q, k_out, value, _, _ = gdn_prefill_post_conv_reference(
        conv_output,
        a,
        b,
        params,
        params,
        num_qk_heads=h,
        num_value_heads=hv,
        key_head_dim=k,
        value_head_dim=vdim,
    )

    assert torch.equal(value.reshape(num_tokens, -1), v_raw)
    expected_q = q_raw.reshape(num_tokens, h, k).float()
    expected_q *= torch.rsqrt(expected_q.square().sum(-1, keepdim=True) + _EPSILON)
    expected_k = k_raw.reshape(num_tokens, h, k).float()
    expected_k *= torch.rsqrt(expected_k.square().sum(-1, keepdim=True) + _EPSILON)
    assert torch.equal(q, expected_q.to(torch.bfloat16))
    assert torch.equal(k_out, expected_k.to(torch.bfloat16))
    assert q_width == 4


def test_tokens_heads_and_gate_entries_are_independent() -> None:
    inputs, kwargs = _inputs(num_tokens=2, num_qk_heads=2, num_value_heads=3)
    baseline = _call(inputs, kwargs)
    changed = [tensor.clone() for tensor in inputs]
    key_dim = kwargs["key_head_dim"]
    changed[0][1, key_dim : 2 * key_dim] += 2
    changed[1][0, 2] += 1
    updated = _call(tuple(changed), kwargs)

    assert torch.equal(baseline[0][0], updated[0][0])
    assert torch.equal(baseline[0][1, 0], updated[0][1, 0])
    assert not torch.equal(baseline[0][1, 1], updated[0][1, 1])
    assert torch.equal(baseline[1], updated[1])
    assert torch.equal(baseline[2], updated[2])
    assert torch.equal(baseline[3][1], updated[3][1])
    assert torch.equal(baseline[3][0, :2], updated[3][0, :2])
    assert baseline[3][0, 2] != updated[3][0, 2]
    assert torch.equal(baseline[4], updated[4])


def test_zero_and_tiny_vectors_use_sum_plus_epsilon_not_f_normalize() -> None:
    h, hv, k, vdim = 1, 1, 2, 1
    conv_output = torch.tensor([[0.0, 0.0, 1e-4, 0.0, 7.0]], dtype=torch.bfloat16)
    a = torch.zeros(1, hv, dtype=torch.bfloat16)
    b = torch.zeros_like(a)
    params = torch.zeros(hv, dtype=torch.float32)

    q, key, _, _, _ = gdn_prefill_post_conv_reference(
        conv_output,
        a,
        b,
        params,
        params,
        num_qk_heads=h,
        num_value_heads=hv,
        key_head_dim=k,
        value_head_dim=vdim,
    )
    wrong = F.normalize(conv_output[:, 2:4].reshape(1, h, k).float(), p=2, dim=-1, eps=1e-6).to(
        torch.bfloat16
    )

    assert torch.count_nonzero(q) == 0
    assert key[0, 0, 0] < torch.tensor(0.11, dtype=torch.bfloat16)
    assert key[0, 0, 0] > torch.tensor(0.09, dtype=torch.bfloat16)
    assert not torch.equal(key, wrong)
    assert wrong[0, 0, 0] == torch.tensor(1.0, dtype=torch.bfloat16)


def test_v_is_bit_exact_but_has_fresh_storage() -> None:
    inputs, kwargs = _inputs()
    q_width = kwargs["num_qk_heads"] * kwargs["key_head_dim"]
    expected = inputs[0][:, 2 * q_width :].reshape(
        inputs[0].shape[0], kwargs["num_value_heads"], kwargs["value_head_dim"]
    )

    value = _call(inputs, kwargs)[2]

    assert torch.equal(value, expected)
    assert value.untyped_storage().data_ptr() != inputs[0].untyped_storage().data_ptr()


def test_beta_zero_is_exactly_one_half() -> None:
    inputs, kwargs = _inputs()
    values = list(inputs)
    values[2] = torch.zeros_like(values[2])

    beta = _call(tuple(values), kwargs)[4]

    assert torch.equal(beta, torch.full_like(beta, 0.5))


def test_softplus_threshold_and_bounded_extremes() -> None:
    h, hv, k, vdim = 1, 5, 1, 1
    conv_output = torch.zeros(1, 2 * h * k + hv * vdim, dtype=torch.bfloat16)
    a = torch.tensor([[-30.0, -1.0, 0.0, 20.0, 30.0]], dtype=torch.bfloat16)
    b = torch.zeros_like(a)
    A_log = torch.zeros(hv, dtype=torch.float32)
    dt_bias = torch.zeros(hv, dtype=torch.float32)

    g = gdn_prefill_post_conv_reference(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_qk_heads=h,
        num_value_heads=hv,
        key_head_dim=k,
        value_head_dim=vdim,
    )[3]
    expected = torch.tensor(
        [
            -math.log1p(math.exp(-30.0)),
            -math.log1p(math.exp(-1.0)),
            -math.log(2.0),
            -(20.0 + math.log1p(math.exp(-20.0))),
            -30.0,
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    assert torch.isfinite(g).all()
    torch.testing.assert_close(g, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("h", "hv", "k", "v"),
    [(1, 2, 3, 1), (3, 2, 2, 4), (2, 5, 5, 3)],
)
def test_distinct_small_geometry_parameterization(h: int, hv: int, k: int, v: int) -> None:
    inputs, kwargs = _inputs(
        num_tokens=2,
        num_qk_heads=h,
        num_value_heads=hv,
        key_head_dim=k,
        value_head_dim=v,
    )

    actual = _call(inputs, kwargs)
    expected = _manual(*inputs, **kwargs)

    for observed, wanted in zip(actual[:3], expected[:3]):
        assert torch.equal(observed, wanted)
    for observed, wanted in zip(actual[3:], expected[3:]):
        torch.testing.assert_close(observed, wanted, rtol=1e-6, atol=1e-6)


def test_noncontiguous_strided_semantic_inputs_are_supported() -> None:
    inputs, kwargs = _inputs(num_tokens=3, num_qk_heads=2, num_value_heads=3)
    conv, a, b, A_log, dt_bias = inputs
    conv = torch.stack((conv, torch.zeros_like(conv)), dim=-1)[..., 0]
    a = torch.stack((a, torch.zeros_like(a)), dim=-1)[..., 0]
    b = torch.stack((b, torch.zeros_like(b)), dim=-1)[..., 0]
    A_log = torch.stack((A_log, torch.zeros_like(A_log)), dim=-1)[..., 0]
    dt_bias = torch.stack((dt_bias, torch.zeros_like(dt_bias)), dim=-1)[..., 0]
    strided = (conv, a, b, A_log, dt_bias)
    assert all(not tensor.is_contiguous() for tensor in strided)

    actual = _call(strided, kwargs)
    expected = _manual(*strided, **kwargs)

    for observed, wanted in zip(actual[:3], expected[:3]):
        assert torch.equal(observed, wanted)
    for observed, wanted in zip(actual[3:], expected[3:]):
        torch.testing.assert_close(observed, wanted, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "name",
    ["num_qk_heads", "num_value_heads", "key_head_dim", "value_head_dim"],
)
@pytest.mark.parametrize("invalid", [0, -1])
def test_rejects_nonpositive_scalar_dimensions(name: str, invalid: int) -> None:
    inputs, kwargs = _inputs()
    kwargs[name] = invalid

    with pytest.raises(ValueError, match=rf"{name} must be positive"):
        _call(inputs, kwargs)


@pytest.mark.parametrize(
    "invalid",
    [True, 1.5, "2", None],
)
@pytest.mark.parametrize(
    "name",
    ["num_qk_heads", "num_value_heads", "key_head_dim", "value_head_dim"],
)
def test_rejects_invalid_scalar_types(name: str, invalid: object) -> None:
    inputs, kwargs = _inputs()
    kwargs[name] = invalid  # type: ignore[assignment]

    with pytest.raises(TypeError, match=rf"{name} must be an int"):
        _call(inputs, kwargs)


def test_rejects_zero_tokens() -> None:
    _, kwargs = _inputs()
    width = (
        2 * kwargs["num_qk_heads"] * kwargs["key_head_dim"]
        + kwargs["num_value_heads"] * kwargs["value_head_dim"]
    )
    values = (
        torch.empty(0, width, dtype=torch.bfloat16),
        torch.empty(0, kwargs["num_value_heads"], dtype=torch.bfloat16),
        torch.empty(0, kwargs["num_value_heads"], dtype=torch.bfloat16),
        torch.zeros(kwargs["num_value_heads"], dtype=torch.float32),
        torch.zeros(kwargs["num_value_heads"], dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="conv_output dimensions must be positive"):
        _call(values, kwargs)


@pytest.mark.parametrize("name", ["conv_output", "a", "b", "A_log", "dt_bias"])
def test_rejects_non_tensor_inputs(name: str) -> None:
    inputs, kwargs = _inputs()
    values = list(inputs)
    values[("conv_output", "a", "b", "A_log", "dt_bias").index(name)] = None

    with pytest.raises(TypeError, match=rf"{name} must be a torch.Tensor"):
        _call(tuple(values), kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "replacement", "rank"),
    [
        ("conv_output", torch.zeros(3, 4, 1, dtype=torch.bfloat16), 2),
        ("a", torch.zeros(3, dtype=torch.bfloat16), 2),
        ("b", torch.zeros(3, 3, 1, dtype=torch.bfloat16), 2),
        ("A_log", torch.zeros(1, 3, dtype=torch.float32), 1),
        ("dt_bias", torch.zeros(3, 1, dtype=torch.float32), 1),
    ],
)
def test_rejects_invalid_ranks(name: str, replacement: torch.Tensor, rank: int) -> None:
    inputs, kwargs = _inputs()
    values = list(inputs)
    values[("conv_output", "a", "b", "A_log", "dt_bias").index(name)] = replacement

    with pytest.raises(ValueError, match=rf"{name} must be rank {rank}"):
        _call(tuple(values), kwargs)


@pytest.mark.parametrize("name", ["conv_output", "a", "b"])
def test_rejects_non_bf16_or_mismatched_activation_dtype(name: str) -> None:
    inputs, kwargs = _inputs()
    values = list(inputs)
    index = ("conv_output", "a", "b").index(name)
    values[index] = values[index].float()

    with pytest.raises(TypeError, match=rf"{name} dtype must be torch.bfloat16"):
        _call(tuple(values), kwargs)


@pytest.mark.parametrize("name", ["A_log", "dt_bias"])
def test_rejects_non_fp32_parameters(name: str) -> None:
    inputs, kwargs = _inputs()
    values = list(inputs)
    index = ("A_log", "dt_bias").index(name) + 3
    values[index] = values[index].to(torch.bfloat16)

    with pytest.raises(TypeError, match=rf"{name} dtype must be torch.float32"):
        _call(tuple(values), kwargs)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda values: values.__setitem__(0, values[0][:, :-1]),
            "conv_output must have shape",
        ),
        (
            lambda values: values.__setitem__(1, values[1][:, :-1]),
            "a must have shape",
        ),
        (
            lambda values: values.__setitem__(2, values[2][:-1]),
            "b must have shape",
        ),
        (
            lambda values: values.__setitem__(3, values[3][:-1]),
            "A_log must have shape",
        ),
        (
            lambda values: values.__setitem__(4, values[4][:-1]),
            "dt_bias must have shape",
        ),
    ],
)
def test_rejects_incompatible_shapes(
    mutate: Callable[[list[torch.Tensor]], None], message: str
) -> None:
    inputs, kwargs = _inputs()
    values = list(inputs)
    mutate(values)

    with pytest.raises(ValueError, match=message):
        _call(tuple(values), kwargs)


def test_rejects_non_strided_tensor() -> None:
    inputs, kwargs = _inputs()
    values = list(inputs)
    dense = values[0]
    indices = torch.nonzero(dense, as_tuple=False).T
    values[0] = torch.sparse_coo_tensor(indices, dense.reshape(-1), dense.shape).coalesce()

    with pytest.raises(ValueError, match="conv_output must have torch.strided layout"):
        _call(tuple(values), kwargs)


def test_rejects_meta_tensor() -> None:
    inputs, kwargs = _inputs()
    values = list(inputs)
    values[0] = torch.empty_like(values[0], device="meta")

    with pytest.raises(ValueError, match="meta tensors are not supported"):
        _call(tuple(values), kwargs)
