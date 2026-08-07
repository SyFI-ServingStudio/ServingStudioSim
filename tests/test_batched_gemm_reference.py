"""CPU contract tests for the layout-agnostic batched GEMM reference."""

from __future__ import annotations

import pytest
import torch

from profiling.runners.gemm.batched_gemm_reference import batched_gemm_reference

_Q_WIDTH = 256
_Q_NOPE_WIDTH = 192
_KV_LORA_RANK = 512
_V_WIDTH = 256
_PACKED_HEAD_WIDTH = 448
_PADDED_HEADS = 64


def _oracle(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    batches = [
        lhs[index].to(torch.float32) @ rhs[index].to(torch.float32)
        for index in range(lhs.shape[0])
    ]
    return torch.stack(batches).to(lhs.dtype)


def _values(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype) * 0.125


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-6, 1e-6),
        (torch.bfloat16, 2e-2, 2e-2),
        (torch.float16, 5e-3, 5e-3),
    ],
)
def test_contiguous_correctness_shape_dtype_and_input_immutability(dtype, rtol, atol):
    lhs = _values((2, 3, 4), dtype, seed=1)
    rhs = _values((2, 4, 5), dtype, seed=2)
    lhs_before = lhs.clone()
    rhs_before = rhs.clone()

    actual = batched_gemm_reference(lhs, rhs)

    torch.testing.assert_close(actual, _oracle(lhs, rhs), rtol=rtol, atol=atol)
    assert actual.shape == (2, 3, 5)
    assert actual.dtype == dtype
    assert actual.data_ptr() not in {lhs.data_ptr(), rhs.data_ptr()}
    assert torch.equal(lhs, lhs_before)
    assert torch.equal(rhs, rhs_before)


def test_out_returns_identical_object_and_storage_and_only_mutates_out():
    lhs = _values((2, 3, 4), torch.float32, seed=3)
    rhs = _values((2, 4, 5), torch.float32, seed=4)
    lhs_before = lhs.clone()
    rhs_before = rhs.clone()
    out = torch.full((2, 3, 5), -17.0)
    storage_ptr = out.untyped_storage().data_ptr()

    actual = batched_gemm_reference(lhs, rhs, out=out)

    assert actual is out
    assert actual.untyped_storage().data_ptr() == storage_ptr
    torch.testing.assert_close(actual, _oracle(lhs, rhs))
    assert torch.equal(lhs, lhs_before)
    assert torch.equal(rhs, rhs_before)


@pytest.mark.parametrize("num_heads", [64, 32, 16])
def test_glm_q_absorption_exact_layout_and_correctness(num_heads):
    m = 2
    dtype = torch.bfloat16
    q_base = _values((m, num_heads, _Q_WIDTH), dtype, seed=10 + num_heads)
    lhs = q_base[..., :_Q_NOPE_WIDTH].transpose(0, 1)

    packed_weight = _values(
        (num_heads * _PACKED_HEAD_WIDTH, _KV_LORA_RANK),
        dtype,
        seed=20 + num_heads,
    )
    packed_view = packed_weight.T.view(
        _KV_LORA_RANK,
        num_heads,
        _PACKED_HEAD_WIDTH,
    )
    w_uk = packed_view[..., :_Q_NOPE_WIDTH]
    rhs = w_uk.permute(1, 2, 0)
    out = torch.empty((num_heads, m, _KV_LORA_RANK), dtype=dtype)
    lhs_before = lhs.clone()
    rhs_before = rhs.clone()

    assert lhs.shape == (num_heads, m, _Q_NOPE_WIDTH)
    assert lhs.stride() == (_Q_WIDTH, num_heads * _Q_WIDTH, 1)
    assert lhs.storage_offset() == 0
    assert lhs.untyped_storage().data_ptr() == q_base.untyped_storage().data_ptr()
    assert not lhs.is_contiguous()

    assert rhs.shape == (num_heads, _Q_NOPE_WIDTH, _KV_LORA_RANK)
    assert rhs.stride() == (
        _PACKED_HEAD_WIDTH * _KV_LORA_RANK,
        _KV_LORA_RANK,
        1,
    )
    assert rhs.storage_offset() == 0
    assert rhs.untyped_storage().data_ptr() == packed_weight.untyped_storage().data_ptr()
    assert rhs.stride(0) - _Q_NOPE_WIDTH * _KV_LORA_RANK == (
        _V_WIDTH * _KV_LORA_RANK
    )
    assert not rhs.is_contiguous()
    assert out.is_contiguous()

    actual = batched_gemm_reference(lhs, rhs, out=out)

    assert actual is out
    torch.testing.assert_close(actual, _oracle(lhs, rhs), rtol=2e-2, atol=2e-2)
    assert torch.equal(lhs, lhs_before)
    assert torch.equal(rhs, rhs_before)


@pytest.mark.parametrize("num_heads", [64, 32, 16])
def test_glm_v_up_exact_layout_and_strided_out_correctness(num_heads):
    m = 2
    dtype = torch.bfloat16
    attention_base = _values(
        (m, _PADDED_HEADS, _KV_LORA_RANK),
        dtype,
        seed=30 + num_heads,
    )
    attention_output = attention_base[:, :num_heads, :]
    lhs = attention_output.view(m, num_heads, _KV_LORA_RANK).transpose(0, 1)

    packed_weight = _values(
        (num_heads * _PACKED_HEAD_WIDTH, _KV_LORA_RANK),
        dtype,
        seed=40 + num_heads,
    )
    packed_view = packed_weight.T.view(
        _KV_LORA_RANK,
        num_heads,
        _PACKED_HEAD_WIDTH,
    )
    w_uv = packed_view[..., _Q_NOPE_WIDTH:]
    rhs = w_uv.transpose(0, 1)

    out_base = torch.full((m, num_heads * _V_WIDTH), -3.0, dtype=dtype)
    out = out_base.view(m, num_heads, _V_WIDTH).transpose(0, 1)
    lhs_before = lhs.clone()
    rhs_before = rhs.clone()
    out_storage_ptr = out.untyped_storage().data_ptr()

    assert attention_output.shape == (m, num_heads, _KV_LORA_RANK)
    assert attention_output.stride() == (
        _PADDED_HEADS * _KV_LORA_RANK,
        _KV_LORA_RANK,
        1,
    )
    assert attention_output.untyped_storage().data_ptr() == (
        attention_base.untyped_storage().data_ptr()
    )
    assert lhs.shape == (num_heads, m, _KV_LORA_RANK)
    assert lhs.stride() == (
        _KV_LORA_RANK,
        _PADDED_HEADS * _KV_LORA_RANK,
        1,
    )
    assert lhs.storage_offset() == 0
    assert not lhs.is_contiguous()

    assert rhs.shape == (num_heads, _KV_LORA_RANK, _V_WIDTH)
    assert rhs.stride() == (
        _PACKED_HEAD_WIDTH * _KV_LORA_RANK,
        1,
        _KV_LORA_RANK,
    )
    assert rhs.storage_offset() == _Q_NOPE_WIDTH * _KV_LORA_RANK
    assert rhs.untyped_storage().data_ptr() == packed_weight.untyped_storage().data_ptr()
    assert rhs.stride(0) - _V_WIDTH * _KV_LORA_RANK == (
        _Q_NOPE_WIDTH * _KV_LORA_RANK
    )
    assert not rhs.is_contiguous()

    assert out.shape == (num_heads, m, _V_WIDTH)
    assert out.stride() == (_V_WIDTH, num_heads * _V_WIDTH, 1)
    assert out.storage_offset() == 0
    assert out.untyped_storage().data_ptr() == out_base.untyped_storage().data_ptr()
    assert int(torch._debug_has_internal_overlap(out)) == 0
    assert not out.is_contiguous()

    actual = batched_gemm_reference(lhs, rhs, out=out)

    assert actual is out
    assert actual.untyped_storage().data_ptr() == out_storage_ptr
    torch.testing.assert_close(actual, _oracle(lhs, rhs), rtol=2e-2, atol=2e-2)
    assert torch.equal(lhs, lhs_before)
    assert torch.equal(rhs, rhs_before)


def _valid_inputs(
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones((2, 3, 4), dtype=dtype), torch.ones((2, 4, 5), dtype=dtype)


@pytest.mark.parametrize(("which", "value"), [("lhs", object()), ("rhs", object())])
def test_rejects_non_tensor_inputs(which, value):
    lhs, rhs = _valid_inputs()
    if which == "lhs":
        lhs = value
    else:
        rhs = value

    with pytest.raises(TypeError, match=rf"{which} must be a torch.Tensor"):
        batched_gemm_reference(lhs, rhs)


def test_rejects_non_tensor_out():
    lhs, rhs = _valid_inputs()

    with pytest.raises(TypeError, match="out must be a torch.Tensor"):
        batched_gemm_reference(lhs, rhs, out=object())


@pytest.mark.parametrize(
    ("lhs", "rhs", "message"),
    [
        (torch.ones((3, 4)), torch.ones((2, 4, 5)), "lhs must be rank 3"),
        (torch.ones((2, 3, 4)), torch.ones((4, 5)), "rhs must be rank 3"),
    ],
)
def test_rejects_non_rank_three_inputs(lhs, rhs, message):
    with pytest.raises(ValueError, match=message):
        batched_gemm_reference(lhs, rhs)


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "message"),
    [
        ((0, 3, 4), (0, 4, 5), "lhs dimensions must be positive"),
        ((2, 0, 4), (2, 4, 5), "lhs dimensions must be positive"),
        ((2, 3, 0), (2, 0, 5), "lhs dimensions must be positive"),
        ((2, 3, 4), (2, 4, 0), "rhs dimensions must be positive"),
    ],
)
def test_rejects_empty_dimensions(lhs_shape, rhs_shape, message):
    lhs = torch.empty(lhs_shape)
    rhs = torch.empty(rhs_shape)

    with pytest.raises(ValueError, match=message):
        batched_gemm_reference(lhs, rhs)


def test_rejects_mismatched_batch_and_inner_dimensions():
    with pytest.raises(ValueError, match="same batch dimension"):
        batched_gemm_reference(torch.ones((2, 3, 4)), torch.ones((3, 4, 5)))
    with pytest.raises(ValueError, match="inner K dimensions"):
        batched_gemm_reference(torch.ones((2, 3, 4)), torch.ones((2, 6, 5)))


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float64,
        torch.int32,
        pytest.param(
            torch.float8_e4m3fn,
            id="float8_e4m3fn",
        ),
    ],
)
def test_rejects_unsupported_dtypes(dtype):
    lhs = torch.empty((2, 3, 4), dtype=dtype)
    rhs = torch.empty((2, 4, 5), dtype=dtype)

    with pytest.raises(TypeError, match="lhs dtype must be"):
        batched_gemm_reference(lhs, rhs)


def test_rejects_mismatched_input_dtypes():
    lhs, _ = _valid_inputs()
    rhs = torch.ones((2, 4, 5), dtype=torch.float16)

    with pytest.raises(TypeError, match="same dtype"):
        batched_gemm_reference(lhs, rhs)


def test_rejects_mismatched_input_devices_before_contraction():
    lhs, _ = _valid_inputs()
    rhs = torch.empty((2, 4, 5), dtype=lhs.dtype, device="meta")

    with pytest.raises(ValueError, match="same device"):
        batched_gemm_reference(lhs, rhs)


def test_rejects_invalid_out_rank_shape_dtype_and_device():
    lhs, rhs = _valid_inputs()

    with pytest.raises(ValueError, match="out must be rank 3"):
        batched_gemm_reference(lhs, rhs, out=torch.empty((2, 15)))
    with pytest.raises(ValueError, match="out must have shape"):
        batched_gemm_reference(lhs, rhs, out=torch.empty((2, 3, 6)))
    with pytest.raises(TypeError, match="out dtype must match"):
        batched_gemm_reference(lhs, rhs, out=torch.empty((2, 3, 5), dtype=torch.float16))
    with pytest.raises(ValueError, match="same device"):
        batched_gemm_reference(
            lhs,
            rhs,
            out=torch.empty((2, 3, 5), dtype=lhs.dtype, device="meta"),
        )


def test_rejects_definite_internal_overlap_in_out():
    lhs, rhs = _valid_inputs()
    out = torch.empty((1, 3, 5)).expand(2, 3, 5)
    assert int(torch._debug_has_internal_overlap(out)) == 1

    with pytest.raises(ValueError, match="internal overlap"):
        batched_gemm_reference(lhs, rhs, out=out)


def test_rejects_out_aliasing_lhs_or_rhs():
    lhs = torch.ones((2, 3, 4))
    rhs = torch.ones((2, 4, 4))
    with pytest.raises(ValueError, match="must not alias"):
        batched_gemm_reference(lhs, rhs, out=lhs)

    lhs = torch.ones((2, 4, 4))
    rhs = torch.ones((2, 4, 5))
    with pytest.raises(ValueError, match="must not alias"):
        batched_gemm_reference(lhs, rhs, out=rhs)
