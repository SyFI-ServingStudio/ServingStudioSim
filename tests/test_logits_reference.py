"""CPU value checks for the logits communication and sampling boundaries."""

import pytest

from profiling.runners.logits.reference import (
    logits_argmax_reference,
    logits_copy_reference,
    vocab_parallel_all_gather_reference,
)

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,world", [(1, 4), (3, 4), (2, 8), (2, 1)])
def test_gather_preserves_rank_order_with_multiple_rows(dtype, rows, world):
    shards = [
        (torch.arange(rows * 5).reshape(rows, 5) + rank * 20).to(dtype)
        for rank in range(world)
    ]
    expected = vocab_parallel_all_gather_reference(torch, shards)
    # The collective writes rank-major rows before the vLLM layout conversion.
    collective_output = torch.cat(shards, dim=0)
    actual = collective_output.reshape(world, rows, 5).movedim(0, 1).reshape(rows, world * 5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert expected.is_contiguous()
    assert expected.dtype == dtype


@pytest.mark.parametrize("rows", [1, 3])
def test_cast_ignores_poisoned_padding_and_returns_contiguous_fp32(rows):
    storage = torch.full((rows, 9), float("nan"), dtype=torch.bfloat16)
    logits = storage[:, :5]
    logits.copy_(torch.tensor([-2.5, 0, 1.25, float("inf"), -float("inf")]))
    result = logits_copy_reference(torch, logits, torch.float32)
    torch.testing.assert_close(result, logits.float(), rtol=0, atol=0)
    assert result.shape == (rows, 5)
    assert result.dtype == torch.float32
    assert result.is_contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 4])
def test_argmax_first_tie_ignores_padding_and_handles_infinity(dtype, rows):
    storage = torch.full((rows, 8), float("nan"), dtype=dtype)
    values = torch.tensor([
        [1, 4, 4, -2, 0],
        [-3, -3, -3, -3, -3],
        [0, float("inf"), 1, float("inf"), 0],
        [-float("inf")] * 5,
    ], dtype=dtype)[:rows]
    logits = storage[:, :5]
    logits.copy_(values)
    result = logits_argmax_reference(torch, logits)
    torch.testing.assert_close(result, torch.tensor([1, 0, 1, 0][:rows]))
    torch.testing.assert_close(result, logits.argmax(dim=-1))
    assert result.dtype == torch.int64


def test_argmax_rejects_nan_in_valid_columns():
    with pytest.raises(ValueError, match="NaN"):
        logits_argmax_reference(torch, torch.tensor([[1.0, float("nan")]]))


def test_gather_and_cast_preserve_nan_values():
    shard = torch.tensor([[float("nan"), float("inf")]], dtype=torch.bfloat16)
    gathered = vocab_parallel_all_gather_reference(torch, [shard, shard])
    result = logits_copy_reference(torch, gathered, torch.float32)
    assert torch.isnan(result[:, [0, 2]]).all()
    assert torch.isposinf(result[:, [1, 3]]).all()


def test_gather_rejects_incompatible_or_strided_shards():
    shard = torch.zeros((2, 3), dtype=torch.bfloat16)
    for shards in ([], [shard, shard.float()], [shard, shard[:1]], [shard[:, :2]]):
        with pytest.raises(ValueError):
            vocab_parallel_all_gather_reference(torch, shards)


@pytest.mark.parametrize("oracle", [
    lambda torch, logits: logits_copy_reference(torch, logits, torch.float32),
    logits_argmax_reference,
])
def test_logits_reject_column_strides_overlapping_rows_and_empty_shapes(oracle):
    storage = torch.zeros((3, 6), dtype=torch.bfloat16)
    for invalid in (storage[:, ::2], storage.as_strided((3, 4), (1, 1)), storage[:0]):
        with pytest.raises(ValueError):
            oracle(torch, invalid)
    with pytest.raises(TypeError):
        oracle(torch, storage.to(torch.float16))


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_copy_never_aliases_input_even_when_dtype_is_unchanged(input_dtype, output_dtype):
    logits = torch.tensor([[1.25, -2.5]], dtype=input_dtype)
    result = logits_copy_reference(torch, logits, output_dtype)
    assert result.dtype == output_dtype
    assert result.is_contiguous()
    torch.testing.assert_close(result, logits.to(output_dtype), rtol=0, atol=0)
    result.zero_()
    torch.testing.assert_close(logits, torch.tensor([[1.25, -2.5]], dtype=input_dtype))


def test_copy_rejects_unsupported_output_dtype():
    with pytest.raises(TypeError, match="output"):
        logits_copy_reference(torch, torch.zeros((1, 3)), torch.float16)
