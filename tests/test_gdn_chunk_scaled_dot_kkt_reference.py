from __future__ import annotations

import math

import pytest
import torch

from profiling.runners.attention.gdn_chunk_scaled_dot_kkt_reference import (
    gdn_chunk_scaled_dot_kkt_reference,
)


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return torch.tensor(boundaries, dtype=torch.int32)


def _inputs(
    *,
    lengths: list[int] | None = None,
    num_key_heads: int = 2,
    num_heads: int = 4,
    key_head_dim: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = lengths or [3, 2]
    num_tokens = sum(lengths)
    k_values = torch.arange(
        num_tokens * num_key_heads * key_head_dim,
        dtype=torch.float32,
    ).reshape(num_tokens, num_key_heads, key_head_dim)
    k = ((k_values % 9) - 4).div(8).to(torch.bfloat16)
    beta = torch.linspace(
        0.125,
        0.875,
        steps=num_tokens * num_heads,
        dtype=torch.float32,
    ).reshape(num_tokens, num_heads)
    g_cumsum = torch.linspace(
        -0.4,
        0.4,
        steps=num_tokens * num_heads,
        dtype=torch.float32,
    ).reshape(num_tokens, num_heads)
    return k, beta, g_cumsum, _cu_seqlens(lengths)


def _manual_reference(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    num_tokens, num_key_heads, key_head_dim = k.shape
    num_heads = beta.shape[1]
    heads_per_key = num_heads // num_key_heads
    output = torch.zeros((num_tokens, num_heads, 64), dtype=torch.float32)
    boundaries = cu_seqlens.tolist()
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            for row in range(chunk_start, chunk_end):
                for column in range(chunk_start, row):
                    local_column = column - chunk_start
                    for head in range(num_heads):
                        key_head = head // heads_per_key
                        dot = torch.tensor(0.0, dtype=torch.float32)
                        for feature in range(key_head_dim):
                            # Match production/reference order: row-owned beta
                            # scales the row key before the FP32 dot reduction.
                            row_key = k[row, key_head, feature].float() * beta[row, head]
                            column_key = k[column, key_head, feature].float()
                            dot = dot + row_key * column_key
                        decay = torch.exp(g_cumsum[row, head] - g_cumsum[column, head])
                        output[row, head, local_column] = dot * decay
    return output


def test_representative_ragged_output_contract_and_immutability() -> None:
    inputs = _inputs(lengths=[3, 65, 2])
    snapshots = tuple(tensor.clone() for tensor in inputs)

    output = gdn_chunk_scaled_dot_kkt_reference(*inputs)
    second = gdn_chunk_scaled_dot_kkt_reference(*inputs)

    assert output.shape == (70, 4, 64)
    assert output.dtype is torch.float32
    assert output.is_contiguous()
    assert output.data_ptr() != second.data_ptr()
    assert all(output.data_ptr() != tensor.data_ptr() for tensor in inputs)
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(tensor, snapshot)


def test_deterministic_manual_fp32_equation_agreement() -> None:
    inputs = _inputs(lengths=[4, 3], key_head_dim=5)

    actual = gdn_chunk_scaled_dot_kkt_reference(*inputs)
    expected = _manual_reference(*inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_grouped_head_mapping_and_row_owned_beta_are_exact() -> None:
    k = torch.zeros((4, 2, 2), dtype=torch.bfloat16)
    k[0:2, 0, 0] = 1
    k[2:4, 1, 1] = 1
    beta = torch.tensor(
        [
            [0.125, 0.25, 0.5, 1.0],
            [2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0],
        ],
        dtype=torch.float32,
    )
    g_cumsum = torch.zeros_like(beta)

    output = gdn_chunk_scaled_dot_kkt_reference(
        k,
        beta,
        g_cumsum,
        _cu_seqlens([4]),
    )

    assert output[1, 0, 0].item() == 2.0
    assert output[1, 1, 0].item() == 3.0
    assert output[3, 2, 2].item() == 12.0
    assert output[3, 3, 2].item() == 13.0
    assert output[1, 2, 0].item() == 0.0
    assert output[3, 0, 2].item() == 0.0


def test_positive_sign_and_gate_difference_direction() -> None:
    k = torch.ones((2, 1, 1), dtype=torch.bfloat16)
    beta = torch.tensor([[0.25], [2.0]], dtype=torch.float32)
    g_cumsum = torch.tensor([[0.0], [-math.log(2.0)]], dtype=torch.float32)

    output = gdn_chunk_scaled_dot_kkt_reference(
        k,
        beta,
        g_cumsum,
        _cu_seqlens([2]),
    )

    # Row 1 owns beta=2 and exp(g_1-g_0)=1/2, producing positive one.
    torch.testing.assert_close(output[1, 0, 0], torch.tensor(1.0), rtol=1e-6, atol=1e-6)


def test_strict_lower_mask_and_partial_chunk_unused_columns_are_exact_zero() -> None:
    k = torch.ones((3, 1, 2), dtype=torch.bfloat16)
    beta = torch.ones((3, 1), dtype=torch.float32)
    g_cumsum = torch.zeros_like(beta)

    output = gdn_chunk_scaled_dot_kkt_reference(
        k,
        beta,
        g_cumsum,
        _cu_seqlens([3]),
    )

    assert torch.equal(output[0, :, :], torch.zeros_like(output[0, :, :]))
    assert output[1, 0, 0].item() == 2.0
    assert output[1, 0, 1].item() == 0.0
    assert output[2, 0, 0].item() == 2.0
    assert output[2, 0, 1].item() == 2.0
    assert torch.count_nonzero(output[:, :, 3:]).item() == 0


def test_sequence_and_chunk_boundaries_reset_strict_lower_columns() -> None:
    k = torch.ones((132, 1, 1), dtype=torch.bfloat16)
    beta = torch.ones((132, 1), dtype=torch.float32)
    g_cumsum = torch.zeros_like(beta)

    output = gdn_chunk_scaled_dot_kkt_reference(
        k,
        beta,
        g_cumsum,
        _cu_seqlens([65, 64, 3]),
    )

    assert output[63, 0, :63].sum().item() == 63.0
    assert output[64, 0].sum().item() == 0.0  # New local chunk.
    assert output[65, 0].sum().item() == 0.0  # New sequence.
    assert output[128, 0, :63].sum().item() == 63.0
    assert output[129, 0].sum().item() == 0.0  # Final sequence.
    assert output[131, 0, :2].sum().item() == 2.0


@pytest.mark.parametrize("lengths", [[64], [65], [130], [64, 64, 2]])
def test_exact_full_multi_and_final_partial_chunks(lengths: list[int]) -> None:
    num_tokens = sum(lengths)
    k = torch.ones((num_tokens, 1, 1), dtype=torch.bfloat16)
    beta = torch.ones((num_tokens, 1), dtype=torch.float32)
    g_cumsum = torch.zeros_like(beta)

    output = gdn_chunk_scaled_dot_kkt_reference(
        k,
        beta,
        g_cumsum,
        _cu_seqlens(lengths),
    )

    for sequence_start, sequence_end in zip(
        _cu_seqlens(lengths).tolist(),
        _cu_seqlens(lengths).tolist()[1:],
    ):
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            assert output[chunk_start, 0].sum().item() == 0.0
            assert output[chunk_end - 1, 0].sum().item() == chunk_end - chunk_start - 1


def test_zero_negative_and_mixed_sign_values_follow_ordinary_fp32_math() -> None:
    k = torch.tensor([[[0.0, 1.0]], [[-2.0, 1.0]], [[1.0, 1.0]]], dtype=torch.bfloat16)
    beta = torch.tensor([[0.0], [-0.5], [2.0]], dtype=torch.float32)
    g_cumsum = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32)
    cu_seqlens = _cu_seqlens([3])

    actual = gdn_chunk_scaled_dot_kkt_reference(k, beta, g_cumsum, cu_seqlens)
    expected = _manual_reference(k, beta, g_cumsum, cu_seqlens)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual[1, 0, 0].item() < 0
    assert actual[2, 0, 1].item() < 0


def test_bounded_random_values_match_independent_fp32_equation() -> None:
    generator = torch.Generator().manual_seed(20260810)
    k = torch.randn((9, 2, 5), generator=generator).mul_(0.25).to(torch.bfloat16)
    beta = torch.rand((9, 4), generator=generator).mul_(0.75)
    g_cumsum = torch.randn((9, 4), generator=generator).mul_(0.2)
    cu_seqlens = _cu_seqlens([2, 5, 2])

    actual = gdn_chunk_scaled_dot_kkt_reference(k, beta, g_cumsum, cu_seqlens)
    expected = _manual_reference(k, beta, g_cumsum, cu_seqlens)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("num_key_heads", "num_heads", "key_head_dim"),
    [(1, 1, 1), (1, 3, 2), (2, 4, 3), (3, 6, 5)],
)
def test_small_distinct_geometry_parameterization(
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
) -> None:
    inputs = _inputs(
        lengths=[2, 3],
        num_key_heads=num_key_heads,
        num_heads=num_heads,
        key_head_dim=key_head_dim,
    )

    output = gdn_chunk_scaled_dot_kkt_reference(*inputs)

    assert output.shape == (5, num_heads, 64)


def test_valid_noncontiguous_semantic_inputs_and_metadata() -> None:
    k_storage = torch.randn((5, 2, 6), dtype=torch.float32).to(torch.bfloat16)
    k = k_storage[..., ::2]
    beta_storage = torch.randn((5, 8), dtype=torch.float32)
    beta = beta_storage[:, ::2]
    gate_storage = torch.randn((5, 8), dtype=torch.float32).mul_(0.1)
    g_cumsum = gate_storage[:, ::2]
    boundary_storage = torch.tensor([0, -99, 2, -99, 5, -99], dtype=torch.int32)
    cu_seqlens = boundary_storage[::2]
    assert not k.is_contiguous()
    assert not beta.is_contiguous()
    assert not g_cumsum.is_contiguous()
    assert not cu_seqlens.is_contiguous()

    actual = gdn_chunk_scaled_dot_kkt_reference(k, beta, g_cumsum, cu_seqlens)
    expected = gdn_chunk_scaled_dot_kkt_reference(
        k.contiguous(),
        beta.contiguous(),
        g_cumsum.contiguous(),
        cu_seqlens.contiguous(),
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()


@pytest.mark.parametrize("argument", ["k", "beta", "g_cumsum", "cu_seqlens"])
def test_rejects_non_tensor_inputs(argument: str) -> None:
    values = dict(zip(("k", "beta", "g_cumsum", "cu_seqlens"), _inputs()))
    values[argument] = []

    with pytest.raises(TypeError, match=rf"{argument} must be a torch.Tensor"):
        gdn_chunk_scaled_dot_kkt_reference(**values)


@pytest.mark.parametrize(
    ("k_shape", "beta_shape", "message"),
    [
        ((0, 1, 2), (0, 1), "token dimension must be positive"),
        ((2, 0, 2), (2, 1), "head dimension must be positive"),
        ((2, 1, 0), (2, 1), "feature dimension must be positive"),
        ((2, 1, 2), (2, 0), "beta head dimension must be positive"),
    ],
)
def test_rejects_nonpositive_geometry(
    k_shape: tuple[int, int, int],
    beta_shape: tuple[int, int],
    message: str,
) -> None:
    k = torch.empty(k_shape, dtype=torch.bfloat16)
    beta = torch.empty(beta_shape, dtype=torch.float32)
    g_cumsum = torch.empty(beta_shape, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, k_shape[0]], dtype=torch.int32)

    with pytest.raises(ValueError, match=message):
        gdn_chunk_scaled_dot_kkt_reference(k, beta, g_cumsum, cu_seqlens)


@pytest.mark.parametrize(
    ("argument", "replacement", "message"),
    [
        ("k", torch.empty((5, 6), dtype=torch.bfloat16), "k must be rank 3"),
        ("beta", torch.empty((5, 4, 1), dtype=torch.float32), "beta must be rank 2"),
        (
            "g_cumsum",
            torch.empty((5, 4, 1), dtype=torch.float32),
            "g_cumsum must be rank 2",
        ),
        ("cu_seqlens", torch.empty((1, 2), dtype=torch.int32), "cu_seqlens must be rank 1"),
    ],
)
def test_rejects_invalid_ranks(argument: str, replacement: torch.Tensor, message: str) -> None:
    values = dict(zip(("k", "beta", "g_cumsum", "cu_seqlens"), _inputs()))
    values[argument] = replacement

    with pytest.raises(ValueError, match=message):
        gdn_chunk_scaled_dot_kkt_reference(**values)


@pytest.mark.parametrize(
    ("argument", "dtype", "message"),
    [
        ("k", torch.float32, "k dtype must be torch.bfloat16"),
        ("beta", torch.bfloat16, "beta dtype must be torch.float32"),
        ("g_cumsum", torch.float64, "g_cumsum dtype must be torch.float32"),
        ("cu_seqlens", torch.int64, "cu_seqlens dtype must be torch.int32"),
    ],
)
def test_rejects_invalid_dtypes(argument: str, dtype: torch.dtype, message: str) -> None:
    values = dict(zip(("k", "beta", "g_cumsum", "cu_seqlens"), _inputs()))
    values[argument] = values[argument].to(dtype)

    with pytest.raises(TypeError, match=message):
        gdn_chunk_scaled_dot_kkt_reference(**values)


def test_rejects_token_and_gate_shape_mismatches() -> None:
    k, beta, g_cumsum, cu_seqlens = _inputs()
    with pytest.raises(ValueError, match="beta token dimension"):
        gdn_chunk_scaled_dot_kkt_reference(k, beta[:-1], g_cumsum[:-1], cu_seqlens)
    with pytest.raises(ValueError, match="g_cumsum must have shape"):
        gdn_chunk_scaled_dot_kkt_reference(k, beta, g_cumsum[:, :-1], cu_seqlens)


def test_rejects_incompatible_grouped_head_divisibility() -> None:
    k = torch.ones((3, 2, 2), dtype=torch.bfloat16)
    beta = torch.ones((3, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="divisible by key head count"):
        gdn_chunk_scaled_dot_kkt_reference(k, beta, beta.clone(), _cu_seqlens([3]))


@pytest.mark.parametrize(
    ("boundaries", "message"),
    [
        ([], "at least one sequence"),
        ([0], "at least one sequence"),
        ([1, 5], "start at zero"),
        ([0, 4], "end at token count 5"),
        ([-1, 5], r"boundaries must be in \[0, 5\]"),
        ([0, 6], r"boundaries must be in \[0, 5\]"),
        ([0, 2, 2, 5], "strictly increasing"),
        ([0, 4, 3, 5], "strictly increasing"),
    ],
)
def test_rejects_invalid_ragged_boundaries(boundaries: list[int], message: str) -> None:
    k, beta, g_cumsum, _ = _inputs()
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)

    with pytest.raises(ValueError, match=message):
        gdn_chunk_scaled_dot_kkt_reference(k, beta, g_cumsum, cu_seqlens)


@pytest.mark.parametrize("argument", ["k", "beta", "g_cumsum", "cu_seqlens"])
def test_rejects_sparse_non_strided_tensors(argument: str) -> None:
    values = dict(zip(("k", "beta", "g_cumsum", "cu_seqlens"), _inputs()))
    values[argument] = values[argument].to_sparse()

    with pytest.raises(ValueError, match=rf"{argument} must have torch.strided layout"):
        gdn_chunk_scaled_dot_kkt_reference(**values)


def test_rejects_meta_tensors() -> None:
    k = torch.empty((2, 1, 2), dtype=torch.bfloat16, device="meta")
    beta = torch.empty((2, 1), dtype=torch.float32, device="meta")
    g_cumsum = torch.empty((2, 1), dtype=torch.float32, device="meta")
    cu_seqlens = torch.empty((2,), dtype=torch.int32, device="meta")

    with pytest.raises(ValueError, match="meta tensors are not supported"):
        gdn_chunk_scaled_dot_kkt_reference(k, beta, g_cumsum, cu_seqlens)


def test_rejects_mismatched_devices_before_reading_metadata() -> None:
    k, beta, g_cumsum, _ = _inputs()
    cu_seqlens = torch.empty((3,), dtype=torch.int32, device="meta")

    with pytest.raises(ValueError, match="all inputs must be on the same device"):
        gdn_chunk_scaled_dot_kkt_reference(k, beta, g_cumsum, cu_seqlens)
