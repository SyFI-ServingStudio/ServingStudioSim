from __future__ import annotations

import math

import pytest
import torch

from profiling.runners.attention.gdn_chunk_recompute_w_u_reference import (
    gdn_chunk_recompute_w_u_reference,
)


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return torch.tensor(boundaries, dtype=torch.int32)


def _inputs(
    lengths: list[int] | None = None,
    *,
    num_key_heads: int = 2,
    num_heads: int = 4,
    key_head_dim: int = 3,
    value_head_dim: int = 5,
) -> tuple[torch.Tensor, ...]:
    lengths = lengths or [3, 2]
    num_tokens = sum(lengths)
    generator = torch.Generator().manual_seed(20260811 + num_tokens)
    k = (
        torch.randn((num_tokens, num_key_heads, key_head_dim), generator=generator)
        .mul_(0.25)
        .to(torch.bfloat16)
    )
    v = (
        torch.randn((num_tokens, num_heads, value_head_dim), generator=generator)
        .mul_(0.25)
        .to(torch.bfloat16)
    )
    beta = torch.randn((num_tokens, num_heads), generator=generator).mul_(0.5)
    g = torch.randn((num_tokens, num_heads), generator=generator).mul_(0.2)
    A = torch.zeros((num_tokens, num_heads, 64), dtype=torch.bfloat16)
    boundaries = _cu_seqlens(lengths)
    for sequence_start, sequence_end in zip(boundaries.tolist(), boundaries.tolist()[1:]):
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            length = chunk_end - chunk_start
            values = torch.randn((length, num_heads, length), generator=generator).mul_(0.125)
            lower = torch.tril(values.permute(1, 0, 2), diagonal=-1).permute(1, 0, 2)
            A[chunk_start:chunk_end, :, :length].copy_(lower.to(torch.bfloat16))
            rows = torch.arange(chunk_start, chunk_end)
            local_rows = torch.arange(length)
            A[rows, :, local_rows] = 1
    return k, v, beta, g, A, boundaries


def _manual_fp32(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    k, v, beta, g, A, cu_seqlens = inputs
    num_tokens, num_key_heads, key_head_dim = k.shape
    num_heads, value_head_dim = v.shape[1:]
    heads_per_key = num_heads // num_key_heads
    w = torch.empty((num_tokens, num_heads, key_head_dim), dtype=torch.float32)
    u = torch.empty((num_tokens, num_heads, value_head_dim), dtype=torch.float32)
    boundaries = cu_seqlens.tolist()
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            length = chunk_end - chunk_start
            for head in range(num_heads):
                key_head = head // heads_per_key
                for row in range(length):
                    global_row = chunk_start + row
                    for feature in range(value_head_dim):
                        value = torch.tensor(0.0)
                        for column in range(length):
                            global_column = chunk_start + column
                            factor = (
                                (
                                    v[global_column, head, feature].float()
                                    * beta[global_column, head]
                                )
                                .to(torch.bfloat16)
                                .float()
                            )
                            value = value + A[global_row, head, column].float() * factor
                        u[global_row, head, feature] = value
                    for feature in range(key_head_dim):
                        value = torch.tensor(0.0)
                        for column in range(length):
                            global_column = chunk_start + column
                            factor = (
                                (
                                    k[global_column, key_head, feature].float()
                                    * beta[global_column, head]
                                    * torch.exp(g[global_column, head])
                                )
                                .to(torch.bfloat16)
                                .float()
                            )
                            value = value + A[global_row, head, column].float() * factor
                        w[global_row, head, feature] = value
    return w, u


def _matrix_oracle_fp32(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    k, v, beta, g, A, cu_seqlens = inputs
    num_tokens, num_key_heads, key_head_dim = k.shape
    num_heads, value_head_dim = v.shape[1:]
    heads_per_key = num_heads // num_key_heads
    w = torch.empty((num_tokens, num_heads, key_head_dim), dtype=torch.float32)
    u = torch.empty((num_tokens, num_heads, value_head_dim), dtype=torch.float32)
    boundaries = cu_seqlens.tolist()
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            length = chunk_end - chunk_start
            solved = A[chunk_start:chunk_end, :, :length].permute(1, 0, 2).float()
            row_beta = beta[chunk_start:chunk_end].T[..., None]
            v_factor = (v[chunk_start:chunk_end].permute(1, 0, 2).float() * row_beta).to(
                torch.bfloat16
            )
            grouped_k = (
                k[chunk_start:chunk_end]
                .repeat_interleave(heads_per_key, dim=1)
                .permute(1, 0, 2)
                .float()
            )
            k_factor = (grouped_k * row_beta * torch.exp(g[chunk_start:chunk_end].T)[..., None]).to(
                torch.bfloat16
            )
            w[chunk_start:chunk_end] = torch.matmul(solved, k_factor.float()).permute(1, 0, 2)
            u[chunk_start:chunk_end] = torch.matmul(solved, v_factor.float()).permute(1, 0, 2)
    return w, u


def test_identity_A_returns_direct_rounded_factors() -> None:
    inputs = list(_inputs([4], key_head_dim=2, value_head_dim=3))
    inputs[4].zero_()
    inputs[4][torch.arange(4), :, torch.arange(4)] = 1
    k, v, beta, g, _, _ = inputs

    w, u = gdn_chunk_recompute_w_u_reference(*inputs)
    expanded_k = k.repeat_interleave(2, dim=1)
    expected_w = (expanded_k.float() * beta.unsqueeze(-1) * torch.exp(g).unsqueeze(-1)).to(
        torch.bfloat16
    )
    expected_u = (v.float() * beta.unsqueeze(-1)).to(torch.bfloat16)

    assert torch.equal(w, expected_w)
    assert torch.equal(u, expected_u)


def test_sparse_exact_manual_equations_and_signs() -> None:
    k = torch.tensor([[[1.0]], [[2.0]], [[4.0]]], dtype=torch.bfloat16)
    v = torch.tensor([[[2.0]], [[4.0]], [[8.0]]], dtype=torch.bfloat16)
    beta = torch.tensor([[1.0], [-1.0], [0.5]])
    g = torch.zeros((3, 1), dtype=torch.float32)
    A = torch.zeros((3, 1, 64), dtype=torch.bfloat16)
    A[0, 0, 0] = A[1, 0, 1] = A[2, 0, 2] = 1
    A[1, 0, 0] = 2
    A[2, 0, 0] = -1
    A[2, 0, 1] = 1

    w, u = gdn_chunk_recompute_w_u_reference(k, v, beta, g, A, _cu_seqlens([3]))

    assert torch.equal(w[:, 0, 0], torch.tensor([1.0, 0.0, -1.0], dtype=torch.bfloat16))
    assert torch.equal(u[:, 0, 0], torch.tensor([2.0, 0.0, -2.0], dtype=torch.bfloat16))


def test_source_row_gate_direction_and_row_owned_beta() -> None:
    k = torch.ones((3, 1, 1), dtype=torch.bfloat16)
    v = torch.ones((3, 1, 1), dtype=torch.bfloat16)
    beta = torch.tensor([[0.0], [2.0], [-1.0]])
    g = torch.tensor([[math.log(2.0)], [-math.log(2.0)], [0.0]])
    A = torch.zeros((3, 1, 64), dtype=torch.bfloat16)
    A[torch.arange(3), :, torch.arange(3)] = 1

    w, u = gdn_chunk_recompute_w_u_reference(k, v, beta, g, A, _cu_seqlens([3]))

    assert torch.equal(w[:, 0, 0], torch.tensor([0.0, 1.0, -1.0], dtype=torch.bfloat16))
    assert torch.equal(u[:, 0, 0], torch.tensor([0.0, 2.0, -1.0], dtype=torch.bfloat16))


def test_grouped_heads_and_feature_independence() -> None:
    inputs = list(_inputs([3], num_key_heads=2, num_heads=4, key_head_dim=2))
    inputs[4].zero_()
    inputs[4][torch.arange(3), :, torch.arange(3)] = 1
    inputs[2].fill_(1)
    inputs[3].zero_()
    inputs[0][:, 0] = torch.tensor([1.0, 2.0])
    inputs[0][:, 1] = torch.tensor([4.0, 8.0])

    w, _ = gdn_chunk_recompute_w_u_reference(*inputs)

    assert torch.equal(w[:, 0], w[:, 1])
    assert torch.equal(w[:, 2], w[:, 3])
    assert not torch.equal(w[:, 0], w[:, 2])
    assert not torch.equal(w[..., 0], w[..., 1])


def test_ragged_chunk_resets_and_output_contract() -> None:
    inputs = _inputs([3, 65, 2])
    snapshots = tuple(tensor.clone() for tensor in inputs)

    w, u = gdn_chunk_recompute_w_u_reference(*inputs)
    w2, u2 = gdn_chunk_recompute_w_u_reference(*inputs)
    expected_w, expected_u = _manual_fp32(inputs)

    assert w.shape == (70, 4, 3)
    assert u.shape == (70, 4, 5)
    assert w.dtype is u.dtype is torch.bfloat16
    assert w.is_contiguous() and u.is_contiguous()
    pointers = {w.data_ptr(), u.data_ptr(), w2.data_ptr(), u2.data_ptr()}
    assert len(pointers) == 4
    assert all(w.data_ptr() != tensor.data_ptr() for tensor in inputs)
    assert all(u.data_ptr() != tensor.data_ptr() for tensor in inputs)
    assert torch.equal(w, expected_w.to(torch.bfloat16))
    assert torch.equal(u, expected_u.to(torch.bfloat16))
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(tensor, snapshot)


@pytest.mark.parametrize("length", [1, 2, 3, 15, 16, 17, 32, 33, 49, 63, 64, 65])
def test_lengths_around_boundaries(length: int) -> None:
    inputs = _inputs([length], num_key_heads=1, num_heads=2, key_head_dim=2, value_head_dim=2)
    actual_w, actual_u = gdn_chunk_recompute_w_u_reference(*inputs)
    expected_w, expected_u = _manual_fp32(inputs)

    assert torch.equal(actual_w, expected_w.to(torch.bfloat16))
    assert torch.equal(actual_u, expected_u.to(torch.bfloat16))


@pytest.mark.parametrize("lengths", [[64], [65], [128], [64, 64, 2], [3, 65, 2]])
def test_exact_multi_partial_and_sequence_resets(lengths: list[int]) -> None:
    inputs = _inputs(lengths, num_key_heads=1, num_heads=2, key_head_dim=2, value_head_dim=2)
    actual = gdn_chunk_recompute_w_u_reference(*inputs)
    expected = tuple(value.to(torch.bfloat16) for value in _manual_fp32(inputs))
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_factor_rounding_precedes_fp32_accumulation() -> None:
    k = torch.ones((3, 1, 1), dtype=torch.bfloat16)
    v = k.clone()
    beta = torch.full((3, 1), 0.502, dtype=torch.float32)
    g = torch.zeros((3, 1), dtype=torch.float32)
    A = torch.zeros((3, 1, 64), dtype=torch.bfloat16)
    A[0, 0, 0] = A[1, 0, 1] = A[2, 0, 2] = 1
    A[2, 0, :2] = 1
    inputs = (k, v, beta, g, A, _cu_seqlens([3]))

    w, u = gdn_chunk_recompute_w_u_reference(*inputs)
    expected_w, expected_u = _manual_fp32(inputs)
    all_fp32_w = torch.tensor([(beta[:, 0] * torch.exp(g[:, 0])).sum()], dtype=torch.float32).to(
        torch.bfloat16
    )
    all_fp32_u = torch.tensor([beta[:, 0].sum()]).to(torch.bfloat16)

    assert w[2, 0, 0] == expected_w[2, 0, 0].to(torch.bfloat16)
    assert u[2, 0, 0] == expected_u[2, 0, 0].to(torch.bfloat16)
    assert w[2, 0, 0] != all_fp32_w[0]
    assert u[2, 0, 0] != all_fp32_u[0]


def test_bounded_random_manual_fp32_and_public_bf16_tolerances() -> None:
    inputs = _inputs([5, 7], key_head_dim=4, value_head_dim=6)
    actual_w, actual_u = gdn_chunk_recompute_w_u_reference(*inputs)
    expected_w, expected_u = _manual_fp32(inputs)

    # The independent oracle reconstructs the same FP32 products after the
    # required BF16 factor casts.
    torch.testing.assert_close(
        actual_w.float(), expected_w.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        actual_u.float(), expected_u.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2
    )
    matrix_w, matrix_u = _matrix_oracle_fp32(inputs)
    torch.testing.assert_close(expected_w, matrix_w, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(expected_u, matrix_u, rtol=1e-5, atol=1e-5)


def test_valid_noncontiguous_inputs_and_metadata() -> None:
    inputs = list(_inputs([3, 2]))
    inputs[0] = inputs[0].transpose(1, 2).contiguous().transpose(1, 2)
    inputs[1] = inputs[1].transpose(1, 2).contiguous().transpose(1, 2)
    inputs[2] = inputs[2].transpose(0, 1).contiguous().transpose(0, 1)
    inputs[3] = inputs[3].transpose(0, 1).contiguous().transpose(0, 1)
    inputs[4] = inputs[4].transpose(1, 2).contiguous().transpose(1, 2)
    metadata_storage = torch.empty(inputs[5].numel() * 2, dtype=torch.int32)
    metadata_storage[::2] = inputs[5]
    inputs[5] = metadata_storage[::2]
    assert all(not tensor.is_contiguous() for tensor in inputs)

    actual = gdn_chunk_recompute_w_u_reference(*inputs)
    expected = tuple(value.to(torch.bfloat16) for value in _manual_fp32(tuple(inputs)))
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def _mutate(
    inputs: tuple[torch.Tensor, ...], index: int, value: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    changed = list(inputs)
    changed[index] = value
    return tuple(changed)


@pytest.mark.parametrize(
    "inputs_factory,match",
    [
        (lambda x: _mutate(x, 0, x[0][:0]), "positive"),
        (lambda x: _mutate(x, 0, x[0][:, :0]), "positive"),
        (lambda x: _mutate(x, 0, x[0][:, :, :0]), "positive"),
        (lambda x: _mutate(x, 1, x[1][:, :0]), "positive"),
        (lambda x: _mutate(x, 1, x[1][:, :, :0]), "positive"),
        (lambda x: _mutate(x, 0, x[0].unsqueeze(0)), "rank 3"),
        (lambda x: _mutate(x, 1, x[1].unsqueeze(0)), "rank 3"),
        (lambda x: _mutate(x, 2, x[2].unsqueeze(0)), "rank 2"),
        (lambda x: _mutate(x, 3, x[3].unsqueeze(0)), "rank 2"),
        (lambda x: _mutate(x, 4, x[4].unsqueeze(0)), "rank 3"),
        (lambda x: _mutate(x, 5, x[5].unsqueeze(0)), "rank 1"),
        (lambda x: _mutate(x, 0, x[0].float()), "bfloat16"),
        (lambda x: _mutate(x, 1, x[1].float()), "bfloat16"),
        (lambda x: _mutate(x, 4, x[4].float()), "bfloat16"),
        (lambda x: _mutate(x, 2, x[2].to(torch.bfloat16)), "float32"),
        (lambda x: _mutate(x, 3, x[3].to(torch.bfloat16)), "float32"),
        (lambda x: _mutate(x, 5, x[5].to(torch.int64)), "int32"),
        (lambda x: _mutate(x, 5, torch.tensor([0], dtype=torch.int32)), "at least"),
        (lambda x: _mutate(x, 5, torch.tensor([1, 5], dtype=torch.int32)), "start"),
        (lambda x: _mutate(x, 5, torch.tensor([0, 4], dtype=torch.int32)), "end"),
        (lambda x: _mutate(x, 5, torch.tensor([-1, 5], dtype=torch.int32)), "in \\[0"),
        (lambda x: _mutate(x, 5, torch.tensor([0, 6], dtype=torch.int32)), "in \\[0"),
        (lambda x: _mutate(x, 5, torch.tensor([0, 2, 2, 5], dtype=torch.int32)), "increasing"),
        (lambda x: _mutate(x, 5, torch.tensor([0, 3, 2, 5], dtype=torch.int32)), "increasing"),
    ],
)
def test_invalid_geometry_dtype_and_metadata(inputs_factory, match: str) -> None:
    inputs = _inputs([3, 2])
    with pytest.raises((TypeError, ValueError), match=match):
        gdn_chunk_recompute_w_u_reference(*inputs_factory(inputs))


def test_incompatible_head_grouping_and_shape_mismatches() -> None:
    inputs = _inputs([5])
    cases = [
        _mutate(inputs, 1, inputs[1][:, :3]),
        _mutate(inputs, 1, inputs[1][:-1]),
        _mutate(inputs, 2, inputs[2][:-1]),
        _mutate(inputs, 3, inputs[3][:, :-1]),
        _mutate(inputs, 4, inputs[4][:, :, :-1]),
    ]
    for case in cases:
        with pytest.raises(ValueError):
            gdn_chunk_recompute_w_u_reference(*case)


@pytest.mark.parametrize("index", range(5))
def test_nonfinite_inputs_are_rejected(index: int) -> None:
    inputs = list(_inputs([3]))
    inputs[index] = inputs[index].clone()
    inputs[index].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        gdn_chunk_recompute_w_u_reference(*inputs)


@pytest.mark.parametrize("kind", ["diagonal", "upper", "unused"])
def test_dirty_solved_matrix_structure_is_rejected(kind: str) -> None:
    inputs = list(_inputs([3]))
    inputs[4] = inputs[4].clone()
    if kind == "diagonal":
        inputs[4][1, 0, 1] = 0
    elif kind == "upper":
        inputs[4][0, 0, 1] = 1
    else:
        inputs[4][2, 0, 63] = 1
    with pytest.raises(ValueError, match="diagonal|lower triangular"):
        gdn_chunk_recompute_w_u_reference(*inputs)


def test_meta_sparse_and_mismatched_devices_are_rejected() -> None:
    inputs = _inputs([3])
    meta = torch.empty_like(inputs[0], device="meta")
    with pytest.raises(ValueError, match="same device"):
        gdn_chunk_recompute_w_u_reference(*_mutate(inputs, 0, meta))

    sparse = inputs[0].to_sparse()
    with pytest.raises(ValueError, match="strided"):
        gdn_chunk_recompute_w_u_reference(*_mutate(inputs, 0, sparse))

    all_meta = tuple(torch.empty_like(tensor, device="meta") for tensor in inputs)
    with pytest.raises(ValueError, match="meta"):
        gdn_chunk_recompute_w_u_reference(*all_meta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_non_cpu_inputs_are_rejected() -> None:
    inputs = tuple(tensor.cuda() for tensor in _inputs([3]))
    with pytest.raises(ValueError, match="requires CPU"):
        gdn_chunk_recompute_w_u_reference(*inputs)
