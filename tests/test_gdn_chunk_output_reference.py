from __future__ import annotations

import math

import pytest
import torch

from profiling.runners.attention.gdn_chunk_output_reference import (
    gdn_chunk_output_reference,
)


def _inputs(
    lengths: list[int], *, hg: int = 2, heads: int = 4, key_dim: int = 2, value_dim: int = 3
) -> tuple[torch.Tensor, ...]:
    tokens = sum(lengths)
    chunks = sum((length + 63) // 64 for length in lengths)
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return (
        torch.zeros((tokens, hg, key_dim), dtype=torch.bfloat16),
        torch.zeros((tokens, hg, key_dim), dtype=torch.bfloat16),
        torch.zeros((tokens, heads, value_dim), dtype=torch.bfloat16),
        torch.zeros((chunks, heads, value_dim, key_dim), dtype=torch.bfloat16),
        torch.zeros((tokens, heads), dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.int32),
    )


def _scalar_oracle(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    q, k, value, snapshots, gate, boundaries = inputs
    tokens, key_heads, key_dim = q.shape
    heads, value_dim = value.shape[1:]
    result = torch.empty((tokens, heads, value_dim), dtype=torch.bfloat16)
    chunk = 0
    bounds = boundaries.tolist()
    for sequence_start, sequence_end in zip(bounds, bounds[1:]):
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            for head in range(heads):
                key_head = head // (heads // key_heads)
                for row in range(chunk_start, chunk_end):
                    state = []
                    for feature in range(value_dim):
                        dot = sum(
                            float(q[row, key_head, d]) * float(snapshots[chunk, head, feature, d])
                            for d in range(key_dim)
                        )
                        state.append(dot * math.exp(float(gate[row, head])))
                    causal = [0.0] * value_dim
                    for source in range(chunk_start, row + 1):
                        score = sum(
                            float(q[row, key_head, d]) * float(k[source, key_head, d])
                            for d in range(key_dim)
                        )
                        score *= math.exp(float(gate[row, head] - gate[source, head]))
                        score = float(torch.tensor(score).to(torch.bfloat16))
                        for feature in range(value_dim):
                            causal[feature] += score * float(value[source, head, feature])
                    result[row, head] = torch.tensor(
                        [(x + y) * key_dim**-0.5 for x, y in zip(state, causal)],
                        dtype=torch.float32,
                    ).to(torch.bfloat16)
            chunk += 1
    return result


def test_state_only_orientation_grouping_and_scale() -> None:
    q, k, value, h, gate, boundaries = _inputs([1], key_dim=4, value_dim=2)
    q[0, 0, 0] = 1
    q[0, 1, 0] = 2
    for head in range(4):
        h[0, head, :, 0] = torch.tensor([2, -3], dtype=torch.bfloat16)
    actual = gdn_chunk_output_reference(q, k, value, h, gate, boundaries)
    expected = torch.tensor([[[1, -1.5], [1, -1.5], [2, -3], [2, -3]]], dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_causal_value_diagonal_order_sign_and_chunk_reset() -> None:
    q, k, value, h, gate, boundaries = _inputs([4], hg=1, heads=1, key_dim=1, value_dim=1)
    q.fill_(1)
    k.fill_(1)
    value[:, 0, 0] = torch.tensor([1, 2, -4, 8], dtype=torch.bfloat16)
    actual = gdn_chunk_output_reference(q, k, value, h, gate, boundaries)
    torch.testing.assert_close(
        actual[:, 0, 0], torch.tensor([1, 3, -1, 7], dtype=torch.bfloat16), rtol=0, atol=0
    )

    q, k, value, h, gate, boundaries = _inputs([65], hg=1, heads=1, key_dim=1, value_dim=1)
    q.fill_(1)
    k.fill_(1)
    value.fill_(1)
    reset = gdn_chunk_output_reference(q, k, value, h, gate, boundaries)
    assert reset[63, 0, 0] == 64
    assert reset[64, 0, 0] == 1


def test_gate_direction_and_ordinary_exp_match_scalar_oracle() -> None:
    inputs = _inputs([3], hg=1, heads=1, key_dim=1, value_dim=2)
    q, k, value, h, gate, _ = inputs
    q.fill_(1)
    k.fill_(1)
    value[:, 0] = torch.tensor([[1, -2], [3, 4], [-1, 2]], dtype=torch.bfloat16)
    h[0, 0, :, 0] = torch.tensor([0.5, -1], dtype=torch.bfloat16)
    gate[:, 0] = torch.tensor([-0.25, 0, 0.5])
    torch.testing.assert_close(
        gdn_chunk_output_reference(*inputs), _scalar_oracle(inputs), rtol=0, atol=0
    )


def test_bf16_score_rounding_precedes_value_dot_and_final_store() -> None:
    inputs = _inputs([2], hg=1, heads=1, key_dim=1, value_dim=1)
    q, k, value, _, gate, _ = inputs
    q[1] = 1
    k[0] = 1
    value[0] = 100
    gate[1] = 0.1
    actual = gdn_chunk_output_reference(*inputs)[1, 0, 0]
    score_fp32 = torch.exp(torch.tensor(0.1))
    all_fp32 = (score_fp32 * 100).to(torch.bfloat16)
    expected = (score_fp32.to(torch.bfloat16).float() * 100).to(torch.bfloat16)
    assert actual == expected
    assert actual != all_fp32
    assert actual.dtype is torch.bfloat16


def test_ragged_partial_global_chunk_order_and_independence() -> None:
    inputs = _inputs([3, 65, 2], hg=1, heads=2, key_dim=1, value_dim=2)
    q, _, _, h, _, _ = inputs
    q.fill_(1)
    for chunk in range(4):
        for head in range(2):
            h[chunk, head, :, 0] = torch.tensor(
                [10 * (chunk + 1) + head, -(chunk + 1)], dtype=torch.bfloat16
            )
    actual = gdn_chunk_output_reference(*inputs)
    expected_chunks = [(0, 3, 0), (3, 67, 1), (67, 68, 2), (68, 70, 3)]
    for start, end, chunk in expected_chunks:
        assert torch.all(actual[start:end, 0, 0] == 10 * (chunk + 1))
        assert torch.all(actual[start:end, 1, 0] == 10 * (chunk + 1) + 1)
        assert torch.all(actual[start:end, :, 1] == -(chunk + 1))


def test_bounded_random_matches_independent_oracle() -> None:
    generator = torch.Generator().manual_seed(123)
    inputs = list(_inputs([3, 2], hg=2, heads=4, key_dim=3, value_dim=2))
    for index in (0, 1, 2, 3):
        inputs[index].copy_(
            (torch.randn(inputs[index].shape, generator=generator) * 0.2).to(torch.bfloat16)
        )
    inputs[4].copy_(torch.randn(inputs[4].shape, generator=generator) * 0.15)
    actual = gdn_chunk_output_reference(*inputs)
    torch.testing.assert_close(actual, _scalar_oracle(tuple(inputs)), rtol=1e-2, atol=1e-2)


def test_output_storage_and_input_immutability() -> None:
    inputs = _inputs([3, 2])
    snapshots = [tensor.clone() for tensor in inputs]
    first = gdn_chunk_output_reference(*inputs)
    second = gdn_chunk_output_reference(*inputs)
    assert first.shape == (5, 4, 3)
    assert first.dtype is torch.bfloat16 and first.is_contiguous()
    assert first.data_ptr() != second.data_ptr()
    assert all(first.data_ptr() != tensor.data_ptr() for tensor in inputs)
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(tensor, snapshot)


def test_valid_noncontiguous_inputs_and_metadata() -> None:
    inputs = list(_inputs([3], hg=1, heads=1, key_dim=2, value_dim=2))
    for index, shape in ((0, (3, 1, 2)), (1, (3, 1, 2)), (2, (3, 1, 2))):
        backing = torch.zeros((*shape[:-1], shape[-1] * 2), dtype=torch.bfloat16)
        inputs[index] = backing[..., ::2]
        assert not inputs[index].is_contiguous()
    h_backing = torch.zeros((1, 1, 2, 4), dtype=torch.bfloat16)
    inputs[3] = h_backing[..., ::2]
    gate_backing = torch.zeros((3, 2), dtype=torch.float32)
    inputs[4] = gate_backing[:, ::2]
    meta_backing = torch.tensor([0, -1, 3, -1], dtype=torch.int32)
    inputs[5] = meta_backing[::2]
    actual = gdn_chunk_output_reference(*inputs)
    assert actual.shape == (3, 1, 2) and actual.is_contiguous()


@pytest.mark.parametrize(
    ("index", "replacement", "error"),
    [
        (0, None, TypeError),
        (0, torch.zeros((1, 1), dtype=torch.bfloat16), ValueError),
        (0, torch.zeros((1, 1, 1), dtype=torch.float32), TypeError),
        (4, torch.zeros((1, 1), dtype=torch.bfloat16), TypeError),
        (5, torch.tensor([0, 1], dtype=torch.int64), TypeError),
    ],
)
def test_rejects_wrong_types_ranks_and_dtypes(
    index: int, replacement: object, error: type[Exception]
) -> None:
    inputs = list(_inputs([1], hg=1, heads=1, key_dim=1, value_dim=1))
    inputs[index] = replacement
    with pytest.raises(error):
        gdn_chunk_output_reference(*inputs)


@pytest.mark.parametrize(
    "boundaries",
    [[1, 2], [0, 1], [0, 2, 2], [0, -1, 2], [0], [0, 2, 1]],
)
def test_rejects_malformed_boundaries(boundaries: list[int]) -> None:
    inputs = list(_inputs([2], hg=1, heads=1, key_dim=1, value_dim=1))
    inputs[5] = torch.tensor(boundaries, dtype=torch.int32)
    with pytest.raises(ValueError):
        gdn_chunk_output_reference(*inputs)


def test_rejects_shapes_heads_chunk_count_and_nonfinite() -> None:
    inputs = list(_inputs([2], hg=1, heads=2, key_dim=1, value_dim=1))
    bad = inputs.copy()
    bad[1] = torch.zeros((1, 1, 1), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        gdn_chunk_output_reference(*bad)
    bad = inputs.copy()
    bad[2] = torch.zeros((2, 3, 1), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        gdn_chunk_output_reference(*bad)
    bad = inputs.copy()
    bad[3] = torch.zeros((2, 2, 1, 1), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        gdn_chunk_output_reference(*bad)
    bad = inputs.copy()
    bad[0][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        gdn_chunk_output_reference(*bad)
    empty = _inputs([1], hg=1, heads=1, key_dim=1, value_dim=1)
    with pytest.raises(ValueError):
        gdn_chunk_output_reference(*(torch.empty((0, 1, 1), dtype=torch.bfloat16), *empty[1:]))


@pytest.mark.parametrize(
    ("q_shape", "value_shape"),
    [
        ((1, 0, 1), (1, 1, 1)),
        ((1, 1, 0), (1, 1, 1)),
        ((1, 2, 1), (1, 3, 1)),
        ((1, 1, 1), (1, 0, 1)),
        ((1, 1, 1), (1, 1, 0)),
    ],
)
def test_rejects_zero_or_incompatible_head_and_feature_dimensions(
    q_shape: tuple[int, int, int], value_shape: tuple[int, int, int]
) -> None:
    q = torch.zeros(q_shape, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    value = torch.zeros(value_shape, dtype=torch.bfloat16)
    heads = value_shape[1]
    h = torch.zeros((1, heads, value_shape[2], q_shape[2]), dtype=torch.bfloat16)
    gate = torch.zeros((1, heads), dtype=torch.float32)
    boundaries = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(ValueError):
        gdn_chunk_output_reference(q, k, value, h, gate, boundaries)


def test_rejects_sparse_meta_and_mismatched_devices() -> None:
    inputs = list(_inputs([1], hg=1, heads=1, key_dim=1, value_dim=1))
    sparse = inputs.copy()
    sparse[0] = sparse[0].to_sparse()
    with pytest.raises(ValueError):
        gdn_chunk_output_reference(*sparse)
    meta = [torch.empty_like(tensor, device="meta") for tensor in inputs]
    with pytest.raises(ValueError):
        gdn_chunk_output_reference(*meta)
    mixed = inputs.copy()
    mixed[0] = torch.empty_like(mixed[0], device="meta")
    with pytest.raises(ValueError):
        gdn_chunk_output_reference(*mixed)
