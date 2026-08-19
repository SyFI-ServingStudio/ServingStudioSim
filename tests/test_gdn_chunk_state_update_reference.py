from __future__ import annotations

import math

import pytest
import torch

from profiling.runners.attention.gdn_chunk_state_update_reference import (
    gdn_chunk_state_update_reference,
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
    value_head_dim: int = 2,
    seed: int = 20260811,
) -> tuple[torch.Tensor, ...]:
    lengths = lengths or [3, 2]
    num_tokens = sum(lengths)
    generator = torch.Generator().manual_seed(seed + num_tokens)
    k = (
        torch.randn((num_tokens, num_key_heads, key_head_dim), generator=generator)
        .mul_(0.08)
        .to(torch.bfloat16)
    )
    w = (
        torch.randn((num_tokens, num_heads, key_head_dim), generator=generator)
        .mul_(0.06)
        .to(torch.bfloat16)
    )
    u = (
        torch.randn((num_tokens, num_heads, value_head_dim), generator=generator)
        .mul_(0.08)
        .to(torch.bfloat16)
    )
    g = torch.empty((num_tokens, num_heads), dtype=torch.float32)
    boundaries = _cu_seqlens(lengths)
    for sequence_start, sequence_end in zip(boundaries.tolist(), boundaries.tolist()[1:]):
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            increments = torch.randn(
                (chunk_end - chunk_start, num_heads), generator=generator
            ).mul_(0.03)
            g[chunk_start:chunk_end].copy_(increments.cumsum(0))
    initial_state = torch.randn(
        (len(lengths), num_heads, value_head_dim, key_head_dim),
        generator=generator,
    ).mul_(0.04)
    return k, w, u, g, initial_state, boundaries


def _manual_reference(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Independent scalar-loop recurrence used only as a test oracle."""
    k, w, u, g, initial_state, cu_seqlens = inputs
    tokens, key_heads, key_dim = k.shape
    heads, value_dim = u.shape[1:]
    heads_per_key = heads // key_heads
    boundaries = cu_seqlens.tolist()
    chunks = sum(math.ceil((right - left) / 64) for left, right in zip(boundaries, boundaries[1:]))
    h_out = torch.empty((chunks, heads, value_dim, key_dim), dtype=torch.bfloat16)
    v_out = torch.empty((tokens, heads, value_dim), dtype=torch.bfloat16)
    final = torch.empty_like(initial_state)
    global_chunk = 0
    for sequence, (sequence_start, sequence_end) in enumerate(zip(boundaries, boundaries[1:])):
        state = initial_state[sequence].clone()
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            for head in range(heads):
                key_head = head // heads_per_key
                snapshot = state[head].to(torch.bfloat16)
                h_out[global_chunk, head].copy_(snapshot)
                residual = torch.empty((chunk_end - chunk_start, value_dim))
                for local_row, token in enumerate(range(chunk_start, chunk_end)):
                    for value_feature in range(value_dim):
                        correction = torch.tensor(0.0)
                        for key_feature in range(key_dim):
                            correction = correction + (
                                w[token, head, key_feature].float()
                                * snapshot[value_feature, key_feature].float()
                            )
                        residual[local_row, value_feature] = (
                            u[token, head, value_feature].float() - correction
                        )
                v_out[chunk_start:chunk_end, head].copy_(residual.to(torch.bfloat16))
                end_gate = g[chunk_end - 1, head]
                factor = torch.empty_like(residual, dtype=torch.bfloat16)
                for local_row, token in enumerate(range(chunk_start, chunk_end)):
                    factor[local_row].copy_(
                        (residual[local_row] * torch.exp(end_gate - g[token, head])).to(
                            torch.bfloat16
                        )
                    )
                next_state = state[head] * torch.exp(end_gate)
                for value_feature in range(value_dim):
                    for key_feature in range(key_dim):
                        update = torch.tensor(0.0)
                        for local_row, token in enumerate(range(chunk_start, chunk_end)):
                            update = update + (
                                factor[local_row, value_feature].float()
                                * k[token, key_head, key_feature].float()
                            )
                        next_state[value_feature, key_feature] += update
                state[head].copy_(next_state)
            global_chunk += 1
        final[sequence].copy_(state)
    return h_out, v_out, final


def _replace(inputs: tuple[torch.Tensor, ...], index: int, value: object):
    changed = list(inputs)
    changed[index] = value
    return tuple(changed)


def _storage_pointer(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def test_zero_state_and_zero_operands() -> None:
    inputs = list(_inputs([3], key_head_dim=2, value_head_dim=2))
    for index in (0, 1, 2, 3, 4):
        inputs[index].zero_()
    h, v_new, final = gdn_chunk_state_update_reference(*inputs)
    assert torch.count_nonzero(h) == 0
    assert torch.count_nonzero(v_new) == 0
    assert torch.count_nonzero(final) == 0


def test_exact_grouped_orientation_subtraction_and_update() -> None:
    k = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bfloat16)
    w = torch.tensor([[[1, 0], [1, 0], [0, 1], [0, 1]]], dtype=torch.bfloat16)
    u = torch.tensor([[[5, 6], [7, 8], [9, 10], [11, 12]]], dtype=torch.bfloat16)
    g = torch.zeros((1, 4), dtype=torch.float32)
    initial = torch.tensor(
        [
            [
                [[1, 2], [3, 4]],
                [[2, 3], [4, 5]],
                [[5, 6], [7, 8]],
                [[6, 7], [8, 9]],
            ]
        ],
        dtype=torch.float32,
    )
    inputs = (k, w, u, g, initial, torch.tensor([0, 1], dtype=torch.int32))
    h, v_new, final = gdn_chunk_state_update_reference(*inputs)

    expected_v = torch.tensor([[[4, 3], [5, 4], [3, 2], [4, 3]]], dtype=torch.bfloat16)
    assert torch.equal(h[0], initial[0].to(torch.bfloat16))
    assert torch.equal(v_new, expected_v)
    expected_final = initial.clone()
    expected_final[0, 0, :, 0] += torch.tensor([4, 3])
    expected_final[0, 1, :, 0] += torch.tensor([5, 4])
    expected_final[0, 2, :, 1] += torch.tensor([3, 2])
    expected_final[0, 3, :, 1] += torch.tensor([4, 3])
    assert torch.equal(final, expected_final)


def test_signed_multipath_gate_direction_matches_manual_oracle() -> None:
    inputs = list(_inputs([3], key_head_dim=2, value_head_dim=2))
    inputs[0] = torch.tensor(
        [
            [[1, -1], [2, 1]],
            [[-2, 1], [1, -2]],
            [[1, 2], [-1, 1]],
        ],
        dtype=torch.bfloat16,
    )
    inputs[1] = torch.tensor(
        [
            [[1, 0], [-1, 1], [0, 1], [1, -1]],
            [[0, 1], [1, 1], [-1, 0], [1, 0]],
            [[1, -1], [0, -1], [1, 1], [-1, 1]],
        ],
        dtype=torch.bfloat16,
    )
    inputs[2] = torch.tensor(
        [
            [[2, -1], [1, 2], [-2, 1], [3, -1]],
            [[-1, 2], [2, -2], [1, 3], [-2, 2]],
            [[3, 1], [-1, 1], [2, -3], [1, 2]],
        ],
        dtype=torch.bfloat16,
    )
    inputs[3] = torch.tensor(
        [[0.0, -0.2, 0.1, 0.0], [0.3, -0.2, -0.1, 0.2], [-0.1, 0.2, 0.0, -0.2]],
        dtype=torch.float32,
    )
    inputs[4] = inputs[4].clone().mul_(0).add_(0.5)
    actual = gdn_chunk_state_update_reference(*inputs)
    expected = _manual_reference(tuple(inputs))
    for result, oracle in zip(actual, expected):
        assert torch.equal(result, oracle)


def test_chunk_end_gate_decay_uses_exp_g_end_minus_g_row() -> None:
    k = torch.ones((2, 1, 1), dtype=torch.bfloat16)
    w = torch.zeros((2, 1, 1), dtype=torch.bfloat16)
    u = torch.ones((2, 1, 1), dtype=torch.bfloat16)
    g = torch.tensor([[0.0], [math.log(2.0)]], dtype=torch.float32)
    initial = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    _, v_new, final = gdn_chunk_state_update_reference(
        k, w, u, g, initial, torch.tensor([0, 2], dtype=torch.int32)
    )
    assert torch.equal(v_new, torch.ones_like(v_new))
    # Row factors are exp(log(2)-0)=2 and exp(log(2)-log(2))=1.
    assert final[0, 0, 0, 0] == 3


def test_head_value_and_key_feature_independence() -> None:
    inputs = list(_inputs([3], key_head_dim=3, value_head_dim=2))
    inputs[0].zero_()
    inputs[0][:, 1, 2] = 1
    baseline = gdn_chunk_state_update_reference(*inputs)
    changed = list(inputs)
    changed[2] = changed[2].clone()
    changed[2][1, 2, 1] += torch.tensor(1.0, dtype=torch.bfloat16)
    perturbed = gdn_chunk_state_update_reference(*changed)

    # Only output head 2 changes; its K update uses grouped key head 1. Other
    # heads and the untouched value feature remain independent.
    assert torch.equal(baseline[0], perturbed[0])
    assert torch.equal(baseline[1][:, :2], perturbed[1][:, :2])
    assert torch.equal(baseline[1][:, 3], perturbed[1][:, 3])
    assert torch.equal(baseline[1][:, 2, 0], perturbed[1][:, 2, 0])
    assert not torch.equal(baseline[1][:, 2, 1], perturbed[1][:, 2, 1])
    assert torch.equal(baseline[2][:, :2], perturbed[2][:, :2])
    assert torch.equal(baseline[2][:, 3], perturbed[2][:, 3])
    assert torch.equal(baseline[2][:, 2, 0], perturbed[2][:, 2, 0])
    assert torch.equal(baseline[2][:, 2, 1, :2], perturbed[2][:, 2, 1, :2])
    assert not torch.equal(baseline[2][:, 2, 1, 2], perturbed[2][:, 2, 1, 2])


def test_snapshot_is_rounded_before_w_dot() -> None:
    k = torch.zeros((1, 1, 1), dtype=torch.bfloat16)
    w = torch.tensor([[[0.5]]], dtype=torch.bfloat16)
    u = torch.tensor([[[0.25]]], dtype=torch.bfloat16)
    g = torch.zeros((1, 1), dtype=torch.float32)
    initial = torch.tensor([[[[0.99]]]], dtype=torch.float32)
    _, v_new, _ = gdn_chunk_state_update_reference(
        k, w, u, g, initial, torch.tensor([0, 1], dtype=torch.int32)
    )
    expected = torch.tensor(-0.244140625, dtype=torch.bfloat16)
    wrong_fp32_state = (u.float() - w.float() * initial[0, 0, 0, 0]).to(torch.bfloat16)
    assert v_new[0, 0, 0] == expected
    assert v_new[0, 0, 0] != wrong_fp32_state[0, 0, 0]


def test_decayed_factor_is_rounded_separately_from_v_new() -> None:
    k = torch.tensor([[[1]], [[0]]], dtype=torch.bfloat16)
    w = torch.tensor([[[-3.9375]], [[0]]], dtype=torch.bfloat16)
    u = torch.tensor([[[-4]], [[0]]], dtype=torch.bfloat16)
    g = torch.tensor([[0.0], [1.2]], dtype=torch.float32)
    initial = torch.tensor([[[[-3.9375]]]], dtype=torch.float32)
    _, v_new, final = gdn_chunk_state_update_reference(
        k, w, u, g, initial, torch.tensor([0, 2], dtype=torch.int32)
    )
    residual = u[0, 0, 0].float() - w[0, 0, 0].float() * initial[0, 0, 0, 0]
    decay = torch.exp(g[1, 0] - g[0, 0])
    correct_factor = (residual * decay).to(torch.bfloat16).float()
    reused_v_new = (v_new[0, 0, 0].float() * decay).to(torch.bfloat16).float()
    all_fp32_factor = residual * decay
    expected = initial[0, 0, 0, 0] * torch.exp(g[1, 0]) + correct_factor
    assert correct_factor != reused_v_new
    assert correct_factor != all_fp32_factor
    assert final[0, 0, 0, 0] == expected
    assert final[0, 0, 0, 0] != initial[0, 0, 0, 0] * torch.exp(g[1, 0]) + reused_v_new


def test_fp32_state_carries_across_chunks() -> None:
    k = torch.zeros((65, 1, 1), dtype=torch.bfloat16)
    k[:64] = 1
    w = torch.zeros((65, 1, 1), dtype=torch.bfloat16)
    u = torch.ones((65, 1, 1), dtype=torch.bfloat16)
    g = torch.zeros((65, 1), dtype=torch.float32)
    initial = torch.tensor([[[[0.003]]]], dtype=torch.float32)
    h, _, final = gdn_chunk_state_update_reference(
        k, w, u, g, initial, torch.tensor([0, 65], dtype=torch.int32)
    )
    assert h.shape[0] == 2
    assert h[0, 0, 0, 0] == initial[0, 0, 0, 0].to(torch.bfloat16)
    assert h[1, 0, 0, 0] == torch.tensor(64.003).to(torch.bfloat16)
    assert final[0, 0, 0, 0] == torch.tensor(64.003)


def test_ragged_resets_and_global_chunk_order() -> None:
    inputs = list(_inputs([3, 65, 2], key_head_dim=1, value_head_dim=1))
    for index in (0, 1, 2, 3):
        inputs[index].zero_()
    inputs[4] = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]]).expand(-1, 4, -1, -1)
    h, v_new, final = gdn_chunk_state_update_reference(*inputs)
    assert h.shape == (4, 4, 1, 1)
    assert torch.equal(h[:, 0, 0, 0], torch.tensor([1, 2, 2, 3], dtype=torch.bfloat16))
    assert torch.count_nonzero(v_new) == 0
    assert torch.equal(final[:, 0, 0, 0], torch.tensor([1, 2, 3], dtype=torch.float32))


@pytest.mark.parametrize("length", [1, 2, 3, 15, 16, 17, 32, 33, 49, 63, 64, 65])
def test_lengths_and_partial_full_chunk_shapes(length: int) -> None:
    inputs = _inputs([length], num_key_heads=1, num_heads=1, key_head_dim=1, value_head_dim=1)
    h, v_new, final = gdn_chunk_state_update_reference(*inputs)
    assert h.shape == (math.ceil(length / 64), 1, 1, 1)
    assert v_new.shape == (length, 1, 1)
    assert final.shape == (1, 1, 1, 1)


@pytest.mark.parametrize(
    "geometry",
    [
        (1, 1, 1, 1),
        (1, 2, 2, 3),
        (2, 4, 3, 2),
        (3, 6, 2, 4),
    ],
)
def test_bounded_random_matches_independent_oracle(geometry: tuple[int, ...]) -> None:
    key_heads, heads, key_dim, value_dim = geometry
    inputs = _inputs(
        [3, 5],
        num_key_heads=key_heads,
        num_heads=heads,
        key_head_dim=key_dim,
        value_head_dim=value_dim,
        seed=sum(geometry),
    )
    actual = gdn_chunk_state_update_reference(*inputs)
    expected = _manual_reference(inputs)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-2, atol=2e-5)


def test_output_storage_and_input_immutability() -> None:
    inputs = _inputs([3, 65, 2])
    snapshots = tuple(tensor.clone() for tensor in inputs)
    outputs = gdn_chunk_state_update_reference(*inputs)
    assert [tensor.dtype for tensor in outputs] == [
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
    ]
    assert all(tensor.is_contiguous() for tensor in outputs)
    input_pointers = {_storage_pointer(tensor) for tensor in inputs}
    output_pointers = [_storage_pointer(tensor) for tensor in outputs]
    assert len(set(output_pointers)) == 3
    assert not input_pointers.intersection(output_pointers)
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(tensor, snapshot)


def _noncontiguous_copy(tensor: torch.Tensor) -> torch.Tensor:
    storage = torch.empty((tensor.shape[0] * 2, *tensor.shape[1:]), dtype=tensor.dtype)
    view = storage[::2]
    view.copy_(tensor)
    assert not view.is_contiguous()
    return view


def test_valid_noncontiguous_semantic_views() -> None:
    inputs = _inputs([3, 2])
    noncontiguous = tuple(_noncontiguous_copy(tensor) for tensor in inputs)
    expected = gdn_chunk_state_update_reference(*inputs)
    actual = gdn_chunk_state_update_reference(*noncontiguous)
    for result, oracle in zip(actual, expected):
        assert torch.equal(result, oracle)
        assert result.is_contiguous()


@pytest.mark.parametrize("index", range(6))
def test_non_tensor_inputs_are_rejected(index: int) -> None:
    inputs = _inputs([3])
    with pytest.raises(TypeError, match="torch.Tensor"):
        gdn_chunk_state_update_reference(*_replace(inputs, index, object()))


@pytest.mark.parametrize("index", range(6))
def test_invalid_ranks_are_rejected(index: int) -> None:
    inputs = _inputs([3])
    with pytest.raises(ValueError, match="rank"):
        gdn_chunk_state_update_reference(*_replace(inputs, index, inputs[index].unsqueeze(0)))


@pytest.mark.parametrize(
    "index,dtype_name",
    [(0, "bfloat16"), (1, "bfloat16"), (2, "bfloat16"), (3, "float32"), (4, "float32")],
)
def test_wrong_floating_dtypes_are_rejected(index: int, dtype_name: str) -> None:
    inputs = _inputs([3])
    replacement = inputs[index].float() if index < 3 else inputs[index].to(torch.bfloat16)
    with pytest.raises(TypeError, match=dtype_name):
        gdn_chunk_state_update_reference(*_replace(inputs, index, replacement))


def test_wrong_metadata_dtype_and_short_metadata_are_rejected() -> None:
    inputs = _inputs([3])
    with pytest.raises(TypeError, match="int32"):
        gdn_chunk_state_update_reference(*_replace(inputs, 5, inputs[5].long()))
    with pytest.raises(ValueError, match="at least"):
        gdn_chunk_state_update_reference(*_replace(inputs, 5, torch.tensor([0], dtype=torch.int32)))


@pytest.mark.parametrize(
    "boundaries,match",
    [
        ([-1, 3], "in \\[0"),
        ([0, 4], "in \\[0"),
        ([1, 3], "start"),
        ([0, 2], "end"),
        ([0, 1, 1, 3], "increasing"),
        ([0, 2, 1, 3], "increasing"),
    ],
)
def test_malformed_boundaries_are_rejected(boundaries: list[int], match: str) -> None:
    inputs = _inputs([3])
    metadata = torch.tensor(boundaries, dtype=torch.int32)
    with pytest.raises(ValueError, match=match):
        gdn_chunk_state_update_reference(*_replace(inputs, 5, metadata))


def test_zero_dimensions_and_incompatible_grouping_are_rejected() -> None:
    base = _inputs([3])
    cases = [
        _replace(base, 0, base[0][:0]),
        _replace(base, 0, base[0][:, :0]),
        _replace(base, 0, base[0][:, :, :0]),
        _replace(base, 1, base[1][:, :0]),
        _replace(base, 2, base[2][:, :, :0]),
        _replace(base, 1, base[1][:, :3]),
    ]
    for case in cases:
        with pytest.raises(ValueError):
            gdn_chunk_state_update_reference(*case)


def test_zero_sequences_and_state_sequence_mismatch_are_rejected() -> None:
    inputs = _inputs([3])
    with pytest.raises(ValueError, match="at least"):
        gdn_chunk_state_update_reference(*_replace(inputs, 5, torch.tensor([0], dtype=torch.int32)))
    with pytest.raises(ValueError, match="initial_state"):
        gdn_chunk_state_update_reference(*_replace(inputs, 4, inputs[4][:0]))


def test_shape_mismatches_are_rejected() -> None:
    inputs = _inputs([3, 2])
    cases = [
        _replace(inputs, 1, inputs[1][:-1]),
        _replace(inputs, 1, inputs[1][:, :, :-1]),
        _replace(inputs, 2, inputs[2][:-1]),
        _replace(inputs, 2, inputs[2][:, :-1]),
        _replace(inputs, 3, inputs[3][:-1]),
        _replace(inputs, 4, inputs[4][:, :, :, :-1]),
    ]
    for case in cases:
        with pytest.raises(ValueError):
            gdn_chunk_state_update_reference(*case)


@pytest.mark.parametrize("index", range(5))
def test_nonfinite_inputs_are_rejected(index: int) -> None:
    inputs = list(_inputs([3]))
    inputs[index] = inputs[index].clone()
    inputs[index].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        gdn_chunk_state_update_reference(*inputs)


def test_meta_sparse_and_mismatched_devices_are_rejected() -> None:
    inputs = _inputs([3])
    all_meta = tuple(torch.empty_like(tensor, device="meta") for tensor in inputs)
    with pytest.raises(ValueError, match="meta"):
        gdn_chunk_state_update_reference(*all_meta)

    sparse = inputs[0].to_sparse()
    with pytest.raises(ValueError, match="strided"):
        gdn_chunk_state_update_reference(*_replace(inputs, 0, sparse))

    meta = torch.empty_like(inputs[0], device="meta")
    with pytest.raises(ValueError, match="same device"):
        gdn_chunk_state_update_reference(*_replace(inputs, 0, meta))


def test_fake_cuda_inputs_are_rejected_without_gpu_execution() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    inputs = _inputs([3])
    with FakeTensorMode():
        fake_cuda = tuple(
            torch.empty(tensor.shape, dtype=tensor.dtype, device="cuda") for tensor in inputs
        )
        with pytest.raises(ValueError, match="requires CPU"):
            gdn_chunk_state_update_reference(*fake_cuda)
