from __future__ import annotations

import pytest
import torch

from profiling.runners.attention.gdn_chunk_solve_tril_reference import (
    gdn_chunk_solve_tril_reference,
)


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return torch.tensor(boundaries, dtype=torch.int32)


def _strict_lower_input(
    lengths: list[int],
    *,
    num_heads: int = 2,
    seed: int = 20260810,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    A = torch.zeros((sum(lengths), num_heads, 64), dtype=torch.float32)
    for sequence_start, sequence_end in zip(
        _cu_seqlens(lengths).tolist(),
        _cu_seqlens(lengths).tolist()[1:],
    ):
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            chunk_length = chunk_end - chunk_start
            values = torch.randn(
                (chunk_length, num_heads, chunk_length),
                generator=generator,
                dtype=torch.float32,
            ).mul_(0.02)
            lower = torch.tril(values.permute(1, 0, 2), diagonal=-1).permute(1, 0, 2)
            A[chunk_start:chunk_end, :, :chunk_length].copy_(lower)
    return A


def _manual_fp32(A: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(A)
    boundaries = cu_seqlens.tolist()
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            chunk_length = chunk_end - chunk_start
            for head in range(A.shape[1]):
                for row in range(chunk_length):
                    output[chunk_start + row, head, row] = 1.0
                    for column in range(row):
                        value = -A[chunk_start + row, head, column]
                        for inner in range(column + 1, row):
                            value = value - (
                                A[chunk_start + row, head, inner]
                                * output[chunk_start + inner, head, column]
                            )
                        output[chunk_start + row, head, column] = value
    return output


def test_exact_one_two_and_sparse_three_by_three_inverses() -> None:
    A = torch.zeros((6, 1, 64), dtype=torch.float32)
    A[2, 0, 0] = 0.5
    A[4, 0, 0] = 0.25
    A[5, 0, 0] = -0.5
    A[5, 0, 1] = 2.0

    output = gdn_chunk_solve_tril_reference(A, _cu_seqlens([1, 2, 3]))

    assert output[0, 0, 0].item() == 1.0
    assert output[1, 0, 0].item() == 1.0
    assert output[2, 0, 0].item() == -0.5
    assert output[3, 0, 0].item() == 1.0
    assert output[4, 0, 0].item() == -0.25
    assert output[4, 0, 1].item() == 1.0
    assert output[5, 0, 1].item() == -2.0
    # M[2,0] = -(-.5) - 2*(-.25) = 1 exactly.
    assert output[5, 0, 0].item() == 1.0


def test_positive_input_has_negative_first_off_diagonal() -> None:
    A = torch.zeros((2, 1, 64), dtype=torch.float32)
    A[1, 0, 0] = 0.75

    output = gdn_chunk_solve_tril_reference(A, _cu_seqlens([2]))

    assert output[1, 0, 0].item() == -0.75


def test_mixed_sign_multi_path_recurrence() -> None:
    A = torch.zeros((4, 1, 64), dtype=torch.float32)
    A[1, 0, 0] = 0.5
    A[2, 0, :2] = torch.tensor([-0.25, 2.0])
    A[3, 0, :3] = torch.tensor([1.0, -0.5, 0.25])

    actual = gdn_chunk_solve_tril_reference(A, _cu_seqlens([4]))
    expected = _manual_fp32(A, _cu_seqlens([4])).to(torch.bfloat16)

    assert torch.equal(actual, expected)
    assert actual[3, 0, 0].item() < 0
    assert actual[3, 0, 1].item() > 0


def test_manual_recurrence_and_triangular_solve_oracle() -> None:
    A = _strict_lower_input([7], num_heads=3)
    cu_seqlens = _cu_seqlens([7])

    actual = gdn_chunk_solve_tril_reference(A, cu_seqlens)
    manual = _manual_fp32(A, cu_seqlens)

    for head in range(3):
        lower = A[:, head, :7]
        identity = torch.eye(7, dtype=torch.float32)
        oracle = torch.linalg.solve_triangular(
            identity + lower,
            identity,
            upper=False,
            unitriangular=False,
        )
        torch.testing.assert_close(manual[:, head, :7], oracle, rtol=1e-5, atol=1e-5)
    assert torch.equal(actual, manual.to(torch.bfloat16))


def test_structural_output_contract_and_input_immutability() -> None:
    A = _strict_lower_input([3, 65, 2], num_heads=2)
    cu_seqlens = _cu_seqlens([3, 65, 2])
    A_snapshot = A.clone()
    metadata_snapshot = cu_seqlens.clone()

    output = gdn_chunk_solve_tril_reference(A, cu_seqlens)
    second = gdn_chunk_solve_tril_reference(A, cu_seqlens)

    assert output.shape == (70, 2, 64)
    assert output.dtype is torch.bfloat16
    assert output.is_contiguous()
    assert output.data_ptr() != second.data_ptr()
    assert output.data_ptr() != A.data_ptr()
    assert torch.equal(A, A_snapshot)
    assert torch.equal(cu_seqlens, metadata_snapshot)

    for sequence_start, sequence_end in zip(cu_seqlens.tolist(), cu_seqlens.tolist()[1:]):
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            length = chunk_end - chunk_start
            block = output[chunk_start:chunk_end, :, :length].permute(1, 0, 2)
            assert torch.equal(
                torch.diagonal(block, dim1=-2, dim2=-1),
                torch.ones((2, length), dtype=torch.bfloat16),
            )
            assert torch.count_nonzero(torch.triu(block, diagonal=1)).item() == 0
            assert torch.count_nonzero(output[chunk_start:chunk_end, :, length:]).item() == 0


@pytest.mark.parametrize("length", [1, 2, 3, 15, 16, 17, 32, 33, 49, 63, 64, 65])
def test_lengths_around_subblock_and_chunk_boundaries(length: int) -> None:
    A = _strict_lower_input([length], num_heads=1, seed=length)

    output = gdn_chunk_solve_tril_reference(A, _cu_seqlens([length]))
    expected = _manual_fp32(A, _cu_seqlens([length])).to(torch.bfloat16)

    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
    if length == 65:
        assert output[64, 0, 0].item() == 1.0
        assert torch.count_nonzero(output[64, 0, 1:]).item() == 0


@pytest.mark.parametrize("lengths", [[64], [65], [128], [64, 64, 2], [3, 65, 2]])
def test_exact_multi_partial_and_ragged_resets(lengths: list[int]) -> None:
    A = torch.zeros((sum(lengths), 2, 64), dtype=torch.float32)
    boundaries = _cu_seqlens(lengths)

    output = gdn_chunk_solve_tril_reference(A, boundaries)

    for sequence_start, sequence_end in zip(boundaries.tolist(), boundaries.tolist()[1:]):
        for chunk_start in range(sequence_start, sequence_end, 64):
            assert output[chunk_start, 0, 0].item() == 1.0
            assert torch.count_nonzero(output[chunk_start, :, 1:]).item() == 0


def test_head_independence() -> None:
    A = torch.zeros((3, 2, 64), dtype=torch.float32)
    A[1, 0, 0] = 0.5
    A[1, 1, 0] = -0.25
    A[2, 0, :2] = torch.tensor([1.0, 2.0])
    A[2, 1, :2] = torch.tensor([-1.0, 0.5])

    output = gdn_chunk_solve_tril_reference(A, _cu_seqlens([3]))

    assert not torch.equal(output[:, 0], output[:, 1])
    for head in range(2):
        single = gdn_chunk_solve_tril_reference(A[:, head : head + 1], _cu_seqlens([3]))
        assert torch.equal(output[:, head : head + 1], single)


def test_zero_positive_negative_sparse_and_random_values() -> None:
    zero = torch.zeros((4, 1, 64), dtype=torch.float32)
    assert torch.equal(
        gdn_chunk_solve_tril_reference(zero, _cu_seqlens([4])),
        _manual_fp32(zero, _cu_seqlens([4])).to(torch.bfloat16),
    )

    for seed in (1, 2, 3):
        A = _strict_lower_input([5, 3], num_heads=2, seed=seed)
        if seed == 2:
            A.neg_()
        if seed == 3:
            A[A.abs() < 0.01] = 0
        actual = gdn_chunk_solve_tril_reference(A, _cu_seqlens([5, 3]))
        expected = _manual_fp32(A, _cu_seqlens([5, 3]))
        torch.testing.assert_close(
            actual.float(), expected.to(torch.bfloat16).float(), rtol=0, atol=0
        )


def test_single_final_bf16_rounding_not_intermediate_rounding() -> None:
    generator = torch.Generator().manual_seed(0)
    A = torch.zeros((8, 1, 64), dtype=torch.float32)
    values = torch.randn((8, 8), generator=generator).mul_(0.2)
    A[:, 0, :8].copy_(torch.tril(values, diagonal=-1))
    expected_fp32 = _manual_fp32(A, _cu_seqlens([8]))

    # Deliberately wrong oracle: round every solved scalar before later rows
    # consume it. This distinguishes the required one final output cast.
    intermediate_bf16 = torch.zeros_like(A, dtype=torch.bfloat16)
    for row in range(8):
        intermediate_bf16[row, 0, row] = 1
        for column in range(row):
            value = -A[row, 0, column]
            for inner in range(column + 1, row):
                value = value - (A[row, 0, inner] * intermediate_bf16[inner, 0, column].float())
            intermediate_bf16[row, 0, column] = value

    output = gdn_chunk_solve_tril_reference(A, _cu_seqlens([8]))

    assert torch.equal(output, expected_fp32.to(torch.bfloat16))
    assert not torch.equal(output, intermediate_bf16)


def test_valid_noncontiguous_semantic_inputs_and_metadata() -> None:
    contiguous = _strict_lower_input([3, 2], num_heads=2)
    storage = torch.zeros((5, 4, 128), dtype=torch.float32)
    storage[:, ::2, ::2].copy_(contiguous)
    A = storage[:, ::2, ::2]
    boundary_storage = torch.tensor([0, -99, 3, -99, 5, -99], dtype=torch.int32)
    cu_seqlens = boundary_storage[::2]
    assert not A.is_contiguous()
    assert not cu_seqlens.is_contiguous()

    actual = gdn_chunk_solve_tril_reference(A, cu_seqlens)
    expected = gdn_chunk_solve_tril_reference(A.contiguous(), cu_seqlens.contiguous())

    assert torch.equal(actual, expected)
    assert actual.is_contiguous()


@pytest.mark.parametrize("argument", ["A", "cu_seqlens"])
def test_rejects_non_tensor_inputs(argument: str) -> None:
    values: dict[str, object] = {
        "A": torch.zeros((2, 1, 64), dtype=torch.float32),
        "cu_seqlens": _cu_seqlens([2]),
    }
    values[argument] = []

    with pytest.raises(TypeError, match=rf"{argument} must be a torch.Tensor"):
        gdn_chunk_solve_tril_reference(**values)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((0, 1, 64), "token dimension must be positive"),
        ((2, 0, 64), "head dimension must be positive"),
        ((2, 1, 63), "last dimension must equal chunk size 64"),
    ],
)
def test_rejects_invalid_geometry(shape: tuple[int, ...], message: str) -> None:
    A = torch.zeros(shape, dtype=torch.float32)
    metadata = torch.tensor([0, shape[0]], dtype=torch.int32)

    with pytest.raises(ValueError, match=message):
        gdn_chunk_solve_tril_reference(A, metadata)


@pytest.mark.parametrize(
    ("A", "metadata", "message"),
    [
        (torch.zeros((2, 64)), _cu_seqlens([2]), "A must be rank 3"),
        (
            torch.zeros((2, 1, 64)),
            torch.tensor([[0, 2]], dtype=torch.int32),
            "cu_seqlens must be rank 1",
        ),
    ],
)
def test_rejects_invalid_ranks(A: torch.Tensor, metadata: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gdn_chunk_solve_tril_reference(A, metadata)


def test_rejects_invalid_dtypes() -> None:
    A = torch.zeros((2, 1, 64), dtype=torch.float32)

    with pytest.raises(TypeError, match="A dtype must be torch.float32"):
        gdn_chunk_solve_tril_reference(A.to(torch.bfloat16), _cu_seqlens([2]))
    with pytest.raises(TypeError, match="cu_seqlens dtype must be torch.int32"):
        gdn_chunk_solve_tril_reference(A, _cu_seqlens([2]).to(torch.int64))


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (torch.tensor([0], dtype=torch.int32), "at least one sequence"),
        (torch.tensor([1, 2], dtype=torch.int32), "start at zero"),
        (torch.tensor([0, 1], dtype=torch.int32), "end at token count 2"),
        (torch.tensor([0, 3, 2], dtype=torch.int32), r"boundaries must be in \[0, 2\]"),
        (torch.tensor([-1, 2], dtype=torch.int32), r"boundaries must be in \[0, 2\]"),
        (torch.tensor([0, 1, 1, 2], dtype=torch.int32), "strictly increasing"),
        (torch.tensor([0, 2, 1, 2], dtype=torch.int32), "strictly increasing"),
    ],
)
def test_rejects_invalid_metadata(metadata: torch.Tensor, message: str) -> None:
    A = torch.zeros((2, 1, 64), dtype=torch.float32)

    with pytest.raises(ValueError, match=message):
        gdn_chunk_solve_tril_reference(A, metadata)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_A(value: float) -> None:
    A = torch.zeros((2, 1, 64), dtype=torch.float32)
    A[1, 0, 0] = value

    with pytest.raises(ValueError, match="only finite values"):
        gdn_chunk_solve_tril_reference(A, _cu_seqlens([2]))


@pytest.mark.parametrize("location", ["diagonal", "upper", "unused"])
def test_rejects_nonzero_invalid_structure(location: str) -> None:
    A = torch.zeros((3, 1, 64), dtype=torch.float32)
    if location == "diagonal":
        A[1, 0, 1] = 1
    elif location == "upper":
        A[0, 0, 1] = 1
    else:
        A[2, 0, 3] = 1

    with pytest.raises(ValueError, match="exactly strict-lower"):
        gdn_chunk_solve_tril_reference(A, _cu_seqlens([3]))


@pytest.mark.parametrize("argument", ["A", "cu_seqlens"])
def test_rejects_sparse_layouts(argument: str) -> None:
    A = torch.zeros((2, 1, 64), dtype=torch.float32)
    metadata = _cu_seqlens([2])
    if argument == "A":
        A = A.to_sparse()
    else:
        metadata = metadata.to_sparse()

    with pytest.raises(ValueError, match=rf"{argument} must have torch.strided layout"):
        gdn_chunk_solve_tril_reference(A, metadata)


def test_rejects_meta_tensors_and_mismatched_devices() -> None:
    meta_A = torch.empty((2, 1, 64), dtype=torch.float32, device="meta")
    meta_metadata = torch.empty((2,), dtype=torch.int32, device="meta")

    with pytest.raises(ValueError, match="meta tensors are not supported"):
        gdn_chunk_solve_tril_reference(meta_A, meta_metadata)
    with pytest.raises(ValueError, match="must be on the same device"):
        gdn_chunk_solve_tril_reference(torch.zeros((2, 1, 64), dtype=torch.float32), meta_metadata)


def test_rejects_non_cpu_fake_tensors() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        A = torch.empty((2, 1, 64), dtype=torch.float32, device="cuda")
        metadata = torch.empty((2,), dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="requires CPU tensors, got cuda"):
            gdn_chunk_solve_tril_reference(A, metadata)
