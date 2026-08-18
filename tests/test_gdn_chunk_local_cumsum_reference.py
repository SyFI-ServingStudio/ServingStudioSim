"""CPU tests for Qwen GDN chunk-local cumulative-decay semantics."""

from __future__ import annotations

from itertools import accumulate

import pytest
import torch

from profiling.runners.attention.gdn_chunk_local_cumsum_reference import (
    gdn_chunk_local_cumsum_reference,
)

_CHUNK_SIZE = 64


def _cu(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)


def _manual(g: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(g)
    boundaries = cu_seqlens.tolist()
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            accumulator = torch.zeros(g.shape[1], dtype=torch.float32)
            for token in range(chunk_start, chunk_end):
                accumulator = accumulator + g[token]
                output[token] = accumulator
    return output


def test_ragged_sequence_and_chunk_resets_are_inclusive() -> None:
    lengths = [3, 65, 2]
    g = torch.arange(70 * 2, dtype=torch.float32).reshape(70, 2).remainder(7)
    cu_seqlens = _cu(lengths)
    expected = _manual(g, cu_seqlens)

    output = gdn_chunk_local_cumsum_reference(g, cu_seqlens)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    for reset in (0, 3, 67, 68):
        assert torch.equal(output[reset], g[reset])
    assert torch.equal(output[1], g[0] + g[1])
    assert torch.equal(output[69], g[68] + g[69])


@pytest.mark.parametrize("length", [64, 128, 130])
def test_exact_chunks_multi_chunk_sequences_and_final_partial_chunk(length: int) -> None:
    g = torch.ones(length, 2, dtype=torch.float32)
    cu_seqlens = _cu([length])

    output = gdn_chunk_local_cumsum_reference(g, cu_seqlens)

    for chunk_start in range(0, length, _CHUNK_SIZE):
        chunk_end = min(chunk_start + _CHUNK_SIZE, length)
        expected = torch.arange(1, chunk_end - chunk_start + 1, dtype=torch.float32)
        assert torch.equal(output[chunk_start:chunk_end, 0], expected)
        assert torch.equal(output[chunk_start:chunk_end, 1], expected)


def test_heads_are_independent_and_values_use_ordinary_arithmetic() -> None:
    g = torch.tensor(
        [
            [0.0, -1.0, 2.0],
            [0.0, -2.0, -3.0],
            [0.0, 4.0, 1.0],
            [0.0, -8.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [
            [0.0, -1.0, 2.0],
            [0.0, -3.0, -1.0],
            [0.0, 1.0, 0.0],
            [0.0, -7.0, 0.0],
        ],
        dtype=torch.float32,
    )

    output = gdn_chunk_local_cumsum_reference(g, _cu([4]))

    assert torch.equal(output, expected)


def test_bounded_random_fp32_matches_independent_chunk_reference() -> None:
    generator = torch.Generator().manual_seed(20260810)
    g = torch.randn(205, 5, generator=generator, dtype=torch.float32) * 0.05
    cu_seqlens = _cu([1, 63, 65, 76])
    expected = _manual(g, cu_seqlens)

    output = gdn_chunk_local_cumsum_reference(g, cu_seqlens)

    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


def test_output_contract_freshness_and_input_immutability() -> None:
    generator = torch.Generator().manual_seed(17)
    g = torch.randn(70, 4, generator=generator, dtype=torch.float32)
    cu_seqlens = _cu([3, 65, 2])
    g_before = g.clone()
    cu_before = cu_seqlens.clone()

    output = gdn_chunk_local_cumsum_reference(g, cu_seqlens)
    another_output = gdn_chunk_local_cumsum_reference(g, cu_seqlens)

    assert output.shape == g.shape
    assert output.dtype is torch.float32
    assert output.is_contiguous()
    assert output.untyped_storage().data_ptr() != g.untyped_storage().data_ptr()
    assert output.untyped_storage().data_ptr() != another_output.untyped_storage().data_ptr()
    assert torch.equal(g, g_before)
    assert torch.equal(cu_seqlens, cu_before)


def test_noncontiguous_strided_inputs_are_supported() -> None:
    base_g = torch.arange(70 * 6, dtype=torch.float32).reshape(70, 6)
    g = base_g[:, ::2]
    base_cu = torch.tensor([0, -1, 3, -1, 68, -1, 70], dtype=torch.int32)
    cu_seqlens = base_cu[::2]
    assert not g.is_contiguous()
    assert not cu_seqlens.is_contiguous()
    expected = _manual(g, cu_seqlens)

    output = gdn_chunk_local_cumsum_reference(g, cu_seqlens)

    assert output.is_contiguous()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["g", "cu_seqlens"])
def test_rejects_non_tensor_inputs(name: str) -> None:
    values: list[object] = [torch.ones(3, 2, dtype=torch.float32), _cu([3])]
    values[("g", "cu_seqlens").index(name)] = None

    with pytest.raises(TypeError, match=rf"{name} must be a torch.Tensor"):
        gdn_chunk_local_cumsum_reference(*values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("g", "message"),
    [
        (torch.empty(0, 2, dtype=torch.float32), "token dimension must be positive"),
        (torch.empty(3, 0, dtype=torch.float32), "head dimension must be positive"),
    ],
)
def test_rejects_nonpositive_g_dimensions(g: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gdn_chunk_local_cumsum_reference(g, torch.tensor([0, g.shape[0]], dtype=torch.int32))


@pytest.mark.parametrize(
    ("g", "cu_seqlens", "message"),
    [
        (torch.ones(2, 3, 1), _cu([2]), "g must be rank 2"),
        (torch.ones(2, 3), torch.tensor([[0, 2]], dtype=torch.int32), "cu_seqlens must be rank 1"),
    ],
)
def test_rejects_invalid_ranks(
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        gdn_chunk_local_cumsum_reference(g, cu_seqlens)


def test_rejects_non_fp32_g() -> None:
    with pytest.raises(TypeError, match="g dtype must be torch.float32"):
        gdn_chunk_local_cumsum_reference(torch.ones(3, 2, dtype=torch.float64), _cu([3]))


def test_rejects_non_int32_metadata() -> None:
    with pytest.raises(TypeError, match="cu_seqlens dtype must be torch.int32"):
        gdn_chunk_local_cumsum_reference(
            torch.ones(3, 2, dtype=torch.float32),
            torch.tensor([0, 3], dtype=torch.int64),
        )


@pytest.mark.parametrize(
    "cu_seqlens",
    [torch.empty(0, dtype=torch.int32), torch.tensor([0], dtype=torch.int32)],
)
def test_rejects_too_short_metadata(cu_seqlens: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="must contain at least one sequence"):
        gdn_chunk_local_cumsum_reference(torch.ones(3, 2), cu_seqlens)


@pytest.mark.parametrize(
    ("cu_seqlens", "message"),
    [
        (torch.tensor([1, 3], dtype=torch.int32), "must start at zero"),
        (torch.tensor([0, 2], dtype=torch.int32), "must end at token count 3"),
        (torch.tensor([-1, 3], dtype=torch.int32), "boundaries must be in"),
        (torch.tensor([0, 4], dtype=torch.int32), "boundaries must be in"),
        (torch.tensor([0, 1, 1, 3], dtype=torch.int32), "strictly increasing"),
        (torch.tensor([0, 2, 1, 3], dtype=torch.int32), "strictly increasing"),
    ],
)
def test_rejects_invalid_boundaries(cu_seqlens: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gdn_chunk_local_cumsum_reference(torch.ones(3, 2), cu_seqlens)


@pytest.mark.parametrize("name", ["g", "cu_seqlens"])
def test_rejects_sparse_non_strided_tensors(name: str) -> None:
    g = torch.ones(3, 2, dtype=torch.float32)
    cu_seqlens = _cu([3])
    if name == "g":
        g = g.to_sparse()
    else:
        cu_seqlens = cu_seqlens.to_sparse()

    with pytest.raises(ValueError, match=rf"{name} must have torch.strided layout"):
        gdn_chunk_local_cumsum_reference(g, cu_seqlens)


def test_rejects_meta_tensors() -> None:
    g = torch.empty(3, 2, dtype=torch.float32, device="meta")
    cu_seqlens = torch.empty(2, dtype=torch.int32, device="meta")

    with pytest.raises(ValueError, match="meta tensors are not supported"):
        gdn_chunk_local_cumsum_reference(g, cu_seqlens)


def test_rejects_mismatched_devices_before_reading_metadata() -> None:
    g = torch.ones(3, 2, dtype=torch.float32)
    cu_seqlens = torch.empty(2, dtype=torch.int32, device="meta")

    with pytest.raises(ValueError, match="must be on the same device"):
        gdn_chunk_local_cumsum_reference(g, cu_seqlens)
