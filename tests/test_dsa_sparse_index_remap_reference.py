from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import profiling.runners.attention.dsa_sparse_index_remap_reference as reference_module
from profiling.runners.attention.dsa_sparse_index_remap_reference import (
    dsa_sparse_index_remap_reference,
)

_BLOCK_SIZE = 64


def _base_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    req_id = torch.tensor([0, 1, 0], dtype=torch.int32)
    block_table = torch.tensor(
        [[7, 2, 11, 5], [3, 13, 1, 17]],
        dtype=torch.int32,
    )
    token_indices = torch.tensor(
        [[0, 63, 64, 129], [65, -1, 191, 255], [128, 1, -7, 256]],
        dtype=torch.int32,
    )
    return req_id, block_table, token_indices


def _manual_oracle(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    block_size: int,
    workspace_ids: torch.Tensor | None = None,
    workspace_starts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent small Python-loop oracle for valid test operands."""
    output = torch.full(token_indices.shape, -1, dtype=torch.int32)
    counts = torch.zeros((token_indices.shape[0],), dtype=torch.int32)
    for row in range(token_indices.shape[0]):
        request = int(req_id[row])
        workspace = -1 if workspace_ids is None else int(workspace_ids[row])
        for column in range(token_indices.shape[1]):
            local_index = int(token_indices[row, column])
            if local_index < 0:
                continue
            local_block, offset = divmod(local_index, block_size)
            if local_block >= block_table.shape[1]:
                continue
            counts[row] += 1
            if workspace >= 0:
                assert workspace_starts is not None
                output[row, column] = int(workspace_starts[workspace]) + local_index
            else:
                page = int(block_table[request, local_block])
                output[row, column] = page * block_size + offset
    return output, counts


def _snapshot(tensor: torch.Tensor) -> tuple[Any, ...]:
    if tensor.device.type == "meta":
        return ("meta", tuple(tensor.shape), tensor.dtype, tensor.layout)
    return (
        tensor.clone(),
        tensor.data_ptr(),
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tensor.stride(),
    )


def _assert_snapshot(tensor: torch.Tensor, snapshot: tuple[Any, ...]) -> None:
    if snapshot[0] == "meta":
        assert ("meta", tuple(tensor.shape), tensor.dtype, tensor.layout) == snapshot
        return
    assert torch.equal(tensor, snapshot[0])
    assert tensor.data_ptr() == snapshot[1]
    assert tensor.untyped_storage().data_ptr() == snapshot[2]
    assert tensor.storage_offset() == snapshot[3]
    assert tensor.stride() == snapshot[4]


def _assert_failure_without_mutation(
    error: type[Exception],
    match: str,
    call: Callable[[], Any],
    tensors: tuple[torch.Tensor, ...],
) -> None:
    snapshots = tuple(_snapshot(tensor) for tensor in tensors)
    with pytest.raises(error, match=match):
        call()
    for tensor, snapshot in zip(tensors, snapshots, strict=True):
        _assert_snapshot(tensor, snapshot)


def test_manual_signed_global_mapping_multiple_requests_and_page_boundaries() -> None:
    req_id, block_table, token_indices = _base_case()

    actual = dsa_sparse_index_remap_reference(
        req_id,
        block_table,
        token_indices,
        block_size=_BLOCK_SIZE,
    )
    expected, _ = _manual_oracle(
        req_id,
        block_table,
        token_indices,
        block_size=_BLOCK_SIZE,
    )

    assert torch.equal(actual, expected)
    assert actual.tolist() == [
        [448, 511, 128, 705],
        [833, -1, 127, 1151],
        [704, 449, -1, -1],
    ]


def test_negative_and_out_of_table_indices_map_to_minus_one() -> None:
    req_id = torch.tensor([0, 0], dtype=torch.int32)
    block_table = torch.tensor([[4, 9]], dtype=torch.int32)
    token_indices = torch.tensor(
        [
            [-2_147_483_648, -99, -1, 128, 2_147_483_647],
            [-7, -6, -5, -4, -3],
        ],
        dtype=torch.int32,
    )

    output = dsa_sparse_index_remap_reference(
        req_id,
        block_table,
        token_indices,
        block_size=_BLOCK_SIZE,
    )

    assert output.tolist() == [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]]


def test_duplicate_indices_preserve_slot_order_and_repeated_outputs() -> None:
    req_id = torch.tensor([0], dtype=torch.int32)
    block_table = torch.tensor([[5, 2]], dtype=torch.int32)
    token_indices = torch.tensor([[65, 3, 65, 64, 3]], dtype=torch.int32)

    output = dsa_sparse_index_remap_reference(
        req_id,
        block_table,
        token_indices,
        block_size=_BLOCK_SIZE,
    )

    assert output.tolist() == [[129, 323, 129, 128, 323]]


def test_mixed_global_workspace_mapping_and_chunk_reset_starts() -> None:
    req_id = torch.tensor([0, 1, 2, 2], dtype=torch.int32)
    block_table = torch.tensor([[4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.int32)
    token_indices = torch.tensor(
        [[0, 64, -1], [1, 65, 129], [2, 66, 130], [63, 127, 191]],
        dtype=torch.int32,
    )
    workspace_ids = torch.tensor([-1, 0, 1, 1], dtype=torch.int32)
    # Workspace 1 models a new prefill chunk whose address space resets.
    workspace_starts = torch.tensor([1000, 0], dtype=torch.int32)

    actual = dsa_sparse_index_remap_reference(
        req_id,
        block_table,
        token_indices,
        block_size=_BLOCK_SIZE,
        prefill_workspace_request_ids=workspace_ids,
        prefill_workspace_starts=workspace_starts,
    )
    expected, _ = _manual_oracle(
        req_id,
        block_table,
        token_indices,
        block_size=_BLOCK_SIZE,
        workspace_ids=workspace_ids,
        workspace_starts=workspace_starts,
    )

    assert torch.equal(actual, expected)
    assert actual.tolist() == [
        [256, 320, -1],
        [1001, 1065, 1129],
        [2, 66, 130],
        [63, 127, 191],
    ]


def test_optional_counts_cover_empty_short_full_duplicates_and_workspace() -> None:
    req_id = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    block_table = torch.tensor([[2, 3], [5, 7]], dtype=torch.int32)
    token_indices = torch.tensor(
        [[-1, -2, 128, -4], [1, -1, 200, -1], [0, 0, 65, -1], [0, 1, 2, 3]],
        dtype=torch.int32,
    )
    workspace_ids = torch.tensor([-1, -1, 0, 0], dtype=torch.int32)
    workspace_starts = torch.tensor([50], dtype=torch.int32)

    no_counts = dsa_sparse_index_remap_reference(
        req_id, block_table, token_indices, block_size=_BLOCK_SIZE
    )
    with_counts = dsa_sparse_index_remap_reference(
        req_id,
        block_table,
        token_indices,
        block_size=_BLOCK_SIZE,
        prefill_workspace_request_ids=workspace_ids,
        prefill_workspace_starts=workspace_starts,
        return_valid_counts=True,
    )

    assert isinstance(no_counts, torch.Tensor)
    assert isinstance(with_counts, tuple) and len(with_counts) == 2
    output, counts = with_counts
    assert counts.tolist() == [0, 1, 3, 4]
    assert output.tolist()[2:] == [[50, 50, 115, -1], [50, 51, 52, 53]]
    assert counts.dtype is torch.int32 and counts.shape == (4,) and counts.is_contiguous()


def test_unused_negative_page_padding_is_ignored() -> None:
    req_id = torch.tensor([0], dtype=torch.int32)
    block_table = torch.tensor([[6, -1, -99]], dtype=torch.int32)
    token_indices = torch.tensor([[0, 63, -1, 192]], dtype=torch.int32)

    output = dsa_sparse_index_remap_reference(
        req_id, block_table, token_indices, block_size=_BLOCK_SIZE
    )

    assert output.tolist() == [[384, 447, -1, -1]]


@pytest.mark.parametrize(
    ("block_table", "token_indices", "match"),
    [
        (
            torch.tensor([[-1, 2]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
            "selected block_table physical page IDs must be nonnegative",
        ),
        (
            torch.tensor([[torch.iinfo(torch.int32).max]], dtype=torch.int32),
            torch.tensor([[63]], dtype=torch.int32),
            "selected global mapped index exceeds int32 range",
        ),
    ],
)
def test_selected_page_metadata_failures_are_precompute_and_actionable(
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    match: str,
) -> None:
    req_id = torch.tensor([0], dtype=torch.int32)
    _assert_failure_without_mutation(
        ValueError,
        match,
        lambda: dsa_sparse_index_remap_reference(
            req_id, block_table, token_indices, block_size=_BLOCK_SIZE
        ),
        (req_id, block_table, token_indices),
    )


@pytest.mark.parametrize(
    ("req_id", "workspace_ids", "workspace_starts", "match"),
    [
        (torch.tensor([-1], dtype=torch.int32), None, None, "req_id values must be"),
        (torch.tensor([1], dtype=torch.int32), None, None, "req_id values must be"),
        (
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([-2], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            "prefill_workspace_request_ids values must be",
        ),
        (
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            "prefill_workspace_request_ids values must be",
        ),
        (
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
            "prefill_workspace_starts values must be nonnegative",
        ),
    ],
)
def test_request_and_workspace_metadata_bounds_fail_before_indexing(
    req_id: torch.Tensor,
    workspace_ids: torch.Tensor | None,
    workspace_starts: torch.Tensor | None,
    match: str,
) -> None:
    block_table = torch.tensor([[1]], dtype=torch.int32)
    token_indices = torch.tensor([[0]], dtype=torch.int32)
    kwargs: dict[str, Any] = {}
    tensors = [req_id, block_table, token_indices]
    if workspace_ids is not None and workspace_starts is not None:
        kwargs = {
            "prefill_workspace_request_ids": workspace_ids,
            "prefill_workspace_starts": workspace_starts,
        }
        tensors.extend((workspace_ids, workspace_starts))
    _assert_failure_without_mutation(
        ValueError,
        match,
        lambda: dsa_sparse_index_remap_reference(
            req_id, block_table, token_indices, block_size=_BLOCK_SIZE, **kwargs
        ),
        tuple(tensors),
    )


def test_workspace_result_overflow_is_rejected() -> None:
    req_id = torch.tensor([0], dtype=torch.int32)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    token_indices = torch.tensor([[1]], dtype=torch.int32)
    workspace_ids = torch.tensor([0], dtype=torch.int32)
    workspace_starts = torch.tensor([torch.iinfo(torch.int32).max], dtype=torch.int32)

    with pytest.raises(ValueError, match="selected workspace mapped index exceeds int32 range"):
        dsa_sparse_index_remap_reference(
            req_id,
            block_table,
            token_indices,
            block_size=_BLOCK_SIZE,
            prefill_workspace_request_ids=workspace_ids,
            prefill_workspace_starts=workspace_starts,
        )


def test_positive_index_inside_table_width_maps_without_context_length() -> None:
    req_id = torch.tensor([0], dtype=torch.int32)
    block_table = torch.tensor([[2, 7, 11, 13]], dtype=torch.int32)
    # No context length is supplied. Index 191 maps through table column 2.
    token_indices = torch.tensor([[191]], dtype=torch.int32)

    output = dsa_sparse_index_remap_reference(
        req_id, block_table, token_indices, block_size=_BLOCK_SIZE
    )

    assert output.item() == 11 * 64 + 63


def test_nonzero_offset_and_padded_outer_stride_views_are_supported() -> None:
    req_backing = torch.tensor([99, 0, 1, 0, 99], dtype=torch.int32)
    req_id = req_backing[1:4]

    block_backing = torch.full((3, 8), -77, dtype=torch.int32)
    block_table = block_backing[1:, 2:6]
    block_table.copy_(torch.tensor([[7, 2, 11, 5], [3, 13, 1, 17]], dtype=torch.int32))

    token_backing = torch.full((4, 9), -88, dtype=torch.int32)
    token_indices = token_backing[1:4, 3:7]
    token_indices.copy_(
        torch.tensor(
            [[0, 63, 64, 129], [65, -1, 191, 255], [128, 1, -7, 256]],
            dtype=torch.int32,
        )
    )

    assert req_id.storage_offset() > 0 and req_id.is_contiguous()
    assert block_table.storage_offset() > 0 and block_table.stride() == (8, 1)
    assert token_indices.storage_offset() > 0 and token_indices.stride() == (9, 1)

    output = dsa_sparse_index_remap_reference(
        req_id, block_table, token_indices, block_size=_BLOCK_SIZE
    )
    expected, _ = _manual_oracle(req_id, block_table, token_indices, block_size=_BLOCK_SIZE)

    assert torch.equal(output, expected)


def test_output_contract_immutability_identity_and_non_aliasing() -> None:
    req_id, block_table, token_indices = _base_case()
    workspace_ids = torch.tensor([-1, 0, 0], dtype=torch.int32)
    starts_backing = torch.tensor([99, 1000, 99], dtype=torch.int32)
    workspace_starts = starts_backing[1:2]
    tensors = (req_id, block_table, token_indices, workspace_ids, workspace_starts)
    snapshots = tuple(_snapshot(tensor) for tensor in tensors)

    output, counts = dsa_sparse_index_remap_reference(
        req_id,
        block_table,
        token_indices,
        block_size=_BLOCK_SIZE,
        prefill_workspace_request_ids=workspace_ids,
        prefill_workspace_starts=workspace_starts,
        return_valid_counts=True,
    )

    assert output.shape == token_indices.shape and output.dtype is torch.int32
    assert output.is_contiguous() and output.storage_offset() == 0
    assert counts.shape == req_id.shape and counts.dtype is torch.int32
    assert counts.is_contiguous() and counts.storage_offset() == 0
    for tensor, snapshot in zip(tensors, snapshots, strict=True):
        _assert_snapshot(tensor, snapshot)
        assert not torch._C._overlaps(output, tensor)
        assert not torch._C._overlaps(counts, tensor)
    assert not torch._C._overlaps(output, counts)


def test_aligned_workspace_metadata_nonzero_offsets_remain_contiguous() -> None:
    req_id = torch.tensor([0, 0], dtype=torch.int32)
    block_table = torch.tensor([[1, 2]], dtype=torch.int32)
    token_indices = torch.tensor([[0, 64], [1, 65]], dtype=torch.int32)
    ids_backing = torch.tensor([99, 0, -1, 99], dtype=torch.int32)
    starts_backing = torch.tensor([99, 100, 99], dtype=torch.int32)
    workspace_ids = ids_backing[1:3]
    workspace_starts = starts_backing[1:2]

    output = dsa_sparse_index_remap_reference(
        req_id,
        block_table,
        token_indices,
        block_size=_BLOCK_SIZE,
        prefill_workspace_request_ids=workspace_ids,
        prefill_workspace_starts=workspace_starts,
    )

    assert output.tolist() == [[100, 164], [65, 129]]


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 64, 1000])
def test_results_are_independent_of_internal_row_chunks(
    monkeypatch: pytest.MonkeyPatch,
    chunk_size: int,
) -> None:
    req_id = torch.tensor([0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)
    block_table = torch.arange(15, dtype=torch.int32).reshape(3, 5).add(2)
    token_indices = torch.tensor(
        [
            [-1, 0, 63, 64, 319, 320],
            [1, 65, 129, 193, 257, -1],
            [2, 66, 130, 194, 258, -2],
            [3, 67, 131, 195, 259, -3],
            [4, 68, 132, 196, 260, -4],
            [5, 69, 133, 197, 261, -5],
            [6, 70, 134, 198, 262, -6],
        ],
        dtype=torch.int32,
    )
    workspace_ids = torch.tensor([-1, 0, 0, -1, 1, 1, -1], dtype=torch.int32)
    workspace_starts = torch.tensor([500, 1000], dtype=torch.int32)
    expected = _manual_oracle(
        req_id,
        block_table,
        token_indices,
        block_size=_BLOCK_SIZE,
        workspace_ids=workspace_ids,
        workspace_starts=workspace_starts,
    )

    monkeypatch.setattr(reference_module, "_ROW_CHUNK_SIZE", chunk_size)
    actual = dsa_sparse_index_remap_reference(
        req_id,
        block_table,
        token_indices,
        block_size=_BLOCK_SIZE,
        prefill_workspace_request_ids=workspace_ids,
        prefill_workspace_starts=workspace_starts,
        return_valid_counts=True,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("name", ["req_id", "block_table", "token_indices"])
def test_non_tensor_principal_inputs_raise_type_error(name: str) -> None:
    req_id, block_table, token_indices = _base_case()
    values: dict[str, Any] = {
        "req_id": req_id,
        "block_table": block_table,
        "token_indices": token_indices,
    }
    values[name] = [0]
    with pytest.raises(TypeError, match=f"{name} must be a torch.Tensor"):
        dsa_sparse_index_remap_reference(
            values["req_id"],
            values["block_table"],
            values["token_indices"],
            block_size=_BLOCK_SIZE,
        )


@pytest.mark.parametrize("name", ["req_id", "block_table", "token_indices"])
def test_wrong_principal_dtypes_raise_type_error(name: str) -> None:
    req_id, block_table, token_indices = _base_case()
    values = {
        "req_id": req_id,
        "block_table": block_table,
        "token_indices": token_indices,
    }
    values[name] = values[name].to(torch.int64)
    with pytest.raises(TypeError, match=f"{name} dtype must be torch.int32"):
        dsa_sparse_index_remap_reference(
            values["req_id"],
            values["block_table"],
            values["token_indices"],
            block_size=_BLOCK_SIZE,
        )


@pytest.mark.parametrize(
    ("name", "replacement", "match"),
    [
        ("req_id", torch.zeros((1, 1), dtype=torch.int32), "req_id must be rank 1"),
        (
            "block_table",
            torch.zeros((2,), dtype=torch.int32),
            "block_table must be rank 2",
        ),
        (
            "token_indices",
            torch.zeros((3,), dtype=torch.int32),
            "token_indices must be rank 2",
        ),
        (
            "req_id",
            torch.empty((0,), dtype=torch.int32),
            "req_id must contain at least one query",
        ),
        (
            "block_table",
            torch.empty((0, 4), dtype=torch.int32),
            "block_table dimensions must be positive",
        ),
        (
            "block_table",
            torch.empty((2, 0), dtype=torch.int32),
            "block_table dimensions must be positive",
        ),
        (
            "token_indices",
            torch.empty((3, 0), dtype=torch.int32),
            "token_indices K dimension must be positive",
        ),
        (
            "token_indices",
            torch.empty((2, 4), dtype=torch.int32),
            "token_indices query dimension must match req_id",
        ),
    ],
)
def test_rank_shape_and_zero_dimension_failures(
    name: str,
    replacement: torch.Tensor,
    match: str,
) -> None:
    req_id, block_table, token_indices = _base_case()
    values = {
        "req_id": req_id,
        "block_table": block_table,
        "token_indices": token_indices,
    }
    values[name] = replacement
    with pytest.raises(ValueError, match=match):
        dsa_sparse_index_remap_reference(
            values["req_id"],
            values["block_table"],
            values["token_indices"],
            block_size=_BLOCK_SIZE,
        )


@pytest.mark.parametrize(
    ("block_size", "error", "match"),
    [
        (True, TypeError, "block_size must be a Python int"),
        (64.0, TypeError, "block_size must be a Python int"),
        (0, ValueError, "block_size must be positive"),
        (-1, ValueError, "block_size must be positive"),
        (torch.iinfo(torch.int64).max + 1, ValueError, "block_size must fit"),
    ],
)
def test_invalid_block_size_types_and_ranges(
    block_size: Any,
    error: type[Exception],
    match: str,
) -> None:
    req_id, block_table, token_indices = _base_case()
    with pytest.raises(error, match=match):
        dsa_sparse_index_remap_reference(req_id, block_table, token_indices, block_size=block_size)


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_return_valid_counts_requires_python_bool(value: Any) -> None:
    req_id, block_table, token_indices = _base_case()
    with pytest.raises(TypeError, match="return_valid_counts must be a Python bool"):
        dsa_sparse_index_remap_reference(
            req_id,
            block_table,
            token_indices,
            block_size=_BLOCK_SIZE,
            return_valid_counts=value,
        )


def test_partial_workspace_arguments_are_rejected() -> None:
    req_id, block_table, token_indices = _base_case()
    workspace_ids = torch.full_like(req_id, -1)
    workspace_starts = torch.tensor([0], dtype=torch.int32)

    with pytest.raises(ValueError, match="must be provided together"):
        dsa_sparse_index_remap_reference(
            req_id,
            block_table,
            token_indices,
            block_size=_BLOCK_SIZE,
            prefill_workspace_request_ids=workspace_ids,
        )
    with pytest.raises(ValueError, match="must be provided together"):
        dsa_sparse_index_remap_reference(
            req_id,
            block_table,
            token_indices,
            block_size=_BLOCK_SIZE,
            prefill_workspace_starts=workspace_starts,
        )


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("ids_non_tensor", TypeError, "prefill_workspace_request_ids must be a torch.Tensor"),
        ("starts_non_tensor", TypeError, "prefill_workspace_starts must be a torch.Tensor"),
        ("ids_dtype", TypeError, "prefill_workspace_request_ids dtype must be torch.int32"),
        ("starts_dtype", TypeError, "prefill_workspace_starts dtype must be torch.int32"),
        ("ids_rank", ValueError, "prefill_workspace_request_ids must be rank 1"),
        ("starts_rank", ValueError, "prefill_workspace_starts must be rank 1"),
        ("ids_q", ValueError, "prefill_workspace_request_ids length must match req_id"),
        ("starts_zero", ValueError, "prefill_workspace_starts must contain at least one entry"),
        ("ids_layout", ValueError, "prefill_workspace_request_ids must be contiguous"),
        ("starts_layout", ValueError, "prefill_workspace_starts must be contiguous"),
    ],
)
def test_workspace_type_shape_dtype_and_layout_failures(
    case: str,
    error: type[Exception],
    match: str,
) -> None:
    req_id, block_table, token_indices = _base_case()
    workspace_ids: Any = torch.tensor([-1, 0, 0], dtype=torch.int32)
    workspace_starts: Any = torch.tensor([100], dtype=torch.int32)
    if case == "ids_non_tensor":
        workspace_ids = [-1, 0, 0]
    elif case == "starts_non_tensor":
        workspace_starts = [100]
    elif case == "ids_dtype":
        workspace_ids = workspace_ids.to(torch.int64)
    elif case == "starts_dtype":
        workspace_starts = workspace_starts.to(torch.int64)
    elif case == "ids_rank":
        workspace_ids = workspace_ids.reshape(1, 3)
    elif case == "starts_rank":
        workspace_starts = workspace_starts.reshape(1, 1)
    elif case == "ids_q":
        workspace_ids = torch.tensor([-1, 0], dtype=torch.int32)
    elif case == "starts_zero":
        workspace_starts = torch.empty((0,), dtype=torch.int32)
    elif case == "ids_layout":
        workspace_ids = torch.tensor([-1, 9, 0, 9, 0, 9], dtype=torch.int32)[::2]
    elif case == "starts_layout":
        workspace_starts = torch.tensor([100, 9, 200, 9], dtype=torch.int32)[::2]

    with pytest.raises(error, match=match):
        dsa_sparse_index_remap_reference(
            req_id,
            block_table,
            token_indices,
            block_size=_BLOCK_SIZE,
            prefill_workspace_request_ids=workspace_ids,
            prefill_workspace_starts=workspace_starts,
        )


@pytest.mark.parametrize("name", ["block_table", "token_indices"])
def test_noncontiguous_inner_dimensions_are_rejected(name: str) -> None:
    req_id, block_table, token_indices = _base_case()
    if name == "block_table":
        block_table = torch.empty((2, 8), dtype=torch.int32)[:, ::2]
    else:
        token_indices = torch.empty((3, 8), dtype=torch.int32)[:, ::2]
    with pytest.raises(ValueError, match=f"{name} innermost stride must be 1"):
        dsa_sparse_index_remap_reference(req_id, block_table, token_indices, block_size=_BLOCK_SIZE)


def test_overlapping_rows_and_aliased_inputs_are_rejected() -> None:
    req_id, block_table, token_indices = _base_case()
    overlapping = torch.arange(8, dtype=torch.int32).as_strided((3, 3), (1, 1))
    with pytest.raises(ValueError, match="outer stride must be at least its row width"):
        dsa_sparse_index_remap_reference(
            req_id,
            block_table,
            overlapping,
            block_size=_BLOCK_SIZE,
        )

    backing = torch.arange(80, dtype=torch.int32)
    aliased_req = backing[:3]
    aliased_tokens = backing[20:32].reshape(3, 4)
    with pytest.raises(ValueError, match="req_id must not alias token_indices"):
        dsa_sparse_index_remap_reference(
            aliased_req,
            block_table,
            aliased_tokens,
            block_size=_BLOCK_SIZE,
        )


def test_meta_tensors_are_rejected() -> None:
    req_id = torch.empty((1,), dtype=torch.int32, device="meta")
    block_table = torch.empty((1, 1), dtype=torch.int32, device="meta")
    token_indices = torch.empty((1, 1), dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="meta tensors are not supported"):
        dsa_sparse_index_remap_reference(req_id, block_table, token_indices, block_size=_BLOCK_SIZE)


def test_device_mismatch_is_rejected_with_cpu_constructible_fake_tensors() -> None:
    with FakeTensorMode():
        req_id = torch.zeros((1,), dtype=torch.int32, device="cpu")
        block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
        token_indices = torch.zeros((1, 1), dtype=torch.int32, device="cpu")
        with pytest.raises(ValueError, match="block_table must be on the same device as req_id"):
            dsa_sparse_index_remap_reference(
                req_id, block_table, token_indices, block_size=_BLOCK_SIZE
            )


def test_sparse_layout_is_rejected() -> None:
    req_id, _, token_indices = _base_case()
    block_table = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]),
        torch.tensor([1], dtype=torch.int32),
        size=(2, 4),
        dtype=torch.int32,
    )
    with pytest.raises(ValueError, match="block_table must have torch.strided layout"):
        dsa_sparse_index_remap_reference(req_id, block_table, token_indices, block_size=_BLOCK_SIZE)


def test_import_has_no_vllm_triton_cuda_registry_or_cache_side_effects(tmp_path: Path) -> None:
    script = """
import json
import sys
import torch

before_cuda = torch.cuda.is_initialized()
before_files = sorted(str(path) for path in __import__('pathlib').Path('.').rglob('*'))
import profiling.runners.attention.dsa_sparse_index_remap_reference as module
after_files = sorted(str(path) for path in __import__('pathlib').Path('.').rglob('*'))
print(json.dumps({
    'cuda_before': before_cuda,
    'cuda_after': torch.cuda.is_initialized(),
    'vllm': any(name == 'vllm' or name.startswith('vllm.') for name in sys.modules),
    'triton': any(name == 'triton' or name.startswith('triton.') for name in sys.modules),
    'registry': any(name.startswith('profiling.kernels') for name in sys.modules),
    'files_changed': before_files != after_files,
    'export': module.__all__,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)

    assert result == {
        "cuda_before": False,
        "cuda_after": False,
        "vllm": False,
        "triton": False,
        "registry": False,
        "files_changed": False,
        "export": ["dsa_sparse_index_remap_reference"],
    }
