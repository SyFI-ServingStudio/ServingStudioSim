"""Torch semantics for request-local to global DSA sparse-index remapping."""

from __future__ import annotations

import torch

__all__ = ["dsa_sparse_index_remap_reference"]

_INT32_MAX = torch.iinfo(torch.int32).max
_INT64_MAX = torch.iinfo(torch.int64).max
_ROW_CHUNK_SIZE = 64


def dsa_sparse_index_remap_reference(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    block_size: int,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
    return_valid_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Map request-local token indices to fresh contiguous global indices.

    Negative token indices and indices whose local block is outside the block
    table width map to ``-1``. Global rows use the selected physical page from
    ``block_table``; workspace rows instead add the local token index to their
    workspace start. Duplicate slots and input order are preserved. Optional
    counts include every post-local-block-mask slot, including duplicates.

    The callable has no request-context-length input. Consequently, a positive
    local index is mapped whenever its block fits the table, even if a higher
    layer would consider it beyond the request's current context. Likewise,
    positive physical page IDs cannot be checked against an external cache pool
    that is not part of this interface; that capacity check remains the caller's
    responsibility.

    This mathematical boundary follows vLLM v0.23.0 commit
    0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665,
    ``vllm/v1/attention/backends/mla/sparse_utils.py``. It never imports or
    calls vLLM, Triton, CUDA extensions, or profiling code. Row chunking bounds
    temporary gathers independently of ``Q`` without changing row semantics.
    """
    tensors = _validate(
        req_id,
        block_table,
        token_indices,
        block_size=block_size,
        prefill_workspace_request_ids=prefill_workspace_request_ids,
        prefill_workspace_starts=prefill_workspace_starts,
        return_valid_counts=return_valid_counts,
    )
    workspace_ids, workspace_starts = tensors

    # Validate every selected page/start and the final int32 range before
    # allocating or populating result storage. Invalid slots never index the
    # block table, so unused negative padding is deliberately ignored.
    _validate_selected_mappings(
        req_id,
        block_table,
        token_indices,
        block_size=block_size,
        workspace_ids=workspace_ids,
        workspace_starts=workspace_starts,
    )

    num_queries, selected_k = token_indices.shape
    output = torch.empty(
        (num_queries, selected_k),
        dtype=torch.int32,
        device=token_indices.device,
    )
    valid_counts = (
        torch.empty((num_queries,), dtype=torch.int32, device=token_indices.device)
        if return_valid_counts
        else None
    )

    for row_start in range(0, num_queries, _ROW_CHUNK_SIZE):
        row_end = min(row_start + _ROW_CHUNK_SIZE, num_queries)
        local_indices = token_indices[row_start:row_end].to(torch.int64)
        local_blocks = torch.div(local_indices, block_size, rounding_mode="floor")
        valid = (local_indices >= 0) & (local_blocks < block_table.shape[1])
        chunk_output = torch.full_like(local_indices, -1)

        if workspace_ids is None:
            global_rows = torch.ones(
                (row_end - row_start,),
                dtype=torch.bool,
                device=token_indices.device,
            )
        else:
            global_rows = workspace_ids[row_start:row_end] == -1

        global_slots = valid & global_rows[:, None]
        _write_global_slots(
            chunk_output,
            global_slots,
            local_indices,
            local_blocks,
            req_id[row_start:row_end],
            block_table,
            block_size=block_size,
        )

        if workspace_ids is not None:
            workspace_slots = valid & ~global_rows[:, None]
            _write_workspace_slots(
                chunk_output,
                workspace_slots,
                local_indices,
                workspace_ids[row_start:row_end],
                workspace_starts,
            )

        output[row_start:row_end].copy_(chunk_output.to(torch.int32))
        if valid_counts is not None:
            valid_counts[row_start:row_end].copy_(valid.sum(dim=1).to(torch.int32))

    if valid_counts is None:
        return output
    return output, valid_counts


def _validate(
    req_id: object,
    block_table: object,
    token_indices: object,
    *,
    block_size: object,
    prefill_workspace_request_ids: object,
    prefill_workspace_starts: object,
    return_valid_counts: object,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    for name, tensor in (
        ("req_id", req_id),
        ("block_table", block_table),
        ("token_indices", token_indices),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert isinstance(req_id, torch.Tensor)
    assert isinstance(block_table, torch.Tensor)
    assert isinstance(token_indices, torch.Tensor)

    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise TypeError("block_size must be a Python int")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_size > _INT64_MAX:
        raise ValueError("block_size must fit in a signed int64")
    if type(return_valid_counts) is not bool:
        raise TypeError("return_valid_counts must be a Python bool")

    workspace_mode = (prefill_workspace_request_ids is not None) or (
        prefill_workspace_starts is not None
    )
    if workspace_mode and (
        prefill_workspace_request_ids is None or prefill_workspace_starts is None
    ):
        raise ValueError(
            "prefill_workspace_request_ids and prefill_workspace_starts must be provided together"
        )
    if workspace_mode:
        if not isinstance(prefill_workspace_request_ids, torch.Tensor):
            raise TypeError("prefill_workspace_request_ids must be a torch.Tensor")
        if not isinstance(prefill_workspace_starts, torch.Tensor):
            raise TypeError("prefill_workspace_starts must be a torch.Tensor")

    workspace_ids = prefill_workspace_request_ids
    workspace_starts = prefill_workspace_starts
    assert workspace_ids is None or isinstance(workspace_ids, torch.Tensor)
    assert workspace_starts is None or isinstance(workspace_starts, torch.Tensor)

    all_tensors = [
        ("req_id", req_id),
        ("block_table", block_table),
        ("token_indices", token_indices),
    ]
    if workspace_ids is not None and workspace_starts is not None:
        all_tensors.extend(
            (
                ("prefill_workspace_request_ids", workspace_ids),
                ("prefill_workspace_starts", workspace_starts),
            )
        )

    _validate_shapes(req_id, block_table, token_indices, workspace_ids, workspace_starts)
    _validate_dtypes_devices_and_layouts(all_tensors, req_id.device)
    _validate_no_aliasing(all_tensors)
    _validate_metadata_bounds(req_id, block_table, workspace_ids, workspace_starts)
    return workspace_ids, workspace_starts


def _validate_shapes(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    workspace_ids: torch.Tensor | None,
    workspace_starts: torch.Tensor | None,
) -> None:
    if req_id.ndim != 1:
        raise ValueError(f"req_id must be rank 1, got rank {req_id.ndim}")
    if req_id.shape[0] <= 0:
        raise ValueError("req_id must contain at least one query")

    if block_table.ndim != 2:
        raise ValueError(f"block_table must be rank 2, got rank {block_table.ndim}")
    if any(dimension <= 0 for dimension in block_table.shape):
        raise ValueError(f"block_table dimensions must be positive, got {tuple(block_table.shape)}")

    if token_indices.ndim != 2:
        raise ValueError(f"token_indices must be rank 2, got rank {token_indices.ndim}")
    if token_indices.shape[0] != req_id.shape[0]:
        raise ValueError(
            "token_indices query dimension must match req_id, "
            f"got {token_indices.shape[0]} and {req_id.shape[0]}"
        )
    if token_indices.shape[1] <= 0:
        raise ValueError("token_indices K dimension must be positive")

    if workspace_ids is None or workspace_starts is None:
        return
    if workspace_ids.ndim != 1:
        raise ValueError(
            f"prefill_workspace_request_ids must be rank 1, got rank {workspace_ids.ndim}"
        )
    if workspace_ids.shape[0] != req_id.shape[0]:
        raise ValueError(
            "prefill_workspace_request_ids length must match req_id, "
            f"got {workspace_ids.shape[0]} and {req_id.shape[0]}"
        )
    if workspace_starts.ndim != 1:
        raise ValueError(
            f"prefill_workspace_starts must be rank 1, got rank {workspace_starts.ndim}"
        )
    if workspace_starts.shape[0] <= 0:
        raise ValueError("prefill_workspace_starts must contain at least one entry")


def _validate_dtypes_devices_and_layouts(
    tensors: list[tuple[str, torch.Tensor]],
    expected_device: torch.device,
) -> None:
    for name, tensor in tensors:
        if tensor.dtype is not torch.int32:
            raise TypeError(f"{name} dtype must be torch.int32, got {tensor.dtype}")
        if tensor.device.type == "meta":
            raise ValueError(f"{name} must be concrete; meta tensors are not supported")
        if tensor.device != expected_device:
            raise ValueError(
                f"{name} must be on the same device as req_id, "
                f"got {tensor.device} and {expected_device}"
            )
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")

    req_id = tensors[0][1]
    block_table = tensors[1][1]
    token_indices = tensors[2][1]
    if req_id.stride(0) != 1:
        raise ValueError("req_id must have contiguous inner stride 1")
    _validate_row_major_view("block_table", block_table)
    _validate_row_major_view("token_indices", token_indices)

    for name, tensor in tensors[3:]:
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _validate_row_major_view(name: str, tensor: torch.Tensor) -> None:
    if tensor.stride(1) != 1:
        raise ValueError(f"{name} innermost stride must be 1")
    if tensor.stride(0) < tensor.shape[1]:
        raise ValueError(f"{name} outer stride must be at least its row width to avoid overlap")
    if int(torch._debug_has_internal_overlap(tensor)) == 1:
        raise ValueError(f"{name} must not have internal overlap")


def _validate_no_aliasing(tensors: list[tuple[str, torch.Tensor]]) -> None:
    for left_index, (left_name, left) in enumerate(tensors):
        for right_name, right in tensors[left_index + 1 :]:
            if torch._C._overlaps(left, right):
                raise ValueError(f"{left_name} must not alias {right_name}")


def _validate_metadata_bounds(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    workspace_ids: torch.Tensor | None,
    workspace_starts: torch.Tensor | None,
) -> None:
    if bool(((req_id < 0) | (req_id >= block_table.shape[0])).any().item()):
        raise ValueError(f"req_id values must be in [0, {block_table.shape[0] - 1}]")

    if workspace_ids is None or workspace_starts is None:
        return
    invalid_workspace_ids = (workspace_ids < -1) | (workspace_ids >= workspace_starts.shape[0])
    if bool(invalid_workspace_ids.any().item()):
        raise ValueError(
            "prefill_workspace_request_ids values must be -1 or in "
            f"[0, {workspace_starts.shape[0] - 1}]"
        )
    if bool((workspace_starts < 0).any().item()):
        raise ValueError("prefill_workspace_starts values must be nonnegative")


def _validate_selected_mappings(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    block_size: int,
    workspace_ids: torch.Tensor | None,
    workspace_starts: torch.Tensor | None,
) -> None:
    for row_start in range(0, token_indices.shape[0], _ROW_CHUNK_SIZE):
        row_end = min(row_start + _ROW_CHUNK_SIZE, token_indices.shape[0])
        local_indices = token_indices[row_start:row_end].to(torch.int64)
        local_blocks = torch.div(local_indices, block_size, rounding_mode="floor")
        valid = (local_indices >= 0) & (local_blocks < block_table.shape[1])

        if workspace_ids is None:
            global_rows = torch.ones(
                (row_end - row_start,),
                dtype=torch.bool,
                device=token_indices.device,
            )
        else:
            global_rows = workspace_ids[row_start:row_end] == -1

        global_slots = valid & global_rows[:, None]
        if bool(global_slots.any().item()):
            row_positions, column_positions = global_slots.nonzero(as_tuple=True)
            request_rows = req_id[row_start:row_end].to(torch.int64)[row_positions]
            block_columns = local_blocks[row_positions, column_positions]
            pages = block_table[request_rows, block_columns].to(torch.int64)
            if bool((pages < 0).any().item()):
                raise ValueError("selected block_table physical page IDs must be nonnegative")
            offsets = local_indices[row_positions, column_positions].remainder(block_size)
            maximum_pages = torch.div(
                _INT32_MAX - offsets,
                block_size,
                rounding_mode="floor",
            )
            if bool((pages > maximum_pages).any().item()):
                raise ValueError("selected global mapped index exceeds int32 range")

        if workspace_ids is not None and workspace_starts is not None:
            workspace_slots = valid & ~global_rows[:, None]
            if bool(workspace_slots.any().item()):
                row_positions, column_positions = workspace_slots.nonzero(as_tuple=True)
                workspace_rows = workspace_ids[row_start:row_end].to(torch.int64)[row_positions]
                starts = workspace_starts.to(torch.int64)[workspace_rows]
                selected_indices = local_indices[row_positions, column_positions]
                if bool((starts > _INT32_MAX - selected_indices).any().item()):
                    raise ValueError("selected workspace mapped index exceeds int32 range")


def _write_global_slots(
    output: torch.Tensor,
    mask: torch.Tensor,
    local_indices: torch.Tensor,
    local_blocks: torch.Tensor,
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_size: int,
) -> None:
    if not bool(mask.any().item()):
        return
    row_positions, column_positions = mask.nonzero(as_tuple=True)
    request_rows = req_id.to(torch.int64)[row_positions]
    block_columns = local_blocks[row_positions, column_positions]
    pages = block_table[request_rows, block_columns].to(torch.int64)
    offsets = local_indices[row_positions, column_positions].remainder(block_size)
    output[row_positions, column_positions] = pages * block_size + offsets


def _write_workspace_slots(
    output: torch.Tensor,
    mask: torch.Tensor,
    local_indices: torch.Tensor,
    workspace_ids: torch.Tensor,
    workspace_starts: torch.Tensor | None,
) -> None:
    if not bool(mask.any().item()):
        return
    assert workspace_starts is not None
    row_positions, column_positions = mask.nonzero(as_tuple=True)
    workspace_rows = workspace_ids.to(torch.int64)[row_positions]
    starts = workspace_starts.to(torch.int64)[workspace_rows]
    output[row_positions, column_positions] = (
        starts + local_indices[row_positions, column_positions]
    )
