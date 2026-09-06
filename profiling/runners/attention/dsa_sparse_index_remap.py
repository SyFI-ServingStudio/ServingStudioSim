"""Torch profiler for GLM-5.2 request-local sparse-index remapping.

The timed callable is a vectorized, row-chunked semantic composite. Request
metadata parsing, page-table/index construction, workspace derivation, output
allocation, and reference correctness remain outside timing. The later native
backend will time vLLM's Triton wrapper as a separate registration row.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "dsa_sparse_index_remap:torch"
_VLLM_BACKEND = "dsa_sparse_index_remap:vllm_triton"
_REQUIRED_GPU = "NVIDIA H200"
_SELECTED_K = 2048
_BLOCK_SIZE = 64
_MAX_BLOCKS_PER_REQUEST = 16384
_MAX_LOCAL_SPAN = _BLOCK_SIZE * _MAX_BLOCKS_PER_REQUEST
_MAX_QUERIES = 16384
_INDEX_DTYPE = "int32"
_SOURCE_BLOCK_N = 128
_ROW_CHUNK_SIZE = 64
_OPERAND_CHECK_CHUNK_SIZE = 256
_INT32_MAX = 2**31 - 1

_UINT = r"(?:0|[1-9][0-9]*)"
_UNIFORM_RE = re.compile(rf"u:({_UINT})x({_UINT})\Z")
_RAMP_RE = re.compile(rf"r:({_UINT})\.\.({_UINT})\Z")
_CLIPPED_RE = re.compile(rf"c:({_UINT})\.\.({_UINT})@({_UINT})\Z")
_GROUP_RE = re.compile(rf"g:\((?P<group>{_UINT}(?:,{_UINT})*)\)x(?P<groups>{_UINT})\Z")
_WORKSPACE_SUFFIX_RE = re.compile(rf"suffix:(?P<decode>{_UINT})@(?P<chunks>{_UINT}(?:,{_UINT})*)\Z")

_INDEX_DISTRIBUTIONS = frozenset(
    {
        "recent_contiguous",
        "unique_scattered_blocks",
        "clustered_blocks",
        "uniform_stride",
    }
)
_PAGE_TABLE_MAPPINGS = frozenset(
    {
        "request_contiguous",
        "interleaved_requests",
        "reverse_within_request",
        "fixed_permutation",
    }
)


@dataclass(frozen=True)
class _WorkspacePartition:
    decode_requests: int
    chunk_sizes: tuple[int, ...]


@dataclass(frozen=True)
class _ValidatedArgs:
    num_queries: int
    num_requests: int
    selected_k: int
    block_size: int
    max_blocks_per_request: int
    request_row_counts: tuple[int, ...]
    request_ids: tuple[int, ...]
    local_span_lengths: tuple[int, ...]
    valid_counts: tuple[int, ...]
    index_distribution: str
    page_table_mapping: str
    workspace_partition: _WorkspacePartition | None
    return_valid_counts: bool


@dataclass(frozen=True)
class _Operands:
    req_id: Any
    block_table: Any
    token_indices: Any
    prefill_workspace_request_ids: Any | None
    prefill_workspace_starts: Any | None
    output: Any
    counts: Any | None


def _encode_vector(
    values: Sequence[int],
    *,
    name: str,
    minimum: int,
    maximum: int,
    allow_clipped: bool = False,
    selected_k: int | None = None,
) -> str:
    """Return the unique canonical compact encoding for an integer vector."""
    vector = tuple(values)
    if not vector:
        raise ValueError(f"{name} must contain at least one value")
    if any(type(value) is not int for value in vector):
        raise TypeError(f"{name} values must be integers")
    if any(value < minimum or value > maximum for value in vector):
        raise ValueError(f"{name} values must be in {minimum}..{maximum}")

    if all(value == vector[0] for value in vector):
        return f"u:{vector[0]}x{len(vector)}"

    if all(vector[index] == vector[0] + index for index in range(len(vector))):
        return f"r:{vector[0]}..{vector[-1]}"

    if allow_clipped:
        if selected_k is None:
            raise ValueError(f"{name} clipped encoding requires selected_k")
        unclipped_last = vector[0] + len(vector) - 1
        clipped = tuple(min(vector[0] + index, selected_k) for index in range(len(vector)))
        if vector[0] < selected_k < unclipped_last <= maximum and vector == clipped:
            return f"c:{vector[0]}..{unclipped_last}@{selected_k}"

    period = len(vector)
    for candidate in range(1, len(vector) + 1):
        if len(vector) % candidate:
            continue
        if all(vector[index] == vector[index % candidate] for index in range(len(vector))):
            period = candidate
            break
    group = ",".join(str(value) for value in vector[:period])
    return f"g:({group})x{len(vector) // period}"


def _decode_vector(
    encoded: str,
    *,
    name: str,
    num_rows: int,
    minimum: int,
    maximum: int,
    allow_clipped: bool = False,
    selected_k: int | None = None,
) -> tuple[int, ...]:
    """Parse one vector and reject every noncanonical equivalent spelling."""
    if not isinstance(encoded, str):
        raise TypeError(f"{name} must be a string")
    if not encoded.isascii():
        raise ValueError(f"{name} must use ASCII compact syntax")

    values: tuple[int, ...]
    if match := _UNIFORM_RE.fullmatch(encoded):
        value, rows = (int(component) for component in match.groups())
        if rows <= 0:
            raise ValueError(f"{name} uniform row count must be positive")
        if rows != num_rows:
            raise ValueError(f"{name} expands to {rows} rows, expected {num_rows}")
        values = (value,) * rows
    elif match := _RAMP_RE.fullmatch(encoded):
        first, last = (int(component) for component in match.groups())
        if first >= last:
            raise ValueError(f"{name} ramp must be a strict +1 ramp")
        rows = last - first + 1
        if rows != num_rows:
            raise ValueError(f"{name} expands to {rows} rows, expected {num_rows}")
        values = tuple(range(first, last + 1))
    elif allow_clipped and (match := _CLIPPED_RE.fullmatch(encoded)):
        first, last, cap = (int(component) for component in match.groups())
        if selected_k is None or cap != selected_k:
            raise ValueError(f"{name} clipped-ramp cap must equal selected_k")
        if not first < selected_k < last <= maximum:
            raise ValueError(f"{name} clipped ramp requires first < selected_k < last <= {maximum}")
        rows = last - first + 1
        if rows != num_rows:
            raise ValueError(f"{name} expands to {rows} rows, expected {num_rows}")
        values = tuple(min(value, selected_k) for value in range(first, last + 1))
    elif match := _GROUP_RE.fullmatch(encoded):
        group = tuple(int(component) for component in match.group("group").split(","))
        groups = int(match.group("groups"))
        if groups <= 0:
            raise ValueError(f"{name} tuple repetition must be positive")
        rows = len(group) * groups
        if rows != num_rows:
            raise ValueError(f"{name} expands to {rows} rows, expected {num_rows}")
        values = group * groups
    else:
        forms = "u:, r:, c:, or g:" if allow_clipped else "u:, r:, or g:"
        raise ValueError(f"{name} must use canonical {forms} compact syntax")

    if any(value < minimum or value > maximum for value in values):
        raise ValueError(f"{name} values must be in {minimum}..{maximum}")
    canonical = _encode_vector(
        values,
        name=name,
        minimum=minimum,
        maximum=maximum,
        allow_clipped=allow_clipped,
        selected_k=selected_k,
    )
    if canonical != encoded:
        raise ValueError(f"{name} is noncanonical; canonical encoding is {canonical!r}")
    return values


def _decode_workspace_partition(
    encoded: str,
    *,
    num_requests: int,
) -> _WorkspacePartition | None:
    if not isinstance(encoded, str):
        raise TypeError("workspace_partition must be a string")
    if not encoded.isascii():
        raise ValueError("workspace_partition must use ASCII compact syntax")
    if encoded == "none":
        return None
    match = _WORKSPACE_SUFFIX_RE.fullmatch(encoded)
    if match is None:
        raise ValueError("workspace_partition must be 'none' or canonical suffix:<D>@<c0>,...,<cn>")
    decode_requests = int(match.group("decode"))
    chunk_sizes = tuple(int(value) for value in match.group("chunks").split(","))
    if not 0 <= decode_requests < num_requests:
        raise ValueError(f"workspace suffix D must be in 0..{num_requests - 1}")
    if any(size <= 0 for size in chunk_sizes):
        raise ValueError("workspace suffix chunk sizes must be positive")
    expected_prefill = num_requests - decode_requests
    if sum(chunk_sizes) != expected_prefill:
        raise ValueError(
            "workspace suffix chunk sizes must sum to num_requests-D "
            f"({expected_prefill}), got {sum(chunk_sizes)}"
        )
    canonical = f"suffix:{decode_requests}@{','.join(str(size) for size in chunk_sizes)}"
    if canonical != encoded:
        raise ValueError(
            f"workspace_partition is noncanonical; canonical encoding is {canonical!r}"
        )
    return _WorkspacePartition(decode_requests, chunk_sizes)


def _validate_args(
    *,
    num_queries: int,
    num_requests: int,
    selected_k: int,
    block_size: int,
    max_blocks_per_request: int,
    request_row_counts: str,
    local_span_lengths: str,
    valid_counts: str,
    index_distribution: str,
    page_table_mapping: str,
    workspace_partition: str,
    return_valid_counts: bool,
    index_dtype: str,
) -> _ValidatedArgs:
    integers = {
        "num_queries": num_queries,
        "num_requests": num_requests,
        "selected_k": selected_k,
        "block_size": block_size,
        "max_blocks_per_request": max_blocks_per_request,
    }
    for name, value in integers.items():
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
    for name, value in (
        ("index_distribution", index_distribution),
        ("page_table_mapping", page_table_mapping),
        ("index_dtype", index_dtype),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
    if type(return_valid_counts) is not bool:
        raise TypeError("return_valid_counts must be a Python bool")

    if not 1 <= num_queries <= _MAX_QUERIES:
        raise ProfilerNotImplemented(f"num_queries must be in 1..{_MAX_QUERIES}")
    if not 1 <= num_requests <= min(num_queries, 256):
        raise ProfilerNotImplemented("num_requests must be in 1..min(num_queries, 256)")
    for name, actual, required in (
        ("selected_k", selected_k, _SELECTED_K),
        ("block_size", block_size, _BLOCK_SIZE),
    ):
        if actual != required:
            raise ProfilerNotImplemented(f"{name} must be {required}, got {actual}")
    if not 1 <= max_blocks_per_request <= _MAX_BLOCKS_PER_REQUEST:
        raise ProfilerNotImplemented(
            f"max_blocks_per_request must be in 1..{_MAX_BLOCKS_PER_REQUEST}, "
            f"got {max_blocks_per_request}"
        )
    if index_dtype != _INDEX_DTYPE:
        raise ProfilerNotImplemented(f"index_dtype must be {_INDEX_DTYPE}")
    if index_distribution not in _INDEX_DISTRIBUTIONS:
        modes = ", ".join(sorted(_INDEX_DISTRIBUTIONS))
        raise ProfilerNotImplemented(f"index_distribution must be one of: {modes}")
    if page_table_mapping not in _PAGE_TABLE_MAPPINGS:
        modes = ", ".join(sorted(_PAGE_TABLE_MAPPINGS))
        raise ProfilerNotImplemented(f"page_table_mapping must be one of: {modes}")

    row_counts = _decode_vector(
        request_row_counts,
        name="request_row_counts",
        num_rows=num_requests,
        minimum=1,
        maximum=num_queries,
    )
    if sum(row_counts) != num_queries:
        raise ValueError(
            f"request_row_counts must sum to num_queries {num_queries}, got {sum(row_counts)}"
        )
    spans = _decode_vector(
        local_span_lengths,
        name="local_span_lengths",
        num_rows=num_queries,
        minimum=0,
        maximum=block_size * max_blocks_per_request,
    )
    counts = _decode_vector(
        valid_counts,
        name="valid_counts",
        num_rows=num_queries,
        minimum=0,
        maximum=block_size * max_blocks_per_request,
        allow_clipped=True,
        selected_k=selected_k,
    )
    for row, (count, span) in enumerate(zip(counts, spans, strict=True)):
        if count > min(selected_k, span):
            raise ValueError(
                f"valid_counts row {row} must be <= min(selected_k, local_span_length)"
            )

    partition = _decode_workspace_partition(
        workspace_partition,
        num_requests=num_requests,
    )
    request_ids = tuple(
        request for request, row_count in enumerate(row_counts) for _ in range(row_count)
    )
    return _ValidatedArgs(
        num_queries=num_queries,
        num_requests=num_requests,
        selected_k=selected_k,
        block_size=block_size,
        max_blocks_per_request=max_blocks_per_request,
        request_row_counts=row_counts,
        request_ids=request_ids,
        local_span_lengths=spans,
        valid_counts=counts,
        index_distribution=index_distribution,
        page_table_mapping=page_table_mapping,
        workspace_partition=partition,
        return_valid_counts=return_valid_counts,
    )


def _require_h200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"CUDA is required for {_BACKEND}")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(f"{_BACKEND} requires {_REQUIRED_GPU}, got {gpu_name!r}")


def _require_b200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"CUDA is required for {_VLLM_BACKEND}")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != "NVIDIA B200":
        raise ProfilerNotImplemented(f"{_VLLM_BACKEND} requires NVIDIA B200, got {gpu_name!r}")


def _coprime_stride(size: int, preferred: int) -> int:
    if size <= 1:
        return 1
    stride = preferred % size or 1
    while math.gcd(stride, size) != 1:
        stride += 1
    return stride


def _build_block_table(torch: Any, validated: _ValidatedArgs, *, device: Any) -> Any:
    table = torch.empty(
        (validated.num_requests, validated.max_blocks_per_request),
        dtype=torch.int32,
        device=device,
    )
    blocks = torch.arange(
        validated.max_blocks_per_request,
        dtype=torch.int64,
        device=device,
    )
    total_pages = validated.num_requests * validated.max_blocks_per_request
    permutation_stride = _coprime_stride(total_pages, 65537)
    permutation_offset = 104729 % total_pages
    for request in range(validated.num_requests):
        if validated.page_table_mapping == "request_contiguous":
            pages = request * validated.max_blocks_per_request + blocks
        elif validated.page_table_mapping == "interleaved_requests":
            pages = blocks * validated.num_requests + request
        elif validated.page_table_mapping == "reverse_within_request":
            pages = (
                request * validated.max_blocks_per_request
                + validated.max_blocks_per_request
                - 1
                - blocks
            )
        else:
            linear = request * validated.max_blocks_per_request + blocks
            pages = (linear * permutation_stride + permutation_offset) % total_pages
        table[request].copy_(pages.to(torch.int32))
    return table


def _row_local_indices(
    torch: Any,
    *,
    row: int,
    count: int,
    span: int,
    distribution: str,
    block_size: int,
    device: Any,
) -> Any:
    if count == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    positions = torch.arange(count, dtype=torch.int64, device=device)
    if distribution == "recent_contiguous":
        return positions + span - count
    if distribution == "uniform_stride":
        return torch.div(positions * span, count, rounding_mode="floor")
    if distribution == "unique_scattered_blocks":
        stride = _coprime_stride(span, block_size + 1 + 2 * row)
        return (positions * stride + 17 * row) % span

    num_blocks = (span + block_size - 1) // block_size
    start = ((row * 11) % num_blocks) * block_size
    return (positions + start) % span


def _derive_workspace_metadata(
    validated: _ValidatedArgs,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    partition = validated.workspace_partition
    if partition is None:
        return None, None

    workspace_ids = tuple(
        -1 if request < partition.decode_requests else request - partition.decode_requests
        for request in validated.request_ids
    )
    max_span_by_request: list[int] = []
    row_offset = 0
    for row_count in validated.request_row_counts:
        request_spans = validated.local_span_lengths[row_offset : row_offset + row_count]
        max_span_by_request.append(max(request_spans))
        row_offset += row_count

    starts: list[int] = []
    compact_request = 0
    for chunk_size in partition.chunk_sizes:
        chunk_offset = 0
        for _ in range(chunk_size):
            starts.append(chunk_offset)
            source_request = partition.decode_requests + compact_request
            chunk_offset += max_span_by_request[source_request]
            if chunk_offset > _INT32_MAX:
                raise ValueError("prefill workspace starts exceed int32 range")
            compact_request += 1
    return workspace_ids, tuple(starts)


def _build_operands(torch: Any, validated: _ValidatedArgs, *, device: Any) -> _Operands:
    req_id = torch.tensor(validated.request_ids, dtype=torch.int32, device=device)
    block_table = _build_block_table(torch, validated, device=device)
    token_indices = torch.full(
        (validated.num_queries, validated.selected_k),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for row, (count, span) in enumerate(
        zip(validated.valid_counts, validated.local_span_lengths, strict=True)
    ):
        if count == 0:
            continue
        indices = _row_local_indices(
            torch,
            row=row,
            count=count,
            span=span,
            distribution=validated.index_distribution,
            block_size=validated.block_size,
            device=device,
        )
        token_indices[row, :count].copy_(indices.to(torch.int32))

    workspace_ids_values, workspace_starts_values = _derive_workspace_metadata(validated)
    workspace_ids = (
        None
        if workspace_ids_values is None
        else torch.tensor(workspace_ids_values, dtype=torch.int32, device=device)
    )
    workspace_starts = (
        None
        if workspace_starts_values is None
        else torch.tensor(workspace_starts_values, dtype=torch.int32, device=device)
    )
    output = torch.empty_like(token_indices)
    counts = (
        torch.empty((validated.num_queries,), dtype=torch.int32, device=device)
        if validated.return_valid_counts
        else None
    )
    operands = _Operands(
        req_id=req_id,
        block_table=block_table,
        token_indices=token_indices,
        prefill_workspace_request_ids=workspace_ids,
        prefill_workspace_starts=workspace_starts,
        output=output,
        counts=counts,
    )
    _validate_operand_invariants(torch, validated, operands)
    return operands


def _validate_operand_invariants(
    torch: Any,
    validated: _ValidatedArgs,
    operands: _Operands,
) -> None:
    if operands.req_id.shape != (validated.num_queries,) or not operands.req_id.is_contiguous():
        raise AssertionError("req_id must be contiguous [num_queries]")
    if (
        operands.block_table.shape
        != (
            validated.num_requests,
            validated.max_blocks_per_request,
        )
        or not operands.block_table.is_contiguous()
    ):
        raise AssertionError("block_table must be contiguous [num_requests, max_blocks]")
    if (
        operands.token_indices.shape
        != (
            validated.num_queries,
            validated.selected_k,
        )
        or not operands.token_indices.is_contiguous()
    ):
        raise AssertionError("token_indices must be contiguous [num_queries, selected_k]")
    for name, tensor in (
        ("req_id", operands.req_id),
        ("block_table", operands.block_table),
        ("token_indices", operands.token_indices),
        ("output", operands.output),
    ):
        if tensor.dtype is not torch.int32:
            raise AssertionError(f"{name} must be int32")
    if not torch.equal(
        operands.req_id,
        torch.tensor(validated.request_ids, dtype=torch.int32, device=operands.req_id.device),
    ):
        raise AssertionError("req_id does not match request_row_counts")
    if bool((operands.block_table < 0).any().item()):
        raise AssertionError("block_table must contain only nonnegative page IDs")
    positions = torch.arange(
        validated.selected_k,
        dtype=torch.int64,
        device=operands.token_indices.device,
    )
    counts = torch.tensor(
        validated.valid_counts,
        dtype=torch.int64,
        device=operands.token_indices.device,
    )
    spans = torch.tensor(
        validated.local_span_lengths,
        dtype=torch.int64,
        device=operands.token_indices.device,
    )
    for row_start in range(0, validated.num_queries, _OPERAND_CHECK_CHUNK_SIZE):
        row_end = min(row_start + _OPERAND_CHECK_CHUNK_SIZE, validated.num_queries)
        rows = operands.token_indices[row_start:row_end].to(torch.int64)
        valid_mask = positions[None, :] < counts[row_start:row_end, None]
        row_spans = spans[row_start:row_end, None]
        layouts_ok = torch.where(
            valid_mask,
            (rows >= 0) & (rows < row_spans),
            rows == -1,
        ).all(dim=1)

        # Invalid slots receive distinct values above the row's valid domain,
        # allowing one sort to prove every generated valid prefix is unique.
        uniqueness_values = torch.where(
            valid_mask,
            rows,
            row_spans + positions[None, :],
        )
        sorted_values = uniqueness_values.sort(dim=1).values
        unique_ok = (sorted_values[:, 1:] != sorted_values[:, :-1]).all(dim=1)
        if not bool((layouts_ok & unique_ok).all().item()):
            raise AssertionError(
                f"token_indices rows {row_start}..{row_end - 1} violate prefix/tail invariants"
            )

    if validated.workspace_partition is None:
        if (
            operands.prefill_workspace_request_ids is not None
            or operands.prefill_workspace_starts is not None
        ):
            raise AssertionError("workspace_partition='none' must omit workspace arrays")
    else:
        if (
            operands.prefill_workspace_request_ids is None
            or operands.prefill_workspace_starts is None
        ):
            raise AssertionError("workspace suffix must create both workspace arrays")
        if not (
            operands.prefill_workspace_request_ids.is_contiguous()
            and operands.prefill_workspace_starts.is_contiguous()
        ):
            raise AssertionError("workspace arrays must be contiguous")
    if operands.counts is not None and not operands.counts.is_contiguous():
        raise AssertionError("counts output must be contiguous")


def _torch_composite(
    torch: Any,
    operands: _Operands,
    *,
    block_size: int,
    row_chunk_size: int = _ROW_CHUNK_SIZE,
) -> Any:
    """Write the preallocated output using vectorized row chunks."""
    if operands.counts is not None:
        operands.counts.zero_()

    for row_start in range(0, operands.token_indices.shape[0], row_chunk_size):
        row_end = min(row_start + row_chunk_size, operands.token_indices.shape[0])
        local_indices = operands.token_indices[row_start:row_end].to(torch.int64)
        local_blocks = torch.div(local_indices, block_size, rounding_mode="floor")
        valid = (local_indices >= 0) & (local_blocks < operands.block_table.shape[1])
        chunk_output = torch.full_like(local_indices, -1)

        workspace_ids = operands.prefill_workspace_request_ids
        if workspace_ids is None:
            global_rows = torch.ones(
                (row_end - row_start,),
                dtype=torch.bool,
                device=local_indices.device,
            )
        else:
            global_rows = workspace_ids[row_start:row_end] == -1

        global_slots = valid & global_rows[:, None]
        if bool(global_slots.any().item()):
            rows, columns = global_slots.nonzero(as_tuple=True)
            requests = operands.req_id[row_start:row_end].to(torch.int64)[rows]
            blocks = local_blocks[rows, columns]
            pages = operands.block_table[requests, blocks].to(torch.int64)
            offsets = local_indices[rows, columns].remainder(block_size)
            chunk_output[rows, columns] = pages * block_size + offsets

        if workspace_ids is not None:
            workspace_slots = valid & ~global_rows[:, None]
            if bool(workspace_slots.any().item()):
                assert operands.prefill_workspace_starts is not None
                rows, columns = workspace_slots.nonzero(as_tuple=True)
                workspace_requests = workspace_ids[row_start:row_end].to(torch.int64)[rows]
                starts = operands.prefill_workspace_starts.to(torch.int64)[workspace_requests]
                chunk_output[rows, columns] = starts + local_indices[rows, columns]

        operands.output[row_start:row_end].copy_(chunk_output.to(torch.int32))
        if operands.counts is not None:
            operands.counts[row_start:row_end].add_(valid.sum(dim=1).to(torch.int32))

    if operands.counts is None:
        return operands.output
    return operands.output, operands.counts


def _check_correctness(torch: Any, validated: _ValidatedArgs, operands: _Operands) -> None:
    from profiling.runners.attention.dsa_sparse_index_remap_reference import (
        dsa_sparse_index_remap_reference,
    )

    input_tensors = [operands.req_id, operands.block_table, operands.token_indices]
    if operands.prefill_workspace_request_ids is not None:
        input_tensors.append(operands.prefill_workspace_request_ids)
    if operands.prefill_workspace_starts is not None:
        input_tensors.append(operands.prefill_workspace_starts)
    snapshots = tuple(tensor.clone() for tensor in input_tensors)
    pointers = tuple(tensor.data_ptr() for tensor in input_tensors)
    storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in input_tensors}
    output_tensors = [operands.output]
    if operands.counts is not None:
        output_tensors.append(operands.counts)
    for output_index, output_tensor in enumerate(output_tensors):
        if any(torch._C._overlaps(output_tensor, input_tensor) for input_tensor in input_tensors):
            raise AssertionError(f"Torch composite output {output_index} aliases an input")
    if len(output_tensors) == 2 and torch._C._overlaps(output_tensors[0], output_tensors[1]):
        raise AssertionError("Torch composite outputs must not alias each other")

    expected = dsa_sparse_index_remap_reference(
        operands.req_id,
        operands.block_table,
        operands.token_indices,
        block_size=validated.block_size,
        prefill_workspace_request_ids=operands.prefill_workspace_request_ids,
        prefill_workspace_starts=operands.prefill_workspace_starts,
        return_valid_counts=validated.return_valid_counts,
    )
    actual = _torch_composite(torch, operands, block_size=validated.block_size)
    repeated = _torch_composite(torch, operands, block_size=validated.block_size)

    actual_values = actual if isinstance(actual, tuple) else (actual,)
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    repeated_values = repeated if isinstance(repeated, tuple) else (repeated,)
    if len(actual_values) != len(expected_values):
        raise AssertionError("Torch composite returned the wrong count/no-count variant")
    for index, (actual_tensor, expected_tensor, repeated_tensor) in enumerate(
        zip(actual_values, expected_values, repeated_values, strict=True)
    ):
        if actual_tensor.dtype is not torch.int32 or not actual_tensor.is_contiguous():
            raise AssertionError(f"Torch composite output {index} must be contiguous int32")
        if actual_tensor.shape != expected_tensor.shape:
            raise AssertionError(
                f"Torch composite output {index} shape {tuple(actual_tensor.shape)} "
                f"does not match {tuple(expected_tensor.shape)}"
            )
        if not torch.equal(actual_tensor, expected_tensor):
            raise AssertionError(f"Torch composite output {index} differs from reference")
        if not torch.equal(repeated_tensor, expected_tensor):
            raise AssertionError(f"Torch composite output {index} is nondeterministic")
        if actual_tensor.untyped_storage().data_ptr() in storage_pointers:
            raise AssertionError(f"Torch composite output {index} aliases an input")

    for name, tensor, snapshot, pointer in zip(
        (
            "req_id",
            "block_table",
            "token_indices",
            "prefill_workspace_request_ids",
            "prefill_workspace_starts",
        ),
        input_tensors,
        snapshots,
        pointers,
        strict=False,
    ):
        if tensor.data_ptr() != pointer or not torch.equal(tensor, snapshot):
            raise AssertionError(f"Torch composite mutated {name}")


def _logical_flops() -> int:
    return 0


def _logical_bytes(
    *,
    num_queries: int,
    selected_k: int,
    valid_counts: Sequence[int],
    workspace_ids: Sequence[int] | None,
    return_valid_counts: bool,
) -> int:
    """Return source-scheduled logical traffic, not physical transactions."""
    if num_queries <= 0 or selected_k <= 0:
        raise ValueError("logical-byte dimensions must be positive")
    if selected_k % _SOURCE_BLOCK_N:
        raise ValueError(f"selected_k must be divisible by source tile {_SOURCE_BLOCK_N}")
    counts = tuple(valid_counts)
    if len(counts) != num_queries or any(count < 0 for count in counts):
        raise ValueError("valid_counts must contain one nonnegative value per query")
    if workspace_ids is not None and len(tuple(workspace_ids)) != num_queries:
        raise ValueError("workspace_ids must contain one value per query")

    tiles = num_queries * selected_k // _SOURCE_BLOCK_N
    if workspace_ids is None:
        global_valid_slots = sum(counts)
        workspace_rows = 0
    else:
        workspace = tuple(workspace_ids)
        global_valid_slots = sum(
            count
            for count, workspace_id in zip(counts, workspace, strict=True)
            if workspace_id == -1
        )
        workspace_rows = sum(workspace_id >= 0 for workspace_id in workspace)

    logical_bytes = 4 * num_queries * selected_k  # token-index reads
    logical_bytes += 4 * num_queries * selected_k  # output writes
    logical_bytes += 4 * tiles  # one request-ID load per source tile
    logical_bytes += 4 * global_valid_slots  # selected block-table reads
    if workspace_ids is not None:
        logical_bytes += 4 * tiles  # one workspace-ID load per tile
        logical_bytes += 4 * workspace_rows * selected_k // _SOURCE_BLOCK_N
    if return_valid_counts:
        logical_bytes += 4 * num_queries  # wrapper zero-initialization writes
        logical_bytes += 8 * tiles  # source-equivalent atomic read/modify/write
    return logical_bytes


def profile_dsa_sparse_index_remap_torch(
    *,
    num_queries: int,
    num_requests: int,
    selected_k: int,
    block_size: int,
    max_blocks_per_request: int,
    request_row_counts: str,
    local_span_lengths: str,
    valid_counts: str,
    index_distribution: str,
    page_table_mapping: str,
    workspace_partition: str,
    return_valid_counts: bool,
    index_dtype: str,
) -> ComputeMetrics:
    """Profile the complete Torch semantic composite on an NVIDIA H200."""
    validated = _validate_args(
        num_queries=num_queries,
        num_requests=num_requests,
        selected_k=selected_k,
        block_size=block_size,
        max_blocks_per_request=max_blocks_per_request,
        request_row_counts=request_row_counts,
        local_span_lengths=local_span_lengths,
        valid_counts=valid_counts,
        index_distribution=index_distribution,
        page_table_mapping=page_table_mapping,
        workspace_partition=workspace_partition,
        return_valid_counts=return_valid_counts,
        index_dtype=index_dtype,
    )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented(f"{_BACKEND} requires PyTorch") from exc

    try:
        _require_h200(torch)
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, validated, device=device)
        _check_correctness(torch, validated, operands)

        latest_output: Any = None

        def kernel() -> Any:
            nonlocal latest_output
            latest_output = _torch_composite(
                torch,
                operands,
                block_size=validated.block_size,
            )
            return latest_output

        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except ProfilerNotImplemented:
        raise
    except KernelLaunchFailed:
        raise
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of GPU memory") from exc
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} semantic composite failed") from exc

    workspace_values, _ = _derive_workspace_metadata(validated)
    logical_bytes = _logical_bytes(
        num_queries=validated.num_queries,
        selected_k=validated.selected_k,
        valid_counts=validated.valid_counts,
        workspace_ids=workspace_values,
        return_valid_counts=validated.return_valid_counts,
    )
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=time_ms,
        energy_j=energy_j,
        tflops=0.0,
        memory_bandwidth_gbps=logical_bytes / seconds / 1e9,
    )


def _launch_vllm_triton(
    callable_: Any,
    operands: _Operands,
    validated: _ValidatedArgs,
) -> Any:
    has_prefill_workspace = operands.prefill_workspace_request_ids is not None
    return callable_(
        operands.req_id,
        operands.block_table,
        operands.token_indices,
        BLOCK_SIZE=validated.block_size,
        NUM_TOPK_TOKENS=validated.selected_k,
        HAS_PREFILL_WORKSPACE=has_prefill_workspace,
        prefill_workspace_request_ids=operands.prefill_workspace_request_ids,
        prefill_workspace_starts=operands.prefill_workspace_starts,
        return_valid_counts=validated.return_valid_counts,
    )


def _check_vllm_triton_correctness(
    torch: Any,
    callable_: Any,
    operands: _Operands,
    validated: _ValidatedArgs,
) -> None:
    from profiling.runners.attention.dsa_sparse_index_remap_reference import (
        dsa_sparse_index_remap_reference,
    )

    actual = _launch_vllm_triton(callable_, operands, validated)
    expected = dsa_sparse_index_remap_reference(
        operands.req_id,
        operands.block_table,
        operands.token_indices,
        block_size=validated.block_size,
        prefill_workspace_request_ids=operands.prefill_workspace_request_ids,
        prefill_workspace_starts=operands.prefill_workspace_starts,
        return_valid_counts=validated.return_valid_counts,
    )
    torch.cuda.synchronize(operands.req_id.device)
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    if len(actual_values) != len(expected_values):
        raise KernelLaunchFailed(f"{_VLLM_BACKEND} returned the wrong output variant")
    for actual_value, expected_value in zip(actual_values, expected_values, strict=True):
        if not torch.equal(actual_value, expected_value):
            raise KernelLaunchFailed(f"{_VLLM_BACKEND} disagrees with the reference")


def profile_dsa_sparse_index_remap_vllm_triton(
    *,
    num_queries: int,
    num_requests: int,
    selected_k: int,
    block_size: int,
    max_blocks_per_request: int,
    request_row_counts: str,
    local_span_lengths: str,
    valid_counts: str,
    index_distribution: str,
    page_table_mapping: str,
    workspace_partition: str,
    return_valid_counts: bool,
    index_dtype: str,
) -> ComputeMetrics:
    """Profile vLLM's production sparse-index Triton wrapper on B200."""
    validated = _validate_args(
        num_queries=num_queries,
        num_requests=num_requests,
        selected_k=selected_k,
        block_size=block_size,
        max_blocks_per_request=max_blocks_per_request,
        request_row_counts=request_row_counts,
        local_span_lengths=local_span_lengths,
        valid_counts=valid_counts,
        index_distribution=index_distribution,
        page_table_mapping=page_table_mapping,
        workspace_partition=workspace_partition,
        return_valid_counts=return_valid_counts,
        index_dtype=index_dtype,
    )
    try:
        import torch
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise ProfilerNotImplemented(f"{_VLLM_BACKEND} requires the repository vllm_env") from exc

    try:
        _require_b200(torch)
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, validated, device=device)
        _check_vllm_triton_correctness(
            torch,
            triton_convert_req_index_to_global_index,
            operands,
            validated,
        )

        def kernel() -> Any:
            return _launch_vllm_triton(
                triton_convert_req_index_to_global_index,
                operands,
                validated,
            )

        time_ms = Timer.cupti(kernel, interval_union=True)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_VLLM_BACKEND} ran out of GPU memory") from exc
    except Exception as exc:
        raise KernelLaunchFailed(f"{_VLLM_BACKEND} native callable failed") from exc

    workspace_values, _ = _derive_workspace_metadata(validated)
    logical_bytes = _logical_bytes(
        num_queries=validated.num_queries,
        selected_k=validated.selected_k,
        valid_counts=validated.valid_counts,
        workspace_ids=workspace_values,
        return_valid_counts=validated.return_valid_counts,
    )
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=0.0,
        memory_bandwidth_gbps=logical_bytes / seconds / 1e9,
    )
