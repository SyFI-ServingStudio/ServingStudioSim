"""Profile one production FlashMLA sparse-decode graph replay."""

import math
from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_sparse_mla_decode:vllm_flashmla_fp8_cudagraph"
_GPU_NAME = "NVIDIA H200"
_PAGE_SIZE = {1: 0, 4: 64, 128: 2}
_PLANNER_MODES = frozenset({"planned", "reused"})
_PRODUCTION_MODEL_SHAPE = (64, 1, 512, 512, 128)
_PRODUCTION_DTYPES = ("bf16", "fp8_e4m3", "bf16")


@dataclass(frozen=True)
class _Shape:
    swa_counts: tuple[int, ...]
    extra_counts: tuple[int, ...]
    num_heads: int
    head_dim: int
    value_dim: int
    swa_window: int
    extra_index_capacity: int
    compress_ratio: int
    planner_mode: str


@dataclass(frozen=True)
class _Operands:
    q: Any
    swa_cache: Any
    swa_indices: Any
    swa_counts: Any
    extra_cache: Any | None
    extra_indices: Any | None
    extra_counts: Any | None
    sink: Any
    output: Any
    expected_output_values: tuple[float, ...]


def _validate_args(
    swa_valid_counts: tuple[int, ...],
    extra_valid_counts: tuple[int, ...],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    value_dim: int,
    swa_window: int,
    extra_index_capacity: int,
    compress_ratio: int,
    q_dtype: object,
    cache_dtype: object,
    output_dtype: object,
    planner_mode: str,
) -> _Shape:
    if not swa_valid_counts or len(swa_valid_counts) != len(extra_valid_counts):
        raise ValueError("valid-count tuples must be non-empty and have equal length")
    if len(swa_valid_counts) > 256:
        raise ProfilerNotImplemented(f"{_BACKEND} supports at most 256 decode rows")
    if compress_ratio not in _PAGE_SIZE:
        raise ProfilerNotImplemented(f"{_BACKEND} supports compress_ratio=1/4/128")
    model_shape = (num_heads, num_kv_heads, head_dim, value_dim, swa_window)
    if model_shape != _PRODUCTION_MODEL_SHAPE:
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports model shape {_PRODUCTION_MODEL_SHAPE}, got {model_shape}"
        )
    dtype_identity = tuple(str(dtype) for dtype in (q_dtype, cache_dtype, output_dtype))
    if dtype_identity != _PRODUCTION_DTYPES:
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports dtypes {_PRODUCTION_DTYPES}, got {dtype_identity}"
        )
    if any(type(count) is not int or not 1 <= count <= swa_window for count in swa_valid_counts):
        raise ValueError("each SWA valid count must be an integer in [1, swa_window]")
    if compress_ratio == 1:
        expected_extra_capacity = 0
    elif compress_ratio == 4:
        expected_extra_capacity = 512
    else:
        expected_extra_capacity = extra_index_capacity
        if not 128 <= extra_index_capacity <= 8192 or extra_index_capacity % 128:
            raise ValueError("C128 extra_index_capacity must be a multiple of 128 in [128, 8192]")
    if extra_index_capacity != expected_extra_capacity:
        raise ValueError(
            f"compress_ratio={compress_ratio} requires "
            f"extra_index_capacity={expected_extra_capacity}"
        )
    if any(
        type(count) is not int or not 0 <= count <= extra_index_capacity
        for count in extra_valid_counts
    ):
        raise ValueError("each extra valid count must fit extra_index_capacity")
    if planner_mode not in _PLANNER_MODES:
        raise ValueError(f"planner_mode must be one of {sorted(_PLANNER_MODES)}")
    return _Shape(
        swa_valid_counts,
        extra_valid_counts,
        num_heads,
        head_dim,
        value_dim,
        swa_window,
        extra_index_capacity,
        compress_ratio,
        planner_mode,
    )


def _require_h200(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    name = str(torch.cuda.get_device_name(device))
    if name != _GPU_NAME or tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        raise ProfilerNotImplemented(f"{_BACKEND} requires {_GPU_NAME} SM90, got {name}")
    return device


def _patterned_cache(
    torch: Any, rows: int, page_size: int, value_base: int, device: Any
) -> tuple[Any, tuple[float, ...]]:
    block_count = max(1, math.ceil(max(1, rows) / page_size))
    cache = torch.empty((block_count, page_size, 1, 584), dtype=torch.uint8, device=device)
    physical_rows = block_count * page_size
    values = tuple(float(value_base + 2 * (row % 4)) for row in range(physical_rows))
    value_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    fp8_bytes = (
        value_tensor[:, None].expand(-1, 448).to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
    )
    rope_bytes = (
        value_tensor[:, None]
        .expand(-1, 64)
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(physical_rows, 128)
    )
    data_rows = torch.cat((fp8_bytes, rope_bytes), dim=1)
    for block in range(block_count):
        raw_block = cache[block].view(-1)
        start = block * page_size
        raw_block[: page_size * 576].copy_(data_rows[start : start + page_size].reshape(-1))
        raw_block[page_size * 576 :].fill_(127)
    return cache, values[:rows]


def _indices(torch: Any, counts: tuple[int, ...], capacity: int, device: Any) -> Any:
    indices = torch.full((len(counts), 1, capacity), -1, dtype=torch.int32, device=device)
    cursor = 0
    for row, count in enumerate(counts):
        indices[row, 0, :count] = torch.arange(
            cursor, cursor + count, dtype=torch.int32, device=device
        )
        cursor += count
    return indices


def _selected_means(
    swa_counts: tuple[int, ...],
    swa_values: tuple[float, ...],
    extra_counts: tuple[int, ...],
    extra_values: tuple[float, ...],
) -> tuple[float, ...]:
    means = []
    swa_cursor = extra_cursor = 0
    for swa_count, extra_count in zip(swa_counts, extra_counts, strict=True):
        selected = swa_values[swa_cursor : swa_cursor + swa_count]
        selected += extra_values[extra_cursor : extra_cursor + extra_count]
        means.append(sum(selected) / len(selected))
        swa_cursor += swa_count
        extra_cursor += extra_count
    return tuple(means)


def _prepare(torch: Any, shape: _Shape, device: Any) -> _Operands:
    swa_rows = sum(shape.swa_counts)
    swa_cache, swa_values = _patterned_cache(torch, swa_rows, 64, 1, device)
    swa_indices = _indices(torch, shape.swa_counts, shape.swa_window, device)
    extra_cache = extra_indices = extra_counts = None
    extra_values: tuple[float, ...] = ()
    if shape.compress_ratio > 1:
        extra_rows = sum(shape.extra_counts)
        page_size = _PAGE_SIZE[shape.compress_ratio]
        extra_cache, extra_values = _patterned_cache(torch, extra_rows, page_size, 9, device)
        extra_indices = _indices(torch, shape.extra_counts, shape.extra_index_capacity, device)
        extra_counts = torch.tensor(shape.extra_counts, dtype=torch.int32, device=device)
    return _Operands(
        q=torch.zeros(
            (len(shape.swa_counts), 1, shape.num_heads, shape.head_dim),
            dtype=torch.bfloat16,
            device=device,
        ),
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_counts=torch.tensor(shape.swa_counts, dtype=torch.int32, device=device),
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_counts=extra_counts,
        sink=torch.full((shape.num_heads,), -float("inf"), dtype=torch.float32, device=device),
        output=torch.full(
            (len(shape.swa_counts), 1, shape.num_heads, shape.value_dim),
            -7.0,
            dtype=torch.bfloat16,
            device=device,
        ),
        expected_output_values=_selected_means(
            shape.swa_counts,
            swa_values,
            shape.extra_counts,
            extra_values,
        ),
    )


def _capture(
    torch: Any, production: Any, metadata_factory: Any, operands: _Operands, shape: _Shape
):
    scheduler = metadata_factory()[0]

    def run():
        return production(
            q=operands.q,
            k_cache=operands.swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=shape.value_dim,
            tile_scheduler_metadata=scheduler,
            softmax_scale=shape.head_dim**-0.5,
            is_fp8_kvcache=True,
            indices=operands.swa_indices,
            attn_sink=operands.sink,
            extra_k_cache=operands.extra_cache,
            extra_indices_in_kvcache=operands.extra_indices,
            topk_length=operands.swa_counts,
            extra_topk_length=operands.extra_counts,
            out=operands.output,
        )

    if shape.planner_mode == "reused":
        run()
        torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_lse = run()
    torch.cuda.synchronize()
    return graph, graph_output, graph_lse


def _check_output(torch: Any, graph: Any, operands: _Operands, lse: Any, shape: _Shape) -> None:
    output = operands.output
    output.fill_(-11.0)
    graph.replay()
    torch.cuda.synchronize()
    expected_output = torch.tensor(
        operands.expected_output_values, dtype=torch.bfloat16, device=output.device
    ).float()[:, None, None, None]
    torch.testing.assert_close(
        output.float(), expected_output.expand_as(output), atol=0.03125, rtol=0.0
    )
    expected_lse = torch.tensor(
        [
            math.log(swa + extra)
            for swa, extra in zip(shape.swa_counts, shape.extra_counts, strict=True)
        ],
        dtype=torch.float32,
        device=lse.device,
    )[:, None, None]
    torch.testing.assert_close(lse, expected_lse.expand_as(lse), atol=2e-5, rtol=2e-5)


def _logical_work(shape: _Shape) -> tuple[int, int]:
    selected = sum(shape.swa_counts) + sum(shape.extra_counts)
    flops = 2 * shape.num_heads * (shape.head_dim + shape.value_dim) * selected
    batch_size = len(shape.swa_counts)
    q_read = batch_size * shape.num_heads * shape.head_dim * 2
    cache_and_indices = selected * (584 + 4)
    count_reads = batch_size * 4 * (2 if shape.compress_ratio > 1 else 1)
    output_write = batch_size * shape.num_heads * (shape.value_dim * 2 + 4)
    sink_read = shape.num_heads * 4
    return flops, q_read + cache_and_indices + count_reads + output_write + sink_read


def profile_deepseek_v4_sparse_mla_decode_flashmla(
    swa_valid_counts: tuple[int, ...],
    extra_valid_counts: tuple[int, ...],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    value_dim: int,
    swa_window: int,
    extra_index_capacity: int,
    compress_ratio: int,
    q_dtype: object,
    cache_dtype: object,
    output_dtype: object,
    planner_mode: str,
) -> ComputeMetrics:
    shape = _validate_args(
        swa_valid_counts,
        extra_valid_counts,
        num_heads,
        num_kv_heads,
        head_dim,
        value_dim,
        swa_window,
        extra_index_capacity,
        compress_ratio,
        q_dtype,
        cache_dtype,
        output_dtype,
        planner_mode,
    )
    try:
        import torch
        from vllm.third_party.flashmla.flash_mla_interface import (
            flash_mla_with_kvcache,
            get_mla_metadata,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires pinned vLLM FlashMLA") from exc
    try:
        device = _require_h200(torch)
        operands = _prepare(torch, shape, device)
        graph, _output, lse = _capture(
            torch, flash_mla_with_kvcache, get_mla_metadata, operands, shape
        )
        _check_output(torch, graph, operands, lse, shape)
        time_ms = Timer.cupti(graph.replay, warmup=3, kernel_name=None)
        energy_j = Energy.perf(graph.replay, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc
    elapsed_seconds = time_ms / 1000.0
    flops, logical_bytes = _logical_work(shape)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_seconds / 1e12),
        memory_bandwidth_gbps=float(logical_bytes / elapsed_seconds / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_deepseek_v4_sparse_mla_decode_flashmla"]
