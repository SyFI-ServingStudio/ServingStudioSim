"""Profile DeepSeek V4's production CuteDSL packed-cache gather."""

import math
from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_packed_cache_gather:vllm_deepseek_v4_cutedsl"
_GPU_NAME = "NVIDIA H200"
_MODEL_IDENTITY = (1, 512, 448, 64)
_STORAGE_IDENTITY = (
    "fp8_ds_mla",
    "bf16",
    "block_segregated_data_then_scales",
    "ue8m0",
)


@dataclass(frozen=True)
class _Shape:
    seq_lens: tuple[int, ...]
    gather_lens: tuple[int, ...] | None
    workspace_rows: int
    block_table_width: int
    block_size: int
    offset: int


@dataclass(frozen=True)
class _Operands:
    output: Any
    cache: Any
    seq_lens: Any
    gather_lens: Any | None
    block_table: Any
    expected_rows: tuple[tuple[float, ...], ...]


def _validate_args(
    seq_lens: tuple[int, ...],
    gather_lens: tuple[int, ...],
    workspace_rows: int,
    block_table_width: int,
    block_size: int,
    offset: int,
    num_kv_heads: int,
    head_dim: int,
    fp8_dim: int,
    quant_group_size: int,
    cache_dtype: object,
    output_dtype: object,
    cache_layout: str,
    scale_format: str,
) -> _Shape:
    if not seq_lens or any(type(length) is not int or length <= 0 for length in seq_lens):
        raise ValueError("seq_lens must be a non-empty tuple of positive integers")
    effective_gather_lens = None if not gather_lens else gather_lens
    if effective_gather_lens is not None and (
        len(effective_gather_lens) != len(seq_lens)
        or any(
            type(length) is not int or not 0 <= length <= seq_len
            for length, seq_len in zip(effective_gather_lens, seq_lens, strict=True)
        )
    ):
        raise ValueError("gather_lens must be empty or match seq_lens within [0, seq_len]")
    gathered = effective_gather_lens or seq_lens
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if type(workspace_rows) is not int or workspace_rows < offset + max(gathered):
        raise ValueError("workspace_rows must cover offset plus the largest gather")
    if block_size not in (2, 64):
        raise ProfilerNotImplemented(f"{_BACKEND} supports block_size=2/64")
    required_width = max(math.ceil(length / block_size) for length in seq_lens)
    if type(block_table_width) is not int or block_table_width < required_width:
        raise ValueError("block_table_width is too small for seq_lens")
    model_identity = (num_kv_heads, head_dim, fp8_dim, quant_group_size)
    if model_identity != _MODEL_IDENTITY:
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports model identity {_MODEL_IDENTITY}, got {model_identity}"
        )
    storage_identity = (
        str(cache_dtype),
        str(output_dtype),
        cache_layout,
        scale_format,
    )
    if storage_identity != _STORAGE_IDENTITY:
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports storage identity {_STORAGE_IDENTITY}, got {storage_identity}"
        )
    return _Shape(
        seq_lens,
        effective_gather_lens,
        workspace_rows,
        block_table_width,
        block_size,
        offset,
    )


def _require_h200_cutedsl(torch: Any, has_cutedsl: Any) -> Any:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    name = str(torch.cuda.get_device_name(device))
    if name != _GPU_NAME or tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        raise ProfilerNotImplemented(f"{_BACKEND} requires {_GPU_NAME} SM90, got {name}")
    if not has_cutedsl():
        raise ProfilerNotImplemented(f"{_BACKEND} requires the production Cutlass DSL path")
    return device


def _block_table(torch: Any, shape: _Shape, device: Any) -> tuple[Any, int]:
    blocks_per_request = tuple(math.ceil(length / shape.block_size) for length in shape.seq_lens)
    total_blocks = sum(blocks_per_request)
    physical_blocks = list(reversed(range(total_blocks)))
    table = torch.full(
        (len(shape.seq_lens), shape.block_table_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    cursor = 0
    for request, block_count in enumerate(blocks_per_request):
        table[request, :block_count] = torch.tensor(
            physical_blocks[cursor : cursor + block_count], dtype=torch.int32, device=device
        )
        cursor += block_count
    return table, total_blocks


def _block_stride(block_size: int) -> int:
    return math.ceil(block_size * 584 / 32) * 32


def _packed_cache(torch: Any, block_count: int, block_size: int, device: Any) -> Any:
    block_stride = _block_stride(block_size)
    storage = torch.empty(block_count * block_stride, dtype=torch.uint8, device=device)
    cache = torch.as_strided(
        storage,
        size=(block_count, block_size, 584),
        stride=(block_stride, 584, 1),
    )
    physical_rows = block_count * block_size
    values = torch.arange(physical_rows, dtype=torch.int32, device=device) % 4 * 2 + 1
    fp8 = values[:, None].expand(-1, 448).to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
    rope = (
        values[:, None]
        .expand(-1, 64)
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(physical_rows, 128)
    )
    data = torch.cat((fp8, rope), dim=1)
    for block in range(block_count):
        raw = cache[block].view(-1)
        start = block * block_size
        raw[: block_size * 576].copy_(data[start : start + block_size].reshape(-1))
        raw[block_size * 576 :].fill_(127)
    return cache


def _expected_rows(
    shape: _Shape, block_table: tuple[tuple[int, ...], ...]
) -> tuple[tuple[float, ...], ...]:
    expected = []
    gather_lens = shape.gather_lens or shape.seq_lens
    for request, (seq_len, gather_len) in enumerate(zip(shape.seq_lens, gather_lens, strict=True)):
        values = []
        for position in range(seq_len - gather_len, seq_len):
            block = block_table[request][position // shape.block_size]
            physical_row = block * shape.block_size + position % shape.block_size
            values.append(float(1 + 2 * (physical_row % 4)))
        expected.append(tuple(values))
    return tuple(expected)


def _prepare(torch: Any, shape: _Shape, device: Any) -> _Operands:
    block_table, block_count = _block_table(torch, shape, device)
    table_host = tuple(tuple(int(value) for value in row) for row in block_table.cpu().tolist())
    return _Operands(
        output=torch.full(
            (len(shape.seq_lens), shape.workspace_rows, 512),
            -17.0,
            dtype=torch.bfloat16,
            device=device,
        ),
        cache=_packed_cache(torch, block_count, shape.block_size, device),
        seq_lens=torch.tensor(shape.seq_lens, dtype=torch.int32, device=device),
        gather_lens=(
            None
            if shape.gather_lens is None
            else torch.tensor(shape.gather_lens, dtype=torch.int32, device=device)
        ),
        block_table=block_table,
        expected_rows=_expected_rows(shape, table_host),
    )


def _check_output(torch: Any, launch: Any, operands: _Operands, shape: _Shape) -> None:
    operands.output.fill_(-17.0)
    launch()
    torch.cuda.synchronize()
    gather_lens = shape.gather_lens or shape.seq_lens
    for request, gather_len in enumerate(gather_lens):
        expected = torch.tensor(
            operands.expected_rows[request], dtype=torch.float32, device=operands.output.device
        )[:, None]
        actual = operands.output[request, shape.offset : shape.offset + gather_len].float()
        torch.testing.assert_close(actual, expected.expand_as(actual), atol=0.0, rtol=0.0)
        assert torch.all(operands.output[request, : shape.offset] == -17)
        assert torch.all(operands.output[request, shape.offset + gather_len :] == -17)


def _logical_work(shape: _Shape) -> tuple[int, int]:
    gathered = sum(shape.gather_lens or shape.seq_lens)
    flops = gathered * 448
    length_bytes = len(shape.seq_lens) * 4 * (2 if shape.gather_lens is not None else 1)
    logical_bytes = gathered * (584 + 4 + 512 * 2) + length_bytes
    return flops, logical_bytes


def profile_deepseek_v4_packed_cache_gather_cutedsl(
    seq_lens: tuple[int, ...],
    gather_lens: tuple[int, ...],
    workspace_rows: int,
    block_table_width: int,
    block_size: int,
    offset: int,
    num_kv_heads: int,
    head_dim: int,
    fp8_dim: int,
    quant_group_size: int,
    cache_dtype: object,
    output_dtype: object,
    cache_layout: str,
    scale_format: str,
) -> ComputeMetrics:
    shape = _validate_args(
        seq_lens,
        gather_lens,
        workspace_rows,
        block_table_width,
        block_size,
        offset,
        num_kv_heads,
        head_dim,
        fp8_dim,
        quant_group_size,
        cache_dtype,
        output_dtype,
        cache_layout,
        scale_format,
    )
    try:
        import torch
        from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
        from vllm.utils.import_utils import has_cutedsl
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires pinned vLLM") from exc
    try:
        device = _require_h200_cutedsl(torch, has_cutedsl)
        operands = _prepare(torch, shape, device)

        def launch():
            return dequantize_and_gather_k_cache(
                operands.output,
                operands.cache,
                operands.seq_lens,
                operands.gather_lens,
                operands.block_table,
                shape.block_size,
                shape.offset,
            )

        launch()
        torch.cuda.synchronize()
        _check_output(torch, launch, operands, shape)
        time_ms = Timer.cupti(launch, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
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


__all__ = ["profile_deepseek_v4_packed_cache_gather_cutedsl"]
