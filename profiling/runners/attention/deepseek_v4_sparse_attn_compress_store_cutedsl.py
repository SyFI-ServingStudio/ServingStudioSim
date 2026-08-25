"""Profile the public DeepSeek V4 main-compressor tail."""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_sparse_attn_compress_store:vllm_deepseek_v4_cutedsl"
_GPU_NAME = "NVIDIA H200"
_MODEL_IDENTITY = (1, 512, 64, 256, 1.0e-6)
_STORAGE_IDENTITY = (
    "fp32",
    "bf16",
    "fp8_ds_mla",
    "block_segregated_data_then_scales",
    "ue8m0",
)
_CACHE_ROW_BYTES = 584
_TOKEN_STRIDE = 576
_SCALE_DIM = 8
_QUANT_BLOCK = 64
_POISON = 0xA5


@dataclass(frozen=True)
class _Shape:
    positions: tuple[int, ...]
    request_ids: tuple[int, ...]
    table_width: int
    ratio: int
    state_width: int
    state_block_size: int
    kv_block_size: int
    head_dim: int = 512
    rope_head_dim: int = 64
    cache_row_bytes: int = _CACHE_ROW_BYTES
    token_stride: int = _TOKEN_STRIDE
    scale_dim: int = _SCALE_DIM
    quant_block: int = _QUANT_BLOCK
    page_alignment: int = 576

    @property
    def active_rows(self) -> tuple[int, ...]:
        return tuple(
            row for row, position in enumerate(self.positions) if (position + 1) % self.ratio == 0
        )

    @property
    def window(self) -> int:
        return self.ratio * (2 if self.ratio == 4 else 1)


@dataclass(frozen=True)
class _Operands:
    kv: Any
    score: Any
    ape: Any
    state_cache: Any
    request_ids: Any
    positions: Any
    slot_mapping: Any
    block_table: Any
    norm_weight: Any
    cos_sin_cache: Any
    kv_cache: Any
    kv_slot_mapping: Any
    active_rows: tuple[int, ...]


def _validate_args(
    row_positions: tuple[int, ...],
    row_request_ids: tuple[int, ...],
    state_block_table_width: int,
    compress_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    rope_head_dim: int,
    logical_block_size: int,
    rms_eps: float,
    state_dtype: object,
    norm_dtype: object,
    cache_dtype: str,
    cache_layout: str,
    scale_format: str,
) -> _Shape:
    if not row_positions or len(row_positions) != len(row_request_ids):
        raise ValueError("row_positions and row_request_ids must be non-empty equal tuples")
    if len(row_positions) > 8192:
        raise ProfilerNotImplemented(f"{_BACKEND} supports at most 8192 token rows")
    if any(type(position) is not int or position < 0 for position in row_positions):
        raise ValueError("row_positions must contain non-negative integers")
    if any(type(request) is not int or request < 0 for request in row_request_ids):
        raise ValueError("row_request_ids must contain non-negative integers")
    request_count = max(row_request_ids) + 1
    if set(row_request_ids) != set(range(request_count)):
        raise ValueError("row_request_ids must densely cover [0, num_requests)")
    if request_count > 64:
        raise ProfilerNotImplemented(f"{_BACKEND} supports at most 64 requests")
    if compress_ratio not in (4, 128):
        raise ProfilerNotImplemented(f"{_BACKEND} supports compress_ratio=4/128")
    if type(state_block_table_width) is not int or state_block_table_width <= 0:
        raise ValueError("state_block_table_width must be a positive integer")
    state_width, state_block_size = {4: (1024, 4), 128: (512, 8)}[compress_ratio]
    required_width = max(position // state_block_size + 1 for position in row_positions)
    if state_block_table_width < required_width:
        raise ValueError("state_block_table_width does not cover row_positions")
    model_identity = (num_kv_heads, head_dim, rope_head_dim, logical_block_size, rms_eps)
    if model_identity != _MODEL_IDENTITY:
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports model identity {_MODEL_IDENTITY}, got {model_identity}"
        )
    storage_identity = (
        str(state_dtype),
        str(norm_dtype),
        cache_dtype,
        cache_layout,
        scale_format,
    )
    if storage_identity != _STORAGE_IDENTITY:
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports storage identity {_STORAGE_IDENTITY}, got {storage_identity}"
        )
    return _Shape(
        row_positions,
        row_request_ids,
        state_block_table_width,
        compress_ratio,
        state_width,
        state_block_size,
        logical_block_size // compress_ratio,
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


def _padded_stride(row_count: int) -> int:
    # The production fp8_ds_mla cache spec pads every page to a 576-byte boundary.
    return math.ceil(row_count * _CACHE_ROW_BYTES / 576) * 576


def _build_operands(torch: Any, shape: _Shape, device: Any) -> _Operands:
    request_count = max(shape.request_ids) + 1
    required_blocks = [0] * request_count
    for position, request in zip(shape.positions, shape.request_ids, strict=True):
        # save_partial_states writes every actual row, including rows that do
        # not cross a compression boundary.
        required_blocks[request] = max(
            required_blocks[request], position // shape.state_block_size + 1
        )
    total_state_blocks = max(1, sum(required_blocks))
    table = torch.zeros((request_count, shape.table_width), dtype=torch.int32, device=device)
    physical_block = 0
    for request, block_count in enumerate(required_blocks):
        if block_count:
            # Reverse each request's pages so correctness depends on the table.
            pages = torch.arange(
                physical_block + block_count - 1,
                physical_block - 1,
                -1,
                dtype=torch.int32,
                device=device,
            )
            table[request, :block_count] = pages
            physical_block += block_count

    state_elements = total_state_blocks * shape.state_block_size * 2 * shape.state_width
    values = torch.arange(state_elements, dtype=torch.int64, device=device)
    state_cache = (((values * 17 + 11).remainder(251).float() - 125.0) / 64.0).reshape(
        total_state_blocks, shape.state_block_size, 2 * shape.state_width
    )
    input_elements = len(shape.positions) * shape.state_width
    input_values = torch.arange(input_elements, dtype=torch.int64, device=device)
    kv = (((input_values * 19 + 7).remainder(239).float() - 119.0) / 64.0).reshape(
        len(shape.positions), shape.state_width
    )
    score = (((input_values * 23 + 5).remainder(241).float() - 120.0) / 96.0).reshape(
        len(shape.positions), shape.state_width
    )
    ape_values = torch.arange(shape.ratio * shape.state_width, dtype=torch.int64, device=device)
    ape = (((ape_values * 29 + 3).remainder(127).float() - 63.0) / 128.0).reshape(
        shape.ratio, shape.state_width
    )
    positions = torch.tensor(shape.positions, dtype=torch.int64, device=device)
    request_ids = torch.tensor(shape.request_ids, dtype=torch.int32, device=device)
    slot_mapping_values = []
    for position, request in zip(shape.positions, shape.request_ids, strict=True):
        page = int(table[request, position // shape.state_block_size].item())
        slot_mapping_values.append(
            page * shape.state_block_size + position % shape.state_block_size
        )
    slot_mapping = torch.tensor(slot_mapping_values, dtype=torch.int64, device=device)
    active_rows = shape.active_rows
    kv_slot_mapping = torch.full((len(shape.positions),), -1, dtype=torch.int64, device=device)
    if active_rows:
        kv_slot_mapping[list(active_rows)] = torch.arange(
            len(active_rows), dtype=torch.int64, device=device
        )

    weight_index = torch.arange(shape.head_dim, dtype=torch.float32, device=device)
    norm_weight = (1.0 + weight_index.remainder(13) / 128.0).to(torch.bfloat16)
    max_position = max(shape.positions)
    pair_index = torch.arange(shape.rope_head_dim // 2, dtype=torch.float32, device=device)
    position_index = torch.arange(max_position + 1, dtype=torch.float32, device=device)
    angles = position_index[:, None] * (pair_index[None, :] + 1.0) / 65536.0
    cos_sin_cache = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)

    output_slots = len(active_rows) + 1
    kv_blocks = max(1, math.ceil(output_slots / shape.kv_block_size))
    unpadded_page_bytes = shape.kv_block_size * shape.cache_row_bytes
    page_stride = math.ceil(unpadded_page_bytes / shape.page_alignment) * shape.page_alignment
    storage = torch.full(
        ((kv_blocks - 1) * page_stride + shape.kv_block_size * shape.cache_row_bytes,),
        _POISON,
        dtype=torch.uint8,
        device=device,
    )
    kv_cache = torch.as_strided(
        storage,
        size=(kv_blocks, shape.kv_block_size, shape.cache_row_bytes),
        stride=(page_stride, shape.cache_row_bytes, 1),
    )
    return _Operands(
        kv,
        score,
        ape,
        state_cache,
        request_ids,
        positions,
        slot_mapping,
        table,
        norm_weight,
        cos_sin_cache,
        kv_cache,
        kv_slot_mapping,
        active_rows,
    )


def _launch(save_op: Any, public_op: Any, operands: _Operands, shape: _Shape) -> None:
    save_op(
        kv=operands.kv,
        score=operands.score,
        ape=operands.ape,
        positions=operands.positions,
        state_cache=operands.state_cache,
        slot_mapping=operands.slot_mapping,
        block_size=shape.state_block_size,
        state_width=shape.state_width,
        compress_ratio=shape.ratio,
        pdl_kwargs={"launch_pdl": False},
    )
    public_op(
        state_cache=operands.state_cache,
        num_actual=len(shape.positions),
        token_to_req_indices=operands.request_ids,
        positions=operands.positions,
        slot_mapping=operands.slot_mapping,
        block_table=operands.block_table,
        block_size=shape.state_block_size,
        state_width=shape.state_width,
        cos_sin_cache=operands.cos_sin_cache,
        kv_cache=operands.kv_cache,
        k_cache_metadata=SimpleNamespace(slot_mapping=operands.kv_slot_mapping),
        pdl_kwargs={"launch_pdl": False},
        head_dim=shape.head_dim,
        rope_head_dim=shape.rope_head_dim,
        compress_ratio=shape.ratio,
        overlap=shape.ratio == 4,
        use_fp4_cache=False,
        rms_norm_weight=operands.norm_weight,
        rms_norm_eps=1.0e-6,
        quant_block=shape.quant_block,
        token_stride=shape.token_stride,
        scale_dim=shape.scale_dim,
        store_full_kv=False,
        store_full_fp8=False,
        fp8_scale=None,
    )


def _cache_row(torch: Any, operands: _Operands, shape: _Shape, slot: int) -> Any:
    page, offset = divmod(slot, shape.kv_block_size)
    raw = operands.kv_cache[page].reshape(-1)
    values = raw[offset * shape.token_stride : (offset + 1) * shape.token_stride]
    scales_start = shape.kv_block_size * shape.token_stride + offset * shape.scale_dim
    return torch.cat((values, raw[scales_start : scales_start + shape.scale_dim]))


def _reference(torch: Any, operands: _Operands, shape: _Shape, row: int) -> Any:
    position = shape.positions[row]
    request = shape.request_ids[row]
    start = position - shape.window + 1
    kv_rows = []
    score_rows = []
    for window_row, logical_position in enumerate(range(start, position + 1)):
        if logical_position < 0:
            continue
        page = int(operands.block_table[request, logical_position // shape.state_block_size].item())
        state = operands.state_cache[page, logical_position % shape.state_block_size].float()
        head_offset = shape.head_dim * (window_row // shape.ratio) if shape.ratio == 4 else 0
        kv_rows.append(state[head_offset : head_offset + shape.head_dim])
        score_start = shape.state_width + head_offset
        score_rows.append(state[score_start : score_start + shape.head_dim])
    kv = torch.stack(kv_rows)
    scores = torch.stack(score_rows)
    compressed = (kv * torch.softmax(scores, dim=0)).sum(dim=0)
    normalized = compressed * torch.rsqrt(compressed.square().mean() + 1.0e-6)
    normalized = normalized * operands.norm_weight.float()
    nope_dim = shape.head_dim - shape.rope_head_dim
    rope_pairs = normalized[nope_dim:].reshape(-1, 2)
    compressed_position = position - shape.ratio + 1
    cosines, sines = operands.cos_sin_cache[compressed_position].chunk(2)
    rotated = torch.stack(
        (
            rope_pairs[:, 0] * cosines - rope_pairs[:, 1] * sines,
            rope_pairs[:, 0] * sines + rope_pairs[:, 1] * cosines,
        ),
        dim=1,
    ).reshape(-1)
    return normalized[:nope_dim].to(torch.bfloat16).float(), rotated.to(torch.bfloat16)


def _check_output(
    torch: Any, save_op: Any, public_op: Any, operands: _Operands, shape: _Shape
) -> None:
    _launch(save_op, public_op, operands, shape)
    torch.cuda.synchronize(operands.kv_cache.device)
    for row, (position, request) in enumerate(zip(shape.positions, shape.request_ids, strict=True)):
        page = int(operands.block_table[request, position // shape.state_block_size].item())
        stored = operands.state_cache[page, position % shape.state_block_size]
        expected = torch.cat(
            (operands.kv[row], operands.score[row] + operands.ape[position % shape.ratio])
        )
        torch.testing.assert_close(stored, expected, rtol=0, atol=0)
    poison = torch.full(
        (shape.cache_row_bytes,), _POISON, dtype=torch.uint8, device=operands.kv_cache.device
    )
    for slot, row in enumerate(operands.active_rows):
        packed = _cache_row(torch, operands, shape, slot)
        if int(packed[-1].item()) != 0:
            raise KernelLaunchFailed(f"{_BACKEND} active row {row} did not write scale padding")
        if slot < 3:
            expected_nope, expected_rope = _reference(torch, operands, shape, row)
            nope_dim = shape.head_dim - shape.rope_head_dim
            scales = torch.exp2(
                packed[
                    shape.token_stride : shape.token_stride + nope_dim // shape.quant_block
                ].float()
                - 127.0
            )
            actual_nope = packed[:nope_dim].view(torch.float8_e4m3fn).float()
            actual_nope = actual_nope * scales.repeat_interleave(shape.quant_block)
            actual_rope = packed[nope_dim : shape.token_stride].view(torch.bfloat16)
            torch.testing.assert_close(actual_nope, expected_nope, rtol=0.20, atol=0.20)
            torch.testing.assert_close(actual_rope, expected_rope, rtol=0.02, atol=0.02)
    if not torch.equal(_cache_row(torch, operands, shape, len(operands.active_rows)), poison):
        raise KernelLaunchFailed(f"{_BACKEND} modified the unreferenced sentinel slot")
    operands.kv_cache.as_strided((operands.kv_cache.untyped_storage().nbytes(),), (1,)).fill_(
        _POISON
    )


def _logical_work(shape: _Shape) -> tuple[int, int]:
    active = len(shape.active_rows)
    flops = active * (shape.window * shape.head_dim * 5 + shape.head_dim * 8)
    flops += len(shape.positions) * shape.state_width
    state_bytes = active * shape.window * shape.head_dim * 2 * 4
    metadata_bytes = len(shape.positions) * (4 + 8 + 8 + 8)
    save_bytes = len(shape.positions) * (shape.state_width * 5 * 4 + 16)
    return flops, state_bytes + active * shape.cache_row_bytes + metadata_bytes + save_bytes


def profile_deepseek_v4_sparse_attn_compress_store_cutedsl(
    row_positions: tuple[int, ...],
    row_request_ids: tuple[int, ...],
    state_block_table_width: int,
    compress_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    rope_head_dim: int,
    logical_block_size: int,
    rms_eps: float,
    state_dtype: object,
    norm_dtype: object,
    cache_dtype: str,
    cache_layout: str,
    scale_format: str,
) -> ComputeMetrics:
    shape = _validate_args(
        row_positions,
        row_request_ids,
        state_block_table_width,
        compress_ratio,
        num_kv_heads,
        head_dim,
        rope_head_dim,
        logical_block_size,
        rms_eps,
        state_dtype,
        norm_dtype,
        cache_dtype,
        cache_layout,
        scale_format,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires PyTorch") from exc
    try:
        from vllm.models.deepseek_v4.common.ops.save_partial_states import save_partial_states
        from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
            compress_norm_rope_store_cutedsl,
        )
        from vllm.utils.import_utils import has_cutedsl

        device = _require_h200_cutedsl(torch, has_cutedsl)
        operands = _build_operands(torch, shape, device)
        _check_output(torch, save_partial_states, compress_norm_rope_store_cutedsl, operands, shape)

        def run() -> None:
            _launch(save_partial_states, compress_norm_rope_store_cutedsl, operands, shape)

        # The semantic tail includes save_partial_states followed by the public
        # compressor/store callable: two launches for C4, three for C128.
        time_ms = Timer.cupti(run, kernel_name=None)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        flops, logical_bytes = _logical_work(shape)
        elapsed_s = time_ms / 1000.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(flops / elapsed_s / 1e12),
            memory_bandwidth_gbps=float(logical_bytes / elapsed_s / 1e9),
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc


__all__ = ["profile_deepseek_v4_sparse_attn_compress_store_cutedsl"]
