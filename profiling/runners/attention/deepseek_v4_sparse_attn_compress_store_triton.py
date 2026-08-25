"""Profile the public DeepSeek V4 indexer-compressor tail."""

from types import SimpleNamespace
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.deepseek_v4_sparse_attn_compress_store_cutedsl import (
    _POISON,
    _build_operands,
    _cache_row,
    _logical_work,
    _reference,
    _Shape,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_sparse_attn_compress_store:vllm_deepseek_v4_triton"
_GPU_NAME = "NVIDIA H200"


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
    identity = (
        compress_ratio,
        num_kv_heads,
        head_dim,
        rope_head_dim,
        logical_block_size,
        rms_eps,
        str(state_dtype),
        str(norm_dtype),
        cache_dtype,
        cache_layout,
        scale_format,
    )
    expected = (
        4,
        1,
        128,
        64,
        256,
        1.0e-6,
        "fp32",
        "bf16",
        "fp8_indexer",
        "block_segregated_data_then_scales",
        "fp32_per_token",
    )
    if identity != expected:
        raise ProfilerNotImplemented(f"{_BACKEND} supports identity {expected}, got {identity}")
    required_width = max(position // 4 + 1 for position in row_positions)
    if state_block_table_width < required_width:
        raise ValueError("state_block_table_width does not cover row_positions")
    return _Shape(
        row_positions,
        row_request_ids,
        state_block_table_width,
        4,
        512,
        4,
        64,
        head_dim=128,
        rope_head_dim=64,
        cache_row_bytes=132,
        token_stride=128,
        scale_dim=4,
        quant_block=128,
        page_alignment=576,
    )


def _require_h200(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    name = str(torch.cuda.get_device_name(device))
    if name != _GPU_NAME or tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        raise ProfilerNotImplemented(f"{_BACKEND} requires {_GPU_NAME} SM90, got {name}")
    return device


def _launch(save_op: Any, compress_op: Any, operands: Any, shape: _Shape) -> None:
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
    compress_op(
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
        overlap=True,
        use_fp4_cache=False,
        rms_norm_weight=operands.norm_weight,
        rms_norm_eps=1.0e-6,
        quant_block=shape.quant_block,
        token_stride=shape.token_stride,
        scale_dim=shape.scale_dim,
    )


def _check_output(torch: Any, save_op: Any, compress_op: Any, operands: Any, shape: _Shape) -> None:
    _launch(save_op, compress_op, operands, shape)
    torch.cuda.synchronize(operands.kv_cache.device)
    for row, (position, request) in enumerate(zip(shape.positions, shape.request_ids, strict=True)):
        page = int(operands.block_table[request, position // shape.state_block_size].item())
        expected_state = torch.cat(
            (operands.kv[row], operands.score[row] + operands.ape[position % shape.ratio])
        )
        torch.testing.assert_close(
            operands.state_cache[page, position % shape.state_block_size],
            expected_state,
            rtol=0,
            atol=0,
        )
    poison = torch.full((132,), _POISON, dtype=torch.uint8, device=operands.kv_cache.device)
    for slot, row in enumerate(operands.active_rows):
        packed = _cache_row(torch, operands, shape, slot)
        if slot < 3:
            expected_nope, expected_rope = _reference(torch, operands, shape, row)
            expected = torch.cat((expected_nope, expected_rope.float()))
            scale = packed[128:].view(torch.float32)
            actual = packed[:128].view(torch.float8_e4m3fn).float() * scale
            torch.testing.assert_close(actual, expected, rtol=0.20, atol=0.20)
    if not torch.equal(_cache_row(torch, operands, shape, len(operands.active_rows)), poison):
        raise KernelLaunchFailed(f"{_BACKEND} modified the unreferenced sentinel slot")
    operands.kv_cache.as_strided((operands.kv_cache.untyped_storage().nbytes(),), (1,)).fill_(
        _POISON
    )


def profile_deepseek_v4_sparse_attn_compress_store_triton(
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
        from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
            compress_norm_rope_store_triton,
        )
        from vllm.models.deepseek_v4.common.ops.save_partial_states import save_partial_states

        operands = _build_operands(torch, shape, _require_h200(torch))
        _check_output(torch, save_partial_states, compress_norm_rope_store_triton, operands, shape)

        def run() -> None:
            _launch(save_partial_states, compress_norm_rope_store_triton, operands, shape)

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


__all__ = ["profile_deepseek_v4_sparse_attn_compress_store_triton"]
