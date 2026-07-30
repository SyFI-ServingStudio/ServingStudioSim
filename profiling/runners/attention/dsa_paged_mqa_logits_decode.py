"""Torch profiler for GLM-5.2's paged-decode DSA MQA logits.

The timed callable is the complete semantic Torch composite, not DeepGEMM.
Accounting follows the production schedule's block-rounded context while
excluding Torch intermediates and physical cache/TMA transactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_NEXT_N = 1
_NUM_HEADS = 64
_HEAD_DIM = 128
_BLOCK_SIZE = 64
_Q_DTYPE = DType.FP8_E4M3
_CACHE_DTYPE = DType.FP8_E4M3
_SCALE_DTYPE = DType.FP32
_WEIGHT_DTYPE = DType.FP32
_OUTPUT_DTYPE = DType.FP32
_CONTEXT_MODE = "uniform"
_PAGE_MAPPING = "unique_scattered"
_CACHE_FORMAT = "page_planar_fp8_fp32_scale"
_REQUIRED_GPU = "NVIDIA H200"


@dataclass(frozen=True)
class _DsaPagedMqaLogitsDecodeOperands:
    q: Any
    cache: Any
    weights: Any
    context_lens: Any
    block_table: Any
    key_view: Any
    scale_view: Any
    logical_pages: int
    padded_context: int


def _validate_args(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    scale_dtype: DType | str,
    weight_dtype: DType | str,
    output_dtype: DType | str,
    context_mode: str,
    page_mapping: str,
    cache_format: str,
    clean_logits: bool,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    DType,
    DType,
    DType,
    DType,
    DType,
    str,
    str,
    str,
    bool,
]:
    batch_size = int(batch_size)
    context_len = int(context_len)
    next_n = int(next_n)
    max_model_len = int(max_model_len)
    num_heads = int(num_heads)
    head_dim = int(head_dim)
    block_size = int(block_size)
    q_dtype = DType.from_value(q_dtype)
    cache_dtype = DType.from_value(cache_dtype)
    scale_dtype = DType.from_value(scale_dtype)
    weight_dtype = DType.from_value(weight_dtype)
    output_dtype = DType.from_value(output_dtype)
    context_mode = str(context_mode)
    page_mapping = str(page_mapping)
    cache_format = str(cache_format)

    if batch_size <= 0 or context_len <= 0 or max_model_len <= 0:
        raise ValueError(
            "batch_size, context_len, and max_model_len must be > 0, got "
            f"{batch_size}, {context_len}, and {max_model_len}"
        )
    if context_len > max_model_len:
        raise ValueError(
            f"context_len must be <= max_model_len, got {context_len} and {max_model_len}"
        )
    if next_n != _NEXT_N:
        raise ValueError(f"dsa_paged_mqa_logits_decode requires next_n=1, got {next_n}")
    if (num_heads, head_dim, block_size) != (_NUM_HEADS, _HEAD_DIM, _BLOCK_SIZE):
        raise ValueError(
            "dsa_paged_mqa_logits_decode requires "
            "(num_heads, head_dim, block_size) == (64, 128, 64), "
            f"got ({num_heads}, {head_dim}, {block_size})"
        )
    if q_dtype is not _Q_DTYPE or cache_dtype is not _CACHE_DTYPE:
        raise ValueError(
            "dsa_paged_mqa_logits_decode requires "
            "q_dtype=cache_dtype=fp8_e4m3, "
            f"got {q_dtype.value} and {cache_dtype.value}"
        )
    if (
        scale_dtype is not _SCALE_DTYPE
        or weight_dtype is not _WEIGHT_DTYPE
        or output_dtype is not _OUTPUT_DTYPE
    ):
        raise ValueError(
            "dsa_paged_mqa_logits_decode requires "
            "scale_dtype=weight_dtype=output_dtype=fp32, got "
            f"{scale_dtype.value}, {weight_dtype.value}, and {output_dtype.value}"
        )
    if context_mode != _CONTEXT_MODE:
        raise ValueError(
            f"dsa_paged_mqa_logits_decode requires context_mode='{_CONTEXT_MODE}', "
            f"got {context_mode!r}"
        )
    if page_mapping != _PAGE_MAPPING:
        raise ValueError(
            f"dsa_paged_mqa_logits_decode requires page_mapping='{_PAGE_MAPPING}', "
            f"got {page_mapping!r}"
        )
    if cache_format != _CACHE_FORMAT:
        raise ValueError(
            f"dsa_paged_mqa_logits_decode requires cache_format='{_CACHE_FORMAT}', "
            f"got {cache_format!r}"
        )
    if not isinstance(clean_logits, bool):
        raise TypeError("clean_logits must be a bool")
    if clean_logits:
        raise ValueError("dsa_paged_mqa_logits_decode requires clean_logits=false")

    return (
        batch_size,
        context_len,
        next_n,
        max_model_len,
        num_heads,
        head_dim,
        block_size,
        q_dtype,
        cache_dtype,
        scale_dtype,
        weight_dtype,
        output_dtype,
        context_mode,
        page_mapping,
        cache_format,
        clean_logits,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch dsa_paged_mqa_logits_decode backend"
        )
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(
            f"torch dsa_paged_mqa_logits_decode is verified only on {_REQUIRED_GPU}, got {gpu_name}"
        )


def _stable_values(torch: Any, shape: tuple[int, ...], *, phase: int, device: str) -> Any:
    total = 1
    for dimension in shape:
        total *= dimension
    values = torch.arange(total, dtype=torch.int64, device=device)
    values = ((values + phase) % 17 - 8).to(torch.float32) / 16.0
    return values.reshape(shape)


def _page_views(
    torch: Any,
    cache: Any,
    *,
    block_size: int,
    head_dim: int,
) -> tuple[Any, Any]:
    """Expose page-planar E4M3 keys and raw-FP32 scales as cache aliases."""
    num_pages = cache.shape[0]
    page_key_bytes = block_size * head_dim
    pages = cache.reshape(num_pages, block_size * (head_dim + 4))
    key_view = (
        pages[:, :page_key_bytes].reshape(num_pages, block_size, head_dim).view(torch.float8_e4m3fn)
    )
    scale_bytes = pages[:, page_key_bytes:].reshape(num_pages, block_size, 4)
    scale_view = scale_bytes.view(torch.float32).squeeze(-1)
    return key_view, scale_view


def _build_operands(
    torch: Any,
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    device: str,
) -> _DsaPagedMqaLogitsDecodeOperands:
    """Construct exact GLM operands and deterministic unique scattered pages."""
    del max_model_len
    logical_pages = (context_len + block_size - 1) // block_size
    padded_context = logical_pages * block_size
    referenced_page_count = batch_size * logical_pages
    num_pages = max(256, 2 * referenced_page_count + 1)

    q = _stable_values(
        torch,
        (batch_size, next_n, num_heads, head_dim),
        phase=1,
        device=device,
    ).to(torch.float8_e4m3fn)
    weights = (
        _stable_values(
            torch,
            (batch_size * next_n, num_heads),
            phase=9,
            device=device,
        )
        / 4.0
    )
    context_lens = torch.full(
        (batch_size, next_n),
        context_len,
        dtype=torch.int32,
        device=device,
    )
    physical_pages = (
        2
        * torch.arange(
            referenced_page_count,
            dtype=torch.int32,
            device=device,
        )
        + 1
    )
    block_table = physical_pages.reshape(batch_size, logical_pages).contiguous()

    cache = torch.empty(
        (num_pages, block_size, 1, head_dim + 4),
        dtype=torch.uint8,
        device=device,
    )
    key_view, scale_view = _page_views(
        torch,
        cache,
        block_size=block_size,
        head_dim=head_dim,
    )
    key_view.copy_(
        _stable_values(
            torch,
            (num_pages, block_size, head_dim),
            phase=5,
            device=device,
        ).to(torch.float8_e4m3fn)
    )
    scale_view.copy_(
        0.75
        + (
            torch.arange(
                num_pages * block_size,
                dtype=torch.float32,
                device=device,
            ).reshape(num_pages, block_size)
            % 7
        )
        / 16.0
    )

    return _DsaPagedMqaLogitsDecodeOperands(
        q=q,
        cache=cache,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        key_view=key_view,
        scale_view=scale_view,
        logical_pages=logical_pages,
        padded_context=padded_context,
    )


def _torch_composite(
    operands: _DsaPagedMqaLogitsDecodeOperands,
    *,
    context_len: int,
    max_model_len: int,
) -> Any:
    """Compute the complete vectorized paged-decode semantic composite."""
    batch_size, next_n, num_heads, head_dim = operands.q.shape
    physical_pages = operands.block_table[:, : operands.logical_pages].long()
    keys = operands.key_view[physical_pages].reshape(batch_size, operands.padded_context, head_dim)
    scales = operands.scale_view[physical_pages].reshape(batch_size, operands.padded_context)
    row_keys = (
        keys.unsqueeze(1)
        .expand(batch_size, next_n, operands.padded_context, head_dim)
        .reshape(batch_size * next_n, operands.padded_context, head_dim)
    )
    row_scales = (
        scales.unsqueeze(1)
        .expand(batch_size, next_n, operands.padded_context)
        .reshape(batch_size * next_n, operands.padded_context)
    )
    queries = operands.q.reshape(batch_size * next_n, num_heads, head_dim)
    dots = torch_einsum(queries.float(), row_keys.float())
    reduced = (dots.relu() * operands.weights.unsqueeze(-1)).sum(dim=1) * row_scales
    output = reduced.new_full((batch_size * next_n, max_model_len), float("nan"))
    output[:, :context_len] = reduced[:, :context_len]
    return output


def torch_einsum(queries: Any, keys: Any) -> Any:
    """Keep the contraction isolated for tests without importing Torch globally."""
    return queries @ keys.transpose(-1, -2)


def _scheduled_work(
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
) -> tuple[int, int, int]:
    """Return ``(logical_pages, C, nominal_flops)`` for the padded schedule."""
    if min(batch_size, context_len, next_n, num_heads, head_dim, block_size) <= 0:
        raise ValueError("scheduled-work dimensions must be > 0")
    logical_pages = (context_len + block_size - 1) // block_size
    scheduled_cells = batch_size * next_n * logical_pages * block_size
    nominal_flops = 2 * scheduled_cells * num_heads * head_dim
    return logical_pages, scheduled_cells, nominal_flops


def _logical_scheduled_bytes(
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
) -> int:
    """Return scheduled logical bytes, excluding Torch/TMA intermediates.

    The count covers Q, weights, block-rounded cache keys/scales, block/context
    metadata, and ``4 * C`` output bytes. ``max_model_len`` deliberately is not
    an argument: it controls the returned allocation shape, not scheduled cache
    work in the production kernel model.
    """
    logical_pages, scheduled_cells, _nominal_flops = _scheduled_work(
        batch_size=batch_size,
        context_len=context_len,
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
    )
    return (
        batch_size * next_n * num_heads * head_dim
        + 4 * batch_size * next_n * num_heads
        + (head_dim + 4) * scheduled_cells
        + 4 * batch_size * logical_pages
        + 4 * batch_size * next_n
        + 4 * scheduled_cells
    )


def profile_dsa_paged_mqa_logits_decode_torch(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    scale_dtype: DType | str,
    weight_dtype: DType | str,
    output_dtype: DType | str,
    context_mode: str,
    page_mapping: str,
    cache_format: str,
    clean_logits: bool,
) -> ComputeMetrics:
    """Profile the complete Torch paged-decode DSA-logits composite."""
    (
        batch_size,
        context_len,
        next_n,
        max_model_len,
        num_heads,
        head_dim,
        block_size,
        _q_dtype,
        _cache_dtype,
        _scale_dtype,
        _weight_dtype,
        _output_dtype,
        _context_mode,
        _page_mapping,
        _cache_format,
        _clean_logits,
    ) = _validate_args(
        batch_size,
        context_len,
        next_n,
        max_model_len,
        num_heads,
        head_dim,
        block_size,
        q_dtype,
        cache_dtype,
        scale_dtype,
        weight_dtype,
        output_dtype,
        context_mode,
        page_mapping,
        cache_format,
        clean_logits,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch dsa_paged_mqa_logits_decode backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(
            torch,
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            max_model_len=max_model_len,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            device="cuda",
        )

        def kernel() -> Any:
            return _torch_composite(
                operands,
                context_len=context_len,
                max_model_len=max_model_len,
            )

        time_ms = Timer.cuda_event(kernel, warmup=5)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )
        _logical_pages, _scheduled_cells, nominal_flops = _scheduled_work(
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
        logical_bytes = _logical_scheduled_bytes(
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
        elapsed_seconds = time_ms / 1000.0
        tflops = nominal_flops / elapsed_seconds / 1e12 if elapsed_seconds > 0 else 0.0
        bandwidth_gbps = logical_bytes / elapsed_seconds / 1e9 if elapsed_seconds > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
