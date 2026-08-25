"""Profile DeepSeek V4's public DeepGEMM paged indexer-decode call."""

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_indexer_mqa_logits_decode:vllm_deepgemm_fp8"
_GPU_NAME = "NVIDIA H200"
_MODEL_IDENTITY = (1, 64, 128, 64)
_STORAGE_IDENTITY = (
    "fp8_e4m3",
    "fp8_e4m3",
    "fp32",
    "fp32",
    "fp32",
    "max_ragged",
    "request_contiguous",
    "fp8_e4m3_ue8m0",
    False,
)
_CACHE_ROW_BYTES = 132
_PAGE_ALIGNMENT = 576
_H200_SMS = 132


@dataclass(frozen=True)
class _Operands:
    q: Any
    cache: Any
    weights: Any
    context_lens: Any
    block_table: Any
    key_view: Any
    scale_view: Any
    page_stride_bytes: int


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _max_ragged_context_lengths(batch_size: int, context_len: int) -> tuple[int, ...]:
    """Keep the largest possible distinct positive lengths under ``context_len``."""
    return tuple(max(1, context_len - request_index) for request_index in range(batch_size))


def _validate_args(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    q_dtype: object,
    cache_dtype: object,
    scale_dtype: object,
    weight_dtype: object,
    output_dtype: object,
    context_mode: str,
    page_mapping: str,
    cache_format: str,
    clean_logits: bool,
) -> tuple[int, ...]:
    integer_identity = (next_n, num_heads, head_dim, block_size)
    if integer_identity != _MODEL_IDENTITY:
        raise ProfilerNotImplemented(f"{_BACKEND} supports model identity {_MODEL_IDENTITY}")
    storage_identity = (
        str(q_dtype),
        str(cache_dtype),
        str(scale_dtype),
        str(weight_dtype),
        str(output_dtype),
        context_mode,
        page_mapping,
        cache_format,
        clean_logits,
    )
    if storage_identity != _STORAGE_IDENTITY:
        raise ProfilerNotImplemented(f"{_BACKEND} supports storage identity {_STORAGE_IDENTITY}")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive int")
    if type(context_len) is not int or context_len < 1:
        raise ValueError("context_len must be a positive int")
    if type(max_model_len) is not int or not context_len <= max_model_len <= 1_048_576:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires context_len <= max_model_len <= 1048576"
        )
    return _max_ragged_context_lengths(batch_size, context_len)


def _stable_values(torch: Any, shape: tuple[int, ...], phase: int, device: Any) -> Any:
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    values = torch.arange(element_count, dtype=torch.int64, device=device)
    return (((values + phase) % 17 - 8).to(torch.float32) / 16.0).reshape(shape)


def _allocate_aligned_cache(
    torch: Any,
    *,
    num_blocks: int,
    block_size: int,
    head_dim: int,
    device: Any,
) -> tuple[Any, Any, Any, int]:
    """Allocate vLLM's page-planar cache with its padded block stride.

    The public tensor keeps the conventional ``[..., head_dim + 4]`` shape,
    but DeepGEMM interprets each page as all FP8 key bytes followed by all FP32
    scale bytes.  The allocator's 576-byte alignment only adds tail padding
    between pages; it does not change that within-page layout.
    """
    page_bytes = block_size * (head_dim + 4)
    page_stride_bytes = _round_up(page_bytes, _PAGE_ALIGNMENT)
    backing = torch.empty(num_blocks * page_stride_bytes, dtype=torch.uint8, device=device)
    pages = torch.as_strided(
        backing,
        size=(num_blocks, page_stride_bytes),
        stride=(page_stride_bytes, 1),
    )
    cache = torch.as_strided(
        backing,
        size=(num_blocks, block_size, 1, head_dim + 4),
        stride=(page_stride_bytes, head_dim + 4, head_dim + 4, 1),
    )
    key_bytes = block_size * head_dim
    key_view = pages[:, :key_bytes].reshape(num_blocks, block_size, 1, head_dim).view(
        torch.float8_e4m3fn
    )
    scale_view = (
        pages[:, key_bytes:page_bytes]
        .reshape(num_blocks, block_size, 1, 4)
        .view(torch.float32)
        .squeeze(-1)
    )
    return cache, key_view, scale_view, page_stride_bytes


def _build_operands(
    torch: Any,
    *,
    context_lengths: tuple[int, ...],
    next_n: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    device: Any,
) -> _Operands:
    batch_size = len(context_lengths)
    pages_per_request = tuple((length + block_size - 1) // block_size for length in context_lengths)
    max_pages = max(pages_per_request)
    num_blocks = sum(pages_per_request)
    block_table = torch.full(
        (batch_size, max_pages), -1, dtype=torch.int32, device=device
    )
    next_block = 0
    for request_index, page_count in enumerate(pages_per_request):
        block_table[request_index, :page_count] = torch.arange(
            next_block, next_block + page_count, dtype=torch.int32, device=device
        )
        next_block += page_count

    cache, key_view, scale_view, page_stride_bytes = _allocate_aligned_cache(
        torch,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        device=device,
    )
    key_template = _stable_values(torch, (block_size, 1, head_dim), 5, device).to(
        torch.float8_e4m3fn
    )
    scale_template = 0.75 + _stable_values(torch, (block_size, 1), 13, device) / 8.0
    key_view.copy_(key_template)
    scale_view.copy_(scale_template)
    q = _stable_values(torch, (batch_size, next_n, num_heads, head_dim), 1, device).to(
        torch.float8_e4m3fn
    )
    weights = _stable_values(torch, (batch_size * next_n, num_heads), 9, device) / 4.0
    context_lens = torch.tensor(context_lengths, dtype=torch.int32, device=device).unsqueeze(1)
    return _Operands(
        q=q,
        cache=cache,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        key_view=key_view,
        scale_view=scale_view,
        page_stride_bytes=page_stride_bytes,
    )


def _launch(public_op: Any, operands: _Operands, schedule_metadata: Any, max_model_len: int) -> Any:
    return public_op(
        (operands.q, None),
        operands.cache,
        operands.weights,
        operands.context_lens,
        operands.block_table,
        schedule_metadata,
        max_model_len,
        clean_logits=False,
    )


def _expected_logits_row(torch: Any, operands: _Operands, request_index: int) -> Any:
    context_len = int(operands.context_lens[request_index, 0].item())
    page_count = (context_len + 63) // 64
    blocks = operands.block_table[request_index, :page_count].long()
    keys = operands.key_view.index_select(0, blocks).reshape(-1, 128)[:context_len].float()
    scales = operands.scale_view.index_select(0, blocks).reshape(-1)[:context_len]
    per_head = operands.q[request_index, 0].float() @ keys.T
    return (per_head.relu() * operands.weights[request_index, :, None]).sum(dim=0) * scales


def _check_output(torch: Any, actual: Any, operands: _Operands, max_model_len: int) -> None:
    batch_size = operands.q.shape[0]
    if actual.dtype != torch.float32 or actual.shape != (batch_size, max_model_len):
        raise KernelLaunchFailed(f"{_BACKEND} returned the wrong output shape or dtype")
    sampled_requests = sorted({0, batch_size // 2, batch_size - 1})
    for request_index in sampled_requests:
        expected = _expected_logits_row(torch, operands, request_index)
        context_len = expected.numel()
        sampled_columns = sorted({0, context_len // 2, context_len - 1})
        columns = torch.tensor(sampled_columns, dtype=torch.int64, device=actual.device)
        torch.testing.assert_close(
            actual[request_index].index_select(0, columns),
            expected.index_select(0, columns),
            atol=2e-4,
            rtol=2e-4,
        )


def profile_deepseek_v4_indexer_mqa_logits_decode_deepgemm(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    q_dtype: object,
    cache_dtype: object,
    scale_dtype: object,
    weight_dtype: object,
    output_dtype: object,
    context_mode: str,
    page_mapping: str,
    cache_format: str,
    clean_logits: bool,
) -> ComputeMetrics:
    context_lengths = _validate_args(
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
        from vllm.utils.deep_gemm import (
            fp8_fp4_paged_mqa_logits,
            get_paged_mqa_logits_metadata,
        )
    except (ImportError, OSError) as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires pinned vLLM DeepGEMM") from exc

    try:
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
        device = torch.device("cuda", torch.cuda.current_device())
        if (
            torch.cuda.get_device_name(device) != _GPU_NAME
            or tuple(torch.cuda.get_device_capability(device)) != (9, 0)
            or torch.cuda.get_device_properties(device).multi_processor_count != _H200_SMS
        ):
            raise ProfilerNotImplemented(f"{_BACKEND} requires the 132-SM {_GPU_NAME}")
        operands = _build_operands(
            torch,
            context_lengths=context_lengths,
            next_n=next_n,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            device=device,
        )
        # Production builds scheduler metadata before entering the public op.
        schedule_metadata = get_paged_mqa_logits_metadata(
            operands.context_lens, block_size=block_size, num_sms=_H200_SMS
        )

        def launch() -> Any:
            return _launch(
                fp8_fp4_paged_mqa_logits, operands, schedule_metadata, max_model_len
            )

        actual = launch()
        torch.cuda.synchronize(device)
        _check_output(torch, actual, operands, max_model_len)
        time_ms = Timer.cupti(launch, kernel_name=None)
        energy_j = Energy.perf(launch, per_iter_time_ms=time_ms)
        # DeepGEMM schedules complete cache blocks even for each ragged tail.
        # Report the same block-rounded work as the timed callable rather than
        # presenting only mathematically valid rows as the hardware workload.
        scheduled_tokens = sum(_round_up(length, block_size) for length in context_lengths)
        nominal_flops = 2 * scheduled_tokens * num_heads * head_dim
        logical_bytes = (
            batch_size * next_n * num_heads * head_dim
            + batch_size * next_n * num_heads * 4
            + scheduled_tokens * _CACHE_ROW_BYTES
            + batch_size * ((context_len + block_size - 1) // block_size) * 4
            + batch_size * 4
            + scheduled_tokens * 4
        )
        elapsed_seconds = time_ms / 1000.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(nominal_flops / elapsed_seconds / 1e12),
            memory_bandwidth_gbps=float(logical_bytes / elapsed_seconds / 1e9),
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc


__all__ = ["profile_deepseek_v4_indexer_mqa_logits_decode_deepgemm"]
