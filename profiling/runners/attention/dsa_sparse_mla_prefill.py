"""Profile the production SM100 varlen sparse-MLA prefill launch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.dsa_sparse_mla_attention import (
    _INDEX_DISTRIBUTIONS,
    _LATENT_DIM,
    _ROPE_DIM,
    _SCORE_DIM,
    _TRTLLM_FP8_CACHE_LAYOUT,
    _TRTLLM_KERNEL_NAME,
    _TRTLLM_PAGE_SIZE,
    _TRTLLM_WORKSPACE_BYTES,
    _VALUE_DIM,
    _launch_trtllm_fp8,
    _logical_bytes,
    _logical_flops,
    _row_indices,
    _TrtllmFp8Operands,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "dsa_sparse_mla_prefill:flashinfer_trtllm_fp8"
_NUM_KV_HEADS = 1


@dataclass(frozen=True)
class _Shape:
    num_heads: int
    selected_k: int
    pairs: tuple[tuple[int, int], ...]
    valid_counts: tuple[int, ...]
    request_page_offsets: tuple[int, ...]
    num_pages: int
    index_distribution: str

    @property
    def num_queries(self) -> int:
        return len(self.valid_counts)


def _validate_args(
    query_context_pairs: tuple[tuple[int, int], ...],
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    index_distribution: str,
    cache_layout: str,
) -> _Shape:
    if not isinstance(query_context_pairs, tuple) or not query_context_pairs:
        raise ValueError("query_context_pairs must be a nonempty tuple")
    if type(selected_k) is not int:
        raise TypeError("selected_k must be an integer")
    # The page table is selected_k wide; FlashInfer only checks that it matches
    # sparse_mla_top_k, so any positive width is a launchable shape.
    if selected_k < 1:
        raise ValueError(f"selected_k must be >= 1, got {selected_k}")

    pairs: list[tuple[int, int]] = []
    valid_counts: list[int] = []
    request_page_offsets: list[int] = []
    num_pages = 0
    for pair in query_context_pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
        ):
            raise TypeError("query_context_pairs must contain integer (query, context) pairs")
        num_queries, context_len = pair
        if num_queries <= 0 or context_len < num_queries:
            raise ValueError("each pair must satisfy 0 < query <= context")
        pairs.append(pair)
        request_page_offsets.append(num_pages)
        num_pages += math.ceil(context_len / _TRTLLM_PAGE_SIZE)
        first_query_position = context_len - num_queries
        valid_counts.extend(
            min(position + 1, selected_k) for position in range(first_query_position, context_len)
        )

    # num_heads is the per-rank q-head count (GLM's 64 heads over TP); the
    # kernel launch, output check and metrics are parametric in it, as in
    # selected_k. The other fields fix the MLA latent+rope cache layout.
    if type(num_heads) is not int:
        raise TypeError("num_heads must be an integer")
    if num_heads < 1:
        raise ValueError(f"num_heads must be >= 1, got {num_heads}")
    model_identity = (
        num_kv_heads,
        latent_dim,
        rope_dim,
        value_dim,
    )
    expected_model_identity = (
        _NUM_KV_HEADS,
        _LATENT_DIM,
        _ROPE_DIM,
        _VALUE_DIM,
    )
    if model_identity != expected_model_identity:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires model identity {expected_model_identity}, got {model_identity}"
        )
    if type(softmax_scale) not in {int, float} or isinstance(softmax_scale, bool):
        raise TypeError("softmax_scale must be a real number")
    # bmm1_scale is a runtime scalar; production passes 1/sqrt(qk_head_dim).
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) <= 0.0:
        raise ValueError(f"softmax_scale must be positive and finite, got {softmax_scale}")
    storage_identity = (
        DType.from_value(q_dtype),
        DType.from_value(cache_dtype),
        index_dtype,
        DType.from_value(output_dtype),
        cache_layout,
    )
    expected_storage_identity = (
        DType.FP8_E4M3,
        DType.FP8_E4M3,
        "int32",
        DType.BF16,
        _TRTLLM_FP8_CACHE_LAYOUT,
    )
    if storage_identity != expected_storage_identity:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires storage identity {expected_storage_identity}, "
            f"got {storage_identity}"
        )
    if index_distribution not in _INDEX_DISTRIBUTIONS:
        raise ProfilerNotImplemented(
            f"index_distribution must be one of: {', '.join(sorted(_INDEX_DISTRIBUTIONS))}"
        )

    return _Shape(
        num_heads=num_heads,
        selected_k=selected_k,
        pairs=tuple(pairs),
        valid_counts=tuple(valid_counts),
        request_page_offsets=tuple(request_page_offsets),
        num_pages=num_pages,
        index_distribution=index_distribution,
    )


def _build_operands(torch: Any, shape: _Shape, *, device: Any) -> _TrtllmFp8Operands:
    fp8 = torch.float8_e4m3fn
    query = torch.empty(
        (shape.num_queries, 1, shape.num_heads, _SCORE_DIM),
        dtype=fp8,
        device=device,
    )
    query.copy_(
        torch.linspace(-0.75, 0.75, _SCORE_DIM, device=device)
        .view(1, 1, 1, _SCORE_DIM)
        .expand_as(query)
    )
    cache = torch.empty(
        (shape.num_pages, 1, _TRTLLM_PAGE_SIZE, _SCORE_DIM),
        dtype=fp8,
        device=device,
    )
    cache.copy_(torch.linspace(-0.5, 0.5, _SCORE_DIM, device=device).view(1, 1, 1, _SCORE_DIM))
    block_tables = torch.zeros(
        (shape.num_queries, 1, shape.selected_k),
        dtype=torch.int32,
        device=device,
    )

    query_row = 0
    for (num_queries, context_len), page_offset in zip(
        shape.pairs, shape.request_page_offsets, strict=True
    ):
        first_query_position = context_len - num_queries
        slot_offset = page_offset * _TRTLLM_PAGE_SIZE
        for position in range(first_query_position, context_len):
            count = shape.valid_counts[query_row]
            indices = _row_indices(
                torch,
                row=query_row,
                count=count,
                num_cache_tokens=position + 1,
                distribution=shape.index_distribution,
                device=device,
            )
            block_tables[query_row, 0, :count].copy_(indices.to(torch.int32) + slot_offset)
            query_row += 1

    return _TrtllmFp8Operands(
        query=query,
        cache=cache,
        block_tables=block_tables,
        seq_lens=torch.tensor(shape.valid_counts, dtype=torch.int32, device=device),
        workspace=torch.zeros(_TRTLLM_WORKSPACE_BYTES, dtype=torch.uint8, device=device),
    )


def profile_dsa_sparse_mla_prefill_flashinfer_trtllm_fp8(
    query_context_pairs: tuple[tuple[int, int], ...],
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    index_distribution: str,
    cache_layout: str,
) -> ComputeMetrics:
    shape = _validate_args(
        query_context_pairs,
        num_heads,
        num_kv_heads,
        selected_k,
        latent_dim,
        rope_dim,
        value_dim,
        softmax_scale,
        q_dtype,
        cache_dtype,
        index_dtype,
        output_dtype,
        index_distribution,
        cache_layout,
    )
    try:
        import torch
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    except (ImportError, ModuleNotFoundError) as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the repository vllm_env") from exc

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, shape, device=device)

        def kernel() -> Any:
            return _launch_trtllm_fp8(
                trtllm_batch_decode_with_kv_cache_mla,
                operands,
                softmax_scale=float(softmax_scale),
                selected_k=shape.selected_k,
            )

        output = kernel()
        torch.cuda.synchronize(device)
        expected_shape = (shape.num_queries, 1, shape.num_heads, _VALUE_DIM)
        if output.dtype is not torch.bfloat16 or tuple(output.shape) != expected_shape:
            raise KernelLaunchFailed(
                f"{_BACKEND} returned {output.dtype} {tuple(output.shape)}, "
                f"expected BF16 {expected_shape}"
            )
        if not torch.isfinite(output).all():
            raise KernelLaunchFailed(f"{_BACKEND} output must be finite")

        time_ms = Timer.cupti(kernel, kernel_name=_TRTLLM_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of GPU memory") from exc
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} native callable failed") from exc

    flops = _logical_flops(num_heads=shape.num_heads, valid_counts=shape.valid_counts)
    logical_bytes = _logical_bytes(
        num_queries=shape.num_queries,
        num_heads=shape.num_heads,
        selected_k=shape.selected_k,
        valid_counts=shape.valid_counts,
        q_bytes=1,
        cache_bytes=1,
    )
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / seconds / 1e12,
        memory_bandwidth_gbps=logical_bytes / seconds / 1e9,
    )


__all__ = ["profile_dsa_sparse_mla_prefill_flashinfer_trtllm_fp8"]
