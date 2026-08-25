"""Profile DeepSeek V4's public BF16 FlashMLA sparse-prefill operation."""

import math
from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_sparse_mla_prefill:vllm_flashmla_bf16"
_GPU_NAME = "NVIDIA H200"
_MODEL_IDENTITY = (4, 128, 64, 1, 512, 512)
_SOFTMAX_SCALE = 1.0 / math.sqrt(512)
_STORAGE_IDENTITY = (
    "bf16",
    "bf16",
    "int32",
    "bf16",
    "request_slot_major_flat_mqa_bf16_d512",
)


@dataclass(frozen=True)
class _Shape:
    pairs: tuple[tuple[int, int], ...]
    max_model_len: int
    max_num_batched_tokens: int
    ratio: int
    selected_k: int
    padded_topk: int

    @property
    def num_queries(self) -> int:
        return sum(query for query, _ in self.pairs)

    @property
    def compressed_capacity(self) -> int:
        return 0 if self.ratio == 1 else math.ceil(self.max_model_len / self.ratio)

    @property
    def request_slot_size(self) -> int:
        return self.compressed_capacity + 128 + self.max_num_batched_tokens


@dataclass(frozen=True)
class _Operands:
    q: Any
    cache: Any
    indices: Any
    valid_lengths: Any
    output: Any


def _validate_args(
    query_context_pairs: tuple[tuple[int, int], ...],
    max_model_len: int,
    max_num_batched_tokens: int,
    prefill_chunk_size: int,
    compress_ratio: int,
    window_size: int,
    selected_k: int,
    selected_index_pattern: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: object,
    cache_dtype: object,
    index_dtype: str,
    output_dtype: object,
    cache_layout: str,
) -> _Shape:
    if not query_context_pairs or len(query_context_pairs) > 64:
        raise ProfilerNotImplemented(f"{_BACKEND} supports 1..64 requests")
    for pair in query_context_pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
        ):
            raise TypeError("query_context_pairs must contain integer (query, context) pairs")
        query, context = pair
        if query <= 0 or context < query:
            raise ValueError("each pair must satisfy 0 < query <= context")
    total_queries = sum(query for query, _ in query_context_pairs)
    if type(max_model_len) is not int or not 1 <= max_model_len <= 1_048_576:
        raise ProfilerNotImplemented(f"{_BACKEND} supports max_model_len <= 1048576")
    if max(context for _, context in query_context_pairs) > max_model_len:
        raise ValueError("context length exceeds max_model_len")
    if (
        type(max_num_batched_tokens) is not int
        or not total_queries <= max_num_batched_tokens <= 32768
    ):
        raise ValueError("max_num_batched_tokens must cover all query tokens and be <=32768")
    model_identity = (
        prefill_chunk_size,
        window_size,
        num_heads,
        num_kv_heads,
        head_dim,
        value_dim,
    )
    if model_identity != _MODEL_IDENTITY:
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports model identity {_MODEL_IDENTITY}, got {model_identity}"
        )
    if not math.isclose(softmax_scale, _SOFTMAX_SCALE, rel_tol=0.0, abs_tol=1e-15):
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires softmax_scale={_SOFTMAX_SCALE}, got {softmax_scale}"
        )
    if compress_ratio not in (1, 4, 128):
        raise ProfilerNotImplemented(f"{_BACKEND} supports compress_ratio=1/4/128")
    expected_selected_k = 0 if compress_ratio == 1 else 512
    if selected_k != expected_selected_k:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires selected_k={expected_selected_k} for C{compress_ratio}"
        )
    if selected_index_pattern != "request_local_topk_plus_swa":
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports request_local_topk_plus_swa selected indices"
        )
    storage_identity = (
        str(q_dtype),
        str(cache_dtype),
        index_dtype,
        str(output_dtype),
        cache_layout,
    )
    if storage_identity != _STORAGE_IDENTITY:
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports storage identity {_STORAGE_IDENTITY}, got {storage_identity}"
        )
    padded_topk = 128 if compress_ratio == 1 else 640
    return _Shape(
        query_context_pairs,
        max_model_len,
        max_num_batched_tokens,
        compress_ratio,
        selected_k,
        padded_topk,
    )


def _chunks(shape: _Shape) -> tuple[_Shape, ...]:
    return tuple(
        _Shape(
            shape.pairs[start : start + 4],
            shape.max_model_len,
            shape.max_num_batched_tokens,
            shape.ratio,
            shape.selected_k,
            shape.padded_topk,
        )
        for start in range(0, len(shape.pairs), 4)
    )


def _build_indices(torch: Any, shape: _Shape, device: Any) -> tuple[Any, Any]:
    indices = torch.full(
        (shape.num_queries, shape.padded_topk), -1, dtype=torch.int32, device=device
    )
    lengths = torch.empty(shape.num_queries, dtype=torch.int32, device=device)
    query_base = 0
    for request, (query_count, context) in enumerate(shape.pairs):
        positions = torch.arange(context - query_count, context, dtype=torch.int64, device=device)
        compressed_available = torch.div(positions + 1, shape.ratio, rounding_mode="floor")
        compressed_lengths = torch.minimum(
            compressed_available,
            torch.full_like(compressed_available, shape.selected_k),
        )
        swa_lengths = torch.minimum(positions + 1, torch.full_like(positions, 128))
        rows = indices[query_base : query_base + query_count]
        request_offset = request * shape.request_slot_size
        if shape.selected_k:
            columns = torch.arange(shape.selected_k, dtype=torch.int64, device=device)
            denominators = torch.clamp(compressed_lengths - 1, min=1)
            spread = torch.div(
                columns[None, :] * torch.clamp(compressed_available - 1, min=0)[:, None],
                denominators[:, None],
                rounding_mode="floor",
            )
            rows[:, : shape.selected_k].copy_(
                torch.where(
                    columns[None, :] < compressed_lengths[:, None],
                    request_offset + spread,
                    -1,
                ).to(torch.int32)
            )
        gather_len = query_count + min(context - query_count, 127)
        gather_start = context - gather_len
        offsets = torch.arange(128, dtype=torch.int64, device=device)
        columns = compressed_lengths[:, None] + offsets[None, :]
        swa_values = (
            request_offset
            + shape.compressed_capacity
            + positions[:, None]
            - swa_lengths[:, None]
            + 1
            - gather_start
            + offsets[None, :]
        )
        rows.scatter_(
            1,
            columns,
            torch.where(offsets[None, :] < swa_lengths[:, None], swa_values, -1).to(torch.int32),
        )
        lengths[query_base : query_base + query_count] = (compressed_lengths + swa_lengths).to(
            torch.int32
        )
        query_base += query_count
    return indices.unsqueeze(1), lengths


def _build_operands(torch: Any, shape: _Shape, device: Any) -> _Operands:
    indices, valid_lengths = _build_indices(torch, shape, device)
    feature = torch.linspace(-0.5, 0.5, 512, dtype=torch.float32, device=device)
    query_rows = torch.arange(shape.num_queries, dtype=torch.float32, device=device)
    heads = torch.arange(64, dtype=torch.float32, device=device)
    q = (
        feature[None, None, :] + query_rows[:, None, None] / 8192.0 + heads[None, :, None] / 2048.0
    ).to(torch.bfloat16)
    cache = torch.zeros((4 * shape.request_slot_size, 1, 512), dtype=torch.bfloat16, device=device)
    selected_rows = torch.unique(indices[indices >= 0].to(torch.int64))
    for start in range(0, selected_rows.numel(), 4096):
        rows = selected_rows[start : start + 4096]
        values = feature[None, :] + rows.float().remainder(257)[:, None] / 512.0
        cache[rows, 0] = values.to(torch.bfloat16)
    output = torch.empty_like(q)
    return _Operands(q, cache, indices, valid_lengths, output)


def _launch(flash_mla_sparse_fwd: Any, operands: _Operands) -> Any:
    return flash_mla_sparse_fwd(
        q=operands.q,
        kv=operands.cache,
        indices=operands.indices,
        sm_scale=512**-0.5,
        d_v=512,
        attn_sink=None,
        topk_length=operands.valid_lengths,
        out=operands.output,
    )


def _reference_rows(torch: Any, operands: _Operands, rows: tuple[int, ...]) -> Any:
    expected = []
    for row in rows:
        length = int(operands.valid_lengths[row].item())
        selected = operands.indices[row, 0, :length].to(torch.int64)
        cache = operands.cache[selected, 0].float()
        scores = torch.einsum("hd,kd->hk", operands.q[row].float(), cache)
        probabilities = torch.softmax(scores * (512**-0.5), dim=-1)
        expected.append(torch.einsum("hk,kd->hd", probabilities, cache).to(torch.bfloat16))
    return torch.stack(expected)


def _check_output(torch: Any, flash_mla_sparse_fwd: Any, operands: _Operands) -> None:
    returned = _launch(flash_mla_sparse_fwd, operands)
    torch.cuda.synchronize(operands.q.device)
    if not isinstance(returned, (tuple, list)) or len(returned) != 3:
        raise KernelLaunchFailed(f"{_BACKEND} returned an invalid result tuple")
    output, max_logits, lse = returned
    if output.data_ptr() != operands.output.data_ptr():
        raise KernelLaunchFailed(f"{_BACKEND} did not use the preallocated output")
    if max_logits.shape != lse.shape or max_logits.shape != operands.q.shape[:2]:
        raise KernelLaunchFailed(f"{_BACKEND} returned invalid diagnostic shapes")
    sample_rows = tuple(sorted({0, operands.q.shape[0] // 2, operands.q.shape[0] - 1}))
    expected = _reference_rows(torch, operands, sample_rows)
    torch.testing.assert_close(
        output[list(sample_rows)].float(), expected.float(), atol=8e-4, rtol=3.01 / 128
    )


def _logical_work(operands: tuple[_Operands, ...]) -> tuple[int, int]:
    queries = sum(chunk.q.shape[0] for chunk in operands)
    valid_pairs = sum(int(chunk.valid_lengths.sum().item()) for chunk in operands)
    padded_pairs = sum(chunk.indices.shape[0] * chunk.indices.shape[-1] for chunk in operands)
    flops = 2 * 64 * valid_pairs * (512 + 512)
    logical_bytes = (
        2 * queries * 64 * 512
        + 4 * padded_pairs
        + 4 * queries
        + 2 * valid_pairs * 512
        + 2 * queries * 64 * 512
        + 8 * queries * 64
    )
    return flops, logical_bytes


def profile_deepseek_v4_sparse_mla_prefill_flashmla(
    query_context_pairs: tuple[tuple[int, int], ...],
    max_model_len: int,
    max_num_batched_tokens: int,
    prefill_chunk_size: int,
    compress_ratio: int,
    window_size: int,
    selected_k: int,
    selected_index_pattern: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: object,
    cache_dtype: object,
    index_dtype: str,
    output_dtype: object,
    cache_layout: str,
) -> ComputeMetrics:
    shape = _validate_args(
        query_context_pairs,
        max_model_len,
        max_num_batched_tokens,
        prefill_chunk_size,
        compress_ratio,
        window_size,
        selected_k,
        selected_index_pattern,
        num_heads,
        num_kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_dtype,
        cache_dtype,
        index_dtype,
        output_dtype,
        cache_layout,
    )
    try:
        import torch
        from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires packaged FlashMLA") from exc
    try:
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
        device = torch.device("cuda", torch.cuda.current_device())
        if str(torch.cuda.get_device_name(device)) != _GPU_NAME or tuple(
            torch.cuda.get_device_capability(device)
        ) != (9, 0):
            raise ProfilerNotImplemented(f"{_BACKEND} requires {_GPU_NAME} SM90")
        operands = tuple(_build_operands(torch, chunk, device) for chunk in _chunks(shape))
        for chunk in operands:
            _check_output(torch, flash_mla_sparse_fwd, chunk)

        def run() -> tuple[Any, ...]:
            return tuple(_launch(flash_mla_sparse_fwd, chunk) for chunk in operands)

        # One semantic prefill slot owns ceil(num_requests / 4) ordered physical
        # FlashMLA launches, matching the public model loop.
        time_ms = Timer.cupti(run, kernel_name=None)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        flops, logical_bytes = _logical_work(operands)
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


__all__ = ["profile_deepseek_v4_sparse_mla_prefill_flashmla"]
