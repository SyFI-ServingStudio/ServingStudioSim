"""Profile the public CUDA prefill top-k calls for one V4 indexer operation."""

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.deepseek_v4_indexer_prefill_workload import (
    IndexerPrefillChunk,
    build_indexer_prefill_chunks,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_indexer_topk_prefill:vllm_cuda"
_GPU_NAME = "NVIDIA H200"
_TOP_K = 512


@dataclass(frozen=True)
class _Operands:
    logits: Any
    row_starts: Any
    row_ends: Any
    output: Any


def _required_logits_row_stride(num_keys: int) -> int:
    return ((num_keys + 255) // 256) * 256 + 256


def _validate_args(
    query_context_pairs: tuple[tuple[int, int], ...],
    max_model_len: int,
    max_num_batched_tokens: int,
    max_logits_bytes: int,
    compress_ratio: int,
    top_k: int,
    logits_dtype: object,
    index_dtype: str,
) -> tuple[IndexerPrefillChunk, ...]:
    if top_k != _TOP_K:
        raise ProfilerNotImplemented(f"{_BACKEND} requires top_k={_TOP_K}")
    if (str(logits_dtype), index_dtype) != ("fp32", "int32"):
        raise ProfilerNotImplemented(f"{_BACKEND} requires FP32 logits and int32 indices")
    return build_indexer_prefill_chunks(
        query_context_pairs,
        max_model_len,
        max_num_batched_tokens,
        max_logits_bytes,
        compress_ratio,
    )


def _build_operands(torch: Any, chunk: IndexerPrefillChunk, device: Any) -> _Operands:
    row_stride = _required_logits_row_stride(chunk.num_keys)
    backing = torch.empty((chunk.num_queries, row_stride), dtype=torch.float32, device=device)
    columns = torch.arange(row_stride, dtype=torch.float32, device=device)
    for query_start in range(0, chunk.num_queries, 512):
        query_stop = min(query_start + 512, chunk.num_queries)
        rows = torch.arange(query_start, query_stop, dtype=torch.float32, device=device)
        backing[query_start:query_stop] = columns[None, :] + rows[:, None] / 16384.0
    return _Operands(
        backing[:, : chunk.num_keys],
        torch.tensor(chunk.row_starts, dtype=torch.int32, device=device),
        torch.tensor(chunk.row_ends, dtype=torch.int32, device=device),
        torch.full((chunk.num_queries, _TOP_K), -1, dtype=torch.int32, device=device),
    )


def _launch(public_op: Any, operands: _Operands) -> None:
    public_op(
        operands.logits,
        operands.row_starts,
        operands.row_ends,
        operands.output,
        operands.logits.shape[0],
        operands.logits.stride(0),
        operands.logits.stride(1),
        _TOP_K,
    )


def _check_output(torch: Any, public_op: Any, operands: _Operands) -> None:
    _launch(public_op, operands)
    torch.cuda.synchronize(operands.logits.device)
    sampled_rows = sorted({0, operands.logits.shape[0] // 2, operands.logits.shape[0] - 1})
    for row_index in sampled_rows:
        start = int(operands.row_starts[row_index].item())
        stop = int(operands.row_ends[row_index].item())
        span_length = stop - start
        actual = operands.output[row_index]
        if span_length <= _TOP_K:
            expected = torch.full_like(actual, -1)
            expected[:span_length] = torch.arange(
                span_length, dtype=torch.int32, device=actual.device
            )
            if not torch.equal(actual, expected):
                raise KernelLaunchFailed(f"{_BACKEND} short-span output differs from Torch")
            continue
        indices = actual.to(torch.int64)
        if bool(((indices < 0) | (indices >= span_length)).any().item()):
            raise KernelLaunchFailed(f"{_BACKEND} returned an out-of-span local index")
        if indices.unique().numel() != _TOP_K:
            raise KernelLaunchFailed(f"{_BACKEND} returned duplicate indices")
        selected = operands.logits[row_index, start:stop].index_select(0, indices).sort().values
        expected = operands.logits[row_index, stop - _TOP_K : stop].sort().values
        if not torch.equal(selected, expected):
            raise KernelLaunchFailed(f"{_BACKEND} selected the wrong value set")


def profile_deepseek_v4_indexer_topk_prefill_cuda(
    query_context_pairs: tuple[tuple[int, int], ...],
    max_model_len: int,
    max_num_batched_tokens: int,
    max_logits_bytes: int,
    compress_ratio: int,
    top_k: int,
    logits_dtype: object,
    index_dtype: str,
) -> ComputeMetrics:
    chunks = _validate_args(
        query_context_pairs,
        max_model_len,
        max_num_batched_tokens,
        max_logits_bytes,
        compress_ratio,
        top_k,
        logits_dtype,
        index_dtype,
    )
    try:
        import torch
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires pinned vLLM") from exc
    try:
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
        device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_name(device) != _GPU_NAME or tuple(
            torch.cuda.get_device_capability(device)
        ) != (9, 0):
            raise ProfilerNotImplemented(f"{_BACKEND} requires {_GPU_NAME} SM90")
        operands = tuple(_build_operands(torch, chunk, device) for chunk in chunks)
        for chunk_operands in operands:
            _check_output(torch, ops.top_k_per_row_prefill, chunk_operands)

        def run() -> None:
            for chunk_operands in operands:
                _launch(ops.top_k_per_row_prefill, chunk_operands)

        time_ms = Timer.cupti(run, kernel_name=None)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        total_queries = sum(chunk.num_queries for chunk in chunks)
        valid_key_pairs = sum(chunk.valid_key_pairs for chunk in chunks)
        logical_bytes = 4 * valid_key_pairs + 8 * total_queries + 4 * total_queries * _TOP_K
        elapsed_seconds = time_ms / 1000.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(logical_bytes / elapsed_seconds / 1e9),
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc


__all__ = ["profile_deepseek_v4_indexer_topk_prefill_cuda"]
