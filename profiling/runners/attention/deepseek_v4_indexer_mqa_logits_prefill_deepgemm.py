"""Profile the public DeepGEMM MQA-logits calls for one V4 prefill operation."""

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

_BACKEND = "deepseek_v4_indexer_mqa_logits_prefill:vllm_deepgemm_fp8"
_GPU_NAME = "NVIDIA H200"
_MODEL_IDENTITY = (64, 128)
_STORAGE_IDENTITY = ("fp8_e4m3", "fp8_e4m3", "fp32", "fp32", "fp32", False)


@dataclass(frozen=True)
class _Operands:
    q: Any
    k: Any
    k_scale: Any
    weights: Any
    row_starts: Any
    row_ends: Any


def _validate_args(
    query_context_pairs: tuple[tuple[int, int], ...],
    max_model_len: int,
    max_num_batched_tokens: int,
    max_logits_bytes: int,
    compress_ratio: int,
    num_heads: int,
    head_dim: int,
    q_dtype: object,
    k_dtype: object,
    k_scale_dtype: object,
    weight_dtype: object,
    output_dtype: object,
    clean_logits: bool,
) -> tuple[IndexerPrefillChunk, ...]:
    if (num_heads, head_dim) != _MODEL_IDENTITY:
        raise ProfilerNotImplemented(f"{_BACKEND} supports model identity {_MODEL_IDENTITY}")
    storage_identity = (
        str(q_dtype),
        str(k_dtype),
        str(k_scale_dtype),
        str(weight_dtype),
        str(output_dtype),
        clean_logits,
    )
    if storage_identity != _STORAGE_IDENTITY:
        raise ProfilerNotImplemented(f"{_BACKEND} supports storage identity {_STORAGE_IDENTITY}")
    return build_indexer_prefill_chunks(
        query_context_pairs,
        max_model_len,
        max_num_batched_tokens,
        max_logits_bytes,
        compress_ratio,
    )


def _build_operands(torch: Any, chunk: IndexerPrefillChunk, device: Any) -> _Operands:
    q = torch.empty((chunk.num_queries, 64, 128), dtype=torch.float8_e4m3fn, device=device)
    weights = torch.empty((chunk.num_queries, 64), dtype=torch.float32, device=device)
    for query_start in range(0, chunk.num_queries, 512):
        query_stop = min(query_start + 512, chunk.num_queries)
        query_rows = torch.arange(query_start, query_stop, dtype=torch.float32, device=device)
        feature = torch.arange(128, dtype=torch.float32, device=device)
        heads = torch.arange(64, dtype=torch.float32, device=device)
        q[query_start:query_stop] = (
            ((query_rows[:, None, None] + heads[None, :, None] + feature[None, None, :]) % 29)
            / 16.0
            - 0.875
        ).to(torch.float8_e4m3fn)
        weights[query_start:query_stop] = (
            0.25 + query_rows[:, None] / 8192.0 + heads[None, :] / 2048.0
        )
    k = torch.empty((chunk.num_keys, 128), dtype=torch.float8_e4m3fn, device=device)
    for key_start in range(0, chunk.num_keys, 4096):
        key_stop = min(key_start + 4096, chunk.num_keys)
        key_rows = torch.arange(key_start, key_stop, dtype=torch.float32, device=device)
        feature = torch.arange(128, dtype=torch.float32, device=device)
        k[key_start:key_stop] = (
            ((key_rows[:, None] * 3 + feature[None, :]) % 31) / 16.0 - 0.9375
        ).to(torch.float8_e4m3fn)
    return _Operands(
        q,
        k,
        0.5 + torch.arange(chunk.num_keys, dtype=torch.float32, device=device).remainder(11) / 16.0,
        weights,
        torch.tensor(chunk.row_starts, dtype=torch.int32, device=device),
        torch.tensor(chunk.row_ends, dtype=torch.int32, device=device),
    )


def _launch(public_op: Any, operands: _Operands) -> Any:
    return public_op(
        (operands.q, None),
        (operands.k, operands.k_scale),
        operands.weights,
        operands.row_starts,
        operands.row_ends,
        clean_logits=False,
    )


def _check_output(torch: Any, public_op: Any, operands: _Operands) -> None:
    actual = _launch(public_op, operands)
    torch.cuda.synchronize(operands.q.device)
    if actual.shape[:2] != (operands.q.shape[0], operands.k.shape[0]):
        raise KernelLaunchFailed(f"{_BACKEND} returned the wrong logits shape")
    sampled_rows = sorted({0, operands.q.shape[0] // 2, operands.q.shape[0] - 1})
    for row_index in sampled_rows:
        start = int(operands.row_starts[row_index].item())
        stop = int(operands.row_ends[row_index].item())
        if start == stop:
            continue
        sampled_columns = sorted({start, (start + stop - 1) // 2, stop - 1})
        column_tensor = torch.tensor(sampled_columns, dtype=torch.int64, device=operands.q.device)
        dequantized_k = operands.k.index_select(0, column_tensor).float()
        dequantized_k *= operands.k_scale.index_select(0, column_tensor)[:, None]
        per_head = operands.q[row_index].float() @ dequantized_k.T
        expected = (per_head.relu() * operands.weights[row_index, :, None]).sum(dim=0)
        torch.testing.assert_close(
            actual[row_index].index_select(0, column_tensor), expected, atol=2e-4, rtol=2e-4
        )


def profile_deepseek_v4_indexer_mqa_logits_prefill_deepgemm(
    query_context_pairs: tuple[tuple[int, int], ...],
    max_model_len: int,
    max_num_batched_tokens: int,
    max_logits_bytes: int,
    compress_ratio: int,
    num_heads: int,
    head_dim: int,
    q_dtype: object,
    k_dtype: object,
    k_scale_dtype: object,
    weight_dtype: object,
    output_dtype: object,
    clean_logits: bool,
) -> ComputeMetrics:
    chunks = _validate_args(
        query_context_pairs,
        max_model_len,
        max_num_batched_tokens,
        max_logits_bytes,
        compress_ratio,
        num_heads,
        head_dim,
        q_dtype,
        k_dtype,
        k_scale_dtype,
        weight_dtype,
        output_dtype,
        clean_logits,
    )
    try:
        import torch
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires pinned vLLM DeepGEMM") from exc
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
            _check_output(torch, fp8_fp4_mqa_logits, chunk_operands)

        def run() -> tuple[Any, ...]:
            return tuple(_launch(fp8_fp4_mqa_logits, chunk_operands) for chunk_operands in operands)

        time_ms = Timer.cupti(run, kernel_name=None)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        total_queries = sum(chunk.num_queries for chunk in chunks)
        valid_key_pairs = sum(chunk.valid_key_pairs for chunk in chunks)
        logical_bytes = (
            total_queries * 64 * 128
            + valid_key_pairs * (128 + 4 + 4)
            + total_queries * (64 * 4 + 8)
        )
        nominal_flops = 2 * valid_key_pairs * 64 * 128
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


__all__ = ["profile_deepseek_v4_indexer_mqa_logits_prefill_deepgemm"]
