"""Profile the public DeepGEMM MQA-logits calls for one V4 prefill operation."""

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.dsa_compressed_prefill_workload import (
    IndexerPrefillChunk,
    build_indexer_prefill_chunks,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "dsa_compressed_mqa_logits_prefill:vllm_deepgemm_fp8"
# DeepGEMM's fp8_fp4_mqa_logits host asserts (csrc/apis/attention.hpp at the
# vLLM-pinned 8b1392b): FP8 head_dim 32/64/128 on every arch, and per-arch
# q-head counts. Other arch majors have no kernel.
_HEAD_DIMS = frozenset({32, 64, 128})
_NUM_HEADS_BY_ARCH_MAJOR = {
    9: frozenset({32, 64}),
    10: frozenset({8, 16, 32, 64}),
    12: frozenset({16, 32, 64}),
}
_STORAGE_IDENTITY = ("fp8_e4m3", "fp8_e4m3", "fp32", "fp32", "fp32", False)


def check_deepgemm_mqa_logits_shape(label: str, num_heads: int, head_dim: int) -> None:
    """Reject a head shape no DeepGEMM build instantiates, before anything loads.

    Shared with dsa_mqa_logits_prefill's deepgemm_fp8 backend, which calls the
    same kernel family.
    """
    all_head_counts = frozenset().union(*_NUM_HEADS_BY_ARCH_MAJOR.values())
    if num_heads not in all_head_counts or head_dim not in _HEAD_DIMS:
        raise ProfilerNotImplemented(
            f"{label} needs num_heads in {sorted(all_head_counts)} and head_dim in "
            f"{sorted(_HEAD_DIMS)}, got ({num_heads}, {head_dim})"
        )


def require_deepgemm_mqa_logits_heads(torch: Any, label: str, num_heads: int) -> None:
    """Require this device's arch build to have a kernel for ``num_heads``.

    The worker has already checked the backend's declared device rule, so CUDA
    is present and the arch major is one DeepGEMM builds.
    """
    arch_major = int(torch.cuda.get_device_capability(torch.cuda.current_device())[0])
    if num_heads not in _NUM_HEADS_BY_ARCH_MAJOR.get(arch_major, frozenset()):
        raise ProfilerNotImplemented(
            f"{label} has no DeepGEMM FP8 MQA-logits kernel for num_heads={num_heads} "
            f"on SM{arch_major}x"
        )


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
    check_deepgemm_mqa_logits_shape(_BACKEND, num_heads, head_dim)
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


def _build_operands(
    torch: Any, chunk: IndexerPrefillChunk, device: Any, num_heads: int, head_dim: int
) -> _Operands:
    q = torch.empty(
        (chunk.num_queries, num_heads, head_dim), dtype=torch.float8_e4m3fn, device=device
    )
    weights = torch.empty((chunk.num_queries, num_heads), dtype=torch.float32, device=device)
    for query_start in range(0, chunk.num_queries, 512):
        query_stop = min(query_start + 512, chunk.num_queries)
        query_rows = torch.arange(query_start, query_stop, dtype=torch.float32, device=device)
        feature = torch.arange(head_dim, dtype=torch.float32, device=device)
        heads = torch.arange(num_heads, dtype=torch.float32, device=device)
        q[query_start:query_stop] = (
            ((query_rows[:, None, None] + heads[None, :, None] + feature[None, None, :]) % 29)
            / 16.0
            - 0.875
        ).to(torch.float8_e4m3fn)
        weights[query_start:query_stop] = (
            0.25 + query_rows[:, None] / 8192.0 + heads[None, :] / 2048.0
        )
    k = torch.empty((chunk.num_keys, head_dim), dtype=torch.float8_e4m3fn, device=device)
    for key_start in range(0, chunk.num_keys, 4096):
        key_stop = min(key_start + 4096, chunk.num_keys)
        key_rows = torch.arange(key_start, key_stop, dtype=torch.float32, device=device)
        feature = torch.arange(head_dim, dtype=torch.float32, device=device)
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


def profile_dsa_compressed_mqa_logits_prefill_deepgemm(
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
        require_deepgemm_mqa_logits_heads(torch, _BACKEND, num_heads)
        device = torch.device("cuda", torch.cuda.current_device())
        operands = tuple(
            _build_operands(torch, chunk, device, num_heads, head_dim) for chunk in chunks
        )
        for chunk_operands in operands:
            _check_output(torch, fp8_fp4_mqa_logits, chunk_operands)

        def run() -> tuple[Any, ...]:
            return tuple(_launch(fp8_fp4_mqa_logits, chunk_operands) for chunk_operands in operands)

        time_ms = Timer.cupti(run, kernel_name=None)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        total_queries = sum(chunk.num_queries for chunk in chunks)
        valid_key_pairs = sum(chunk.valid_key_pairs for chunk in chunks)
        logical_bytes = (
            total_queries * num_heads * head_dim
            + valid_key_pairs * (head_dim + 4 + 4)
            + total_queries * (num_heads * 4 + 8)
        )
        nominal_flops = 2 * valid_key_pairs * num_heads * head_dim
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


__all__ = ["profile_dsa_compressed_mqa_logits_prefill_deepgemm"]
