"""Torch profiling runner for GLM-5.2's prefill DSA MQA logits.

The timed callable is a semantic Torch composite, not the production
DeepGEMM kernel. Scheduled-work accounting follows the latter's two-query and
256-key tiles while excluding Torch intermediates and physical transactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_NUM_SEQUENCES = 1
_NUM_HEADS = 64
_HEAD_DIM = 128
_Q_DTYPE = DType.FP8_E4M3
_K_DTYPE = DType.FP8_E4M3
_K_SCALE_DTYPE = DType.FP32
_WEIGHT_DTYPE = DType.FP32
_OUTPUT_DTYPE = DType.FP32
_SPAN_MODE = "single_causal_tail"
_REQUIRED_GPU = "NVIDIA H200"
_QUERY_TILE = 2
_KEY_TILE = 256


@dataclass(frozen=True)
class _DsaMqaLogitsPrefillOperands:
    q: Any
    k: Any
    k_scale: Any
    weights: Any
    k_start: Any
    k_end: Any
    valid_mask: Any


def _validate_args(
    num_queries: int,
    num_keys: int,
    num_sequences: int,
    num_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    k_dtype: DType | str,
    k_scale_dtype: DType | str,
    weight_dtype: DType | str,
    output_dtype: DType | str,
    span_mode: str,
    clean_logits: bool,
) -> tuple[int, int, int, int, int, DType, DType, DType, DType, DType, str, bool]:
    num_queries = int(num_queries)
    num_keys = int(num_keys)
    num_sequences = int(num_sequences)
    num_heads = int(num_heads)
    head_dim = int(head_dim)
    q_dtype = DType.from_value(q_dtype)
    k_dtype = DType.from_value(k_dtype)
    k_scale_dtype = DType.from_value(k_scale_dtype)
    weight_dtype = DType.from_value(weight_dtype)
    output_dtype = DType.from_value(output_dtype)
    span_mode = str(span_mode)

    if num_queries <= 0 or num_keys <= 0:
        raise ValueError(f"num_queries and num_keys must be > 0, got {num_queries} and {num_keys}")
    if num_queries > num_keys:
        raise ValueError(f"num_queries must be <= num_keys, got {num_queries} and {num_keys}")
    if num_sequences != _NUM_SEQUENCES:
        raise ValueError(
            f"torch dsa_mqa_logits_prefill requires num_sequences=1, got {num_sequences}"
        )
    if (num_heads, head_dim) != (_NUM_HEADS, _HEAD_DIM):
        raise ValueError(
            "torch dsa_mqa_logits_prefill requires "
            "(num_heads, head_dim) == (64, 128), "
            f"got ({num_heads}, {head_dim})"
        )
    if q_dtype is not _Q_DTYPE or k_dtype is not _K_DTYPE:
        raise ValueError(
            "torch dsa_mqa_logits_prefill requires "
            "q_dtype=k_dtype=fp8_e4m3, "
            f"got {q_dtype.value} and {k_dtype.value}"
        )
    if (
        k_scale_dtype is not _K_SCALE_DTYPE
        or weight_dtype is not _WEIGHT_DTYPE
        or output_dtype is not _OUTPUT_DTYPE
    ):
        raise ValueError(
            "torch dsa_mqa_logits_prefill requires "
            "k_scale_dtype=weight_dtype=output_dtype=fp32, got "
            f"{k_scale_dtype.value}, {weight_dtype.value}, and "
            f"{output_dtype.value}"
        )
    if span_mode != _SPAN_MODE:
        raise ValueError(
            f"torch dsa_mqa_logits_prefill requires span_mode='{_SPAN_MODE}', got {span_mode!r}"
        )
    if not isinstance(clean_logits, bool):
        raise TypeError("clean_logits must be a bool")
    if clean_logits:
        raise ValueError("torch dsa_mqa_logits_prefill requires clean_logits=false")

    return (
        num_queries,
        num_keys,
        num_sequences,
        num_heads,
        head_dim,
        q_dtype,
        k_dtype,
        k_scale_dtype,
        weight_dtype,
        output_dtype,
        span_mode,
        clean_logits,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch dsa_mqa_logits_prefill backend"
        )
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(
            f"torch dsa_mqa_logits_prefill is verified only on {_REQUIRED_GPU}, got {gpu_name}"
        )


def _stable_values(torch: Any, shape: tuple[int, ...], *, phase: int, device: str) -> Any:
    """Create deterministic, finite FP32 values with both signs."""
    total = 1
    for dimension in shape:
        total *= dimension
    values = torch.arange(total, dtype=torch.int64, device=device)
    values = ((values + phase) % 17 - 8).to(torch.float32) / 16.0
    return values.reshape(shape)


def _build_operands(
    torch: Any,
    *,
    num_queries: int,
    num_keys: int,
    num_heads: int,
    head_dim: int,
    device: str,
) -> _DsaMqaLogitsPrefillOperands:
    """Build exact contiguous GLM tensors and causal-tail span metadata."""
    q = _stable_values(
        torch,
        (num_queries, num_heads, head_dim),
        phase=1,
        device=device,
    ).to(torch.float8_e4m3fn)
    k = _stable_values(
        torch,
        (num_keys, head_dim),
        phase=5,
        device=device,
    ).to(torch.float8_e4m3fn)
    k_scale = 0.75 + (torch.arange(num_keys, dtype=torch.float32, device=device) % 7) / 16.0
    weights = (
        _stable_values(
            torch,
            (num_queries, num_heads),
            phase=9,
            device=device,
        )
        / 4.0
    )
    k_start = torch.zeros(num_queries, dtype=torch.int32, device=device)
    k_end = torch.arange(
        num_keys - num_queries + 1,
        num_keys + 1,
        dtype=torch.int32,
        device=device,
    )
    key_positions = torch.arange(num_keys, dtype=torch.int32, device=device)
    valid_mask = (key_positions.unsqueeze(0) >= k_start.unsqueeze(1)) & (
        key_positions.unsqueeze(0) < k_end.unsqueeze(1)
    )
    return _DsaMqaLogitsPrefillOperands(
        q=q,
        k=k,
        k_scale=k_scale,
        weights=weights,
        k_start=k_start,
        k_end=k_end,
        valid_mask=valid_mask,
    )


def _torch_composite(
    operands: _DsaMqaLogitsPrefillOperands,
) -> Any:
    """Compute the complete FP32 Torch semantic composite."""
    dot_products = operands.q.float() @ operands.k.float().T
    reduced = (dot_products.relu() * operands.weights.unsqueeze(-1)).sum(
        dim=1
    ) * operands.k_scale.unsqueeze(0)
    return reduced.masked_fill(~operands.valid_mask, float("nan"))


def _scheduled_work(
    *,
    num_queries: int,
    num_keys: int,
    num_heads: int,
    head_dim: int,
) -> tuple[int, int, int]:
    """Return ``(sum_w, C, nominal_flops)`` for DeepGEMM's scheduled tiles.

    Queries are paired in source two-query tiles. Each tile schedules the
    zero-based union through its greatest real causal-tail end, rounded to a
    256-key tile. An odd final tile still occupies a two-query source tile.
    """
    if min(num_queries, num_keys, num_heads, head_dim) <= 0:
        raise ValueError("scheduled-work dimensions must be > 0")
    if num_queries > num_keys:
        raise ValueError("num_queries must be <= num_keys")

    first_end = num_keys - num_queries + 1
    sum_w = 0
    for first_query in range(0, num_queries, _QUERY_TILE):
        final_query = min(first_query + _QUERY_TILE - 1, num_queries - 1)
        union_width = first_end + final_query
        rounded_width = ((union_width + _KEY_TILE - 1) // _KEY_TILE) * _KEY_TILE
        sum_w += rounded_width
    scheduled_cells = _QUERY_TILE * sum_w
    nominal_flops = 2 * scheduled_cells * num_heads * head_dim
    return sum_w, scheduled_cells, nominal_flops


def _logical_scheduled_bytes(
    *,
    num_queries: int,
    num_keys: int,
    num_heads: int,
    head_dim: int,
) -> int:
    """Return scheduled logical bytes, excluding Torch/TMA intermediates."""
    sum_w, scheduled_cells, _nominal_flops = _scheduled_work(
        num_queries=num_queries,
        num_keys=num_keys,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    return (
        num_queries * num_heads * head_dim
        + head_dim * sum_w
        + 4 * sum_w
        + 4 * num_queries * num_heads
        + 8 * num_queries
        + 4 * scheduled_cells
    )


def profile_dsa_mqa_logits_prefill_torch(
    num_queries: int,
    num_keys: int,
    num_sequences: int,
    num_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    k_dtype: DType | str,
    k_scale_dtype: DType | str,
    weight_dtype: DType | str,
    output_dtype: DType | str,
    span_mode: str,
    clean_logits: bool,
) -> ComputeMetrics:
    """Profile the complete Torch prefill DSA-logits semantic composite."""
    (
        num_queries,
        num_keys,
        _num_sequences,
        num_heads,
        head_dim,
        _q_dtype,
        _k_dtype,
        _k_scale_dtype,
        _weight_dtype,
        _output_dtype,
        _span_mode,
        _clean_logits,
    ) = _validate_args(
        num_queries,
        num_keys,
        num_sequences,
        num_heads,
        head_dim,
        q_dtype,
        k_dtype,
        k_scale_dtype,
        weight_dtype,
        output_dtype,
        span_mode,
        clean_logits,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch dsa_mqa_logits_prefill backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(
            torch,
            num_queries=num_queries,
            num_keys=num_keys,
            num_heads=num_heads,
            head_dim=head_dim,
            device="cuda",
        )

        def kernel() -> Any:
            return _torch_composite(operands)

        time_ms = Timer.cuda_event(kernel, warmup=5)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )
        _sum_w, _scheduled_cells, nominal_flops = _scheduled_work(
            num_queries=num_queries,
            num_keys=num_keys,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        logical_bytes = _logical_scheduled_bytes(
            num_queries=num_queries,
            num_keys=num_keys,
            num_heads=num_heads,
            head_dim=head_dim,
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
