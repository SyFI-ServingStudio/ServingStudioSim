"""Analytic memory-roofline *floor* adapters for the GLM-5.3-Flash MI300X port.

Several composed kinds of the ``glm53_flash_vllm_fp8_kda_dsa_moe`` arch carry a
negligible predicted share of iteration time (Phase-4 ranking, decision #38) yet
have no MI300X-native backend written. Rather than leave them pinned to an
NVIDIA-only backend — which a real MI300X ``timing-predict`` would reject at
``BackendSupport.allows`` — each is floored here with a CLOSED-FORM memory
roofline over the kind's HBM byte footprint.

Mechanism (decision #37 "mechanism B", converted to analytic by decision #49): a
per-kind ``elementwise_floor`` backend, ``gpus={MI300X}`` and compute-dtype
agnostic, whose runner derives the kind's memory-bound byte footprint from its
*shape* args and converts it to a time arithmetically:

    t_ms = launch_latency_ms + (read_bytes + write_bytes) / effective_hbm_bw

This mirrors the analytic Infinity-Fabric all-reduce roofline
(``profiling.runners.comm.fabric_roofline``): the number is DERIVED, not timed.
There is NO per-cell GPU allocation and NO ``rocprofv3`` capture, so a cell whose
derived output footprint exceeds the 192 GB HBM (e.g. the DSA prefill indexer
logits, whose naive ``num_queries · num_keys · 4`` materialization is ~275 GB)
yields a finite, plausible time instead of OOM-ing on ``torch.empty``. The brute
force the measured path required — ~37 GPU-hours over the floor grid, and an
impossible allocation for the >192 GB cells — is gone.

``effective_hbm_bw`` is an EMPIRICAL effective HBM bandwidth fitted from the
already-measured MI300X ``elementwise`` ``torch_rocm`` byte-mover rows (a linear
``bytes -> t_ms`` fit: the slope is the bandwidth, the intercept the fixed kernel
launch latency so small footprints are not under-counted). Those cells are all
small (<= ~48 MB) and so latency-bound, which makes the fitted slope run slightly
above the achievable HBM ceiling; the bandwidth is therefore clamped to an
achievable fraction of the MI300X peak (``gpu/spec.json`` ``mem_bandwidth_gbps``
= 5300 GB/s decimal), while the empirically measured launch-latency intercept is
kept. With no measured rows on-host the whole roofline falls back to the
documented achievable-fraction constant plus a nominal launch latency. The source
is labelled on the returned metrics' bandwidth field (achieved = traffic / time),
exactly as the fabric roofline reports a derived ``busbw``.

These are intentionally approximate: a memory-bound read-inputs + write-outputs
estimate is the designed floor for a negligible-share kind. A kind whose share
later proves material is promoted to a real measured backend (decision #29).
Energy is not modelled on this analytic path (``energy_j = 0.0``); it is a
timing-only floor.

The runner touches neither torch nor the GPU, so it fills profile.db rows from
pure arithmetic on any host (no OOM, no measurement), just like the all-reduce
fabric roofline.
"""

from __future__ import annotations

import math
import sqlite3
from functools import lru_cache
from typing import Any, NamedTuple

from profiling.db.args import DType
from profiling.runners.metrics import ComputeMetrics

# Canonical ``gpu/spec.json`` key and aliases for the MI300X; this backend is
# MI300X-gated, so this is the only target whose rows/peak it reads.
_MI300X_GPU_NAMES: tuple[str, ...] = ("MI300X", "AMD MI300X", "MI300")

# MI300X peak HBM3 bandwidth, decimal GB/s (``gpu/spec.json`` ``mem_bandwidth_gbps``);
# the constant below is only a fallback for when the catalog cannot be read.
_DEFAULT_PEAK_HBM_GBPS = 5300.0

# Fraction of peak HBM a well-formed streaming byte-mover actually sustains on
# MI300X. The measured ``elementwise`` byte-mover's largest cells (~40-48 MB)
# achieve ~4300-4900 GB/s, i.e. ~0.81-0.92 of the 5300 GB/s peak; 0.9 is the
# documented achievable ceiling used both as the clamp on the empirical slope and
# as the no-rows fallback bandwidth.
_ACHIEVABLE_HBM_FRACTION = 0.9

# Nominal kernel launch latency (ms) when no measured rows are available to fit an
# intercept. ~2 us matches the measured byte-mover's fitted intercept.
_DEFAULT_LAUNCH_LATENCY_MS = 0.002

# Fewer measured rows than this cannot anchor a two-parameter (slope, intercept)
# fit, so fall back to the documented constant roofline instead.
_MIN_FIT_ROWS = 8


class _Roofline(NamedTuple):
    """Resolved MI300X memory roofline: ``t_ms = latency_ms + traffic / bw``."""

    bw_bytes_per_ms: float  # effective HBM bandwidth, bytes per millisecond
    latency_ms: float  # fixed per-launch latency floor
    source: str  # provenance label ("empirical_fit" / "spec_roofline")


def _dt_bytes(value: Any) -> float:
    """Byte width of a dtype arg given as a wire string or ``DType``."""
    return DType.from_value(value).size_bytes()


def _index_bytes(index_dtype: str) -> int:
    """Byte width of an integer index dtype arg (a plain string, not ``DType``)."""
    token = str(index_dtype).lower()
    if "64" in token:
        return 8
    if "16" in token:
        return 2
    return 4  # int32 / uint32 and anything unrecognized


def _peak_hbm_gbps() -> float:
    """MI300X peak HBM bandwidth from ``gpu/spec.json``; constant on failure."""
    try:
        from profiling.gpu_catalog import resolve_gpu_spec

        resolution = resolve_gpu_spec(_MI300X_GPU_NAMES[0])
        if resolution is not None and resolution.mem_bandwidth_gbps:
            return float(resolution.mem_bandwidth_gbps)
    except Exception:  # pragma: no cover - catalog is optional at runner time
        pass
    return _DEFAULT_PEAK_HBM_GBPS


def _measured_byte_mover_rows() -> list[tuple[int, float]]:
    """``(traffic_bytes, time_ms)`` for the measured MI300X elementwise rows.

    Reads the shared L1 cache directly (read-only) so the fit sees whatever the
    ``elementwise`` ``torch_rocm`` byte-mover has already measured on this host.
    Any failure (missing DB/table/columns) yields an empty list and the constant
    fallback roofline.
    """
    try:
        from profiling.perf_api import DB_PATH

        placeholders = ",".join("?" for _ in _MI300X_GPU_NAMES)
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            cursor = conn.execute(
                "SELECT input_size_bytes, output_size_bytes, time_ms "
                "FROM elementwise "
                "WHERE backend = 'torch_rocm' "
                f"AND gpu_name IN ({placeholders}) "
                "AND (is_outlier IS NULL OR is_outlier = 0) "
                "AND time_ms > 0",
                _MI300X_GPU_NAMES,
            )
            return [(int(in_b) + int(out_b), float(t)) for in_b, out_b, t in cursor]
        finally:
            conn.close()
    except Exception:  # pragma: no cover - DB is optional at runner time
        return []


@lru_cache(maxsize=1)
def _hbm_roofline() -> _Roofline:
    """Resolve the MI300X memory roofline once per process.

    Prefers an empirical ``bytes -> t_ms`` least-squares fit over the measured
    byte-mover rows (slope = bandwidth, intercept = launch latency), clamping the
    fitted bandwidth to the achievable HBM ceiling because the measured cells are
    too small to saturate HBM. Falls back to the documented constant roofline when
    too few rows exist or the fit is degenerate.
    """
    peak_bytes_per_ms = _peak_hbm_gbps() * 1e6  # GB/s (1e9 B/s) -> bytes/ms
    ceiling_bytes_per_ms = _ACHIEVABLE_HBM_FRACTION * peak_bytes_per_ms
    fallback = _Roofline(ceiling_bytes_per_ms, _DEFAULT_LAUNCH_LATENCY_MS, "spec_roofline")

    rows = _measured_byte_mover_rows()
    if len(rows) < _MIN_FIT_ROWS:
        return fallback

    n = len(rows)
    sum_x = sum(x for x, _ in rows)
    sum_y = sum(y for _, y in rows)
    sum_xx = sum(x * x for x, _ in rows)
    sum_xy = sum(x * y for x, y in rows)
    denom = n * sum_xx - sum_x * sum_x
    if denom <= 0:
        return fallback
    slope = (n * sum_xy - sum_x * sum_y) / denom  # ms per byte
    intercept = (sum_y - slope * sum_x) / n  # ms
    if slope <= 0:
        # A non-positive slope means the fit found no bandwidth signal (all cells
        # latency-bound); the empirical bandwidth is meaningless, so use the ceiling.
        return _Roofline(ceiling_bytes_per_ms, max(0.0, intercept), "spec_roofline")
    empirical_bytes_per_ms = 1.0 / slope
    effective = min(empirical_bytes_per_ms, ceiling_bytes_per_ms)
    return _Roofline(effective, max(0.0, intercept), "empirical_fit")


def _floor(input_size_bytes: float, output_size_bytes: float) -> ComputeMetrics:
    """Closed-form memory-roofline time for a derived HBM byte footprint.

    ``traffic = read_bytes + write_bytes`` is the total HBM movement (both the
    input read and the output write pass), which measurement showed is the cost
    driver — total footprint, not fan-in. The output is forced to at least one
    byte so even a pure sink has a write pass, mirroring the measured byte-mover's
    contract. No tensor is allocated and no kernel is launched.
    """
    out_bytes = max(1, int(math.ceil(output_size_bytes)))
    in_bytes = max(0, int(math.ceil(input_size_bytes)))
    traffic_bytes = in_bytes + out_bytes

    roofline = _hbm_roofline()
    time_ms = roofline.latency_ms + traffic_bytes / roofline.bw_bytes_per_ms
    bandwidth_gbps = traffic_bytes / (time_ms * 1e-3) / 1e9 if time_ms > 0 else 0.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(bandwidth_gbps),
        energy_j=0.0,
    )


# ── fp8 activation quantization ──────────────────────────────────────────────


def profile_fp8_per_token_group_quant_floor(
    *,
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    input_dtype: Any,
    scale_format: Any,
    **_: Any,
) -> ComputeMetrics:
    rows = num_tokens * hidden_size
    in_bytes = rows * _dt_bytes(input_dtype)
    # fp8 output (1 byte/elem) plus one fp32 scale per group.
    scales = num_tokens * math.ceil(hidden_size / max(1, group_size))
    out_bytes = rows * 1 + scales * 4
    return _floor(in_bytes, out_bytes)


# ── dense / router GEMM with fp32 output ─────────────────────────────────────


def profile_gemm_fp32_output_floor(
    *, m: int, n: int, k: int, input_dtype: Any, **_: Any
) -> ComputeMetrics:
    in_bytes = (m * k + n * k) * _dt_bytes(input_dtype)
    out_bytes = m * n * 4  # fp32 output
    return _floor(in_bytes, out_bytes)


# ── DSA indexer logits (decode / prefill) ────────────────────────────────────


def profile_dsa_paged_mqa_logits_decode_floor(
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    q_dtype: Any,
    cache_dtype: Any,
    output_dtype: Any,
    **_: Any,
) -> ComputeMetrics:
    queries = batch_size * next_n * num_heads * head_dim * _dt_bytes(q_dtype)
    keys = batch_size * context_len * head_dim * _dt_bytes(cache_dtype)
    out_bytes = batch_size * next_n * max_model_len * _dt_bytes(output_dtype)
    return _floor(queries + keys, out_bytes)


def profile_dsa_mqa_logits_prefill_floor(
    *,
    num_queries: int,
    num_keys: int,
    num_heads: int,
    head_dim: int,
    q_dtype: Any,
    k_dtype: Any,
    output_dtype: Any,
    **_: Any,
) -> ComputeMetrics:
    queries = num_queries * num_heads * head_dim * _dt_bytes(q_dtype)
    keys = num_keys * head_dim * _dt_bytes(k_dtype)
    out_bytes = num_queries * num_keys * _dt_bytes(output_dtype)
    return _floor(queries + keys, out_bytes)


# ── DSA top-k selection (decode / prefill) ───────────────────────────────────


def profile_dsa_persistent_topk_decode_floor(
    *,
    batch_size: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_dtype: Any,
    index_dtype: str,
    **_: Any,
) -> ComputeMetrics:
    in_bytes = batch_size * next_n * max_model_len * _dt_bytes(logits_dtype)
    out_bytes = batch_size * next_n * top_k * _index_bytes(index_dtype)
    return _floor(in_bytes, out_bytes)


def profile_dsa_topk_prefill_floor(
    *,
    num_queries: int,
    num_keys: int,
    top_k: int,
    logits_dtype: Any,
    index_dtype: str,
    **_: Any,
) -> ComputeMetrics:
    in_bytes = num_queries * num_keys * _dt_bytes(logits_dtype)
    out_bytes = num_queries * top_k * _index_bytes(index_dtype)
    return _floor(in_bytes, out_bytes)


# ── MLA cache append ─────────────────────────────────────────────────────────


def profile_mla_cache_append_floor(
    *,
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    input_dtype: Any,
    kv_dtype: Any,
    **_: Any,
) -> ComputeMetrics:
    width = kv_lora_rank + rope_dim
    in_bytes = num_tokens * width * _dt_bytes(input_dtype)
    out_bytes = num_tokens * width * _dt_bytes(kv_dtype)
    return _floor(in_bytes, out_bytes)


# ── DSA sparse index remap ───────────────────────────────────────────────────


def profile_dsa_sparse_index_remap_floor(
    *, num_queries: int, selected_k: int, index_dtype: str, **_: Any
) -> ComputeMetrics:
    slot_bytes = num_queries * selected_k * _index_bytes(index_dtype)
    # Read local indices, write mapped indices: one pass each.
    return _floor(slot_bytes, slot_bytes)


# ── mHC residual-stream RMS norms ────────────────────────────────────────────


def profile_mhc_pre_rms_norm_floor(
    *, num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: Any, **_: Any
) -> ComputeMetrics:
    stream_bytes = num_tokens * hidden_size * hc_mult * _dt_bytes(hidden_dtype)
    return _floor(stream_bytes, stream_bytes)


def profile_mhc_fused_post_pre_rms_norm_floor(
    *, num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: Any, **_: Any
) -> ComputeMetrics:
    stream_bytes = num_tokens * hidden_size * hc_mult * _dt_bytes(hidden_dtype)
    # Fused post+pre reads the prior stream and its residual, writes the normed
    # stream: two read passes, one write.
    return _floor(2 * stream_bytes, stream_bytes)


# ── TP all-reduce fusion ─────────────────────────────────────────────────────


def profile_all_reduce_fusion_floor(
    *, num_tokens: int, hidden_dim: int, dtype: Any, **_: Any
) -> ComputeMetrics:
    tensor_bytes = num_tokens * hidden_dim * _dt_bytes(dtype)
    # A ring all-reduce moves roughly twice the tensor over the fabric; floor it
    # as two read passes and one write of the reduced tensor.
    return _floor(2 * tensor_bytes, tensor_bytes)
