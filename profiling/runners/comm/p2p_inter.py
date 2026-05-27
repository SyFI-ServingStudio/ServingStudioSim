"""Inter-domain p2p runner — analytical lookup table, NOT real profiling.

Cross-NVL-domain (cross-node NIC) p2p cannot be measured on a single-node box,
so instead of spawning 2 ranks and timing a real send/recv this runner returns a
*modeled* ``CommMetrics`` from a measured latency curve — the same analytical
model ref uses (``ref/profile/network/inter_device_p2p.py``).

For Hopper/Blackwell datacenter parts (H100/H200/B200) the curve below
(``message_size_bytes`` → ``time_us``) is linearly interpolated. Other GPUs fall
back to a flat bandwidth model with a 32 KB minimum transfer. ``gpu_name`` is
read from the reserved device so the modeled number matches the cache key L1b
records (the worker keys results by the same device name); when no CUDA device
is visible the model degrades to the default bandwidth fallback.

p2p is one hop, so ``busbw == algbw`` (no ring factor). ``fabric`` is a cache-key
namespace only and is not used by the model.
"""

from __future__ import annotations

from bisect import bisect_left

from profiling.db.args import DType
from profiling.runners.metrics import CommMetrics

# Fallback bandwidth model constants (used when gpu_name is not a profiled part).
MIN_SIZE_BYTES = 32 * 1024  # smaller transfers are padded to this size
DEFAULT_BANDWIDTH_GBPS = 44.0

# Measured inter-node latency curve for Hopper/Blackwell datacenter parts;
# entries are (message_size_bytes, time_us). Ported from ref.
PROFILED_GPU_FAMILIES = ("H100", "H200", "B200")
PROFILED_LATENCY_CURVE_US: tuple[tuple[int, float], ...] = (
    (1024, 30.19),
    (2048, 29.76),
    (4096, 29.89),
    (8192, 29.98),
    (16384, 30.43),
    (32768, 30.88),
    (65536, 30.92),
    (131072, 32.10),
    (262144, 39.80),
    (524288, 46.92),
    (1048576, 62.84),
    (2097152, 89.32),
    (4194304, 134.3),
    (8388608, 223.8),
    (16777216, 407.8),
    (33554432, 752.0),
    (67108864, 1437.4),
    (134217728, 2796.2),
    (268435456, 5512.3),
    (536870912, 10936.0),
    (1073741824, 21759.0),
)
PROFILED_LATENCY_SIZES = tuple(size for size, _ in PROFILED_LATENCY_CURVE_US)

# Substring → bandwidth (GB/s) for non-profiled parts, matched case-insensitively.
GPU_BANDWIDTH_GBPS = {
    "H800": 44.0,
    "H20": 44.0,
    "B300": 44.0,
    "GB200": 44.0,
    "A100": 22.0,
    "A40": 22.0,
}


def profile_p2p(
    message_size_bytes: int,
    dtype: DType | str,
    fabric: str,
    *,
    warmup: int = 50,
    rep: int = 100,
) -> CommMetrics:
    """Return a modeled inter-device p2p time (no real communication)."""
    del dtype, fabric, warmup, rep  # cache-key axes only; the model ignores them
    gpu_name = _current_gpu_name()
    time_ms = _compute_inter_device_p2p_time_ms(message_size_bytes, gpu_name)

    if time_ms <= 0.0:
        bw_gbps = 0.0
    else:
        effective_bytes = (
            message_size_bytes
            if _uses_profiled_curve(gpu_name)
            else max(message_size_bytes, MIN_SIZE_BYTES)
        )
        bw_gbps = (effective_bytes / (time_ms / 1000.0)) / 1e9

    return CommMetrics(time_ms=time_ms, algbw_gbps=bw_gbps, busbw_gbps=bw_gbps)


def _compute_inter_device_p2p_time_ms(
    message_size_bytes: int, gpu_name: str | None
) -> float:
    if _uses_profiled_curve(gpu_name):
        return _interpolate_profiled_time_ms(message_size_bytes)
    bandwidth = _fallback_bandwidth_gbps(gpu_name)
    effective_size = max(message_size_bytes, MIN_SIZE_BYTES)
    return effective_size / (bandwidth * 1e9) * 1000.0


def _uses_profiled_curve(gpu_name: str | None) -> bool:
    if gpu_name is None:
        return False
    name = gpu_name.upper()
    return any(family in name for family in PROFILED_GPU_FAMILIES)


def _fallback_bandwidth_gbps(gpu_name: str | None) -> float:
    if gpu_name is None:
        return DEFAULT_BANDWIDTH_GBPS
    name = gpu_name.upper()
    for key, bw in GPU_BANDWIDTH_GBPS.items():
        if key.upper() in name:
            return bw
    return DEFAULT_BANDWIDTH_GBPS


def _interpolate_profiled_time_ms(message_size_bytes: int) -> float:
    if message_size_bytes <= 0:
        return 0.0
    if message_size_bytes <= PROFILED_LATENCY_SIZES[0]:
        return PROFILED_LATENCY_CURVE_US[0][1] / 1000.0

    idx = bisect_left(PROFILED_LATENCY_SIZES, message_size_bytes)
    if idx < len(PROFILED_LATENCY_CURVE_US):
        size, time_us = PROFILED_LATENCY_CURVE_US[idx]
        if size == message_size_bytes:
            return time_us / 1000.0
        prev_size, prev_time_us = PROFILED_LATENCY_CURVE_US[idx - 1]
        return _linear_interpolate(
            message_size_bytes, prev_size, prev_time_us, size, time_us
        ) / 1000.0

    # Past the last grid point: extrapolate along the final segment.
    prev_size, prev_time_us = PROFILED_LATENCY_CURVE_US[-2]
    last_size, last_time_us = PROFILED_LATENCY_CURVE_US[-1]
    return _linear_interpolate(
        message_size_bytes, prev_size, prev_time_us, last_size, last_time_us
    ) / 1000.0


def _linear_interpolate(
    x: int, x0: int, y0_us: float, x1: int, y1_us: float
) -> float:
    if x1 == x0:
        return y0_us
    return y0_us + (y1_us - y0_us) * ((x - x0) / (x1 - x0))


def _current_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except ImportError:
        pass
    return None
