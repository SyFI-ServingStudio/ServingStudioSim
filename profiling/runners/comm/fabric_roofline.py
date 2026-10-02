"""Analytic Infinity-Fabric all-reduce roofline for the GLM-5.3-Flash MI300X port.

vLLM-ROCm ends a tensor-parallel block with an RCCL / aiter custom all-reduce
over AMD Infinity Fabric, not FlashInfer MNNVL (which is NVLink-multicast only).
A *measured* RCCL all-reduce needs a 4+ GPU Infinity-Fabric group; where that
measurement is unavailable, this backend supplies a principled analytic cost so
the kind resolves on MI300X instead of staying pinned to the NVIDIA-only MNNVL
backend (which a real MI300X ``timing-predict`` rejects at
``BackendSupport.allows``).

This is an ANALYTIC FABRIC ROOFLINE, explicitly NOT a measured multi-GPU row.
The cost comes from the ring all-reduce data-movement law and the MI300X
Infinity-Fabric bandwidth: a ring all-reduce moves ``2(N-1)/N`` of each rank's
tensor across the fabric, so at the bandwidth ceiling

    time = 2(N-1)/N · (num_tokens · hidden_dim · dtype_bytes) / fabric_bw

with ``fabric_bw = 896 GB/s`` (``gpu/spec.json`` MI300X
``interconnect_bandwidth_gbps``, the bidirectional Infinity-Fabric bandwidth).
The returned ``busbw_gbps`` is the fabric ceiling by construction and
``algbw_gbps`` is the implied per-rank algorithm bandwidth (``data / time``); the
simulator reads ``time_ms`` as the cost.

Because the number is derived, not timed, the runner needs no multi-GPU rank
group and no device work: the spec declares no ``gpu_count_fn`` (so ``gpu_count``
is 1) and the runner returns one ``CommMetrics`` per spec from pure arithmetic.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.runners.metrics import CommMetrics

# MI300X Infinity-Fabric bidirectional bandwidth in GB/s (decimal, 1e9 bytes/s),
# read from ``gpu/spec.json`` MI300X ``interconnect_bandwidth_gbps``. This backend
# is Infinity-Fabric / MI300X-gated, so this is the only fabric it models.
INFINITY_FABRIC_GBPS = 896.0
FABRIC = "infinity_fabric"


def profile_all_reduce_fusion_fabric_roofline(
    *,
    num_gpus: int,
    num_tokens: int,
    hidden_dim: int,
    dtype: Any,
    fabric: str,
    **_: Any,
) -> CommMetrics:
    """Ring all-reduce roofline over MI300X Infinity Fabric (analytic, not timed)."""
    if fabric != FABRIC:
        raise ValueError(
            f"fabric roofline all-reduce models {FABRIC!r} only, got {fabric!r}"
        )
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    data_bytes = num_tokens * hidden_dim * DType.from_value(dtype).size_bytes()
    # Ring all-reduce fabric traffic: each rank sends+receives 2(N-1)/N of the
    # tensor. A single rank is a no-op (no fabric traffic, zero cost).
    ring_factor = 2.0 * (num_gpus - 1) / num_gpus if num_gpus > 1 else 0.0
    moved_bytes = data_bytes * ring_factor
    time_ms = moved_bytes / (INFINITY_FABRIC_GBPS * 1e9) * 1e3
    if time_ms > 0.0:
        # busbw is the fabric ceiling by construction; algbw is the implied
        # per-rank algorithm bandwidth (full tensor moved per unit time).
        busbw_gbps = INFINITY_FABRIC_GBPS
        algbw_gbps = data_bytes / (time_ms * 1e-3) / 1e9
    else:
        busbw_gbps = 0.0
        algbw_gbps = 0.0
    return CommMetrics(time_ms=time_ms, algbw_gbps=algbw_gbps, busbw_gbps=busbw_gbps)
