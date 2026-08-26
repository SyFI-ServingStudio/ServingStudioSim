"""Shared payload helpers for L1 execution backends.

Agent note:
- This module owns the JSON-friendly process-boundary schema shared by
  controller modules and worker entrypoints.
- Keep GPU selection, subprocess spawning, and runner execution out of this
  file. Those belong in ``local.py`` / ``local_worker.py`` or future backend
  implementations.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from profiling.db.kind import KernelKind
from profiling.db.registry import resolve_spec_backend
from profiling.exec.pool import ChunkResult
from profiling.runners.metrics import CommMetrics, ComputeMetrics, Metrics


def resolve_chunk_backend(kernel_kind: KernelKind, chunk_specs: list[dict]) -> str:
    """Validate that one worker payload contains exactly one backend."""

    backend = resolve_spec_backend(kernel_kind, chunk_specs[0])
    for spec_index, chunk_spec in enumerate(chunk_specs[1:], start=1):
        spec_backend = resolve_spec_backend(kernel_kind, chunk_spec)
        if spec_backend != backend:
            raise ValueError(
                "chunk specs must share one backend; "
                f"spec 0 uses {backend!r}, spec {spec_index} uses {spec_backend!r}"
            )
    return backend


def metrics_to_payload(metrics: Metrics) -> dict[str, Any]:
    if isinstance(metrics, ComputeMetrics):
        return {"kind": "compute", **asdict(metrics)}
    if isinstance(metrics, CommMetrics):
        return {"kind": "comm", **asdict(metrics)}
    raise TypeError(f"unsupported metrics type {type(metrics).__name__}")


def chunk_result_from_payload(result_payload: dict[str, Any]) -> ChunkResult:
    if not result_payload.get("ok"):
        return ChunkResult(metrics=None, error=result_payload.get("error"))

    metrics_payload = dict(result_payload["metrics"])
    kind = metrics_payload.pop("kind")
    metrics: Metrics
    if kind == "compute":
        metrics = ComputeMetrics(**metrics_payload)
    elif kind == "comm":
        metrics = CommMetrics(**metrics_payload)
    else:
        raise ValueError(f"unknown metrics payload kind {kind!r}")
    # The worker's ``gpu_name`` is the physical GPU it observed. That is
    # provenance (``observed_gpu_name``), not the DB cache key; the requested
    # cache key is supplied by the controller/executor and set by
    # the batch controller. When no explicit key was requested, the validated
    # observed name becomes the key.
    return ChunkResult(
        metrics=metrics,
        observed_gpu_name=result_payload.get("gpu_name"),
        cuda_version=result_payload.get("cuda_version"),
        backend_version=result_payload.get("backend_version"),
    )


__all__ = [
    "chunk_result_from_payload",
    "metrics_to_payload",
    "resolve_chunk_backend",
]
