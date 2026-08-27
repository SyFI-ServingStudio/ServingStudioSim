"""Unit tests for the list-runner contract adapter + worker payload round-trip.

Pure CPU (no GPU, no subprocess): exercises ``batched`` — the single-spec → list
adapter every compute runner is wrapped in — and the
``RunnerResult -> worker payload -> ChunkResult`` round-trip the collapsed worker
loop depends on. The runner contract is ``run(list[dict]) -> list[RunnerResult]``,
1:1 and in order; these lock that in.
"""

from __future__ import annotations

from profiling.exec.local_worker import _to_payload
from profiling.exec.payload import chunk_result_from_payload
from profiling.runners.batched import _empty_cuda_cache, batched
from profiling.runners.metrics import ComputeMetrics, RunnerResult


def _compute(value: int) -> ComputeMetrics:
    return ComputeMetrics(time_ms=float(value), tflops=1.0, memory_bandwidth_gbps=2.0)


def test_batched_preserves_order_and_count():
    calls: list[dict] = []

    def single_fn(**kwargs) -> ComputeMetrics:
        calls.append(kwargs)
        return _compute(kwargs["value"])

    results = batched(single_fn)([{"value": 1}, {"value": 2}, {"value": 3}])

    assert len(results) == 3
    assert [r.metrics.time_ms for r in results] == [1.0, 2.0, 3.0]
    assert all(r.error is None for r in results)
    # Coerced kwargs are forwarded verbatim, in order (the worker already coerced).
    assert calls == [{"value": 1}, {"value": 2}, {"value": 3}]


def test_batched_captures_per_item_error_without_aborting():
    def single_fn(**kwargs) -> ComputeMetrics:
        if kwargs["value"] == 2:
            raise RuntimeError("boom")
        return _compute(kwargs["value"])

    results = batched(single_fn)([{"value": 1}, {"value": 2}, {"value": 3}])

    assert len(results) == 3
    # One bad shape is captured as an error; its neighbours still succeed, in order.
    assert results[0].metrics is not None and results[0].error is None
    assert results[1].metrics is None and results[1].error == "boom"
    assert results[2].metrics is not None and results[2].error is None


def test_batched_empty_list_returns_empty():
    assert batched(lambda **_: _compute(0))([]) == []


def test_to_payload_success_round_trips_through_chunk_result():
    metrics = _compute(5)
    payload = _to_payload(
        RunnerResult(metrics=metrics),
        gpu_name="NVIDIA H100",
        runtime_versions={"cuda_version": "13.0", "backend_version": "0.23.0"},
    )

    assert payload == {
        "ok": True,
        "metrics": {
            "kind": "compute",
            "time_ms": 5.0,
            "tflops": 1.0,
            "memory_bandwidth_gbps": 2.0,
            "energy_j": 0.0,
        },
        "gpu_name": "NVIDIA H100",
        "cuda_version": "13.0",
        "backend_version": "0.23.0",
    }
    chunk = chunk_result_from_payload(payload)
    assert chunk.metrics == metrics
    # The worker's reported name is physical-GPU provenance, not the DB cache key.
    assert chunk.observed_gpu_name == "NVIDIA H100"
    assert chunk.cuda_version == "13.0"
    assert chunk.backend_version == "0.23.0"
    assert chunk.error is None


def test_to_payload_error_round_trips_through_chunk_result():
    payload = _to_payload(RunnerResult(error="kernel failed"), gpu_name="NVIDIA H100")
    assert payload == {"ok": False, "error": "kernel failed"}

    chunk = chunk_result_from_payload(payload)
    assert chunk.metrics is None
    assert chunk.error == "kernel failed"


def test_empty_cuda_cache_is_safe_without_a_device():
    # The adapter flushes the CUDA cache after a failed launch; on a host with no
    # torch / no device that must be a silent no-op, not an error.
    _empty_cuda_cache()
