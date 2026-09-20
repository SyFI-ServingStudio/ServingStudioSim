from __future__ import annotations

import logging
import signal
import subprocess
from collections import Counter

import pytest

from profiling.db.batch import _log_batch_failures, execute_profile_batch
from profiling.exec.local import _worker_failure_message
from profiling.exec.pool import ChunkResult, GpuChunk, GpuPool
from profiling.runners.metrics import ComputeMetrics


def _spec(m: int) -> dict:
    return {"m": m, "n": 8, "k": 8, "dtype": "fp16", "backend": "torch"}


class _ResultChunk(GpuChunk):
    def __init__(self, results: list[ChunkResult]):
        self.results = results

    def run(self, kernel_kind: str, specs: list[dict]) -> list[ChunkResult]:
        assert kernel_kind == "single_gemm"
        assert len(specs) == len(self.results)
        return self.results


class _ResultPool(GpuPool):
    def __init__(self, results: list[ChunkResult]):
        self.results = results

    def acquire_chunks(self, k: int, max_concurrent: int):
        assert k == 1
        assert max_concurrent == len(self.results)
        yield _ResultChunk(self.results)


class _VariableGpuPool(GpuPool):
    def acquire_chunks(self, k: int, max_concurrent: int):
        assert max_concurrent == 1
        if k == 2:
            yield _ResultChunk([ChunkResult(metrics=_metrics(), observed_gpu_name="NVIDIA H200")])
        elif k == 4:
            yield _ResultChunk([ChunkResult(metrics=None, error="collective failed")])
        else:
            raise AssertionError(f"unexpected GPU count: {k}")


def _metrics() -> ComputeMetrics:
    return ComputeMetrics(
        time_ms=1.0,
        tflops=0.1,
        memory_bandwidth_gbps=0.2,
        energy_j=0.0,
    )


def test_all_failed_specs_log_the_grouped_worker_reason_at_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = _ResultPool(
        [
            ChunkResult(metrics=None, error="CUDA kernel\nis unavailable"),
            ChunkResult(metrics=None, error="CUDA kernel is unavailable"),
            ChunkResult(metrics=None),
        ]
    )

    with caplog.at_level(logging.INFO, logger="profiling.db.batch"):
        outcome = execute_profile_batch(
            "single_gemm",
            [_spec(1), _spec(2), _spec(3)],
            pool=pool,
            db_path=None,
            gpu_name="NVIDIA H200",
        )

    assert outcome.results == [None, None, None]
    assert outcome.provenance.source == "cache_key"
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.ERROR
    assert record.getMessage() == (
        "single_gemm:torch profiled 0/3 specs; 3 failed and were not inserted "
        "(2x CUDA kernel is unavailable; 1x runner returned no metrics and no error)"
    )


def test_partial_batch_failure_logs_warning_and_preserves_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = _ResultPool(
        [
            ChunkResult(metrics=_metrics(), observed_gpu_name="NVIDIA H200"),
            ChunkResult(metrics=None, error="shape unsupported"),
        ]
    )

    with caplog.at_level(logging.INFO, logger="profiling.db.batch"):
        outcome = execute_profile_batch(
            "single_gemm",
            [_spec(1), _spec(2)],
            pool=pool,
            db_path=None,
            gpu_name="NVIDIA H200",
        )

    assert outcome.results[0] == _metrics()
    assert outcome.results[1] is None
    assert outcome.provenance.source == "measurement"
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "profiled 1/2 specs" in caplog.records[0].getMessage()
    assert "1x shape unsupported" in caplog.records[0].getMessage()


def test_same_backend_across_gpu_counts_logs_once_at_backend_severity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="profiling.db.batch"):
        outcome = execute_profile_batch(
            "single_gemm",
            [_spec(1), _spec(2)],
            pool=_VariableGpuPool(),
            gpu_count_fn=lambda spec: 2 if spec["m"] == 1 else 4,
            db_path=None,
            gpu_name="NVIDIA H200",
        )

    assert outcome.results[0] == _metrics()
    assert outcome.results[1] is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert caplog.records[0].getMessage() == (
        "single_gemm:torch profiled 1/2 specs; 1 failed and were not inserted "
        "(1x collective failed)"
    )


def test_failure_summary_limits_distinct_reasons_and_reason_length(
    caplog: pytest.LogCaptureFixture,
) -> None:
    failures = Counter(
        {
            "most common": 4,
            "second reason": 3,
            "x" * 300: 2,
            "omitted": 1,
        }
    )

    with caplog.at_level(logging.INFO, logger="profiling.db.batch"):
        _log_batch_failures("single_gemm", "torch", 11, failures)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "4x most common" in message
    assert "3x second reason" in message
    assert "2x " + "x" * 237 + "..." in message
    assert "1 additional reason(s)" in message
    assert "omitted" not in message


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["worker"], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def test_worker_failure_keeps_stdout_when_a_container_banner_fills_stderr() -> None:
    """Container workers print a wrapper banner to stderr on every run, so
    `stderr or stdout` always short-circuited and the real cause on stdout was
    never reported."""

    message = _worker_failure_message(
        _completed(
            1,
            stdout="Traceback (most recent call last):\nValueError: bad shape",
            stderr="--- Wrapper: Applying owner=kanzhu, memory=432834647654 bytes ---",
        )
    )

    assert "ValueError: bad shape" in message
    assert "Wrapper: Applying owner" in message


def test_worker_failure_names_the_signal_that_killed_the_worker() -> None:
    """A killed worker leaves no traceback, so the exit status is the only
    evidence there is; the old message dropped it."""

    message = _worker_failure_message(_completed(-signal.SIGKILL, stderr="banner only"))

    assert "SIGKILL" in message
    assert f"returncode {-signal.SIGKILL}" in message


def test_worker_failure_marks_an_empty_stream_rather_than_omitting_it() -> None:
    message = _worker_failure_message(_completed(2, stdout="", stderr="boom"))

    assert "stdout: <empty>" in message
    assert "stderr: boom" in message


def test_worker_failure_tails_a_long_stream_because_the_exception_is_last() -> None:
    message = _worker_failure_message(_completed(1, stderr="x" * 5000 + "RuntimeError: last"))

    assert "RuntimeError: last" in message
    assert "truncated" in message
