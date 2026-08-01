"""Phase A contract tests: kernel_profile / kernel_measure first-class resource
metadata, GPU provenance, cached-only vs measured, identity-mismatch rejection,
development identity, and managed-registration id routing.

These cover the Python artifact + execution-layer half. The Rust analyzer
discovery/curve/plot/hardware half is tested in
``analyzer/rust/src/ui_service/tests.rs``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib import request

import pytest

from launcher.managed_job import MANAGED_JOB_CONTEXT_ENV
from profiling import perf_api
from profiling.artifacts import (
    MEASUREMENT_METADATA_FILENAME,
    PROFILE_JOB_METADATA_FILENAME,
    PROFILE_METADATA_FILENAME,
    _write_json_atomic,
)
from profiling.db.batch import (
    ProfileProvenance,
    execute_profile_batch,
    run_profile_batch,
)
from profiling.db.registry import find_kernel_profiler_spec
from profiling.exec.pool import ChunkResult, GpuChunk, GpuPool
from profiling.facade import KindTimesResult, run_kind_times
from profiling.gpu_catalog import resolve_gpu_spec
from profiling.runners.metrics import ComputeMetrics


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class FakeUrlopenResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _managed_context(tmp_path: Path, backend_url: str = "http://backend.test") -> Path:
    context_path = tmp_path / "managed-job.json"
    context_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_url": backend_url,
                "capability_token": "secret",
            }
        )
    )
    return context_path


def _run_args(output_dir: Path | None, **overrides) -> SimpleNamespace:
    base = dict(
        table="single_gemm",
        backend="torch",
        spec=[
            json.dumps({"m": 1, "n": 128, "k": 64, "dtype": "bfloat16"}),
            json.dumps({"m": 2, "n": 128, "k": 64, "dtype": "bfloat16"}),
        ],
        specs=None,
        db=None,
        gpu_name="NVIDIA H200",
        force=False,
        output_dir=output_dir,
        json=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_run_kind_times(
    kernel_kind,
    specs,
    *,
    backend,
    gpu_name=None,
    db_path,
    jit_enabled,
    force=False,
    provenance=None,
):
    del kernel_kind, jit_enabled, backend, db_path, force
    results = [
        ComputeMetrics(
            time_ms=float(index + 1),
            tflops=1.0,
            memory_bandwidth_gbps=2.0,
            energy_j=0.1,
        )
        for index, _ in enumerate(specs)
    ]
    return KindTimesResult(
        results=results,
        provenance=provenance or ProfileProvenance(source="cache_key", requested_gpu_name=gpu_name),
    )


def _development_cmd_run(monkeypatch, tmp_path: Path, output_dir: Path):
    monkeypatch.delenv(MANAGED_JOB_CONTEXT_ENV, raising=False)
    import profiling.cli as cli

    monkeypatch.setattr(cli, "_set_db_path", lambda _path: None)
    monkeypatch.setattr(cli, "run_kind_times", _fake_run_kind_times)
    monkeypatch.setattr(cli, "_count_facade", lambda _table: lambda *_a, **_k: 0)
    return cli._cmd_run(_run_args(output_dir))


def test_resolve_gpu_spec_maps_aliases_and_dense_peaks() -> None:
    resolved = resolve_gpu_spec("NVIDIA H200")
    assert resolved is not None
    assert resolved.canonical_name == "H200-SXM-141GB"
    assert resolved.matched_alias == "NVIDIA H200"
    assert resolved.mem_bandwidth_gbps == 4800.0
    assert resolved.bf16_tflops == 990.0
    assert resolved.dense_peak_tflops("bf16") == 990.0
    assert resolved.dense_peak_tflops("fp8_e4m3") == 1979.0
    assert resolved.dense_peak_tflops("totally-made-up-dtype") is None
    assert resolve_gpu_spec("Totally Made Up GPU") is None


def test_development_profile_writes_identity_and_cached_only_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "profile-artifact"
    assert _development_cmd_run(monkeypatch, tmp_path, output_dir) == 0

    metadata = json.loads((output_dir / PROFILE_METADATA_FILENAME).read_text())
    assert metadata["schema_version"] == 1
    assert metadata["profile_id"].startswith("kp_")
    assert len(metadata["profile_id"]) > 3
    assert metadata["kernel"] == {
        "kind": "single_gemm",
        "table": "single_gemm",
        "backend": "torch",
        "metric_family": "compute",
    }
    assert metadata["gpu"] == {"cache_key": "NVIDIA H200", "observed_name": None, "count": 1}
    # A run whose facade never observed a worker is honest cache provenance: the
    # cache key is kept and no physical GPU is fabricated.
    assert metadata["provenance"] == {
        "source": "cache_key",
        "resolved_canonical_name": "H200-SXM-141GB",
    }
    assert metadata["mode"] == "jit-fill"
    assert metadata["created_at"]
    assert metadata["artifacts"] == {
        "request": "request.json",
        "results": "results.json",
        "curve": "curve.json",
        "job_metadata": "job.meta.json",
    }
    # The legacy-compat file is still written beside the new discovery metadata.
    job_metadata = json.loads((output_dir / PROFILE_JOB_METADATA_FILENAME).read_text())
    assert job_metadata["origin"] == {"kind": "development"}
    assert job_metadata["resourceId"] == metadata["profile_id"]


def test_managed_profile_registration_carries_analyzer_resource_id(
    monkeypatch, tmp_path: Path
) -> None:
    import profiling.cli as cli

    context_path = _managed_context(tmp_path)
    monkeypatch.setenv(MANAGED_JOB_CONTEXT_ENV, str(context_path))
    artifact_root = tmp_path / "logs" / "profile"
    captured: list[request.Request] = []

    def fake_urlopen(http_request: request.Request, timeout: int):
        captured.append(http_request)
        assert timeout == 15
        if http_request.full_url.endswith("/register"):
            return FakeUrlopenResponse(
                {
                    "jobId": "j_profile",
                    "resourceId": "kp_registered",
                    "approvedRoot": str(artifact_root),
                }
            )
        return FakeUrlopenResponse({"ok": True})

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    monkeypatch.setattr(cli, "_set_db_path", lambda _path: None)

    def measured_run_kind_times(
        kernel_kind,
        specs,
        *,
        backend,
        gpu_name=None,
        db_path,
        jit_enabled,
        force=False,
    ):
        del kernel_kind, jit_enabled, db_path, force
        return KindTimesResult(
            results=[
                ComputeMetrics(
                    time_ms=float(index + 1),
                    tflops=1.0,
                    memory_bandwidth_gbps=2.0,
                    energy_j=0.1,
                )
                for index, _ in enumerate(specs)
            ],
            provenance=ProfileProvenance(
                source="measurement",
                requested_gpu_name=gpu_name or "NVIDIA H200",
                observed_gpu_name="NVIDIA H200",
                gpu_count=1,
            ),
        )

    monkeypatch.setattr(cli, "run_kind_times", measured_run_kind_times)
    monkeypatch.setattr(cli, "_count_facade", lambda _table: lambda *_a, **_k: 0)
    exit_code = cli._cmd_run(_run_args(artifact_root))
    assert exit_code == 0

    registration = json.loads(captured[0].data or b"{}")
    assert registration["jobKind"] == "kernel_profile"
    assert registration["analyzerResourceId"].startswith("kp_")
    metadata = json.loads((artifact_root / PROFILE_METADATA_FILENAME).read_text())
    assert registration["analyzerResourceId"] == metadata["profile_id"]
    job_metadata = json.loads((artifact_root / PROFILE_JOB_METADATA_FILENAME).read_text())
    assert job_metadata["origin"]["kind"] == "managed"
    assert job_metadata["origin"]["jobId"] == "j_profile"


def test_profile_id_is_stable_across_reruns(tmp_path: Path) -> None:
    from profiling.cli import _measurement_id, _profile_id

    assert _profile_id(tmp_path / "missing").startswith("kp_")
    first = _profile_id(tmp_path / "dir")
    metadata = {"schema_version": 1, "profile_id": first}
    _write_json_atomic(tmp_path / "dir" / PROFILE_METADATA_FILENAME, metadata)
    assert _profile_id(tmp_path / "dir") == first

    assert _measurement_id(tmp_path / "missing").startswith("km_")
    measurement_metadata = {"schema_version": 1, "measurement_id": "km_second"}
    _write_json_atomic(
        tmp_path / "mdir" / MEASUREMENT_METADATA_FILENAME,
        measurement_metadata,
    )
    assert _measurement_id(tmp_path / "mdir") == "km_second"
    assert not _profile_id(tmp_path / "mismatched") == "gp_bad"


def test_run_profile_batch_records_observed_and_requested_cache_key(tmp_path: Path) -> None:
    class ProvenanceChunk(GpuChunk):
        def run(self, kernel_kind, specs):
            return [
                ChunkResult(
                    metrics=ComputeMetrics(
                        time_ms=1.0,
                        tflops=0.5,
                        memory_bandwidth_gbps=2.0,
                        energy_j=0.0,
                    ),
                    observed_gpu_name="NVIDIA H200",
                )
                for _ in specs
            ]

    class ProvenancePool(GpuPool):
        def acquire_chunks(self, k, max_concurrent):
            yield ProvenanceChunk()

    provenance: dict = {}
    db_path = tmp_path / "profile.db"
    results = run_profile_batch(
        "single_gemm",
        [{"m": 4, "n": 8, "k": 16, "dtype": "fp16", "backend": "torch"}],
        pool=ProvenancePool(),
        db_path=db_path,
        gpu_name="H200-SXM-141GB",
        provenance=provenance,
    )
    assert results[0] is not None
    assert provenance["source"] == "measurement"
    assert provenance["requested_gpu_name"] == "H200-SXM-141GB"
    assert provenance["observed_gpu_name"] == "NVIDIA H200"
    assert provenance["gpu_count"] == 1

    profiler_spec = find_kernel_profiler_spec("single_gemm", "torch")
    from profiling.db.table import Table

    table = Table(profiler_spec, db_path)
    from profiling.db.args import DType
    from profiling.kernels.single_gemm import SingleGemmArgs

    saved = table.query(
        [SingleGemmArgs(m=4, n=8, k=16, dtype=DType.FP16)],
        backend="torch",
        gpu_name="H200-SXM-141GB",
    )[0]
    assert saved is not None and getattr(saved, "time_ms", None) == 1.0
    # The typed funnel returns the same values without an ambient side channel.
    outcome = execute_profile_batch(
        "single_gemm",
        [{"m": 4, "n": 8, "k": 16, "dtype": "fp16", "backend": "torch"}],
        pool=ProvenancePool(),
        db_path=db_path,
        gpu_name="H200-SXM-141GB",
    )
    assert outcome.results[0] is not None
    assert outcome.provenance == ProfileProvenance(
        source="measurement",
        requested_gpu_name="H200-SXM-141GB",
        observed_gpu_name="NVIDIA H200",
        gpu_count=1,
    )


def test_mismatch_rejects_before_any_db_insert(tmp_path: Path) -> None:
    """A requested key / observed GPU that disagree leave no row (fresh DB)."""

    class MismatchedChunk(GpuChunk):
        def run(self, kernel_kind, specs):
            return [
                ChunkResult(
                    metrics=ComputeMetrics(
                        time_ms=1.0,
                        tflops=0.5,
                        memory_bandwidth_gbps=2.0,
                        energy_j=0.0,
                    ),
                    observed_gpu_name="NVIDIA H100",
                )
                for _ in specs
            ]

    class MismatchedPool(GpuPool):
        def acquire_chunks(self, k, max_concurrent):
            yield MismatchedChunk()

    db_path = tmp_path / "profile.db"
    with pytest.raises(ValueError, match="do not resolve to the same canonical SKU"):
        run_profile_batch(
            "single_gemm",
            [{"m": 4, "n": 8, "k": 16, "dtype": "fp16", "backend": "torch"}],
            pool=MismatchedPool(),
            db_path=db_path,
            gpu_name="H200-SXM-141GB",
        )
    from profiling.db.table import Table

    profiler_spec = find_kernel_profiler_spec("single_gemm", "torch")
    assert Table(profiler_spec, db_path).metadata().row_count == 0


def test_two_distinct_observed_skus_reject_not_first_choice(tmp_path: Path) -> None:
    """Two chunks observing different physical GPU SKUs reject instead of silently
    picking the first observation."""

    class SkewedChunk(GpuChunk):
        def __init__(self, observed: str):
            self.observed = observed

        def run(self, kernel_kind, specs):
            return [
                ChunkResult(
                    metrics=ComputeMetrics(
                        time_ms=1.0,
                        tflops=0.5,
                        memory_bandwidth_gbps=2.0,
                        energy_j=0.0,
                    ),
                    observed_gpu_name=self.observed,
                )
                for _ in specs
            ]

    class TwoSkuPool(GpuPool):
        def acquire_chunks(self, k, max_concurrent):
            yield SkewedChunk("NVIDIA H200")
            yield SkewedChunk("NVIDIA H100")

    db_path = tmp_path / "profile.db"
    with pytest.raises(ValueError, match="different physical GPU SKUs"):
        run_profile_batch(
            "single_gemm",
            [
                {"m": 1, "n": 8, "k": 16, "dtype": "fp16", "backend": "torch"},
                {"m": 2, "n": 8, "k": 16, "dtype": "fp16", "backend": "torch"},
            ],
            pool=TwoSkuPool(),
            db_path=db_path,
            gpu_name="H200-SXM-141GB",
        )
    from profiling.db.table import Table

    profiler_spec = find_kernel_profiler_spec("single_gemm", "torch")
    assert Table(profiler_spec, db_path).metadata().row_count == 0


def test_unmatched_requested_key_rejects_before_insert(tmp_path: Path) -> None:
    """A requested cache key nobody can canonicalize fails a measured batch; the
    unmatched GPU is never defaulted to H200."""

    class ObservedChunk(GpuChunk):
        def run(self, kernel_kind, specs):
            return [
                ChunkResult(
                    metrics=ComputeMetrics(
                        time_ms=1.0,
                        tflops=0.5,
                        memory_bandwidth_gbps=2.0,
                        energy_j=0.0,
                    ),
                    observed_gpu_name="NVIDIA H200",
                )
                for _ in specs
            ]

    class ObservedPool(GpuPool):
        def acquire_chunks(self, k, max_concurrent):
            yield ObservedChunk()

    db_path = tmp_path / "profile.db"
    with pytest.raises(ValueError, match="not in gpu/spec.json"):
        run_profile_batch(
            "single_gemm",
            [{"m": 4, "n": 8, "k": 16, "dtype": "fp16", "backend": "torch"}],
            pool=ObservedPool(),
            db_path=db_path,
            gpu_name="Totally Made Up GPU",
        )
    from profiling.db.table import Table

    profiler_spec = find_kernel_profiler_spec("single_gemm", "torch")
    assert Table(profiler_spec, db_path).metadata().row_count == 0


def test_successful_execution_without_observed_gpu_rejects_before_insert(
    tmp_path: Path,
) -> None:
    """Cached-only means no worker ran; a successful execution needs provenance."""

    class FakeKeyChunk(GpuChunk):
        def run(self, kernel_kind, specs):
            return [
                ChunkResult(
                    metrics=ComputeMetrics(
                        time_ms=1.0,
                        tflops=0.5,
                        memory_bandwidth_gbps=2.0,
                        energy_j=0.0,
                    ),
                )
                for _ in specs
            ]

    class FakeKeyPool(GpuPool):
        def acquire_chunks(self, k, max_concurrent):
            yield FakeKeyChunk()

    database_path = tmp_path / "profile.db"
    with pytest.raises(RuntimeError, match="did not report observed_gpu_name"):
        execute_profile_batch(
            "single_gemm",
            [{"m": 4, "n": 8, "k": 16, "dtype": "fp16", "backend": "torch"}],
            pool=FakeKeyPool(),
            db_path=database_path,
        )

    from profiling.db.table import Table

    profiler_spec = find_kernel_profiler_spec("single_gemm", "torch")
    assert Table(profiler_spec, database_path).metadata().row_count == 0


def test_execution_without_database_still_reports_measurement_provenance() -> None:
    class ObservedChunk(GpuChunk):
        def run(self, kernel_kind, specs):
            return [
                ChunkResult(
                    metrics=ComputeMetrics(
                        time_ms=1.0,
                        tflops=0.5,
                        memory_bandwidth_gbps=2.0,
                        energy_j=0.0,
                    ),
                    observed_gpu_name="NVIDIA H200",
                )
                for _ in specs
            ]

    class ObservedPool(GpuPool):
        def acquire_chunks(self, k, max_concurrent):
            yield ObservedChunk()

    outcome = execute_profile_batch(
        "single_gemm",
        [{"m": 4, "n": 8, "k": 16, "dtype": "fp16", "backend": "torch"}],
        pool=ObservedPool(),
        db_path=None,
    )
    assert outcome.provenance == ProfileProvenance(
        source="measurement",
        requested_gpu_name="NVIDIA H200",
        observed_gpu_name="NVIDIA H200",
        gpu_count=1,
    )


def test_measured_provenance_without_worker_observation_fails_loudly() -> None:
    from profiling.cli import _outcome_provenance

    with pytest.raises(ValueError, match="no worker-observed physical GPU"):
        _outcome_provenance(
            ProfileProvenance(source="measurement", requested_gpu_name="NVIDIA H200"),
            "NVIDIA H200",
        )
    # A forced measurement that would have downgraded to cache-only must fail too.
    with pytest.raises(ValueError, match="refusing to stamp cache-only provenance"):
        _outcome_provenance(
            ProfileProvenance(source="cache_key", requested_gpu_name="NVIDIA H200"),
            "NVIDIA H200",
            forced=True,
        )
    # A non-forced cached-only job keeps its requested key and observes nothing.
    assert _outcome_provenance(
        ProfileProvenance(source="cache_key", requested_gpu_name="NVIDIA H200"),
        "NVIDIA H200",
    ) == ("NVIDIA H200", None, None)


def test_measured_gpu_identity_mismatch_rejected_programmatically() -> None:
    from profiling.cli import _validate_measured_gpu_identity

    # Same canonical SKU via alias → ok.
    _validate_measured_gpu_identity("H200-SXM-141GB", "NVIDIA H200")

    # Cache-only (no observed GPU) → ok, nothing fabricated.
    _validate_measured_gpu_identity("NVIDIA H200", None)

    # Different canonical SKU → the job must fail.
    try:
        _validate_measured_gpu_identity("H200-SXM-141GB", "NVIDIA H100")
    except ValueError as exc:
        assert "do not resolve to the same canonical SKU" in str(exc)
    else:
        raise AssertionError("expected mismatch rejection")

    # An observed GPU nobody can canonicalize → fail, never a default H200.
    try:
        _validate_measured_gpu_identity("NVIDIA H200", "Totally Made Up GPU")
    except ValueError as exc:
        assert "do not resolve to the same canonical SKU" in str(exc)
    else:
        raise AssertionError("expected unmatched observation rejection")


def test_managed_measure_registration_carries_analyzer_resource_id(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import profiling.cli as cli

    context_path = _managed_context(tmp_path)
    monkeypatch.setenv(MANAGED_JOB_CONTEXT_ENV, str(context_path))
    artifact_root = tmp_path / "logs" / "measure"
    captured: list[request.Request] = []

    def fake_urlopen(http_request: request.Request, timeout: int):
        captured.append(http_request)
        assert timeout == 15
        if http_request.full_url.endswith("/register"):
            return FakeResponse(
                {
                    "jobId": "j_measure",
                    "resourceId": "km_registered",
                    "approvedRoot": str(artifact_root),
                }
            )
        return FakeResponse({"ok": True})

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    monkeypatch.setattr(cli, "_set_db_path", lambda _path: None)

    def fake_measure(kernel_kind, spec, **kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            '{"schema_version": 1, "runtime_ms": {"median": 1.0}}'
        )
        return {
            "kernel_kind": "single_gemm",
            "backend": "torch",
            "gpu_index": 0,
            "gpu_name": "NVIDIA H200",
            "observed_gpu_name": "NVIDIA H200",
            "output_dir": str(kwargs["output_dir"]),
            "time_ms": 1.0,
            "artifacts": [str((output_dir / "summary.json").resolve())],
            "metrics": None,
            "runner_error": None,
        }

    monkeypatch.setattr(perf_api, "measure_kernel", fake_measure)
    exit_code = cli._cmd_measure(
        SimpleNamespace(
            table="single_gemm",
            backend="torch",
            spec=[json.dumps({"m": 8, "n": 8, "k": 8, "dtype": "bfloat16"})],
            specs=None,
            db=None,
            gpu_name="NVIDIA H200",
            output_dir=artifact_root,
            json=True,
            duration_s=10.0,
            telemetry_hz=20.0,
            telemetry=True,
            clear_l2=True,
        )
    )
    assert exit_code == 0
    registration = json.loads(captured[0].data or b"{}")
    assert registration["jobKind"] == "kernel_measure"
    assert registration["analyzerResourceId"].startswith("km_")
    metadata = json.loads((artifact_root / "kernel-measurement.meta.json").read_text())
    assert registration["analyzerResourceId"] == metadata["measurement_id"]
    assert metadata["gpu"]["observed_name"] == "NVIDIA H200"
    assert metadata["gpu"]["cache_key"] == "NVIDIA H200"


def test_measure_keeps_worker_observed_gpu_and_writes_metadata(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import profiling.cli as cli

    monkeypatch.delenv(MANAGED_JOB_CONTEXT_ENV, raising=False)
    monkeypatch.setattr(cli, "_set_db_path", lambda _path: None)
    calls: dict = {}

    def fake_measure(kernel_kind, spec, **kwargs):
        calls["kwargs"] = kwargs
        output_dir = kwargs["output_dir"]
        return {
            "kernel_kind": "single_gemm",
            "backend": "torch",
            "gpu_index": 0,
            "gpu_name": "NVIDIA H200",
            # The worker response carried the observed physical GPU; measure must
            # keep it rather than drop it.
            "observed_gpu_name": "NVIDIA H200",
            "output_dir": str(output_dir),
            "time_ms": 1.23,
            "artifacts": [
                str((output_dir / "runtimes.csv").resolve()),
                str((output_dir / "telemetry.csv").resolve()),
                str((output_dir / "summary.json").resolve()),
                str((output_dir / "runtime_trend.png").resolve()),
                str((output_dir / "runtime_telemetry.png").resolve()),
            ],
            "metrics": {
                "time_ms": 1.23,
                "tflops": 2.0,
                "memory_bandwidth_gbps": 3.0,
                "energy_j": 0.1,
            },
            "runner_error": None,
        }

    monkeypatch.setattr(perf_api, "measure_kernel", fake_measure)
    output_dir = tmp_path / "measure-out"
    exit_code = cli.main(
        [
            "measure",
            "single_gemm",
            "--backend",
            "torch",
            "--output-dir",
            str(output_dir),
            "--gpu-name",
            "NVIDIA H200",
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
            "--json",
        ],
        prog="python -m profiling",
    )
    assert exit_code == 0
    assert calls["kwargs"]["gpu_name"] == "NVIDIA H200"

    metadata = json.loads((output_dir / MEASUREMENT_METADATA_FILENAME).read_text())
    assert metadata["schema_version"] == 1
    assert metadata["measurement_id"].startswith("km_")
    assert metadata["kernel"]["table"] == "single_gemm"
    assert metadata["gpu"] == {
        "cache_key": "NVIDIA H200",
        "observed_name": "NVIDIA H200",
        "count": 1,
    }
    assert metadata["shape"] == {"m": 8, "n": 8, "k": 8, "dtype": "fp16"}
    assert metadata["plots"] == ["runtime_telemetry.png", "runtime_trend.png"]
    assert metadata["summary_file"] == "summary.json"
    assert "runtime_trend.png" in metadata["artifacts"]

    payload = json.loads(capsys.readouterr().out)
    assert payload["observed_gpu_name"] == "NVIDIA H200"
    assert payload["gpu_name"] == "NVIDIA H200"
    assert payload["resource_id"].startswith("km_")


def test_measure_rejects_artifact_outside_output_directory(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import profiling.cli as cli

    output_dir = tmp_path / "measure-out"

    def fake_measure(kernel_kind, spec, **kwargs):
        return {
            "kernel_kind": "single_gemm",
            "backend": "torch",
            "gpu_index": 0,
            "gpu_name": "NVIDIA H200",
            "observed_gpu_name": "NVIDIA H200",
            "output_dir": str(output_dir),
            "time_ms": 1.0,
            "artifacts": [str(tmp_path / "outside" / "summary.json")],
            "metrics": None,
            "runner_error": None,
        }

    monkeypatch.setattr(perf_api, "measure_kernel", fake_measure)
    exit_code = cli.main(
        [
            "measure",
            "single_gemm",
            "--backend",
            "torch",
            "--output-dir",
            str(output_dir),
            "--gpu-name",
            "NVIDIA H200",
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
        ]
    )
    assert exit_code == 2
    assert "outside output directory" in capsys.readouterr().err


def test_measured_facade_provenance_records_worker_observation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(MANAGED_JOB_CONTEXT_ENV, raising=False)
    from profiling import perf_api as py_perf_api
    from profiling.exec import set_default_pool

    monkeypatch.setattr(py_perf_api, "DB_PATH", tmp_path / "profile.db")

    class ObservedChunk(GpuChunk):
        def run(self, kernel_kind, specs):
            return [
                ChunkResult(
                    metrics=ComputeMetrics(
                        time_ms=1.0,
                        tflops=0.5,
                        memory_bandwidth_gbps=2.0,
                        energy_j=0.0,
                    ),
                    observed_gpu_name="NVIDIA H200",
                )
                for _ in specs
            ]

    class ObservedPool(GpuPool):
        def acquire_chunks(self, k, max_concurrent):
            yield ObservedChunk()

    set_default_pool(ObservedPool())
    try:
        # CLI and public facade share one typed per-invocation core; the CLI's
        # internal entry returns results AND provenance for this exact call.
        outcome = run_kind_times(
            "single_gemm",
            [{"m": 4, "n": 8, "k": 16, "dtype": "fp16"}],
            backend="torch",
            gpu_name="H200-SXM-141GB",
            db_path=py_perf_api.DB_PATH,
            jit_enabled=False,
            force=True,
        )
        assert outcome.results[0] is not None
        assert outcome.provenance == ProfileProvenance(
            source="measurement",
            requested_gpu_name="H200-SXM-141GB",
            observed_gpu_name="NVIDIA H200",
            gpu_count=1,
        )
        # The public generated facade returns the same list — no ambient channel.
        public_result = py_perf_api.get_single_gemm_times(
            [{"m": 4, "n": 8, "k": 16, "dtype": "fp16"}],
            backend="torch",
            gpu_name="H200-SXM-141GB",
            force=False,
        )[0]
        assert public_result is not None
    finally:
        set_default_pool(None)
