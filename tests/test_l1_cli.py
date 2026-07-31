from __future__ import annotations

import json

from profiling import cli, perf_api
from profiling.db import DType, ProfileRow, Table
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.runners.metrics import ComputeMetrics


def _json_stdout(capsys):
    return json.loads(capsys.readouterr().out)


def test_cli_count_missing_json_uses_perf_api_read_only(tmp_path, capsys):
    db_path = tmp_path / "profile.db"
    exit_code = cli.main(
        [
            "count-missing",
            "single_gemm",
            "--backend",
            "torch",
            "--gpu-name",
            "FakeGPU",
            "--db",
            str(db_path),
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
            "--json",
        ]
    )

    assert exit_code == 0
    payload = _json_stdout(capsys)
    assert payload["ok"] is True
    assert payload["missing_count"] == 1
    assert payload["spec_count"] == 1
    assert not db_path.exists()


def test_cli_query_json_reads_existing_row(tmp_path, capsys):
    db_path = tmp_path / "profile.db"
    profiler_spec = find_kernel_profiler_spec("single_gemm", "torch")
    Table(profiler_spec, db_path).insert(
        [
            ProfileRow(
                args=SingleGemmArgs(m=8, n=8, k=8, dtype=DType.FP16),
                metrics=ComputeMetrics(
                    time_ms=2.5,
                    tflops=0.1,
                    memory_bandwidth_gbps=0.2,
                    energy_j=0.3,
                ),
                gpu_name="FakeGPU",
                backend="torch",
            )
        ]
    )

    exit_code = cli.main(
        [
            "query",
            "single_gemm",
            "--backend",
            "torch",
            "--gpu-name",
            "FakeGPU",
            "--db",
            str(db_path),
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
            "--json",
        ]
    )

    assert exit_code == 0
    payload = _json_stdout(capsys)
    assert payload["ok"] is True
    assert payload["missing_count"] == 0
    assert payload["results"][0]["status"] == "ok"
    assert payload["results"][0]["metric_family"] == "compute"
    assert payload["results"][0]["metrics"]["time_ms"] == 2.5
    assert payload["results"][0]["metrics"]["energy_j"] == 0.3


def test_cli_run_force_calls_generated_perf_api(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_get(specs, *, backend: str, gpu_name: str | None = None, force: bool = False):
        calls["get"] = {
            "specs": specs,
            "backend": backend,
            "gpu_name": gpu_name,
            "force": force,
            "db_path": str(perf_api.DB_PATH),
        }
        return [
            ComputeMetrics(
                time_ms=1.0,
                tflops=2.0,
                memory_bandwidth_gbps=3.0,
                energy_j=4.0,
            )
        ]

    def fake_count(specs, *, backend: str, gpu_name: str | None = None):
        calls["count"] = {
            "specs": specs,
            "backend": backend,
            "gpu_name": gpu_name,
        }
        return 0

    monkeypatch.setattr(perf_api, "get_single_gemm_times", fake_get)
    monkeypatch.setattr(perf_api, "count_missing_single_gemm", fake_count)

    exit_code = cli.main(
        [
            "run",
            "single_gemm",
            "--backend",
            "torch",
            "--gpu-name",
            "FakeGPU",
            "--db",
            str(tmp_path / "profile.db"),
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
            "--force",
            "--json",
        ]
    )

    assert exit_code == 0
    assert calls["get"]["force"] is True
    assert calls["get"]["backend"] == "torch"
    assert calls["get"]["gpu_name"] == "FakeGPU"
    assert calls["get"]["db_path"] == str(tmp_path / "profile.db")
    payload = _json_stdout(capsys)
    assert payload["ok"] is True
    assert payload["mode"] == "force-refresh"
    assert payload["results"][0]["metrics"]["energy_j"] == 4.0


def test_cli_run_accepts_batched_specs_from_flags_and_file(
    monkeypatch,
    tmp_path,
    capsys,
):
    specs_path = tmp_path / "specs.jsonl"
    specs_path.write_text(
        '{"m": 16, "n": 8, "k": 8, "dtype": "fp16"}\n{"m": 32, "n": 8, "k": 8, "dtype": "fp16"}\n',
        encoding="utf-8",
    )
    captured_specs = []

    def fake_get(specs, *, backend: str, gpu_name: str | None = None, force: bool = False):
        del backend, gpu_name, force
        captured_specs.extend(specs)
        return [
            ComputeMetrics(
                time_ms=float(spec["m"]),
                tflops=2.0,
                memory_bandwidth_gbps=3.0,
                energy_j=4.0,
            )
            for spec in specs
        ]

    def fake_count(specs, *, backend: str, gpu_name: str | None = None):
        del specs, backend, gpu_name
        return 0

    monkeypatch.setattr(perf_api, "get_single_gemm_times", fake_get)
    monkeypatch.setattr(perf_api, "count_missing_single_gemm", fake_count)

    exit_code = cli.main(
        [
            "run",
            "single_gemm",
            "--backend",
            "torch",
            "--gpu-name",
            "FakeGPU",
            "--db",
            str(tmp_path / "profile.db"),
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
            "--specs",
            str(specs_path),
            "--force",
            "--json",
        ]
    )

    assert exit_code == 0
    assert [spec["m"] for spec in captured_specs] == [8, 16, 32]
    payload = _json_stdout(capsys)
    assert payload["ok"] is True
    assert payload["spec_count"] == 3
    assert [result["metrics"]["time_ms"] for result in payload["results"]] == [
        8.0,
        16.0,
        32.0,
    ]


def test_cli_run_returns_nonzero_when_rows_remain_missing(monkeypatch, tmp_path, capsys):
    def fake_get(specs, *, backend: str, gpu_name: str | None = None, force: bool = False):
        del specs, backend, gpu_name, force
        return [
            ComputeMetrics(
                time_ms=1.0,
                tflops=2.0,
                memory_bandwidth_gbps=3.0,
                energy_j=4.0,
            )
        ]

    def fake_count(specs, *, backend: str, gpu_name: str | None = None):
        del specs, backend, gpu_name
        return 1

    monkeypatch.setattr(perf_api, "get_single_gemm_times", fake_get)
    monkeypatch.setattr(perf_api, "count_missing_single_gemm", fake_count)

    exit_code = cli.main(
        [
            "run",
            "single_gemm",
            "--backend",
            "torch",
            "--gpu-name",
            "FakeGPU",
            "--db",
            str(tmp_path / "profile.db"),
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
            "--json",
        ]
    )

    assert exit_code == 1
    payload = _json_stdout(capsys)
    assert payload["ok"] is False
    assert payload["missing_count"] == 1
