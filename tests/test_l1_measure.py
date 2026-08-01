"""Tests for the ``measure`` trend+telemetry diagnostic.

CPU tier: the measure context roundtrip, the pure runtime/telemetry summaries,
and the CLI wiring (perf_api facade is monkeypatched, no GPU). GPU tier: one real
end-to-end smoke run that asserts the artifact set lands on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from profiling import cli, perf_api
from profiling.profilers import trend
from profiling.profilers.measure_context import (
    MeasureContext,
    clear_measure_context,
    get_measure_context,
    set_measure_context,
)

# ── CPU tier: measure context is inert unless explicitly set ─────────────────


def test_measure_context_defaults_inert():
    clear_measure_context()
    assert get_measure_context() is None


def test_measure_context_set_get_clear_roundtrip(tmp_path):
    context = MeasureContext(output_dir=tmp_path, label="single_gemm:torch", shape={"m": 8})
    set_measure_context(context)
    try:
        active = get_measure_context()
        assert active is context
        assert active.consumed is False
        assert active.clear_l2 is True
    finally:
        clear_measure_context()
    assert get_measure_context() is None


# ── CPU tier: pure summaries over synthetic samples ──────────────────────────


def _synthetic_series(count: int = 60) -> list[dict[str, object]]:
    return [
        {
            "sample_index": index,
            "start_s": index * 0.001,
            "duration_ms": 1.0 + 0.01 * index,
            "kernel_count": 1,
            "kernel_names": "gemm",
        }
        for index in range(count)
    ]


def test_trend_summarize_shape():
    metadata = {"label": "single_gemm:torch", "shape": {"m": 8}, "requested_duration_s": 1.0}
    summary = trend.summarize(_synthetic_series(), metadata)

    assert summary["schema_version"] == 1
    runtime = summary["runtime_ms"]
    assert runtime["min"] <= runtime["median"] <= runtime["max"]
    assert runtime["mean"] > 0
    for key in ("p10", "p90", "p99", "first_1s_mean", "last_1s_mean", "linear_slope_ms_per_s"):
        assert key in runtime
    assert summary["one_second_bins"]
    assert summary["metadata"]["label"] == "single_gemm:torch"


def test_align_telemetry_bins_runtimes_onto_samples():
    from profiling.profilers import telemetry

    runtimes = [
        {"start_s": 0.1, "duration_ms": 1.0},
        {"start_s": 0.6, "duration_ms": 2.0},
        {"start_s": 1.1, "duration_ms": 3.0},
    ]
    samples = [
        {"time_s": 0.0, "power_w": 100.0},
        {"time_s": 0.5, "power_w": 110.0},
        {"time_s": 1.0, "power_w": 120.0},
    ]

    aligned = telemetry.align_telemetry(runtimes, samples)

    assert [row["runtime_count"] for row in aligned] == [1, 1, 1]
    assert aligned[0]["runtime_mean_ms"] == 1.0
    assert aligned[2]["runtime_mean_ms"] == 3.0


def test_summarize_telemetry_shape_without_pynvml():
    from profiling.profilers import telemetry

    # No clock_throttle_reasons / pstate keys, so the pynvml-backed decode path is
    # never reached — this stays a pure CPU test.
    aligned = [
        {"time_s": t, "power_w": 300.0 + t, "sm_clock_mhz": 1800.0, "runtime_mean_ms": 2.0 + t}
        for t in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    summary = telemetry.summarize_telemetry(aligned)

    assert summary["sample_count"] == 5
    assert summary["metrics"]["power_w"]["max"] >= summary["metrics"]["power_w"]["min"]
    assert summary["observed_clock_throttle_reasons"] == []


# ── CPU tier: per-launch record splitting (no GPU; fabricated records) ───────


def _record(name: str, start_ns: int, duration_ns: int, correlation_id: int):
    from profiling.profilers.cupti_kernel_profiler import KernelRecord

    return KernelRecord(
        name=name,
        device_id=0,
        stream_id=0,
        correlation_id=correlation_id,
        start_ns=start_ns,
        end_ns=start_ns + duration_ns,
        duration_ns=duration_ns,
    )


def test_split_launch_series_warm_layout():
    from profiling.profilers.cupti_kernel_profiler import _LaunchPattern, split_launch_series

    records = [_record("gemm", 1000 + 1000 * i, 100, i) for i in range(3)]
    series = split_launch_series(
        records,
        launch_pattern=_LaunchPattern(callable_kernel_names=("gemm",), clear_kernel_names=()),
        launches_per_run=3,
        clear_l2_between_launches=False,
    )

    assert [sample["start_ns"] for sample in series] == [1000, 2000, 3000]
    assert all(sample["duration_ms"] == 100 / 1e6 for sample in series)
    assert all(sample["kernel_count"] == 1 for sample in series)


def test_split_launch_series_cold_layout_excludes_clears():
    from profiling.profilers.cupti_kernel_profiler import _LaunchPattern, split_launch_series

    # Cold layout: gemm, reduce, gemm, reduce, gemm (clear between, none trailing).
    records = [
        _record("gemm", 1000, 100, 0),
        _record("reduce", 1500, 40, 1),
        _record("gemm", 2000, 100, 2),
        _record("reduce", 2500, 40, 3),
        _record("gemm", 3000, 100, 4),
    ]
    series = split_launch_series(
        records,
        launch_pattern=_LaunchPattern(
            callable_kernel_names=("gemm",), clear_kernel_names=("reduce",)
        ),
        launches_per_run=3,
        clear_l2_between_launches=True,
    )

    assert [sample["start_ns"] for sample in series] == [1000, 2000, 3000]
    assert all(sample["kernel_names"] == "gemm" for sample in series)


def test_split_launch_series_multi_kernel_callable_sums_duration():
    from profiling.profilers.cupti_kernel_profiler import _LaunchPattern, split_launch_series

    records = [
        _record("gemm", 1000, 100, 0),
        _record("epilogue", 1100, 50, 1),
        _record("gemm", 2000, 100, 2),
        _record("epilogue", 2100, 50, 3),
    ]
    series = split_launch_series(
        records,
        launch_pattern=_LaunchPattern(
            callable_kernel_names=("gemm", "epilogue"), clear_kernel_names=()
        ),
        launches_per_run=2,
        clear_l2_between_launches=False,
    )

    assert len(series) == 2
    assert all(sample["duration_ms"] == 150 / 1e6 for sample in series)
    assert all(sample["kernel_count"] == 2 for sample in series)


# ── CPU tier: CLI wiring (perf_api facade monkeypatched) ──────────────────────


def _fake_measure_result(output_dir) -> dict:
    output_path = Path(output_dir)
    return {
        "kernel_kind": "single_gemm",
        "backend": "torch_linear",
        "gpu_index": 0,
        "gpu_name": None,
        "observed_gpu_name": "NVIDIA H200",
        "output_dir": str(output_path),
        "time_ms": 1.23,
        "artifacts": [
            str((output_path / "runtimes.csv").resolve()),
            str((output_path / "summary.json").resolve()),
        ],
        "metrics": None,
        "runner_error": None,
    }


def test_cli_measure_forwards_defaults(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_measure(kernel_kind, spec, **kwargs):
        calls["kernel_kind"] = kernel_kind
        calls["spec"] = spec
        calls["kwargs"] = kwargs
        return _fake_measure_result(kwargs["output_dir"])

    monkeypatch.setattr(perf_api, "measure_kernel", fake_measure)
    exit_code = cli.main(
        [
            "measure",
            "single_gemm",
            "--backend",
            "torch_linear",
            "--output-dir",
            str(tmp_path / "out"),
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
            "--json",
        ]
    )

    assert exit_code == 0
    assert calls["kernel_kind"] == "single_gemm"
    assert calls["kwargs"]["backend"] == "torch_linear"
    assert str(calls["kwargs"]["output_dir"]) == str(tmp_path / "out")
    assert calls["kwargs"]["duration_s"] == 10.0
    assert calls["kwargs"]["telemetry_hz"] == 20.0
    assert calls["kwargs"]["clear_l2"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["time_ms"] == 1.23


def test_cli_measure_no_clear_l2_flag(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_measure(kernel_kind, spec, **kwargs):
        calls["kwargs"] = kwargs
        return _fake_measure_result(kwargs["output_dir"])

    monkeypatch.setattr(perf_api, "measure_kernel", fake_measure)
    monkeypatch.chdir(tmp_path)
    exit_code = cli.main(
        [
            "measure",
            "single_gemm",
            "--backend",
            "torch_linear",
            "--no-clear-l2",
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
            "--json",
        ]
    )

    assert exit_code == 0
    assert calls["kwargs"]["clear_l2"] is False


def test_cli_measure_default_output_dir(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_measure(kernel_kind, spec, **kwargs):
        calls["kwargs"] = kwargs
        return _fake_measure_result(kwargs["output_dir"])

    monkeypatch.setattr(perf_api, "measure_kernel", fake_measure)
    monkeypatch.chdir(tmp_path)
    cli.main(
        [
            "measure",
            "single_gemm",
            "--backend",
            "torch_linear",
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
            "--json",
        ]
    )
    assert str(calls["kwargs"]["output_dir"]) == "measure_single_gemm_torch_linear"


def test_cli_measure_rejects_multiple_specs(monkeypatch, capsys):
    monkeypatch.setattr(
        perf_api,
        "measure_kernel",
        lambda *args, **kwargs: _fake_measure_result(kwargs["output_dir"]),
    )
    exit_code = cli.main(
        [
            "measure",
            "single_gemm",
            "--backend",
            "torch_linear",
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
            "--spec",
            '{"m": 16, "n": 8, "k": 8, "dtype": "fp16"}',
            "--json",
        ]
    )
    assert exit_code == 2


# ── GPU tier: real end-to-end smoke ──────────────────────────────────────────


@pytest.mark.gpu
def test_measure_kernel_writes_artifacts_end_to_end(tmp_path):
    from pathlib import Path

    pytest.importorskip("torch")
    from profiling.exec.local import find_idle_gpus
    from profiling.measure import measure_kernel
    from profiling.profilers import cupti_kernel_profiler

    try:
        cupti_kernel_profiler._resolve_cupti_paths()
    except RuntimeError as exc:
        pytest.skip(f"CUPTI headers/libs are not available: {exc}")
    if not find_idle_gpus():
        pytest.skip("no idle GPU available for the measure smoke run")

    output_dir = tmp_path / "measure_smoke"
    result = measure_kernel(
        "single_gemm",
        {"m": 1024, "n": 4096, "k": 4096, "dtype": "fp16"},
        backend="torch_linear",
        output_dir=output_dir,
        duration_s=0.5,
    )

    assert result["time_ms"] is not None and result["time_ms"] > 0
    for name in ("runtimes.csv", "summary.json", "runtime_trend.png"):
        assert (output_dir / name).exists(), name
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["runtime_ms"]["median"] > 0
    assert Path(result["output_dir"]) == output_dir.resolve()
