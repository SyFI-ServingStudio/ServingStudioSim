from __future__ import annotations

import json

import pytest

from profiling import cli, facade, perf_api
from profiling.db import DType, ProfileRow, Table
from profiling.db.batch import ProfileBatchOutcome, ProfileProvenance
from profiling.db.registry import find_kernel_profiler_spec
from profiling.facade import KindTimesResult
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.profilers import energy
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


def test_cli_run_force_calls_shared_internal_facade_op(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_run_kind_times(
        kernel_kind,
        specs,
        *,
        backend: str,
        gpu_name: str | None = None,
        db_path,
        jit_enabled: bool,
        force: bool = False,
        persist: bool = True,
    ):
        del jit_enabled
        calls["get"] = {
            "specs": specs,
            "backend": backend,
            "gpu_name": gpu_name,
            "force": force,
            "persist": persist,
            "db_path": str(db_path),
        }
        return KindTimesResult(
            results=[
                ComputeMetrics(
                    time_ms=1.0,
                    tflops=2.0,
                    memory_bandwidth_gbps=3.0,
                    energy_j=4.0,
                )
            ],
            provenance=ProfileProvenance(source="cache_key", requested_gpu_name=gpu_name),
        )

    def fake_count(specs, *, backend: str, gpu_name: str | None = None):
        calls["count"] = {
            "specs": specs,
            "backend": backend,
            "gpu_name": gpu_name,
        }
        return 0

    monkeypatch.setattr(cli, "run_kind_times", fake_run_kind_times)
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


def test_cli_run_fresh_asks_the_facade_to_measure_without_persisting(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_run_kind_times(
        kernel_kind,
        specs,
        *,
        backend: str,
        gpu_name: str | None = None,
        db_path,
        jit_enabled: bool,
        force: bool = False,
        persist: bool = True,
    ):
        del kernel_kind, specs, backend, gpu_name, db_path, jit_enabled
        calls["force"] = force
        calls["persist"] = persist
        return KindTimesResult(
            results=[
                ComputeMetrics(
                    time_ms=1.0,
                    tflops=2.0,
                    memory_bandwidth_gbps=3.0,
                    energy_j=4.0,
                )
            ],
            provenance=ProfileProvenance(source="measurement", requested_gpu_name="FakeGPU"),
        )

    monkeypatch.setattr(cli, "run_kind_times", fake_run_kind_times)

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
            "--fresh",
            "--json",
        ]
    )

    assert exit_code == 0
    # --fresh measures every spec exactly like --force; only the keeping differs.
    assert calls == {"force": True, "persist": False}
    payload = _json_stdout(capsys)
    assert payload["mode"] == "fresh"
    assert payload["persisted"] is False
    assert payload["results"][0]["metrics"]["time_ms"] == 1.0


def test_cli_run_fresh_returns_measured_rows_and_writes_no_db(monkeypatch, tmp_path, capsys):
    """The whole point of --fresh: real results back, profile.db never created."""

    db_path = tmp_path / "profile.db"
    seen = {}

    def fake_execute_profile_batch(kernel_kind, specs, *, db_path=None, gpu_name=None, **kwargs):
        del kernel_kind, kwargs
        seen["db_path"] = db_path
        return ProfileBatchOutcome(
            results=[
                ComputeMetrics(
                    time_ms=float(spec["m"]),
                    tflops=2.0,
                    memory_bandwidth_gbps=3.0,
                    energy_j=4.0,
                )
                for spec in specs
            ],
            provenance=ProfileProvenance(
                source="measurement",
                requested_gpu_name=gpu_name,
                observed_gpu_name=gpu_name,
                gpu_count=1,
            ),
        )

    monkeypatch.setattr(facade, "execute_profile_batch", fake_execute_profile_batch)

    exit_code = cli.main(
        [
            "run",
            "single_gemm",
            "--backend",
            "torch",
            "--gpu-name",
            "FakeGPU",
            "--db",
            str(db_path),
            "--spec",
            '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
            "--spec",
            '{"m": 16, "n": 8, "k": 8, "dtype": "fp16"}',
            "--fresh",
            "--json",
        ]
    )

    assert exit_code == 0
    assert seen["db_path"] is None
    assert not db_path.exists()
    payload = _json_stdout(capsys)
    # Results come straight from the batch, not from a read-back of inserted rows.
    assert [result["metrics"]["time_ms"] for result in payload["results"]] == [8.0, 16.0]
    assert payload["missing_count"] == 0


def test_cli_run_fresh_counts_specs_the_runner_failed_on(monkeypatch, tmp_path, capsys):
    """With nothing persisted, missing_count must describe this run, not the cache."""

    def fake_execute_profile_batch(kernel_kind, specs, *, db_path=None, gpu_name=None, **kwargs):
        del kernel_kind, db_path, kwargs
        return ProfileBatchOutcome(
            # Second spec failed in its runner, so the batch has no metrics for it.
            results=[
                ComputeMetrics(
                    time_ms=1.0,
                    tflops=2.0,
                    memory_bandwidth_gbps=3.0,
                    energy_j=4.0,
                ),
                None,
            ][: len(specs)],
            provenance=ProfileProvenance(
                source="measurement",
                requested_gpu_name=gpu_name,
                observed_gpu_name=gpu_name,
                gpu_count=1,
            ),
        )

    monkeypatch.setattr(facade, "execute_profile_batch", fake_execute_profile_batch)

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
            "--spec",
            '{"m": 16, "n": 8, "k": 8, "dtype": "fp16"}',
            "--fresh",
            "--json",
        ]
    )

    assert exit_code == 1
    payload = _json_stdout(capsys)
    assert payload["ok"] is False
    assert payload["missing_count"] == 1
    assert [result["status"] for result in payload["results"]] == ["ok", "missing"]


def test_cli_run_rejects_force_with_fresh(tmp_path):
    """Keeping and not keeping the rows are exclusive; argparse must say so."""

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "run",
                "single_gemm",
                "--backend",
                "torch",
                "--db",
                str(tmp_path / "profile.db"),
                "--spec",
                '{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}',
                "--force",
                "--fresh",
            ]
        )
    assert excinfo.value.code == 2


def test_run_kind_times_rejects_unpersisted_without_force(tmp_path):
    """persist=False only has a defined meaning for the force path."""

    with pytest.raises(ValueError, match="persist=False"):
        facade.run_kind_times(
            "single_gemm",
            [{"m": 8, "n": 8, "k": 8, "dtype": "fp16"}],
            backend="torch",
            gpu_name="FakeGPU",
            db_path=tmp_path / "profile.db",
            jit_enabled=True,
            force=False,
            persist=False,
        )


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

    def fake_get(
        kernel_kind,
        specs,
        *,
        backend: str,
        gpu_name: str | None = None,
        db_path,
        jit_enabled: bool,
        force: bool = False,
        persist: bool = True,
    ):
        del kernel_kind, backend, gpu_name, db_path, jit_enabled, force, persist
        captured_specs.extend(specs)
        return KindTimesResult(
            results=[
                ComputeMetrics(
                    time_ms=float(spec["m"]),
                    tflops=2.0,
                    memory_bandwidth_gbps=3.0,
                    energy_j=4.0,
                )
                for spec in specs
            ],
            provenance=ProfileProvenance(source="cache_key", requested_gpu_name=None),
        )

    def fake_count(specs, *, backend: str, gpu_name: str | None = None):
        del specs, backend, gpu_name
        return 0

    monkeypatch.setattr(cli, "run_kind_times", fake_get)
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
    def fake_get(
        kernel_kind,
        specs,
        *,
        backend: str,
        gpu_name: str | None = None,
        db_path,
        jit_enabled: bool,
        force: bool = False,
        persist: bool = True,
    ):
        del kernel_kind, specs, backend, gpu_name, db_path, jit_enabled, force, persist
        return KindTimesResult(
            results=[
                ComputeMetrics(
                    time_ms=1.0,
                    tflops=2.0,
                    memory_bandwidth_gbps=3.0,
                    energy_j=4.0,
                )
            ],
            provenance=ProfileProvenance(source="cache_key", requested_gpu_name=None),
        )

    def fake_count(specs, *, backend: str, gpu_name: str | None = None):
        del specs, backend, gpu_name
        return 1

    monkeypatch.setattr(cli, "run_kind_times", fake_get)
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


def test_cli_run_no_energy_records_the_policy_for_the_worker(monkeypatch, tmp_path, capsys):
    """The flag has to reach a profiler that runs several frames below, in another
    process, so it is carried in the environment rather than through signatures."""

    monkeypatch.delenv(energy.ENERGY_ENV, raising=False)
    seen = {}

    def fake_run_kind_times(kernel_kind, specs, **kwargs):
        del kernel_kind, specs, kwargs
        # Sampled here because the worker would read it at this point in the run.
        seen["enabled"] = energy.energy_enabled()
        return KindTimesResult(
            results=[
                ComputeMetrics(time_ms=1.0, tflops=2.0, memory_bandwidth_gbps=3.0, energy_j=0.0)
            ],
            provenance=ProfileProvenance(source="measurement", requested_gpu_name="FakeGPU"),
        )

    monkeypatch.setattr(cli, "run_kind_times", fake_run_kind_times)

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
            "--fresh",
            "--no-energy",
            "--json",
        ]
    )

    assert exit_code == 0
    assert seen["enabled"] is False
    payload = _json_stdout(capsys)
    assert payload["energy_measured"] is False


def test_cli_run_keeps_an_energy_opt_out_that_came_from_the_environment(
    monkeypatch, tmp_path, capsys
):
    """A caller that exports VIBESIM_PROFILE_ENERGY=0 and never passes a flag must
    still get the window skipped.

    ``set_energy_enabled`` writes the policy into the environment, so calling it
    with an argparse default of True overwrote the caller's "0" with "1" and
    silently re-enabled a ~500 ms per-spec window -- half the cost of a fill. It
    was measured, not reasoned about: a 64-spec run with the variable exported
    took 87.45 s where the same specs with the window genuinely off took 3.57 s.
    """

    monkeypatch.setenv(energy.ENERGY_ENV, "0")
    seen = {}

    def fake_run_kind_times(kernel_kind, specs, **kwargs):
        del kernel_kind, specs, kwargs
        seen["enabled"] = energy.energy_enabled()
        return KindTimesResult(
            results=[
                ComputeMetrics(time_ms=1.0, tflops=2.0, memory_bandwidth_gbps=3.0, energy_j=0.0)
            ],
            provenance=ProfileProvenance(source="measurement", requested_gpu_name="FakeGPU"),
        )

    monkeypatch.setattr(cli, "run_kind_times", fake_run_kind_times)

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
            "--fresh",
            "--json",
        ]
    )

    assert exit_code == 0
    assert seen["enabled"] is False
    # The artifact records what was in force, not the flag that was never passed.
    assert _json_stdout(capsys)["energy_measured"] is False


def test_cli_run_skips_energy_by_default(monkeypatch, tmp_path, capsys):
    """No flag and no environment means no energy window, matching the launcher.

    The two entry points used to disagree: the launcher's `energy:` key defaulted
    to false while the library default was on, so the same specs cost a second
    ~500 ms loop each depending on which door they came through.
    """

    monkeypatch.delenv(energy.ENERGY_ENV, raising=False)
    seen = {}

    def fake_run_kind_times(kernel_kind, specs, **kwargs):
        del kernel_kind, specs, kwargs
        seen["enabled"] = energy.energy_enabled()
        return KindTimesResult(
            results=[
                ComputeMetrics(time_ms=1.0, tflops=2.0, memory_bandwidth_gbps=3.0, energy_j=0.0)
            ],
            provenance=ProfileProvenance(source="measurement", requested_gpu_name="FakeGPU"),
        )

    monkeypatch.setattr(cli, "run_kind_times", fake_run_kind_times)

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
            "--fresh",
            "--json",
        ]
    )

    assert exit_code == 0
    assert seen["enabled"] is False
    assert _json_stdout(capsys)["energy_measured"] is False


def test_cli_run_energy_flag_opts_back_in(monkeypatch, tmp_path, capsys):
    """`--energy` is the only way to get the window from the CLI now."""

    monkeypatch.delenv(energy.ENERGY_ENV, raising=False)
    seen = {}

    def fake_run_kind_times(kernel_kind, specs, **kwargs):
        del kernel_kind, specs, kwargs
        seen["enabled"] = energy.energy_enabled()
        return KindTimesResult(
            results=[
                ComputeMetrics(time_ms=1.0, tflops=2.0, memory_bandwidth_gbps=3.0, energy_j=4.0)
            ],
            provenance=ProfileProvenance(source="measurement", requested_gpu_name="FakeGPU"),
        )

    monkeypatch.setattr(cli, "run_kind_times", fake_run_kind_times)

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
            "--fresh",
            "--energy",
            "--json",
        ]
    )

    assert exit_code == 0
    assert seen["enabled"] is True
    assert _json_stdout(capsys)["energy_measured"] is True


def test_energy_perf_skips_every_launch_when_disabled(monkeypatch):
    """Skipping must skip the warmup launches too, or it saves almost nothing."""

    monkeypatch.setenv(energy.ENERGY_ENV, "0")
    calls = 0

    def kernel():
        nonlocal calls
        calls += 1

    assert energy.Energy.perf(kernel, per_iter_time_ms=1.0) == 0.0
    assert calls == 0
