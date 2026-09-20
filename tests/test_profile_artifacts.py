from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from launcher.artifact_kind import ARTIFACT_METADATA_FILENAME
from profiling import cli
from profiling.artifacts import (
    PROFILE_CURVE_FILENAME,
    PROFILE_JOB_METADATA_FILENAME,
    PROFILE_METADATA_FILENAME,
    PROFILE_REQUEST_FILENAME,
    PROFILE_RESULTS_FILENAME,
    build_curve_payload,
)
from profiling.db.batch import ProfileProvenance
from profiling.db.registry import iter_kernel_profiler_specs
from profiling.facade import KindTimesResult
from profiling.runners.metrics import ComputeMetrics


def _single_gemm_torch_spec():
    return next(
        profiler_spec
        for profiler_spec in iter_kernel_profiler_specs()
        if profiler_spec.table_name == "single_gemm" and profiler_spec.backend == "torch"
    )


def test_curve_axes_follow_kernel_args_declaration_order() -> None:
    profiler_spec = _single_gemm_torch_spec()
    specs = [
        {"m": 1, "n": 128, "k": 64, "dtype": "bfloat16"},
        {"m": 2, "n": 128, "k": 64, "dtype": "bfloat16"},
        {"m": 1, "n": 256, "k": 64, "dtype": "bfloat16"},
        {"m": 2, "n": 256, "k": 64, "dtype": "bfloat16"},
    ]
    result_payload = {
        "results": [
            {
                "index": index,
                "status": "ok",
                "metrics": {
                    "time_ms": float(index + 1),
                    "tflops": 1.0,
                    "memory_bandwidth_gbps": 2.0,
                    "energy_j": 0.1,
                },
            }
            for index in range(len(specs))
        ]
    }

    curve = build_curve_payload(profiler_spec, specs, result_payload)

    assert [axis["key"] for axis in curve["axes"]] == ["m", "n"]
    assert curve["layout"] == {"xAxis": "m", "yAxis": "n", "facets": []}
    assert curve["fixedArgs"] == {"k": 64, "dtype": "bfloat16"}
    assert curve["series"][0] == {
        "metric": "time_ms",
        "unit": "ms",
        "lowerIsBetter": True,
    }


def test_profile_run_writes_immutable_snapshot(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VIBESIM_MANAGED_JOB_CONTEXT", raising=False)
    monkeypatch.delenv("VIBESIM_MANAGED_RUN_CONTEXT", raising=False)
    monkeypatch.setattr(cli, "_set_db_path", lambda _path: None)

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
        del kernel_kind, backend, gpu_name, db_path, jit_enabled, force, persist
        return KindTimesResult(
            results=[
                ComputeMetrics(
                    time_ms=float(index + 1),
                    tflops=1.0,
                    memory_bandwidth_gbps=2.0,
                    energy_j=0.1,
                )
                for index, _spec in enumerate(specs)
            ],
            provenance=ProfileProvenance(source="cache_key", requested_gpu_name="NVIDIA H200"),
        )

    monkeypatch.setattr(cli, "run_kind_times", fake_run_kind_times)
    monkeypatch.setattr(cli, "_count_facade", lambda _table: lambda *_args, **_kwargs: 0)
    output_dir = tmp_path / "profile-artifact"
    args = SimpleNamespace(
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
        fresh=False,
        energy=True,
        output_dir=output_dir,
        json=True,
    )

    assert cli._cmd_run(args) == 0

    assert {path.name for path in output_dir.iterdir()} == {
        ARTIFACT_METADATA_FILENAME,
        PROFILE_REQUEST_FILENAME,
        PROFILE_RESULTS_FILENAME,
        PROFILE_CURVE_FILENAME,
        PROFILE_JOB_METADATA_FILENAME,
        PROFILE_METADATA_FILENAME,
    }
    curve = json.loads((output_dir / PROFILE_CURVE_FILENAME).read_text())
    assert [axis["key"] for axis in curve["axes"]] == ["m"]
    assert curve["rows"][1]["metrics"]["time_ms"] == 2.0
    metadata = json.loads((output_dir / PROFILE_METADATA_FILENAME).read_text())
    assert metadata["profile_id"].startswith("kp_")
    assert metadata["kernel"] == {
        "kind": "single_gemm",
        "table": "single_gemm",
        "backend": "torch",
        "metric_family": "compute",
    }
    assert metadata["gpu"] == {"cache_key": "NVIDIA H200", "observed_name": None, "count": 1}
    assert metadata["provenance"]["source"] == "cache_key"
    assert metadata["mode"] == "jit-fill"
    assert metadata["args"] == [
        {"m": 1, "n": 128, "k": 64, "dtype": "bfloat16"},
        {"m": 2, "n": 128, "k": 64, "dtype": "bfloat16"},
    ]


def test_launcher_kernel_profile_dispatch_uses_shared_cli(monkeypatch) -> None:
    from launcher import __main__ as launcher_main

    captured = {}

    def fake_profile_main(argv, *, prog):
        captured["argv"] = argv
        captured["prog"] = prog
        return 7

    monkeypatch.setattr(cli, "main", fake_profile_main)

    assert launcher_main.main(["kernel-profile", "list", "--json"]) == 7
    assert captured == {
        "argv": ["list", "--json"],
        "prog": "python -m launcher kernel-profile",
    }
