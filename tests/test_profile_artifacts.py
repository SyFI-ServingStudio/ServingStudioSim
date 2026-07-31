from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from profiling import cli
from profiling.artifacts import (
    PROFILE_CURVE_FILENAME,
    PROFILE_JOB_METADATA_FILENAME,
    PROFILE_REQUEST_FILENAME,
    PROFILE_RESULTS_FILENAME,
    build_curve_payload,
)
from profiling.db.registry import iter_kernel_profiler_specs
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
    monkeypatch.setattr(
        cli,
        "_get_facade",
        lambda _table: (
            lambda specs, **_kwargs: [
                ComputeMetrics(
                    time_ms=float(index + 1),
                    tflops=1.0,
                    memory_bandwidth_gbps=2.0,
                    energy_j=0.1,
                )
                for index, _spec in enumerate(specs)
            ]
        ),
    )
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
        force=True,
        output_dir=output_dir,
        json=True,
    )

    assert cli._cmd_run(args) == 0

    assert {path.name for path in output_dir.iterdir()} == {
        PROFILE_REQUEST_FILENAME,
        PROFILE_RESULTS_FILENAME,
        PROFILE_CURVE_FILENAME,
        PROFILE_JOB_METADATA_FILENAME,
    }
    curve = json.loads((output_dir / PROFILE_CURVE_FILENAME).read_text())
    assert [axis["key"] for axis in curve["axes"]] == ["m"]
    assert curve["rows"][1]["metrics"]["time_ms"] == 2.0


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
