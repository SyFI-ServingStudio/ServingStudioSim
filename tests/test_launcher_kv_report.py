"""`python -m launcher <preset> --kv-report[-json]`: one KV report per distinct
candidate, structure only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from launcher import kv_report
from launcher.__main__ import parse_simulation_args

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(log_dir: str, *, rate: float = 1.0, arch_type: str = "llama3_dense") -> dict:
    return {
        "deployment": "unified",
        "workload": {"request_rate": rate},
        "io": {"log_dir": log_dir},
        "pools": {"main": {"groups": [{"arch": {"type": arch_type}, "worker": {}}]}},
    }


@pytest.fixture
def binary(monkeypatch):
    """The `kv-cache` binary, answering from `outputs` keyed by arch type."""
    calls = SimpleNamespace(argv=[], outputs={}, fail=set())

    def run_sync(spec):
        calls.argv.append(list(spec.argv))
        config = yaml.safe_load(Path(spec.argv[2]).read_text())
        arch = config["pools"]["main"]["groups"][0]["arch"]["type"]
        failed = arch in calls.fail
        return SimpleNamespace(
            output="" if failed else calls.outputs[arch],
            stderr="boom\n" if failed else "",
            exit_code=1 if failed else 0,
            succeeded=not failed,
        )

    monkeypatch.setattr(kv_report._PROCESS_SUPERVISOR, "run_sync", run_sync)
    monkeypatch.setattr(kv_report, "_build_subprocess_env", lambda: {})
    return calls


def test_workload_and_io_never_split_a_report():
    groups = kv_report.group_by_kv_report(
        [
            _config("logs/a", rate=1.0),
            _config("logs/b", rate=2.0),
            _config("logs/c", arch_type="qwen36_local"),
        ]
    )
    assert [[c["io"]["log_dir"] for c in g] for g in groups] == [
        ["logs/a", "logs/b"],
        ["logs/c"],
    ]


def test_text_runs_the_binary_once_per_distinct_config(binary, capsys):
    binary.outputs = {"llama3_dense": "KV cache · pool main\n"}

    configs = [_config("logs/a", rate=1.0), _config("logs/b", rate=2.0)]
    assert kv_report.report_kv_cache(configs, "debug") == 0

    assert len(binary.argv) == 1
    assert binary.argv[0][1] == "kv-cache" and "--json" not in binary.argv[0]
    out = capsys.readouterr().out
    assert "== 2 runs share this report: logs/a, logs/b\nKV cache · pool main\n" in out


def test_identical_reports_from_different_configs_print_once(binary, capsys):
    report = json.dumps({"schema_version": 1, "pools": []})
    binary.outputs = {"llama3_dense": report, "llama3_dense_tp": report}

    configs = [_config("logs/a"), _config("logs/b", arch_type="llama3_dense_tp")]
    assert kv_report.report_kv_cache(configs, "debug", as_json=True) == 0

    assert all("--json" in argv for argv in binary.argv) and len(binary.argv) == 2
    document = json.loads(capsys.readouterr().out)
    assert document == {
        "schema_version": kv_report.SCHEMA_VERSION,
        "reports": [{"log_dirs": ["logs/a", "logs/b"], "kv_cache": json.loads(report)}],
    }


def test_a_failed_report_sets_the_exit_code_and_keeps_the_rest(binary, capsys):
    binary.outputs = {"llama3_dense": "ok\n"}
    binary.fail = {"qwen36_local"}

    configs = [_config("logs/a"), _config("logs/b", arch_type="qwen36_local")]
    assert kv_report.report_kv_cache(configs, "debug") == 1

    captured = capsys.readouterr()
    assert "== logs/a\nok\n" in captured.out
    assert "kv-report failed for logs/b:\nboom" in captured.err


def test_flags_parse():
    args = parse_simulation_args(["p.yaml", "--kv-report-json"])
    assert args.kv_report_json and not args.kv_report


@pytest.mark.needs_binary
def test_a_sweep_reports_once_without_profile_db(tmp_path, monkeypatch, capsys):
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    preset = yaml.safe_load((REPO_ROOT / "presets/glm52_nvfp4_b200_ep4.yaml").read_text())
    preset["workload"]["request_rate"] = "${rate}"
    preset["io"]["log_dir"] = str(tmp_path / "rate{rate}")
    preset["sweep"] = {"rate": [1.0, 2.0]}
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(yaml.safe_dump(preset))
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setenv("VIBESIM_PROFILE_DB", str(tmp_path / "absent.db"))

    argv = [str(preset_path), "--kv-report-json", "--build-type", "release"]
    assert main_module.main(argv) == 0

    (report,) = json.loads(capsys.readouterr().out)["reports"]
    assert len(report["log_dirs"]) == 2
    (pool,) = report["kv_cache"]["pools"]
    assert pool["result"]["max_concurrency_at_max_model_len"] > 1
    assert not any(Path(d).exists() for d in report["log_dirs"])
    assert not (tmp_path / "absent.db").exists()
