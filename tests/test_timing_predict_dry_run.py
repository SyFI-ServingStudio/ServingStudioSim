"""`python -m launcher timing-predict --dry-run`: validate, report, write nothing."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from launcher import timing_predict


@pytest.fixture
def launcher(monkeypatch):
    calls = SimpleNamespace(builds=[], processes=[], jobs=[], exit_code=0)
    monkeypatch.setattr(
        timing_predict,
        "cargo_build",
        lambda build_type, *, build_analyzer: calls.builds.append(build_analyzer) or True,
    )

    def run_sync(spec):
        calls.processes.append(spec)
        return SimpleNamespace(
            output="timing-predict dry run: 1 case(s) valid\n",
            succeeded=calls.exit_code == 0,
        )

    monkeypatch.setattr(timing_predict._PROCESS_SUPERVISOR, "run_sync", run_sync)
    monkeypatch.setattr(
        timing_predict, "prepare_managed_job", lambda *args, **kwargs: calls.jobs.append(args)
    )
    return calls


def _config(tmp_path, cases=({"groups": []},)):
    (tmp_path / "cases.json").write_text(json.dumps(list(cases)))
    config = tmp_path / "predict.json"
    config.write_text(
        json.dumps(
            {
                "arch": {"iter": {"type": "llama3"}},
                "gpu": "NVIDIA H200",
                "log_dir": str(tmp_path / "out"),
                "cases_file": "cases.json",
            }
        )
    )
    return config


def test_dry_run_asks_the_binary_to_validate_and_writes_nothing(tmp_path, launcher, capsys):
    config = _config(tmp_path)

    assert timing_predict.main(["--dry-run", str(config)]) == 0

    [spec] = launcher.processes
    assert spec.argv[1:] == ["timing-predict", "--dry-run", str(config)]
    assert spec.env["RUST_LOG"] == "warn"
    assert launcher.builds == [False]
    assert launcher.jobs == []
    assert not (tmp_path / "out").exists()
    assert "1 case(s) valid" in capsys.readouterr().out


def test_a_user_log_level_wins_over_the_dry_run_default(tmp_path, launcher, monkeypatch):
    monkeypatch.setenv("RUST_LOG", "debug")

    assert timing_predict.main(["--dry-run", str(_config(tmp_path))]) == 0

    assert launcher.processes[0].env["RUST_LOG"] == "debug"


def test_a_cases_file_that_is_not_a_list_is_invalid_before_the_binary(tmp_path, launcher, capsys):
    config = _config(tmp_path)
    (tmp_path / "cases.json").write_text("{}")

    assert timing_predict.main(["--dry-run", str(config)]) == 2

    assert launcher.processes == []
    assert "must contain a list" in capsys.readouterr().err


def test_a_rejected_case_fails_the_dry_run(tmp_path, launcher):
    launcher.exit_code = 1

    assert timing_predict.main(["--dry-run", str(_config(tmp_path))]) == 1
