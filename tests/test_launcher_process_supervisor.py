"""Fault-oriented tests for the launcher's root-PID completion contract."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from launcher.process import ProcessSpec, ProcessSupervisor
from launcher.process.artifacts import (
    ArtifactValidationError,
    validate_simulation_artifacts,
)
from launcher.process.journal import RunJournal, StageState
from launcher.workflow import StageKind, simulation_workflow


def _spec(arguments: list[str], working_directory: Path, **overrides) -> ProcessSpec:
    return ProcessSpec(
        argv=[sys.executable, "-c", *arguments],
        cwd=working_directory,
        name="test-process",
        **overrides,
    )


def test_capture_uses_root_process_completion(tmp_path: Path) -> None:
    result = asyncio.run(
        ProcessSupervisor().run(
            _spec(
                ["import sys; print('stdout'); print('stderr', file=sys.stderr)"],
                tmp_path,
                capture_output=True,
            )
        )
    )

    assert result.succeeded
    assert "stdout" in result.output
    assert "stderr" in result.output


def test_root_exit_with_inherited_output_descriptor_cleans_descendant(tmp_path: Path) -> None:
    started = time.monotonic()
    code = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "print('root exited')"
    )

    result = asyncio.run(
        ProcessSupervisor().run(_spec([code], tmp_path, log_path=tmp_path / "stage.log"))
    )

    assert time.monotonic() - started < 5
    assert result.exit_code == 0
    assert result.leaked_descendants
    assert not result.succeeded
    assert "root exited" in (tmp_path / "stage.log").read_text()


def test_large_output_cannot_fill_a_launcher_pipe(tmp_path: Path) -> None:
    result = asyncio.run(
        ProcessSupervisor().run(
            _spec(
                ["import sys; sys.stdout.write('x' * 2_000_000)"],
                tmp_path,
                log_path=tmp_path / "large.log",
            )
        )
    )

    assert result.succeeded
    assert (tmp_path / "large.log").stat().st_size == 2_000_000


def test_joined_process_pool_is_not_reported_as_a_leak(tmp_path: Path) -> None:
    code = (
        "from multiprocessing import Pool; "
        "pool = Pool(2); print(pool.map(abs, [-1, -2])); pool.close(); pool.join()"
    )

    result = asyncio.run(ProcessSupervisor().run(_spec([code], tmp_path, capture_output=True)))

    assert result.succeeded
    assert not result.leaked_descendants
    assert "[1, 2]" in result.output


def test_cancellation_terminates_the_process_group(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )

    async def run_and_cancel() -> int:
        task = asyncio.create_task(
            ProcessSupervisor().run(_spec([code], tmp_path, log_path=tmp_path / "cancel.log"))
        )
        deadline = time.monotonic() + 5
        while not child_pid_path.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert child_pid_path.exists()
        child_pid = int(child_pid_path.read_text())
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return child_pid

    child_pid = asyncio.run(run_and_cancel())
    deadline = time.monotonic() + 2
    while _pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _pid_exists(child_pid)


def test_sync_adapter_uses_same_process_group_contract(tmp_path: Path) -> None:
    result = ProcessSupervisor().run_sync(_spec(["print('sync')"], tmp_path, capture_output=True))

    assert result.succeeded
    assert result.output.strip() == "sync"


def test_repeated_short_processes_do_not_leak_fds_or_threads(tmp_path: Path) -> None:
    descriptor_count_before = len(list(Path("/proc/self/fd").iterdir()))
    thread_count_before = threading.active_count()
    supervisor = ProcessSupervisor()

    for _iteration in range(30):
        result = supervisor.run_sync(_spec(["print('short')"], tmp_path, capture_output=True))
        assert result.succeeded

    descriptor_count_after = len(list(Path("/proc/self/fd").iterdir()))
    assert descriptor_count_after <= descriptor_count_before + 1
    assert threading.active_count() == thread_count_before


def test_cross_process_exclusive_lease_singleflights_one_build(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache.lock"
    cache_state = tmp_path / "cache.ready"
    build_log = tmp_path / "build.log"
    code = "\n".join(
        [
            "import asyncio, pathlib, sys",
            "from launcher.process.leases import ResourceLease",
            "lock_path, cache_state, build_log = map(pathlib.Path, sys.argv[1:])",
            "async def main():",
            "    async with ResourceLease(lock_path, 'test-cache', 'exclusive'):",
            "        if not cache_state.exists():",
            "            with build_log.open('a') as stream: stream.write('build\\n')",
            "            await asyncio.sleep(0.25)",
            "            cache_state.write_text('ready')",
            "asyncio.run(main())",
        ]
    )
    arguments = [code, str(lock_path), str(cache_state), str(build_log)]
    child_environment = os.environ.copy()
    repository_root = Path(__file__).resolve().parents[1]
    child_environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repository_root), child_environment.get("PYTHONPATH", "")])
    )

    async def run_both():
        return await asyncio.gather(
            ProcessSupervisor().run(
                _spec(
                    arguments,
                    tmp_path,
                    capture_output=True,
                    env=child_environment,
                )
            ),
            ProcessSupervisor().run(
                _spec(
                    arguments,
                    tmp_path,
                    capture_output=True,
                    env=child_environment,
                )
            ),
        )

    results = asyncio.run(run_both())

    assert all(result.succeeded for result in results)
    assert build_log.read_text().splitlines() == ["build"]


def test_run_journal_records_process_identity_atomically(tmp_path: Path) -> None:
    journal = RunJournal(tmp_path)
    spec = _spec(["print('done')"], tmp_path, capture_output=True)
    result = ProcessSupervisor().run_sync(spec)

    journal.update("simulate", StageState.SUCCEEDED, spec=spec, result=result)

    stage = json.loads((tmp_path / ".launcher/stages/simulate.json").read_text())
    run_state = json.loads((tmp_path / ".launcher/run_state.json").read_text())
    assert stage["state"] == "SUCCEEDED"
    assert stage["child_pid"] == result.pid
    assert stage["process_group_id"] == result.process_group_id
    assert run_state["stages"]["simulate"]["state"] == "SUCCEEDED"


def test_run_journal_increments_attempt_on_refresh(tmp_path: Path) -> None:
    journal = RunJournal(tmp_path)
    journal.begin_attempts(["simulate"])
    journal.update("simulate", StageState.SUCCEEDED)
    journal.begin_attempts(["simulate"])

    stage = json.loads((tmp_path / ".launcher/stages/simulate.json").read_text())
    assert stage["attempt"] == 2
    assert stage["state"] == "PENDING"
    assert stage["started_at"] is None
    assert stage["exit_code"] is None


def test_simulation_artifact_contract_accepts_zero_request_run(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "cause": "DrainComplete",
                "requests_total": 0,
                "requests_finished": 0,
                "total_tokens": 0,
                "sim_ms": 0.0,
            }
        )
    )
    (tmp_path / "raw/run_meta.json").write_text(json.dumps({"schema_version": 1, "workers": []}))

    validation = validate_simulation_artifacts(tmp_path)

    assert set(validation.paths) == {
        tmp_path / "summary.json",
        tmp_path / "raw/run_meta.json",
    }


def test_simulation_artifact_contract_rejects_truncated_json(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text("{")

    try:
        validate_simulation_artifacts(tmp_path)
    except ArtifactValidationError as error:
        assert "invalid JSON artifact" in str(error)
    else:
        raise AssertionError("truncated summary must fail validation")


def test_no_analyze_workflow_removes_optional_stages() -> None:
    without_analysis = simulation_workflow(analyze=False)
    with_analysis = simulation_workflow(analyze=True)

    assert not without_analysis.contains(StageKind.ANALYZE_COMPUTE)
    assert not without_analysis.contains(StageKind.RENDER)
    assert not without_analysis.contains(StageKind.TRACE)
    assert with_analysis.contains(StageKind.ANALYZE_COMPUTE)
    assert with_analysis.contains(StageKind.RENDER)
    assert with_analysis.contains(StageKind.TRACE)


def _pid_exists(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize("synchronous", [True, False])
def test_exited_unreaped_group_is_not_a_live_descendant(synchronous: bool) -> None:
    # Keep a real zombie until assertions finish, without leaving it to PID 1.
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    child = process.pid
    try:
        os.waitid(os.P_PID, child, os.WEXITED | os.WNOWAIT)
        os.killpg(child, 0)  # The old existence-only check reports this group.
        supervisor = ProcessSupervisor()
        if synchronous:
            leaked = supervisor._clean_leaked_descendants_sync(child)
        else:
            leaked = asyncio.run(supervisor._clean_leaked_descendants(child))
        assert not leaked
    finally:
        process.wait()


def test_sync_root_exit_with_live_descendant_still_fails(tmp_path: Path) -> None:
    result = ProcessSupervisor().run_sync(
        _spec(
            [
                "import subprocess, sys; subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(30)'])"
            ],
            tmp_path,
            capture_output=True,
        )
    )
    assert result.exit_code == 0
    assert result.leaked_descendants
    assert not result.succeeded


@pytest.mark.parametrize("stage", ["simulator compilation", "schema discovery"])
def test_build_error_identifies_stage_and_cleanup_failure(stage, capsys) -> None:
    from launcher.exec import _report_build_failure
    from launcher.process import ProcessResult

    result = ProcessResult(
        argv=("tool",),
        pid=123,
        process_group_id=123,
        exit_code=0,
        elapsed_seconds=1,
        leaked_descendants=True,
        output="tool diagnostic",
    )
    _report_build_failure(stage, result)
    error = capsys.readouterr().err
    assert stage in error
    assert "live descendants" in error
    assert "exit_code=0" in error
    assert "process_group_id=123" in error
    assert "tool diagnostic" in error
