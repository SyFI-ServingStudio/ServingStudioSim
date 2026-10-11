"""`--remote` building blocks that need no remote: source identity, job scripts,
status parsing, ssh arguments, and the node-side job runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from launcher.remote import launch, node, source
from launcher.remote.config import Remote, Slurm, Ssh
from launcher.remote.transport import Transport


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    frontend = repo / source.REQ_FRONTEND
    frontend.mkdir(parents=True)
    _git(repo.parent, "init", "-q", repo.name)
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text("logs/\ntrace/big.csv\nCargo.lock\n")
    (repo / "Cargo.lock").write_text("# lock\n")
    (repo / "a.py").write_text("A = 1\n")
    (repo / "trace").mkdir()
    (repo / "trace" / "big.csv").write_text("x\n")
    (repo / "logs").mkdir()
    (repo / "logs" / "run.txt").write_text("out\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    _git(frontend, "init", "-q")
    (frontend / "Cargo.toml").write_text("[package]\n")
    return repo


def test_the_shipped_tree_has_code_req_frontend_and_trace_data_but_not_logs(repo):
    (repo / "new.py").write_text("B = 2\n")
    files = source.shipped_files(repo)
    assert "a.py" in files and "new.py" in files and ".gitignore" in files
    assert f"{source.REQ_FRONTEND}/Cargo.toml" in files
    assert "trace/big.csv" in files and "Cargo.lock" in files
    assert not any(name.startswith("logs/") for name in files)


def test_the_version_follows_content_and_commit(repo):
    files = source.shipped_files(repo)
    first = source.tree_version(repo, files, "h1")
    assert source.tree_version(repo, files, "h1") == first
    assert source.tree_version(repo, files, "h2") != first
    (repo / "a.py").write_text("A = 2\n")
    assert source.tree_version(repo, files, "h1") != first


def test_the_venv_key_ignores_code_and_follows_the_lock(repo):
    (repo / "uv.lock").write_text("v1\n")
    key = source.venv_key(repo, source.CPU_GROUPS)
    (repo / "a.py").write_text("A = 3\n")
    assert source.venv_key(repo, source.CPU_GROUPS) == key
    assert source.venv_key(repo, ("launcher",)) != key
    (repo / "uv.lock").write_text("v2\n")
    assert source.venv_key(repo, source.CPU_GROUPS) != key


def test_a_missing_req_frontend_checkout_is_named(repo):
    (repo / source.REQ_FRONTEND / "Cargo.toml").unlink()
    with pytest.raises(source.SourceError, match="git submodule update --init"):
        source.shipped_files(repo)


def _remote(**fields) -> Remote:
    defaults = dict(name="r", workdir=PurePosixPath("/w"), ssh=None, slurm=None, files="rsync")
    return Remote(**{**defaults, **fields})


def test_ssh_never_prompts_and_pins_host_keys_beside_remote_yaml(tmp_path):
    (tmp_path / ".remote").mkdir()
    (tmp_path / ".remote" / "known_hosts").write_text("")
    transport = Transport(
        _remote(ssh=Ssh("me@h", port=2222, identity_file="/k"), config_dir=tmp_path)
    )
    argv = transport.command_argv("echo 'hi there'")
    assert argv[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert f"UserKnownHostsFile={tmp_path}/.remote/known_hosts" in argv
    assert "StrictHostKeyChecking=yes" in argv
    assert argv[argv.index("-p") + 1] == "2222" and argv[argv.index("-i") + 1] == "/k"
    assert argv[-2:] == ["me@h", "bash -lc 'echo '\"'\"'hi there'\"'\"''"]
    assert transport.remote_spec("/w/x", directory=True) == "me@h:/w/x/"
    assert Transport(_remote()).command_argv("true") == ["bash", "-lc", "true"]


def test_the_sbatch_script_clears_the_managed_context_and_states_a_time_limit():
    script = launch.sbatch_script(
        Slurm(time="01:00:00", partition="cpu", cpus_per_task=8, extra=("--qos=low",)),
        name="vibesim-j",
        jobdir=PurePosixPath("/w/jobs/j"),
        cwd=PurePosixPath("/w/src/v"),
        command="python -m launcher.remote.node run-job /w/jobs/j",
    )
    assert "#SBATCH --time=01:00:00" in script
    assert "#SBATCH --partition=cpu" in script and "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --qos=low" in script and "#SBATCH --chdir=/w/src/v" in script
    unset = script.index("unset VIBESIM_MANAGED_JOB_CONTEXT VIBESIM_MANAGED_RUN_CONTEXT")
    assert unset < script.index("exec python -m launcher.remote.node")
    assert "--gres" not in script


def test_a_gpu_job_asks_slurm_for_its_cards():
    script = launch.sbatch_script(
        Slurm(time="01:00:00", gpus=2),
        name="vibesim-j",
        jobdir=PurePosixPath("/w/jobs/j"),
        cwd=PurePosixPath("/w/src/v"),
        command="true",
    )
    assert "#SBATCH --gres=gpu:2" in script


def test_status_reports_the_next_log_offset_in_bytes():
    running = launch.parse_status("alive=1\ndetail=RUNNING\nsize=10\n---\n0123456789", 0)
    assert (running.exit_code, running.lost, running.log_size, running.log_total) == (
        None,
        False,
        10,
        10,
    )
    assert running.log_chunk == "0123456789" and running.detail == "RUNNING"
    done = launch.parse_status("exit=3\nsize=10\n---\n", 10)
    assert (done.exit_code, done.lost, done.log_size) == (3, False, 10)
    lost = launch.parse_status("alive=0\nsize=0\n---\n", 0)
    assert lost.lost and lost.exit_code is None
    big = launch.parse_status(f"alive=1\nsize={launch.LOG_CHUNK_BYTES * 3}\n---\nx", 5)
    assert big.log_size == 5 + launch.LOG_CHUNK_BYTES


def test_a_direct_submit_returns_while_the_job_still_runs(tmp_path):
    import time

    jobdir = PurePosixPath(tmp_path / "jobs" / "j")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/bin/bash\nj="${@: -1}"; echo $$ > "$j/pid"; '
        "trap 'echo 143 > \"$j/exit_code\"; exit 143' TERM; sleep 30 & wait\n"
    )
    fake_python.chmod(0o755)
    transport = Transport(_remote())
    began = time.monotonic()
    submitted = launch.submit(
        transport,
        _remote(),
        job_id="j",
        jobdir=jobdir,
        cwd=PurePosixPath(tmp_path),
        remote_python=PurePosixPath(fake_python),
    )
    assert time.monotonic() - began < 10
    for _ in range(100):
        if (tmp_path / "jobs" / "j" / "pid").is_file():
            break
        time.sleep(0.05)
    assert launch.poll(transport, submitted, 0).exit_code is None
    launch.cancel(transport, submitted)
    for _ in range(100):
        status = launch.poll(transport, submitted, 0)
        if status.exit_code is not None:
            break
        time.sleep(0.05)
    assert status.exit_code == 143


def test_a_direct_job_runs_detached_and_is_polled_to_its_exit(tmp_path, monkeypatch):
    jobdir = PurePosixPath(tmp_path / "jobs" / "j")
    fake_python = tmp_path / "python"
    # Stands in for `python -m launcher.remote.node run-job JOBDIR`.
    fake_python.write_text(
        '#!/bin/bash\nj="${@: -1}"; echo $$ > "$j/pid"; echo hello > "$j/stdout.log"; '
        'echo 4 > "$j/exit_code"\n'
    )
    fake_python.chmod(0o755)
    transport = Transport(_remote())
    submitted = launch.submit(
        transport,
        _remote(),
        job_id="j",
        jobdir=jobdir,
        cwd=PurePosixPath(tmp_path),
        remote_python=PurePosixPath(fake_python),
    )
    assert not submitted.slurm and submitted.handle.isdigit()
    for _ in range(100):
        status = launch.poll(transport, submitted, 0)
        if status.exit_code is not None:
            break
        import time

        time.sleep(0.05)
    assert status.exit_code == 4 and status.log_chunk == "hello\n"


def test_run_job_isolates_the_database_and_drops_the_managed_context(tmp_path, monkeypatch):
    src = tmp_path / "src"
    (src / "profiling").mkdir(parents=True)
    (src / "profiling" / "profile.db").write_bytes(b"db")
    (src / ".venv" / "bin").mkdir(parents=True)
    # A fake interpreter that records the environment the job sees.
    (src / ".venv" / "bin" / "python").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "keys = ('TMPDIR', 'VIBESIM_BIN_DIR', 'VIBESIM_PROFILE_DB', "
        "'VIBESIM_MANAGED_JOB_CONTEXT', 'HF_HOME', 'UV_NO_SYNC')\n"
        "print(json.dumps({'argv': sys.argv[1:], **{k: os.environ.get(k) for k in keys}}))\n"
        "print(open(os.environ['VIBESIM_PROFILE_DB'], 'rb').read().decode())\n"
        "sys.exit(5)\n"
    )
    (src / ".venv" / "bin" / "python").chmod(0o755)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    job = {
        "job_id": "j",
        "src": str(src),
        "argv": ["p.yaml", "--no-gpu"],
        "env": {"HF_HOME": "/hf"},
        "node_local": str(tmp_path / "local"),
        "bin_dir": "/w/bin/h",
    }
    (jobdir / "job.json").write_text(json.dumps(job))
    monkeypatch.setenv("VIBESIM_MANAGED_JOB_CONTEXT", "/secret")

    assert node.run_job(jobdir) == 5

    first, second = (jobdir / "stdout.log").read_text().splitlines()
    seen = json.loads(first)
    scratch = tmp_path / "local" / "vibesim-job-j"
    assert seen["argv"] == ["-m", "launcher", "p.yaml", "--no-gpu"]
    assert seen["VIBESIM_PROFILE_DB"] == str(scratch / "profile.db") and second == "db"
    assert seen["TMPDIR"] == str(scratch / "tmp") and seen["VIBESIM_BIN_DIR"] == "/w/bin/h"
    assert seen["VIBESIM_MANAGED_JOB_CONTEXT"] is None and seen["HF_HOME"] == "/hf"
    assert seen["UV_NO_SYNC"] == "1"
    assert (jobdir / "exit_code").read_text() == "5\n"
    assert not scratch.exists()


def test_a_job_with_its_own_venv_runs_and_keeps_its_database(tmp_path):
    src = tmp_path / "src"
    (src / "profiling").mkdir(parents=True)
    (src / "profiling" / "profile.db").write_bytes(b"db")
    venv = tmp_path / "venvs" / "gpu"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text(
        f"#!{sys.executable}\n"
        "import json, os\n"
        "keys = ('VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT')\n"
        "print(json.dumps({k: os.environ.get(k) for k in keys}))\n"
        "open(os.environ['VIBESIM_PROFILE_DB'], 'ab').write(b'+row')\n"
    )
    (venv / "bin" / "python").chmod(0o755)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    job = {
        "job_id": "j",
        "src": str(src),
        "venv": str(venv),
        "argv": ["kernel-profile", "run"],
        "node_local": str(tmp_path / "local"),
        "bin_dir": "/w/bin/h",
        "keep_db": True,
    }
    (jobdir / "job.json").write_text(json.dumps(job))

    assert node.run_job(jobdir) == 0

    seen = json.loads((jobdir / "stdout.log").read_text())
    assert seen == {"VIRTUAL_ENV": str(venv), "UV_PROJECT_ENVIRONMENT": str(venv)}
    assert (jobdir / "profile.db").read_bytes() == b"db+row"


def test_prune_keeps_recent_and_busy_versions_and_their_binaries(tmp_path):
    import os
    import time

    for index, version in enumerate(("old", "busy", "mid", "new")):
        src = tmp_path / "src" / version
        src.mkdir(parents=True)
        (tmp_path / "bin" / version).mkdir(parents=True)
        (src / "bin").symlink_to(tmp_path / "bin" / version)
        marker = src / node.SHIPPED_MARKER
        marker.touch()
        os.utime(marker, (time.time() + index, time.time() + index))
    job = tmp_path / "jobs" / "j"
    job.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"version": "busy"}))

    node.prune(tmp_path, keep=2)

    assert sorted(p.name for p in (tmp_path / "src").iterdir()) == ["busy", "mid", "new"]
    assert sorted(p.name for p in (tmp_path / "bin").iterdir()) == ["busy", "mid", "new"]


def _copy_db_with_new_gemm_row(source_db: Path, target: Path, *, time_ms: float) -> None:
    import shutil
    import sqlite3

    shutil.copy2(source_db, target)
    with sqlite3.connect(target) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(single_gemm)")]
        row = dict(
            zip(columns, conn.execute("SELECT * FROM single_gemm LIMIT 1").fetchone(), strict=True)
        )
        row.pop("id")
        row["m"], row["args_hash"], row["time_ms"] = 999_983, "remote-merge-test", time_ms
        conn.execute(
            f"INSERT INTO single_gemm ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
            list(row.values()),
        )


def test_a_remote_jobs_rows_merge_into_the_local_database(tmp_path, monkeypatch):
    from launcher.remote import run

    repo_db = source.REPO_ROOT / "profiling" / "profile.db"
    local = tmp_path / "profile.db"
    import shutil

    shutil.copy2(repo_db, local)
    local.chmod(0o664)
    job_db = tmp_path / "job.db"
    _copy_db_with_new_gemm_row(repo_db, job_db, time_ms=1.0)
    monkeypatch.setenv("VIBESIM_PROFILE_DB", str(local))
    monkeypatch.setattr(run, "REPO_ROOT", tmp_path)

    message = run._merge_profile_db(job_db, tmp_path / "report.json")

    assert message.startswith("1 new row(s) merged")
    import sqlite3

    with sqlite3.connect(local) as conn:
        assert conn.execute("SELECT time_ms FROM single_gemm WHERE m = 999983").fetchall() == [
            (1.0,)
        ]
    assert local.stat().st_mode & 0o777 == 0o664

    # The same key with another value is a conflict: nothing changes here.
    before = local.read_bytes()
    conflicting = tmp_path / "conflict.db"
    _copy_db_with_new_gemm_row(repo_db, conflicting, time_ms=2.0)
    with pytest.raises(run.RemoteRunError, match="conflict"):
        run._merge_profile_db(conflicting, tmp_path / "report2.json")
    assert local.read_bytes() == before
