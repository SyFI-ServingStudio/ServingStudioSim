"""`profile_runtime: full-remote-profile`: each remote job is one profiler container."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import time
import types
from pathlib import Path, PurePosixPath

import pytest

from launcher.remote import launch
from launcher.remote.config import (
    FULL_REMOTE_PROFILE,
    Remote,
    RemoteConfigError,
    load_remote,
)

DOCUMENT = """\
remotes:
  gpu:
    access: {ssh: me@gpu}
    launch: {slurm: {time: "01:00:00", partition: main}}
    workdir: /data/vr
    profile_runtime: full-remote-profile
    mounts: [/data/hf, /data/corpus]
  direct:
    access: {ssh: me@gpu}
    workdir: /data/vr
    profile_runtime: full-remote-profile
    gpus: [2, 3]
  plain:
    access: {ssh: me@gpu}
    workdir: /data/vr
"""


@pytest.fixture
def remote_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "remote.yaml"
    path.write_text(DOCUMENT)
    return path


def test_a_full_remote_profile_remote_parses(remote_yaml):
    gpu = load_remote("gpu", path=remote_yaml)
    assert gpu.profile_runtime == FULL_REMOTE_PROFILE
    assert gpu.mounts == (PurePosixPath("/data/hf"), PurePosixPath("/data/corpus"))
    assert load_remote("direct", path=remote_yaml).gpus == ("2", "3")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("mounts=relative", "mounts\\[\\] must be an absolute path"),
        ("gpus=true", "gpus must be a GPU index"),
    ],
)
def test_bad_container_settings_are_named(remote_yaml, override, message):
    with pytest.raises(RemoteConfigError, match=message):
        load_remote("gpu", (override,), path=remote_yaml)


def test_container_settings_need_the_full_remote_profile_runtime(remote_yaml):
    with pytest.raises(RemoteConfigError, match="apply only to profile_runtime"):
        load_remote("plain", ("gpus=0",), path=remote_yaml)


def _container(tmp_path: Path, **fields) -> launch.Container:
    defaults = dict(
        image="img:t",
        python=PurePosixPath("/opt/v/.venv/bin/python"),
        workdir=PurePosixPath(tmp_path / "w"),
        node_local=PurePosixPath(tmp_path / "scratch"),
    )
    return launch.Container(**{**defaults, **fields})


def _fake_docker(tmp_path: Path, body: str) -> dict[str, str]:
    """A `docker` on PATH that logs each call's argv, one JSON list per line."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys, time\n"
        f"open({str(tmp_path / 'docker.calls')!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        + body
    )
    docker.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def _calls(tmp_path: Path) -> list[list[str]]:
    return [json.loads(line) for line in (tmp_path / "docker.calls").read_text().splitlines()]


def _script(tmp_path: Path, container: launch.Container) -> tuple[Path, Path]:
    jobdir = tmp_path / "w" / "jobs" / "J1"
    jobdir.mkdir(parents=True)
    script = jobdir / "container.sh"
    script.write_text(
        launch.container_script(
            container, job_id="J1", jobdir=PurePosixPath(jobdir), cwd=PurePosixPath(tmp_path)
        )
    )
    return jobdir, script


def test_the_container_mounts_paths_where_they_are_and_takes_slurms_gpus(tmp_path):
    # The container stands in for node.py: it records the job's exit code.
    env = _fake_docker(
        tmp_path,
        "if sys.argv[1] == 'run':\n"
        "    jobdir = sys.argv[-1]\n"
        "    open(os.path.join(jobdir, 'exit_code'), 'w').write('0\\n')\n",
    )
    container = _container(tmp_path, mounts=(PurePosixPath("/data/hf"),))
    jobdir, script = _script(tmp_path, container)
    done = subprocess.run(["bash", str(script)], env={**env, "SLURM_JOB_GPUS": "1,3"})
    assert done.returncode == 0
    (argv,) = _calls(tmp_path)
    assert argv[argv.index("--gpus") + 1] == '"device=1,3"'
    volumes = [argv[i + 1] for i, a in enumerate(argv) if a == "--volume"]
    assert volumes == [f"{tmp_path}/w:{tmp_path}/w", f"{tmp_path}/scratch:{tmp_path}/scratch",
                       "/data/hf:/data/hf"]
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert argv[argv.index("--entrypoint") + 1] == "/opt/v/.venv/bin/python"
    assert argv[-5:] == ["img:t", "-m", "launcher.remote.node", "run-job", str(jobdir)]
    assert (jobdir / "exit_code").read_text() == "0\n"
    assert (tmp_path / "w" / "cache" / "home").is_dir()


def test_a_job_with_no_gpus_fails_with_a_reason(tmp_path):
    env = _fake_docker(tmp_path, "")
    jobdir, script = _script(tmp_path, _container(tmp_path))
    clean = {k: v for k, v in env.items() if not k.startswith(("SLURM_", "CUDA_VISIBLE"))}
    assert subprocess.run(["bash", str(script)], env=clean).returncode == 2
    assert (jobdir / "exit_code").read_text() == "2\n"
    assert "no GPUs for this job" in (jobdir / "stdout.log").read_text()


def test_a_container_that_never_starts_records_dockers_status(tmp_path):
    env = _fake_docker(tmp_path, "sys.exit(125)\n")
    jobdir, script = _script(tmp_path, _container(tmp_path, gpus=("0",)))
    assert subprocess.run(["bash", str(script)], env=env).returncode == 125
    assert (jobdir / "exit_code").read_text() == "125\n"


def test_stopping_the_job_stops_its_container(tmp_path):
    # `run` waits until `kill` names its container, as a real container would.
    env = _fake_docker(
        tmp_path,
        "if sys.argv[1] == 'run':\n"
        f"    while 'kill' not in open({str(tmp_path / 'docker.calls')!r}).read():\n"
        "        time.sleep(0.05)\n"
        "    sys.exit(143)\n",
    )
    jobdir, script = _script(tmp_path, _container(tmp_path, gpus=("0",)))
    job = subprocess.Popen(["bash", str(script)], env=env)
    deadline = time.monotonic() + 10
    while not (tmp_path / "docker.calls").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    job.send_signal(signal.SIGTERM)
    assert job.wait(timeout=10) == 143
    assert ["kill", "--signal", "TERM", "vibesim-J1"] in _calls(tmp_path)
    assert (jobdir / "exit_code").read_text() == "143\n"


def test_a_container_jobs_liveness_and_stop_use_the_host_pid():
    submitted = launch.Submitted(PurePosixPath("/w/jobs/J"), "77", slurm=False, container=True)
    assert "cat /w/jobs/J/pid " not in launch.status_script(submitted, 0)
    plain = dataclasses.replace(submitted, container=False)
    assert "cat /w/jobs/J/pid " in launch.status_script(plain, 0)


def _remote(tmp_path: Path, **fields) -> Remote:
    defaults = dict(
        name="gpu",
        workdir=PurePosixPath(tmp_path / "w"),
        ssh=None,
        slurm=None,
        files="rsync",
        profile_runtime=FULL_REMOTE_PROFILE,
    )
    return Remote(**{**defaults, **fields})


def test_preparing_a_container_job_reads_the_images_manifest(tmp_path, monkeypatch):
    from launcher.remote.run import RemoteRunError, Session

    manifest = {"inputs": "abc", "envs": ["vllm_env"], "python": "/opt/v/.venv/bin/python"}
    env = _fake_docker(
        tmp_path,
        "if sys.argv[1] == 'run':\n"
        "    print('--- Wrapper: Applying owner=me ---')\n"
        f"    print({json.dumps(manifest)!r})\n"
        "elif sys.argv[1:3] == ['image', 'inspect']:\n"
        "    print('sha256:1')\n",
    )
    monkeypatch.setenv("PATH", env["PATH"])
    session = Session(_remote(tmp_path, mounts=(PurePosixPath("/data/hf"),), gpus=("2",)))
    session.source = types.SimpleNamespace(version="v1")
    (tmp_path / "w" / "src" / "v1").mkdir(parents=True)

    with pytest.raises(RemoteRunError, match="serves \\['vllm_env'\\]"):
        session.prepare_full_remote_profile("default_env", "vibesim-profiler-vllm:abc")
    assert session.container is None

    assert session.prepare_full_remote_profile("vllm_env", "vibesim-profiler-vllm:abc") == "sha256:1"
    assert session.container == launch.Container(
        image="vibesim-profiler-vllm:abc",
        python=PurePosixPath("/opt/v/.venv/bin/python"),
        workdir=PurePosixPath(tmp_path / "w"),
        node_local=PurePosixPath("/tmp"),
        mounts=(PurePosixPath("/data/hf"),),
        gpus=("2",),
    )
    assert (tmp_path / "w" / "src" / "v1" / ".venv").readlink() == Path("/opt/v/.venv")


def test_a_remote_without_docker_access_is_refused_by_name(tmp_path, monkeypatch):
    from launcher.remote.run import RemoteRunError, Session

    env = _fake_docker(tmp_path, "sys.exit(1)\n")
    monkeypatch.setenv("PATH", env["PATH"])
    session = Session(_remote(tmp_path))
    with pytest.raises(RemoteRunError, match="missing docker-access"):
        session.check_container_host()


def test_a_full_remote_profile_remote_runs_gpu_work_only(tmp_path, monkeypatch, capsys):
    from launcher.cli import GlobalOptions
    from launcher.remote import run as remote_run

    monkeypatch.setattr(remote_run, "_require_repo_cwd", lambda: None)
    monkeypatch.setattr(remote_run, "load_remote", lambda name, sets: _remote(tmp_path))
    assert remote_run.run(None, ["presets/x.yaml"], GlobalOptions(remote="gpu")) == 2
    assert "run `simulation` here" in capsys.readouterr().err
    assert remote_run.run("timing-predict", ["p.json"], GlobalOptions(remote="gpu")) == 2


@pytest.mark.parametrize(
    ("runtime", "argv", "message"),
    [
        (FULL_REMOTE_PROFILE, ["label", "--pack", "p", "--out-root", "logs/x"], "label runs here"),
        ("docker", ["run", "--pack", "p", "--out-root", "logs/x", "--phase", "profile_nsys"],
         "needs profile_runtime: full-remote-profile"),
        (FULL_REMOTE_PROFILE,
         ["run", "--pack", "p", "--out-root", "logs/x", "--phase", "profile_nsys", "--dry-run"],
         "--dry-run stays local"),
    ],
)
def test_only_a_campaigns_capture_goes_to_a_remote(tmp_path, runtime, argv, message):
    from launcher.remote.run import RemoteRunError, Session, _alignment_campaign

    session = Session(_remote(tmp_path, profile_runtime=runtime))
    with pytest.raises(RemoteRunError, match=message):
        _alignment_campaign(session, argv)


def test_a_campaigns_analysis_phase_is_refused_remotely(tmp_path):
    from launcher.alignment_campaign.cli import PACK_ROOT
    from launcher.remote.run import RemoteRunError, Session, _alignment_campaign

    pack = next(path for path in sorted(PACK_ROOT.iterdir()) if (path / "campaign.yaml").is_file())
    session = Session(_remote(tmp_path))
    with pytest.raises(RemoteRunError, match="not a profile pass"):
        _alignment_campaign(
            session,
            ["run", "--pack", pack.name, "--out-root", "logs/x", "--phase", "timing_predict"],
        )
