"""GPU jobs on a remote: one profiler container per job (`docker`), or the pod itself (`in-image`)."""

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
from launcher.remote.config import Remote, RemoteConfigError, load_remote

DOCUMENT = """\
remotes:
  gpu:
    access: {ssh: me@gpu}
    launch: {slurm: {time: "01:00:00", partition: main}}
    workdir: /data/vr
    mounts: [/data/hf, /data/corpus]
  direct:
    access: {ssh: me@gpu}
    workdir: /data/vr
    gpus: [2, 3]
  pod:
    access: {ssh: me@pod}
    workdir: /workspace/vr
    profile_runtime: in-image
"""


@pytest.fixture
def remote_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "remote.yaml"
    path.write_text(DOCUMENT)
    return path


def test_a_docker_remote_takes_container_settings(remote_yaml):
    gpu = load_remote("gpu", path=remote_yaml)
    assert gpu.profile_runtime == "docker"
    assert gpu.mounts == (PurePosixPath("/data/hf"), PurePosixPath("/data/corpus"))
    assert load_remote("direct", path=remote_yaml).gpus == ("2", "3")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("mounts=relative", "mounts\\[\\] must be an absolute path"),
        ("gpus=true", "gpus must be a GPU index"),
        ("profile_runtime=full-remote-profile", "profile_runtime must be one of docker, in-image"),
    ],
)
def test_bad_container_settings_are_named(remote_yaml, override, message):
    with pytest.raises(RemoteConfigError, match=message):
        load_remote("gpu", (override,), path=remote_yaml)


def test_container_settings_need_the_docker_runtime(remote_yaml):
    with pytest.raises(RemoteConfigError, match="apply only to profile_runtime: docker"):
        load_remote("pod", ("gpus=0",), path=remote_yaml)


def _container(tmp_path: Path, **fields) -> launch.Container:
    defaults = dict(
        image="img:t",
        python=PurePosixPath("/usr/bin/python3"),
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
    _fake_nvidia_smi(tmp_path)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def _fake_nvidia_smi(
    tmp_path: Path,
    free_mib: list[int] = (80000,),
    *,
    caps: dict[str, str] | None = None,
    listing: list[str] = (),
    busy: str = "",
) -> None:
    """An `nvidia-smi` for the container script's queries.

    A card is named `GPU-<index>` by UUID. `free_mib` gives the next
    free-memory reading on each call (the last repeats); `caps` a card's
    compute capability (9.0 by default); `listing` the rows of the idle-card
    query, and `busy` the UUIDs running a compute process.
    """

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    state = tmp_path / "smi.state"
    state.write_text("0")
    smi = bin_dir / "nvidia-smi"
    smi.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"readings, caps, listing, busy = {list(free_mib)!r}, {caps or {}!r}, {list(listing)!r}, {busy!r}\n"
        f"state = {str(state)!r}\n"
        "args = sys.argv[1:]\n"
        "cards = args[args.index('-i') + 1].split(',') if '-i' in args else []\n"
        "uuid = lambda c: c if c.startswith('GPU-') else 'GPU-' + c\n"
        "if '--query-compute-apps=pid,used_memory' in args:\n"
        "    print('4242, 30000 MiB')\n"
        "elif '--query-compute-apps=gpu_uuid' in args:\n"
        "    print(busy)\n"
        "elif '--query-gpu=uuid' in args:\n"
        "    print('\\n'.join(uuid(c) for c in cards))\n"
        "elif '--query-gpu=compute_cap' in args:\n"
        "    print('\\n'.join(caps.get(uuid(c), '9.0') for c in cards))\n"
        "elif '--query-gpu=memory.free' in args:\n"
        "    i = int(open(state).read())\n"
        "    print(readings[min(i, len(readings) - 1)])\n"
        "    open(state, 'w').write(str(i + 1))\n"
        "else:\n"
        "    print('\\n'.join(listing))\n"
    )
    smi.chmod(0o755)


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


def _without_gpu_context(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if not k.startswith(("SLURM_", "CUDA_VISIBLE"))}


_RECORD_EXIT = (
    "if sys.argv[1] == 'run':\n"
    "    open(os.path.join(sys.argv[-1], 'exit_code'), 'w').write('0\\n')\n"
)


def test_the_container_names_slurms_gpus_by_uuid_and_keys_its_cache(tmp_path):
    # The container stands in for node.py: it records the job's exit code.
    env = _fake_docker(tmp_path, _RECORD_EXIT)
    container = _container(
        tmp_path, mounts=(PurePosixPath("/data/hf"),), cache_label="vllm-0123456789ab"
    )
    jobdir, script = _script(tmp_path, container)
    job_env = {**_without_gpu_context(env), "CUDA_VISIBLE_DEVICES": "0,1", "SLURM_JOB_GPUS": "1,3"}
    assert subprocess.run(["bash", str(script)], env=job_env).returncode == 0
    (argv,) = _calls(tmp_path)
    # Slurm's CUDA_VISIBLE_DEVICES is in nvidia-smi's numbering; docker gets UUIDs.
    assert argv[argv.index("--gpus") + 1] == '"device=GPU-0,GPU-1"'
    assert "CUDA_VISIBLE_DEVICES=GPU-0,GPU-1" in argv
    cache = f"{tmp_path}/w/cache/vllm-0123456789ab-sm90"
    volumes = [argv[i + 1] for i, a in enumerate(argv) if a == "--volume"]
    assert volumes == [
        f"{tmp_path}/w:{tmp_path}/w",
        f"{tmp_path}/scratch:{tmp_path}/scratch",
        "/data/hf:/data/hf",
        f"{cache}:/cache",
    ]
    assert "HOME=/cache/home" in argv
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert argv[argv.index("--entrypoint") + 1] == "/usr/bin/python3"
    assert argv[-5:] == ["img:t", "-m", "launcher.remote.node", "run-job", str(jobdir)]
    assert (jobdir / "exit_code").read_text() == "0\n"
    assert Path(cache, "home").is_dir()


def test_an_image_without_a_cache_label_keeps_one_cache_per_tag(tmp_path):
    env = _fake_docker(tmp_path, _RECORD_EXIT)
    jobdir, script = _script(tmp_path, _container(tmp_path, image="reg/img:t", gpus=("0",)))
    assert subprocess.run(["bash", str(script)], env=_without_gpu_context(env)).returncode == 0
    (argv,) = _calls(tmp_path)
    assert f"{tmp_path}/w/cache/reg_img_t:/cache" in argv


def test_cards_of_two_architectures_are_refused(tmp_path):
    env = _fake_docker(tmp_path, "")
    _fake_nvidia_smi(tmp_path, caps={"GPU-1": "8.9"})
    container = _container(tmp_path, gpus=("0", "1"), cache_label="vllm-0123456789ab")
    jobdir, script = _script(tmp_path, container)
    assert subprocess.run(["bash", str(script)], env=_without_gpu_context(env)).returncode == 2
    assert "mix compute capabilities" in (jobdir / "stdout.log").read_text()
    assert not (tmp_path / "docker.calls").exists()


def test_a_job_with_no_gpus_fails_with_a_reason(tmp_path):
    env = _fake_docker(tmp_path, "")
    jobdir, script = _script(tmp_path, _container(tmp_path))
    assert subprocess.run(["bash", str(script)], env=_without_gpu_context(env)).returncode == 2
    assert (jobdir / "exit_code").read_text() == "2\n"
    assert "no GPUs for this job" in (jobdir / "stdout.log").read_text()


def test_a_container_that_never_starts_records_dockers_status(tmp_path):
    env = _fake_docker(tmp_path, "sys.exit(125)\n")
    jobdir, script = _script(tmp_path, _container(tmp_path, gpus=("0",)))
    assert subprocess.run(["bash", str(script)], env=_without_gpu_context(env)).returncode == 125
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
    job = subprocess.Popen(["bash", str(script)], env=_without_gpu_context(env))
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
        profile_runtime="docker",
    )
    return Remote(**{**defaults, **fields})


MANIFEST = {
    "image": "vllm",
    "inputs": "abc",
    "envs": ["vllm_env"],
    "python": "/usr/bin/python3",
    "capture_python": "/opt/vibesim-capture/.venv/bin/python",
}


def test_preparing_a_container_job_reads_the_images_manifest(tmp_path, monkeypatch):
    from launcher.remote.run import RemoteRunError, Session

    env = _fake_docker(
        tmp_path,
        "if sys.argv[1] == 'run':\n"
        "    print('--- Wrapper: Applying owner=me ---')\n"
        f"    print({json.dumps(MANIFEST)!r})\n"
        "elif sys.argv[1:3] == ['image', 'inspect'] and '{{.Id}}' in sys.argv[-2]:\n"
        "    print('sha256:1')\n"
        "elif sys.argv[1:3] == ['image', 'inspect']:\n"
        "    print('vllm-0123456789ab')\n",
    )
    monkeypatch.setenv("PATH", env["PATH"])
    session = Session(_remote(tmp_path, mounts=(PurePosixPath("/data/hf"),), gpus=("2",)))
    session.source = types.SimpleNamespace(version="v1")

    with pytest.raises(RemoteRunError, match="serves \\['vllm_env'\\]"):
        session.prepare_image_job("default_env", "vibesim-profiler-vllm:abc")
    assert session.container is None

    tag = "vibesim-profiler-vllm:abc"
    assert session.prepare_image_job("vllm_env", tag, gpu_count=2) == "sha256:1"
    assert session.image_python == PurePosixPath("/usr/bin/python3")
    assert session.container == launch.Container(
        image=tag,
        python=PurePosixPath("/usr/bin/python3"),
        workdir=PurePosixPath(tmp_path / "w"),
        node_local=PurePosixPath("/tmp"),
        mounts=(PurePosixPath("/data/hf"),),
        gpus=("2",),
        gpu_count=2,
        cache_label="vllm-0123456789ab",
    )
    # A capture runs on the image's capture interpreter, not the engine's.
    session.prepare_image_job("vllm_env", tag, capture=True)
    assert session.container.python == PurePosixPath("/opt/vibesim-capture/.venv/bin/python")


def test_a_pod_runs_jobs_on_its_own_interpreter(tmp_path):
    from launcher.remote.run import RemoteRunError, Session

    session = Session(_remote(tmp_path, profile_runtime="in-image"))
    session.source = types.SimpleNamespace(version="v1")
    replies = iter([json.dumps(MANIFEST), json.dumps({**MANIFEST, "inputs": "old"})])
    session.transport = types.SimpleNamespace(
        run=lambda command, check=True: types.SimpleNamespace(returncode=0, stdout=next(replies))
    )
    assert session.prepare_image_job("vllm_env", "vibesim-profiler-vllm:abc", capture=True) == "abc"
    assert session.image_python == PurePosixPath("/opt/vibesim-capture/.venv/bin/python")
    assert session.container is None
    with pytest.raises(RemoteRunError, match="runs an image with inputs old"):
        session.prepare_image_job("vllm_env", "vibesim-profiler-vllm:abc")


def test_a_remote_without_docker_access_is_refused_by_name(tmp_path, monkeypatch):
    from launcher.remote.run import RemoteRunError, Session

    env = _fake_docker(tmp_path, "sys.exit(1)\n")
    monkeypatch.setenv("PATH", env["PATH"])
    session = Session(_remote(tmp_path))
    with pytest.raises(RemoteRunError, match="missing docker-access"):
        session.check_container_host()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["label", "--pack", "p", "--out-root", "logs/x"], "label runs here"),
        (
            ["run", "--pack", "p", "--out-root", "logs/x", "--phase", "profile_nsys", "--dry-run"],
            "--dry-run stays local",
        ),
    ],
)
def test_only_a_campaigns_capture_goes_to_a_remote(tmp_path, argv, message):
    from launcher.remote.run import RemoteRunError, Session, _alignment_campaign

    session = Session(_remote(tmp_path))
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


def test_a_job_waits_for_free_gpu_memory_before_its_container_starts(tmp_path):
    from launcher.remote.config import GpuWait

    env = _fake_docker(tmp_path, _RECORD_EXIT)
    _fake_nvidia_smi(tmp_path, [1000, 70000])
    container = _container(tmp_path, gpus=("2",), gpu_wait=GpuWait(60000, quiet_s=0, timeout_s=60))
    jobdir, script = _script(tmp_path, container)
    done = subprocess.run(["bash", str(script)], env=_without_gpu_context(env), timeout=60)
    assert done.returncode == 0
    log = (jobdir / "stdout.log").read_text()
    assert "waiting for 60000 MiB free on GPU(s) 2" in log
    assert len(_calls(tmp_path)) == 1  # the container ran once the memory was free


def test_a_job_whose_gpu_stays_busy_gives_up_with_a_reason(tmp_path):
    from launcher.remote.config import GpuWait

    env = _fake_docker(tmp_path, "")
    _fake_nvidia_smi(tmp_path, [1000])
    container = _container(tmp_path, gpus=("2",), gpu_wait=GpuWait(60000, quiet_s=0, timeout_s=0))
    jobdir, script = _script(tmp_path, container)
    done = subprocess.run(["bash", str(script)], env=_without_gpu_context(env), timeout=60)
    assert done.returncode == 75
    assert "never had 60000 MiB free" in (jobdir / "stdout.log").read_text()
    assert not (tmp_path / "docker.calls").exists()


def test_auto_gpus_take_an_idle_card(tmp_path):
    from launcher.remote.config import GpuWait

    env = _fake_docker(tmp_path, _RECORD_EXIT)
    # GPU 0 runs a process, GPU 1 holds memory, GPU 2 is idle.
    _fake_nvidia_smi(
        tmp_path,
        busy="GPU-0",
        listing=["0, GPU-0, 100, 80000", "1, GPU-1, 30000, 50000", "2, GPU-2, 5, 80000"],
    )
    container = _container(
        tmp_path, gpus=("auto",), gpu_wait=GpuWait(60000, quiet_s=5, timeout_s=60)
    )
    jobdir, script = _script(tmp_path, container)
    done = subprocess.run(["bash", str(script)], env=_without_gpu_context(env), timeout=60)
    assert done.returncode == 0
    (argv,) = _calls(tmp_path)
    assert argv[argv.index("--gpus") + 1] == '"device=GPU-2"'


def test_auto_gpus_cannot_be_mixed_with_names(tmp_path):
    path = tmp_path / "remote.yaml"
    path.write_text("remotes:\n  d:\n    workdir: /w\n    gpus: [auto, 1]\n")
    with pytest.raises(RemoteConfigError, match="cannot be combined"):
        load_remote("d", path=path)
