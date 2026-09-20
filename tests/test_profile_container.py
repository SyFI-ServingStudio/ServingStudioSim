"""Container-specific profiling provenance and path contracts."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from profiling.exec.env import ContainerProfileEnv
from profiling.exec.local import _container_worker_command, gpu_uuids_for_indices
from profiling.profilers.energy import ENERGY_ENV
from profiling.profilers.timer import CUPTI_TRACE_ENV


def _fake_smi_xml(count: int = 8) -> ET.Element:
    gpus = "".join(f"<gpu><uuid>GPU-fake-{index}</uuid></gpu>" for index in range(count))
    return ET.fromstring(f"<nvidia_smi_log>{gpus}</nvidia_smi_log>")


@pytest.fixture(autouse=True)
def stub_nvidia_smi(monkeypatch):
    """Give every container-command test a deterministic index-to-UUID map.

    The command builder now resolves UUIDs, so these unit tests would otherwise
    depend on the host's real cards.
    """

    monkeypatch.setattr("profiling.exec.local._nvidia_smi_xml", _fake_smi_xml)


def test_container_worker_mounts_worktree_source_read_only(tmp_path: Path, monkeypatch) -> None:
    exchange_dir = tmp_path / "exchange"
    cache_dir = tmp_path / "cache"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(cache_dir))

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test"),
        [3, 1],
        exchange_dir,
    )

    assert '"device=GPU-fake-3,GPU-fake-1"' in command
    assert "HOME=/cache/home" in command
    assert "USER=vibesim" in command
    assert "LOGNAME=vibesim" in command
    assert f"{exchange_dir}:/io" in command
    assert f"{cache_dir}:/cache" in command
    project_root = Path(__file__).parents[1].resolve()
    for directory in ("profiling", "launcher", "gpu"):
        assert f"{project_root / directory}:/opt/vibesim/{directory}:ro" in command
    assert not any("profile.db" in argument for argument in command)
    assert command[-4:] == [
        "--worker-input",
        "/io/input.json",
        "--worker-output",
        "/io/output.json",
    ]


def test_container_worker_can_use_frozen_image_source(tmp_path: Path, monkeypatch) -> None:
    exchange_dir = tmp_path / "exchange"
    cache_dir = tmp_path / "cache"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("VIBESIM_PROFILE_SOURCE_MODE", "image")

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test"),
        [3],
        exchange_dir,
    )

    assert not any("/opt/vibesim/profiling:ro" in argument for argument in command)
    assert not any("/opt/vibesim/launcher:ro" in argument for argument in command)
    assert not any("/opt/vibesim/gpu:ro" in argument for argument in command)


def test_container_worker_rejects_unknown_source_mode(tmp_path: Path, monkeypatch) -> None:
    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("VIBESIM_PROFILE_SOURCE_MODE", "surprise")

    try:
        _container_worker_command(
            ContainerProfileEnv("vllm_env", "profiler:test"),
            [3],
            exchange_dir,
        )
    except ValueError as exc:
        assert "VIBESIM_PROFILE_SOURCE_MODE" in str(exc)
    else:
        raise AssertionError("unknown source mode must fail")


def test_container_worker_mounts_explicit_additional_volume(tmp_path: Path, monkeypatch) -> None:
    exchange_dir = tmp_path / "exchange"
    cache_dir = tmp_path / "cache"
    measurement_dir = tmp_path / "measurement"
    exchange_dir.mkdir()
    measurement_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(cache_dir))

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test"),
        [4],
        exchange_dir,
        additional_volumes=((measurement_dir, measurement_dir.resolve()),),
    )

    measurement_mount = f"{measurement_dir.resolve()}:{measurement_dir.resolve()}"
    assert measurement_mount in command
    assert command.index(measurement_mount) < command.index("profiler:test")


def test_container_worker_never_passes_a_bare_gpu_index(tmp_path: Path, monkeypatch) -> None:
    """The container runtime resolves indices in its own device view, not ours.

    Under a device cgroup the allocated card enumerates locally as index 0 while
    ``dockerd``, outside the cgroup, reads ``device=0`` as the host's first card,
    so every worker silently lands on GPU 0. Only a UUID survives the crossing.
    """

    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test"),
        [0],
        exchange_dir,
    )

    gpu_request = command[command.index("--gpus") + 1]
    assert gpu_request == '"device=GPU-fake-0"'
    assert "device=0" not in gpu_request


def test_container_worker_fails_when_the_uuid_is_unknown(tmp_path: Path, monkeypatch) -> None:
    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr("profiling.exec.local._nvidia_smi_xml", lambda: _fake_smi_xml(2))

    with pytest.raises(RuntimeError, match="no UUID for GPU index"):
        _container_worker_command(
            ContainerProfileEnv("vllm_env", "profiler:test"),
            [5],
            exchange_dir,
        )


def test_gpu_uuids_require_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr("profiling.exec.local._nvidia_smi_xml", lambda: None)

    with pytest.raises(RuntimeError, match="nvidia-smi is required"):
        gpu_uuids_for_indices([0])


def test_container_worker_does_not_forward_the_energy_policy(tmp_path, monkeypatch) -> None:
    """The energy policy rides the JSON payload, and must ride only that.

    It used to be forwarded as ``--env`` here as well. Two channels for one
    policy is how they drift: a container run whose payload said off while its
    environment said on would measure a different thing than the host rows
    sitting beside it in the same table.
    """

    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv(ENERGY_ENV, "1")

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test"),
        [0],
        exchange_dir,
    )

    assert not any(argument.startswith(ENERGY_ENV) for argument in command)


def test_container_worker_mounts_the_trace_directory(tmp_path, monkeypatch) -> None:
    """Forwarding the trace path is useless unless the path also exists inside.

    Without the mount the container write fails, ``_emit_trace`` swallows it,
    and the container groups come back blank while the host groups beside them
    trace normally.
    """

    exchange_dir = tmp_path / "exchange"
    trace_dir = tmp_path / "traces"
    exchange_dir.mkdir()
    trace_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv(CUPTI_TRACE_ENV, str(trace_dir / "run.jsonl"))

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test"),
        [0],
        exchange_dir,
    )

    assert f"{CUPTI_TRACE_ENV}={trace_dir / 'run.jsonl'}" in command
    assert f"{trace_dir.resolve()}:{trace_dir.resolve()}" in command


def test_container_worker_redirects_the_inductor_cache_into_the_mount(
    tmp_path: Path, monkeypatch
) -> None:
    """Inductor's default cache dir is under /tmp, which dies with the container.

    Left alone, every submission recompiles from nothing while the host workers
    beside it reuse a warm cache -- a difference in cost only, but an unbounded
    one, and invisible because a recompile looks exactly like a slow import.
    """

    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test"),
        [0],
        exchange_dir,
    )

    cache_dir = command[command.index("HOME=/cache/home") - 1 :]
    assert "TORCHINDUCTOR_CACHE_DIR=/cache/home/.cache/torchinductor" in command
    # Whatever it points at has to be inside the one directory that is mounted.
    inductor = next(a for a in command if a.startswith("TORCHINDUCTOR_CACHE_DIR="))
    assert inductor.split("=", 1)[1].startswith("/cache/")
    assert any(a.endswith(":/cache") for a in cache_dir)
