"""Container-specific profiling provenance and path contracts."""

from __future__ import annotations

from pathlib import Path

from profiling.exec.env import ContainerProfileEnv
from profiling.exec.local import _container_worker_command


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

    assert '"device=3,1"' in command
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
