"""Container-specific profiling provenance and path contracts."""

from __future__ import annotations

from pathlib import Path

from profiling.exec.env import ContainerProfileEnv
from profiling.exec.local import _container_worker_command


def test_container_worker_mounts_only_exchange_and_cache(tmp_path: Path, monkeypatch) -> None:
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
    assert not any("profile.db" in argument for argument in command)
    assert command[-4:] == [
        "--worker-input",
        "/io/input.json",
        "--worker-output",
        "/io/output.json",
    ]
