"""Profiling subprocess environment registry."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfileEnv:
    name: str
    python_executable: Path

    def validate_python_executable(self) -> None:
        if not self.python_executable.exists():
            raise FileNotFoundError(
                f"profiling env {self.name!r} python executable does not exist: "
                f"{self.python_executable}"
            )
        if not os.access(self.python_executable, os.X_OK):
            raise PermissionError(
                f"profiling env {self.name!r} python executable is not executable: "
                f"{self.python_executable}"
            )


_PROFILE_ENVS_ROOT = Path.home() / "profile_envs"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_UV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"


def _profile_env_python(name: str) -> Path:
    return _PROFILE_ENVS_ROOT / name / "bin" / "python"


def _default_python() -> Path:
    # Env selection is intentionally passive: callers create/sync the uv env
    # ahead of time (`uv sync --group profiling`), then the execution backend
    # reuses that interpreter for subprocess workers.
    if _PROJECT_UV_PYTHON.exists():
        return _PROJECT_UV_PYTHON
    configured = _profile_env_python("default_env")
    return configured if configured.exists() else Path(sys.executable)


ENV_REGISTRY: dict[str, ProfileEnv] = {
    "default_env": ProfileEnv("default_env", _default_python()),
    # Most profilers should stay on default_env via subprocess_env=None. In this
    # repo, default_env is the uv-managed project .venv when it exists.
    # Add or use a named env only for real import/linker/dependency isolation.
    # FlashInfer pip is intentionally the default stack. Keep a registry alias
    # so KernelProfilerSpec rows can state the dependency intent without
    # forcing a separate venv.
    "flashinfer_pip_env": ProfileEnv("flashinfer_pip_env", _default_python()),
    "flashinfer_local": ProfileEnv(
        "flashinfer_local",
        _profile_env_python("flashinfer_local"),
    ),
    "vllm_env": ProfileEnv("vllm_env", _profile_env_python("vllm_env")),
}


def register_profile_env(name: str, python_executable: Path | str) -> None:
    ENV_REGISTRY[name] = ProfileEnv(name=name, python_executable=Path(python_executable))


def resolve_profile_env(name: str | None) -> ProfileEnv:
    env_name = name or "default_env"
    try:
        return ENV_REGISTRY[env_name]
    except KeyError as exc:
        known_envs = sorted(ENV_REGISTRY)
        raise ValueError(f"unknown profiling env {env_name!r}; known envs: {known_envs}") from exc


__all__ = [
    "ENV_REGISTRY",
    "ProfileEnv",
    "register_profile_env",
    "resolve_profile_env",
]
