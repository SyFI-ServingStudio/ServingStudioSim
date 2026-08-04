"""Profiling subprocess environment registry."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfileEnv:
    name: str
    python_executable: Path
    additional_python_paths: tuple[Path, ...] = ()
    additional_library_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "python_executable", Path(self.python_executable))
        object.__setattr__(
            self,
            "additional_python_paths",
            tuple(Path(path) for path in self.additional_python_paths),
        )
        object.__setattr__(
            self,
            "additional_library_paths",
            tuple(Path(path) for path in self.additional_library_paths),
        )

    def validate(self) -> None:
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
        for path in self.additional_python_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"profiling env {self.name!r} additional Python path does not exist: "
                    f"{path}"
                )
            if not path.is_dir():
                raise NotADirectoryError(
                    f"profiling env {self.name!r} additional Python path is not a directory: "
                    f"{path}"
                )
        for path in self.additional_library_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"profiling env {self.name!r} additional library path does not exist: "
                    f"{path}"
                )
            if not path.is_dir():
                raise NotADirectoryError(
                    f"profiling env {self.name!r} additional library path is not a directory: "
                    f"{path}"
                )

    def validate_python_executable(self) -> None:
        """Backward-compatible validation entry point for existing callers."""
        self.validate()


_PROFILE_ENVS_ROOT = Path.home() / "profile_envs"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_UV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_VLLM_ROOT = _PROJECT_ROOT / "alignment" / "profiler" / "vllm"
_WORKER_PYTHON_DIR = f"python{sys.version_info.major}.{sys.version_info.minor}"
_VLLM_SITE_PACKAGES = _VLLM_ROOT / ".venv" / "lib" / _WORKER_PYTHON_DIR / "site-packages"
_VLLM_TORCH_LIB = _VLLM_SITE_PACKAGES / "torch" / "lib"


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
    # The exact vLLM CUDA runners must use the same instrumented fork and wheel
    # stack as alignment profiling. The checkout is intentionally read-only and
    # its venv interpreter symlink is not portable, so use the working project
    # interpreter while importing the intact vLLM environment explicitly.
    "vllm_env": ProfileEnv(
        "vllm_env",
        _default_python(),
        (_VLLM_SITE_PACKAGES, _VLLM_ROOT),
        (_VLLM_TORCH_LIB,),
    ),
}


def register_profile_env(
    name: str,
    python_executable: Path | str,
    additional_python_paths: Iterable[Path | str] = (),
    additional_library_paths: Iterable[Path | str] = (),
) -> None:
    ENV_REGISTRY[name] = ProfileEnv(
        name=name,
        python_executable=Path(python_executable),
        additional_python_paths=tuple(Path(path) for path in additional_python_paths),
        additional_library_paths=tuple(Path(path) for path in additional_library_paths),
    )


def compose_pythonpath(profile_env: ProfileEnv, existing: str | None) -> str:
    """Compose the worker import path without mutating the parent interpreter."""
    entries = [
        str(_PROJECT_ROOT),
        *(str(path) for path in profile_env.additional_python_paths),
    ]
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def compose_library_path(profile_env: ProfileEnv, existing: str | None) -> str:
    """Prepend env-owned shared libraries exactly as the framework launch does.

    CUDA extension runners must resolve the Torch wheel's CUDA runtime before a
    system toolkit runtime. In particular, the instrumented vLLM launch puts
    ``torch/lib`` first; reproducing only its Python import path can load a
    different ``libcudart`` and invalidate the measured kernel environment.
    """
    entries = [str(path) for path in profile_env.additional_library_paths]
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


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
    "compose_library_path",
    "compose_pythonpath",
    "register_profile_env",
    "resolve_profile_env",
]
