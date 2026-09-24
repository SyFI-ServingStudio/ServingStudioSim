"""Profiling subprocess environment registry."""

from __future__ import annotations

import os
import shutil
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
    # True for an env whose interpreter owns a separate venv. The worker then
    # drops inherited site-packages from PYTHONPATH: a parent venv's third-party
    # stack (e.g. the project Torch that the PyO3 bridge exports) would shadow
    # the env's own ABI-matched builds. Repo source stays importable.
    isolated_site_packages: bool = False

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
                    f"profiling env {self.name!r} additional Python path does not exist: {path}"
                )
            if not path.is_dir():
                raise NotADirectoryError(
                    f"profiling env {self.name!r} additional Python path is not a directory: {path}"
                )
        for path in self.additional_library_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"profiling env {self.name!r} additional library path does not exist: {path}"
                )
            if not path.is_dir():
                raise NotADirectoryError(
                    f"profiling env {self.name!r} additional library path is not a directory: "
                    f"{path}"
                )

    def validate_python_executable(self) -> None:
        """Backward-compatible validation entry point for existing callers."""
        self.validate()


@dataclass(frozen=True)
class ContainerProfileEnv:
    """Immutable image used to execute one profiling worker chunk."""

    name: str
    image: str

    def validate(self) -> None:
        if not self.image:
            raise ValueError(f"profiling env {self.name!r} has no container image")
        if shutil.which("docker") is None:
            raise FileNotFoundError("docker is required for container profiling")


_PROFILE_ENVS_ROOT = Path.home() / "profile_envs"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_UV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_SGLANG_CHECKOUT = _PROJECT_ROOT / "alignment" / "profiler" / "sglang"
# The instrumented vLLM fork is the production engine for alignment captures.
# A separate worktree without the initialized submodule points here at the
# checkout that owns the built `.venv`.
_VLLM_FORK_CHECKOUT = Path(
    os.environ.get("VIBESIM_VLLM_FORK_ROOT", _PROJECT_ROOT / "alignment" / "profiler" / "vllm")
)
_VLLM_FORK_PYTHON = _VLLM_FORK_CHECKOUT / ".venv" / "bin" / "python"
_VLLM_FORK_TORCH_LIB = (
    _VLLM_FORK_CHECKOUT / ".venv" / "lib" / "python3.12" / "site-packages" / "torch" / "lib"
)
_SGLANG_PYTHON_ROOT = _SGLANG_CHECKOUT / "python"
_SGLANG_PYTHON = _SGLANG_PYTHON_ROOT / ".venv-sglang" / "bin" / "python"


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


ENV_REGISTRY: dict[str, ProfileEnv | ContainerProfileEnv] = {
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
    # SGLang's CUDA extensions are built against the checkout-local Torch and
    # FlashInfer stack. Keep the interpreter and source tree from the same
    # checkout so profiling cannot silently import a globally installed build.
    "sglang_env": ProfileEnv(
        "sglang_env",
        _SGLANG_PYTHON,
        additional_python_paths=(_SGLANG_PYTHON_ROOT,),
    ),
    # The alignment fork's own venv (alignment/profiler/README.md). Source and
    # native extensions come from that checkout, and Torch's CUDA runtime is
    # resolved first, as the alignment server launch does.
    "vllm_fork_env": ProfileEnv(
        "vllm_fork_env",
        _VLLM_FORK_PYTHON,
        additional_python_paths=(_VLLM_FORK_CHECKOUT,),
        additional_library_paths=(_VLLM_FORK_TORCH_LIB,),
        isolated_site_packages=True,
    ),
    # vLLM runners execute in the pinned image; host source and Python packages
    # are deliberately outside this environment boundary.
    "vllm_env": ContainerProfileEnv(
        "vllm_env",
        os.environ.get("VIBESIM_VLLM_PROFILE_IMAGE", "vibesim-profiler-vllm:cu130"),
    ),
}


def register_profile_env(
    name: str,
    python_executable: Path | str,
    additional_python_paths: Iterable[Path | str] = (),
    additional_library_paths: Iterable[Path | str] = (),
    isolated_site_packages: bool = False,
) -> None:
    ENV_REGISTRY[name] = ProfileEnv(
        name=name,
        python_executable=Path(python_executable),
        additional_python_paths=tuple(Path(path) for path in additional_python_paths),
        additional_library_paths=tuple(Path(path) for path in additional_library_paths),
        isolated_site_packages=isolated_site_packages,
    )


def compose_pythonpath(profile_env: ProfileEnv, existing: str | None) -> str:
    """Compose the worker import path without mutating the parent interpreter."""
    entries = [
        str(_PROJECT_ROOT),
        *(str(path) for path in profile_env.additional_python_paths),
    ]
    if existing and profile_env.isolated_site_packages:
        entries.extend(
            entry
            for entry in existing.split(os.pathsep)
            if entry and not _is_site_packages(entry)
        )
    elif existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def _is_site_packages(entry: str) -> bool:
    return any(part in ("site-packages", "dist-packages") for part in Path(entry).parts)


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


def resolve_profile_env(name: str | None) -> ProfileEnv | ContainerProfileEnv:
    env_name = name or "default_env"
    try:
        return ENV_REGISTRY[env_name]
    except KeyError as exc:
        known_envs = sorted(ENV_REGISTRY)
        raise ValueError(f"unknown profiling env {env_name!r}; known envs: {known_envs}") from exc


__all__ = [
    "ENV_REGISTRY",
    "ContainerProfileEnv",
    "ProfileEnv",
    "compose_library_path",
    "compose_pythonpath",
    "register_profile_env",
    "resolve_profile_env",
]
