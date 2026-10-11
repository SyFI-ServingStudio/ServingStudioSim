"""Profiling subprocess environment registry."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, MutableMapping
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
        inspect = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
            capture_output=True,
            text=True,
            check=False,
        )
        if inspect.returncode != 0:
            # Never pulled: the tag names this checkout's inputs, so a registry
            # copy under it would be some other build.
            raise FileNotFoundError(
                f"profiling env {self.name!r} needs image {self.image}, which this host "
                "does not have; build it with `python -m profiling.container.images "
                f"build {self.image.split(':')[0].removeprefix('vibesim-profiler-')}`"
            )


_PROFILE_ENVS_ROOT = Path.home() / "profile_envs"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_UV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_SGLANG_CHECKOUT = _PROJECT_ROOT / "alignment" / "profiler" / "sglang"
# The alignment vLLM checkout (alignment/profiler/vllm) and its own built
# `.venv`. A separate worktree without the initialized submodule points
# VIBESIM_VLLM_FORK_ROOT at the checkout that owns the `.venv`.
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


ENV_REGISTRY: dict[str, ProfileEnv] = {
    "default_env": ProfileEnv("default_env", _default_python()),
    # Most profilers should stay on default_env via subprocess_env=None. In this
    # repo, default_env is the uv-managed project .venv when it exists.
    # Add or use a named env only for real import/linker/dependency isolation.
    # FlashInfer pip is intentionally the default stack. Keep a registry alias
    # so KernelProfilerSpec rows can state the dependency intent without
    # forcing a separate venv.
    "flashinfer_pip_env": ProfileEnv("flashinfer_pip_env", _default_python()),
    # FlashInfer's Blackwell KDA prefill backends (TIRx, CuTe DSL persistent)
    # are on FlashInfer main after the 0.7.1 release branch was cut, and need
    # Torch cu130, CuTe DSL >= 4.7 and a CUDA-enabled TVM with TIRx -- none of
    # which the project venv (Torch 2.10 cu128, FlashInfer 0.6) can take. It is
    # its own venv, built by `profiling/exec/flashinfer_kda_env.sh`.
    "flashinfer_kda_env": ProfileEnv(
        "flashinfer_kda_env",
        _profile_env_python("flashinfer_kda"),
        isolated_site_packages=True,
    ),
    # SGLang's CUDA extensions are built against the checkout-local Torch and
    # FlashInfer stack. Keep the interpreter and source tree from the same
    # checkout so profiling cannot silently import a globally installed build.
    "sglang_env": ProfileEnv(
        "sglang_env",
        _SGLANG_PYTHON,
        additional_python_paths=(_SGLANG_PYTHON_ROOT,),
    ),
    # Dao-AILab causal-conv1d's prebuilt wheel, installed with `--no-deps
    # --target` into one directory and imported on the project interpreter.
    # The wheel is built against the project's Torch (2.10, cu12, cxx11 ABI),
    # so it needs no venv of its own; a Torch bump fails its import loudly.
    # Setup: profiling/README.md "causal_conv1d_env".
    "causal_conv1d_env": ProfileEnv(
        "causal_conv1d_env",
        _default_python(),
        additional_python_paths=(_PROFILE_ENVS_ROOT / "causal_conv1d",),
    ),
    # The alignment vLLM checkout (alignment/profiler/README.md): upstream
    # vLLM 04730e8 plus the alignment commits, with the models and the
    # FlashInfer/DeepGEMM/FlashMLA builds its backends call. Its image builds
    # the checkout at the gitlink (profiling/container/vllm.Dockerfile). On the
    # host it is the checkout's own `.venv`: source and native extensions come
    # from the checkout, Torch's CUDA runtime resolves first as in the alignment
    # server launch, and inherited site-packages are dropped so the project
    # Torch cannot shadow the venv's ABI-matched build.
    "vllm_env": ProfileEnv(
        "vllm_env",
        _VLLM_FORK_PYTHON,
        additional_python_paths=(_VLLM_FORK_CHECKOUT,),
        additional_library_paths=(_VLLM_FORK_TORCH_LIB,),
        isolated_site_packages=True,
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
            entry for entry in existing.split(os.pathsep) if entry and not _is_site_packages(entry)
        )
    elif existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def _is_site_packages(entry: str) -> bool:
    return any(part in ("site-packages", "dist-packages") for part in Path(entry).parts)


# Interpreter-selection variables a host worker inherits from its caller. The
# launcher's PyO3 child sets them to the project venv's stdlib and site-packages.
_INTERPRETER_PATH_VARS = ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")


def set_worker_python_env(profile_env: ProfileEnv, env: MutableMapping[str, str]) -> None:
    """Set a host worker's import path in `env` (a copy of the caller's environment).

    A worker on the project interpreter keeps the caller's PYTHONPATH after its
    own entries. A worker on its own interpreter (`sglang_env`, `vllm_env`) drops the
    caller's interpreter variables: inherited, they would put the project venv's
    Torch and FlashInfer ahead of the env's own stack.
    """
    if profile_env.python_executable != _default_python():
        for key in _INTERPRETER_PATH_VARS:
            env.pop(key, None)
    env["PYTHONPATH"] = compose_pythonpath(profile_env, env.get("PYTHONPATH"))


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


# Every env runs in the image `profiling/container/images.py` assigns it
# ("container"). "in-image" is for a process that already runs inside one of
# those images (a container host such as a RunPod pod): the envs that image
# serves run on its own interpreter, and any other env is refused. "host" runs
# an env on the host interpreter its entry above names, for debugging an env
# against a local build; rows it writes share the image's DB key, so compare
# against the image before keeping them.
PROFILE_RUNTIME_ENV = "VIBESIM_PROFILE_RUNTIME"
_PROFILE_RUNTIMES = ("container", "in-image", "host")
# Written by every profiler image's Dockerfile: the image's input hash (its
# tag's suffix), the envs it serves, its interpreter and import path.
IMAGE_MANIFEST_ENV = "VIBESIM_PROFILE_IMAGE_MANIFEST"
IMAGE_MANIFEST = Path("/etc/vibesim-profiler-image.json")


def profile_runtime() -> str:
    runtime = os.environ.get(PROFILE_RUNTIME_ENV, "").strip() or "container"
    if runtime not in _PROFILE_RUNTIMES:
        raise ValueError(
            f"{PROFILE_RUNTIME_ENV} must be one of {_PROFILE_RUNTIMES}, got {runtime!r}"
        )
    return runtime


def resolve_profile_env(name: str | None) -> ProfileEnv | ContainerProfileEnv:
    env_name = name or "default_env"
    try:
        env = ENV_REGISTRY[env_name]
    except KeyError as exc:
        known_envs = sorted(ENV_REGISTRY)
        raise ValueError(f"unknown profiling env {env_name!r}; known envs: {known_envs}") from exc
    runtime = profile_runtime()
    if runtime == "container":
        return ContainerProfileEnv(env_name, container_image(env_name))
    if runtime == "in-image":
        return in_image_env(env_name)
    return env


def in_image_env(env_name: str) -> ProfileEnv:
    """Profile env ``env_name`` on the interpreter of the image this process runs in.

    Only an env that image serves, and only when the image is the one this
    checkout names for it: another image's rows would share the same DB key.
    """
    path = Path(os.environ.get(IMAGE_MANIFEST_ENV) or IMAGE_MANIFEST)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(
            f"{PROFILE_RUNTIME_ENV}=in-image, but {path} is not a profiler image "
            f"manifest ({exc}); this process does not run in a profiler image"
        ) from None
    if env_name not in manifest["envs"]:
        raise ValueError(
            f"profiling env {env_name!r} needs image {container_image(env_name)}; "
            f"the image this process runs in serves only {manifest['envs']}"
        )
    expected = container_image(env_name)
    if expected.rpartition(":")[2] != manifest["inputs"]:
        raise ValueError(
            f"profiling env {env_name!r} needs image {expected}, but this process "
            f"runs in an image built from inputs {manifest['inputs']}"
        )
    return ProfileEnv(
        env_name,
        Path(manifest["python"]),
        additional_python_paths=tuple(Path(p) for p in manifest["python_paths"]),
        isolated_site_packages=True,
    )


def container_image(name: str | None) -> str:
    """The image that runs profile env ``name``."""
    from profiling.container.images import image_for_env

    env_name = name or "default_env"
    image = image_for_env(env_name, _PROJECT_ROOT)
    if image is None:
        raise ValueError(
            f"profiling env {env_name!r} has no image (profiling/container/images.py); "
            f"only {PROFILE_RUNTIME_ENV}=host can run it"
        )
    return image


__all__ = [
    "ENV_REGISTRY",
    "ContainerProfileEnv",
    "IMAGE_MANIFEST",
    "IMAGE_MANIFEST_ENV",
    "PROFILE_RUNTIME_ENV",
    "ProfileEnv",
    "compose_library_path",
    "compose_pythonpath",
    "container_image",
    "in_image_env",
    "profile_runtime",
    "register_profile_env",
    "resolve_profile_env",
    "set_worker_python_env",
]
