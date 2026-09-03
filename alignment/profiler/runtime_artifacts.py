"""Prepare immutable wheel-backed artifacts before a serving process starts.

This module deliberately does not import the managed packages.  In particular,
importing FlashInfer's JIT registry can download metadata or build extensions,
which would make a supposedly read-only preflight mutate runtime state.  Package
metadata and wheel file lists are enough to prove that the pinned artifacts are
present.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import PythonPackageArtifact, PythonRuntimeConfig

_INSPECT_SCRIPT = r"""
import json
import sys
import re
from importlib.metadata import distributions
from pathlib import Path

specs = json.loads(sys.argv[1])
result = {}
installed_distributions = tuple(distributions())
def canonical_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()

for spec in specs:
    name = spec["name"]
    matches = [
        dist
        for dist in installed_distributions
        if canonical_name(dist.metadata["Name"]) == canonical_name(name)
    ]
    if not matches:
        result[name] = None
        continue
    if len(matches) != 1:
        result[name] = {
            "distribution_count": len(matches),
            "installed_versions": [dist.version for dist in matches],
            "required_files": {},
        }
        continue
    dist = matches[0]
    files = tuple(dist.files or ())
    required = {}
    for requested in spec["required_files"]:
        matches = []
        for item in files:
            item_text = str(item)
            if item_text == requested:
                candidate = Path(dist.locate_file(item)).resolve()
                if candidate.is_file():
                    matches.append(
                        {"path": str(candidate), "size_bytes": candidate.stat().st_size}
                    )
        required[requested] = matches
    result[name] = {
        "distribution_count": 1,
        "installed_version": dist.version,
        "required_files": required,
    }
print(json.dumps(result, sort_keys=True))
"""


def _expected_version(package: PythonPackageArtifact) -> str:
    if package.local_version is None:
        return package.version
    return f"{package.version}+{package.local_version}"


def _inspect(
    fork_python: str, packages: list[PythonPackageArtifact]
) -> dict[str, dict[str, Any] | None]:
    specs = [
        {"name": package.name, "required_files": package.required_files} for package in packages
    ]
    probe = subprocess.run(
        [fork_python, "-c", _INSPECT_SCRIPT, json.dumps(specs)],
        capture_output=True,
        text=True,
        env=_sanitized_subprocess_env(),
        timeout=120,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip() or probe.stdout.strip() or "metadata probe failed"
        raise RuntimeError(f"could not inspect runtime artifacts with {fork_python}: {detail}")
    try:
        return json.loads(probe.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"runtime-artifact probe returned invalid JSON: {probe.stdout!r}"
        ) from exc


def _sanitized_subprocess_env() -> dict[str, str]:
    """Keep host tool access but exclude Python-environment inheritance."""
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _acquire_lock(lock_file, lock_path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for Python-runtime preparation lock: {lock_path}"
                ) from None
            time.sleep(0.1)


def _verify_versions(
    packages: list[PythonPackageArtifact],
    state: dict[str, dict[str, Any] | None],
) -> list[PythonPackageArtifact]:
    missing = []
    for package in packages:
        installed = state[package.name]
        if installed is None:
            missing.append(package)
            continue
        distribution_count = installed.get("distribution_count", 1)
        if distribution_count != 1:
            raise RuntimeError(
                f"runtime package {package.name!r} resolves to {distribution_count} "
                f"installed distributions with versions "
                f"{installed.get('installed_versions', [])}. Refusing to select one "
                "implicitly; rebuild the fork venv."
            )
        expected = _expected_version(package)
        actual = installed["installed_version"]
        if actual != expected:
            raise RuntimeError(
                f"runtime package {package.name!r} has version {actual!r}, expected "
                f"{expected!r}. Refusing to mutate an existing environment; rebuild the "
                "fork venv or correct python_runtime in the profile YAML."
            )
    return missing


def _verify_files(
    packages: list[PythonPackageArtifact],
    state: dict[str, dict[str, Any] | None],
) -> None:
    for package in packages:
        installed = state[package.name]
        if installed is None:
            raise RuntimeError(f"runtime package {package.name!r} is still missing")
        absent = [name for name, matches in installed["required_files"].items() if not matches]
        if absent:
            raise RuntimeError(
                f"runtime package {package.name!r} is present but lacks required wheel "
                f"artifacts {absent}. Refusing to refresh/reinstall an existing package."
            )


def _install_missing(
    fork_python: str,
    missing: list[PythonPackageArtifact],
    timeout: float,
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "runtime artifacts are missing and `uv` is not on PATH; install uv or "
            "prepare the exact wheels declared by python_runtime"
        )
    for package in missing:
        requirement = f"{package.name}=={package.version}"
        try:
            install = subprocess.run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    fork_python,
                    "--no-deps",
                    "--only-binary",
                    ":all:",
                    requirement,
                    "--index-url",
                    package.index_url,
                ],
                text=True,
                env=_sanitized_subprocess_env(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"timed out after {timeout}s installing runtime artifact "
                f"{requirement!r} from {package.index_url}"
            ) from exc
        if install.returncode != 0:
            raise RuntimeError(
                f"failed to install missing runtime artifact {requirement!r} from "
                f"{package.index_url}"
            )


def prepare_python_runtime(
    fork_python: str, config: PythonRuntimeConfig | None
) -> dict[str, Any] | None:
    """Ensure one interpreter's declared wheels exactly once, then prove them.

    Concurrent profiles sharing the same interpreter serialize only this short
    preparation section.  Existing mismatched or incomplete packages are never
    refreshed in place, preventing two configs from repeatedly replacing one
    another's runtime.  The lock is released before any server is launched.
    """
    if config is None:
        return None

    venv_root = Path(fork_python).parent.parent
    lock_path = venv_root / ".vibesim-python-runtime.lock"
    with lock_path.open("a+") as lock_file:
        _acquire_lock(lock_file, lock_path, config.lock_timeout_seconds)
        state = _inspect(fork_python, config.packages)
        missing = _verify_versions(config.packages, state)
        if missing:
            _install_missing(fork_python, missing, config.install_timeout_seconds)
            state = _inspect(fork_python, config.packages)
            remaining = _verify_versions(config.packages, state)
            if remaining:
                raise RuntimeError(
                    "runtime artifact installation completed but packages remain missing: "
                    + ", ".join(package.name for package in remaining)
                )
        _verify_files(config.packages, state)

    return {
        "lock_path": str(lock_path.resolve()),
        "policy": "install-missing-only; reject-version-conflict; verify-wheel-files",
        "packages": [
            {
                "name": package.name,
                "requested_version": package.version,
                "required_version": _expected_version(package),
                "index_url": package.index_url,
                **state[package.name],
            }
            for package in config.packages
        ],
    }
