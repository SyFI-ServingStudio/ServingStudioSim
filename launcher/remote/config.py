"""`remote.yaml`: the machines `--remote NAME` can run on.

The file is untracked and per user. It is found at ``$VIBESIM_REMOTE_CONFIG``,
else at this checkout's root, else at the root of the repository's main
worktree, so one file serves every worktree. ``remote.example.yaml`` documents
each field; this module is the authority on what is accepted.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from ..cli import UsageError, apply_dotted

CONFIG_ENV = "VIBESIM_REMOTE_CONFIG"
CONFIG_NAME = "remote.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]


class RemoteConfigError(ValueError):
    """``remote.yaml`` is missing, or the selected remote is malformed."""


@dataclass(frozen=True)
class Ssh:
    host: str
    port: int | None = None
    identity_file: str | None = None
    # Extra ssh arguments, appended after the defaults (`transport.py`).
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class Slurm:
    time: str
    partition: str | None = None
    cpus_per_task: int | None = None
    mem: str | None = None
    account: str | None = None
    # GPUs per job (`--gres=gpu:N`). A GPU command asks for at least what its
    # specs need; 0 leaves a CPU job without any.
    gpus: int = 0
    # Extra `#SBATCH` arguments, verbatim (e.g. "--qos=foo").
    extra: tuple[str, ...] = ()


@dataclass(frozen=True)
class Remote:
    name: str
    # Absolute path on the remote that holds code versions, builds, venvs,
    # jobs and experiments (`layout.py`).
    workdir: PurePosixPath
    # None: the command runs on this machine (`access: local`).
    ssh: Ssh | None
    # None: the job runs directly on the access host (`launch: direct`).
    slurm: Slurm | None
    # "rsync": ship inputs and fetch results over the transport. "shared": the
    # workdir is visible here at the same path, so files move by local copy.
    files: str
    # Node-local scratch for each job's private profile.db copy and TMPDIR.
    node_local: PurePosixPath = PurePosixPath("/tmp")
    env: dict[str, str] = field(default_factory=dict)
    keep_versions: int = 10
    profile_runtime: str = "docker"
    # The remote may lose its disk (a container pod): copy results back whole.
    ephemeral: bool = False
    # Directory of remote.yaml; `.remote/known_hosts` beside it pins host keys.
    config_dir: Path = REPO_ROOT


def find_config(repo_root: Path = REPO_ROOT) -> Path:
    explicit = os.environ.get(CONFIG_ENV, "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise RemoteConfigError(f"{CONFIG_ENV}={explicit}: no such file")
        return path
    candidates = [repo_root / CONFIG_NAME]
    main_root = _main_worktree_root(repo_root)
    if main_root is not None and main_root != repo_root:
        candidates.append(main_root / CONFIG_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RemoteConfigError(
        f"no {CONFIG_NAME} (looked at {', '.join(str(c) for c in candidates)}); "
        f"copy remote.example.yaml to {repo_root / CONFIG_NAME} or set {CONFIG_ENV}"
    )


def _main_worktree_root(repo_root: Path) -> Path | None:
    try:
        listing = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in listing.splitlines():
        if line.startswith("worktree "):
            return Path(line.removeprefix("worktree ")).resolve()
    return None


def load_remote(name: str, overrides: tuple[str, ...] = (), *, path: Path | None = None) -> Remote:
    path = find_config() if path is None else path
    try:
        document = yaml.safe_load(path.read_text("utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RemoteConfigError(f"{path}: not valid YAML: {exc}") from None
    remotes = document.get("remotes") if isinstance(document, dict) else None
    if not isinstance(remotes, dict):
        raise RemoteConfigError(f"{path}: expected a top-level `remotes:` mapping")
    if name not in remotes:
        known = ", ".join(sorted(remotes)) or "none"
        raise RemoteConfigError(f"{path}: no remote {name!r} (known: {known})")
    raw = remotes[name]
    if not isinstance(raw, dict):
        raise RemoteConfigError(f"{path}: remote {name!r} must be a mapping")
    try:
        raw = apply_dotted(raw, list(overrides))
    except (UsageError, IndexError, ValueError, TypeError) as exc:
        raise RemoteConfigError(f"--remote-set: {exc}") from exc
    try:
        return _parse(name, raw, path.parent.resolve())
    except RemoteConfigError as exc:
        raise RemoteConfigError(f"{path}: remote {name!r}: {exc}") from None


def _parse(name: str, raw: dict[str, Any], config_dir: Path) -> Remote:
    _reject_unknown(
        raw,
        {
            "access",
            "launch",
            "files",
            "workdir",
            "node_local",
            "env",
            "keep_versions",
            "profile_runtime",
            "ephemeral",
        },
        "",
    )
    workdir = _absolute(raw.get("workdir"), "workdir")
    ssh = _parse_access(raw.get("access", "local"))
    slurm = _parse_launch(raw.get("launch", "direct"))
    files = raw.get("files", "rsync")
    if files not in ("rsync", "shared"):
        raise RemoteConfigError(f"files must be rsync or shared, got {files!r}")
    env = raw.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
        raise RemoteConfigError("env must map variable names to values")
    keep_versions = raw.get("keep_versions", 10)
    if not isinstance(keep_versions, int) or isinstance(keep_versions, bool) or keep_versions < 1:
        raise RemoteConfigError(f"keep_versions must be a positive integer, got {keep_versions!r}")
    profile_runtime = raw.get("profile_runtime", "docker")
    if profile_runtime not in ("docker", "in-image"):
        raise RemoteConfigError(
            f"profile_runtime must be docker or in-image, got {profile_runtime!r}"
        )
    ephemeral = raw.get("ephemeral", False)
    if not isinstance(ephemeral, bool):
        raise RemoteConfigError(f"ephemeral must be true or false, got {ephemeral!r}")
    return Remote(
        name=name,
        workdir=workdir,
        ssh=ssh,
        slurm=slurm,
        files=files,
        node_local=_absolute(raw.get("node_local", "/tmp"), "node_local"),
        env={key: str(value) for key, value in env.items()},
        keep_versions=keep_versions,
        profile_runtime=profile_runtime,
        ephemeral=ephemeral,
        config_dir=config_dir,
    )


def _parse_access(raw: Any) -> Ssh | None:
    if raw == "local":
        return None
    if not isinstance(raw, dict) or "ssh" not in raw:
        raise RemoteConfigError("access must be `local` or a mapping with `ssh: user@host`")
    _reject_unknown(raw, {"ssh", "port", "identity_file", "options"}, "access.")
    host = raw["ssh"]
    if not isinstance(host, str) or not host:
        raise RemoteConfigError("access.ssh must be a host (user@host or an ssh_config alias)")
    port = raw.get("port")
    if port is not None and (not isinstance(port, int) or isinstance(port, bool)):
        raise RemoteConfigError(f"access.port must be an integer, got {port!r}")
    identity_file = raw.get("identity_file")
    if identity_file is not None and not isinstance(identity_file, str):
        raise RemoteConfigError("access.identity_file must be a path")
    options = raw.get("options", [])
    if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
        raise RemoteConfigError("access.options must be a list of ssh arguments")
    return Ssh(host, port, identity_file and os.path.expanduser(identity_file), tuple(options))


def _parse_launch(raw: Any) -> Slurm | None:
    if raw == "direct":
        return None
    if not isinstance(raw, dict) or set(raw) != {"slurm"} or not isinstance(raw["slurm"], dict):
        raise RemoteConfigError("launch must be `direct` or `{slurm: {time: ..., ...}}`")
    body = raw["slurm"]
    _reject_unknown(
        body,
        {"time", "partition", "cpus_per_task", "mem", "account", "gpus", "extra"},
        "launch.slurm.",
    )
    time = body.get("time")
    if not isinstance(time, str) or not time:
        # Every submission must carry a limit; some clusters reject jobs without one.
        raise RemoteConfigError('launch.slurm.time is required (e.g. "04:00:00")')
    cpus = body.get("cpus_per_task")
    if cpus is not None and (not isinstance(cpus, int) or isinstance(cpus, bool) or cpus < 1):
        raise RemoteConfigError(
            f"launch.slurm.cpus_per_task must be a positive integer, got {cpus!r}"
        )
    gpus = body.get("gpus", 0)
    if not isinstance(gpus, int) or isinstance(gpus, bool) or gpus < 0:
        raise RemoteConfigError(f"launch.slurm.gpus must be a non-negative integer, got {gpus!r}")
    for key in ("partition", "mem", "account"):
        if body.get(key) is not None and not isinstance(body[key], str):
            raise RemoteConfigError(f"launch.slurm.{key} must be a string")
    extra = body.get("extra", [])
    if not isinstance(extra, list) or not all(isinstance(e, str) for e in extra):
        raise RemoteConfigError("launch.slurm.extra must be a list of sbatch arguments")
    return Slurm(
        time=time,
        partition=body.get("partition"),
        cpus_per_task=cpus,
        mem=body.get("mem"),
        account=body.get("account"),
        gpus=gpus,
        extra=tuple(extra),
    )


def _absolute(raw: Any, key: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise RemoteConfigError(f"{key} must be an absolute path on the remote, got {raw!r}")
    return PurePosixPath(raw)


def _reject_unknown(raw: dict, known: set[str], prefix: str) -> None:
    unknown = sorted(set(raw) - known)
    if unknown:
        raise RemoteConfigError(f"unknown field(s) {', '.join(prefix + u for u in unknown)}")
