"""What `--remote` ships of this checkout, and the identity of what it ships.

The remote gets the file tree, not the repository: every tracked or untracked
non-ignored file, the req-frontend submodule the Rust build depends on, the
ignored ``Cargo.lock``, and the ignored workload data under ``trace/``. The
other submodules (the vLLM and SGLang profiler checkouts) are only for
alignment profiling on this machine.

``version`` names the shipped tree: it changes whenever any shipped file or the
commit it sits on changes, and two checkouts with the same commit and edits
share it, so a remote keeps one copy per version. ``venv_key`` names the Python
environment: only the lock, the project file and the selected groups decide
it, so code edits never rebuild a venv.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..source_stamp import STAMP_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
REQ_FRONTEND = "alignment/load_generator/req-frontend"
DATA_DIRS = ("trace",)
# Ignored here but build inputs: the Rust build must resolve the same crates.
IGNORED_INPUTS = ("Cargo.lock",)
# Ignored files under DATA_DIRS that are not data.
_NOT_DATA = ("__pycache__/", ".pyc")
# Groups a CPU host needs to simulate, analyze and test: no torch/CUDA wheels
# and no DeepGEMM build (pyproject `profiling` and `public-api` stay out).
CPU_GROUPS = ("launcher", "analyze", "dev")
VENV_INPUTS = ("uv.lock", "pyproject.toml", ".python-version")


class SourceError(RuntimeError):
    """This checkout cannot be shipped as is."""


@dataclass(frozen=True)
class Source:
    root: Path
    files: tuple[str, ...]
    version: str
    venv_key: str
    groups: tuple[str, ...]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SourceError(f"git {' '.join(args)} in {root}: {result.stderr.strip()}")
    return result.stdout


def _ls_files(root: Path, *args: str) -> list[str]:
    return [name for name in _git(root, "ls-files", "-z", *args).split("\0") if name]


def shipped_files(root: Path = REPO_ROOT) -> list[str]:
    gitlinks = {
        line.split("\t", 1)[1]
        for line in _git(root, "ls-files", "-s").splitlines()
        if line.startswith("160000 ")
    }
    names = set(_ls_files(root, "-c", "-o", "--exclude-standard")) - gitlinks
    frontend = root / REQ_FRONTEND
    if not (frontend / "Cargo.toml").is_file():
        raise SourceError(
            f"{REQ_FRONTEND} is not checked out; the Rust build needs it "
            f"(git submodule update --init {REQ_FRONTEND})"
        )
    names.update(
        f"{REQ_FRONTEND}/{name}" for name in _ls_files(frontend, "-c", "-o", "--exclude-standard")
    )
    for directory in DATA_DIRS:
        names.update(
            name
            for name in _ls_files(root, "-o", "-i", "--exclude-standard", "--", directory)
            if not any(marker in name for marker in _NOT_DATA)
        )
    names.update(name for name in IGNORED_INPUTS if (root / name).is_file())
    # `ls-files -c` still lists a tracked file deleted in the worktree, and
    # `-o` lists a nested repository as its directory.
    return sorted(name for name in names if _is_file_or_link(root / name))


def _is_file_or_link(path: Path) -> bool:
    return path.is_symlink() or path.is_file()


def _file_digest(path: Path) -> bytes:
    if path.is_symlink():
        return b"link\0" + os.readlink(path).encode()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    executable = b"x" if os.access(path, os.X_OK) else b"-"
    return executable + digest.digest()


def tree_version(root: Path, files: list[str], head: str) -> str:
    digest = hashlib.sha256(f"head\0{head}\0".encode())
    for name in files:
        digest.update(name.encode() + b"\0" + _file_digest(root / name))
    return digest.hexdigest()[:16]


def venv_key(root: Path, groups: tuple[str, ...]) -> str:
    digest = hashlib.sha256(("groups\0" + ",".join(sorted(groups)) + "\0").encode())
    for name in VENV_INPUTS:
        path = root / name
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"-")
    return digest.hexdigest()[:16]


def collect(root: Path = REPO_ROOT, groups: tuple[str, ...] = CPU_GROUPS) -> Source:
    files = shipped_files(root)
    head = _git(root, "rev-parse", "HEAD").strip()
    return Source(
        root=root,
        files=tuple(files),
        version=tree_version(root, files, head),
        venv_key=venv_key(root, groups),
        groups=groups,
    )


def source_stamp(source: Source) -> dict:
    """The stamp shipped at the tree's root (`launcher/source_stamp.py`)."""

    from profiling.db.table import _git_tree_stamp

    from ..metadata import GIT_SNAPSHOT_MAX_CHANGED_LINES_PER_FILE, _git_diff_capped

    root = source.root
    status = _git(root, "status", "--short").rstrip("\n")
    return {
        "schema_version": STAMP_SCHEMA_VERSION,
        "version": source.version,
        "venv_key": source.venv_key,
        "groups": list(source.groups),
        "head": _git(root, "rev-parse", "HEAD").strip(),
        "head_oneline": _git(root, "log", "-1", "--oneline").strip(),
        "dirty": bool(status),
        "status": status,
        "diff": _git_diff_capped(GIT_SNAPSHOT_MAX_CHANGED_LINES_PER_FILE),
        "profiler_git_hash": _git_tree_stamp(root),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
