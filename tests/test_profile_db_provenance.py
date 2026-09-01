from __future__ import annotations

import subprocess
from pathlib import Path

from profiling.db.table import _git_tree_stamp


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "profiling" / "runners").mkdir(parents=True)
    _git(repo.parent, "init", "-q", repo.name)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    (repo / "profiling" / "runners" / "kernel.py").write_text("TIME_MS = 1.0\n")
    (repo / "README.md").write_text("unrelated\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial")
    return repo


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_clean_profiling_tree_stamps_the_bare_commit(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)

    assert _git_tree_stamp(repo) == _head(repo)


def test_modified_runner_is_stamped_dirty(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "profiling" / "runners" / "kernel.py").write_text("TIME_MS = 2.0\n")

    assert _git_tree_stamp(repo) == f"{_head(repo)}-dirty"


def test_staged_runner_is_stamped_dirty(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "profiling" / "runners" / "kernel.py").write_text("TIME_MS = 2.0\n")
    _git(repo, "add", "profiling/runners/kernel.py")

    assert _git_tree_stamp(repo) == f"{_head(repo)}-dirty"


def test_untracked_runner_is_stamped_dirty(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "profiling" / "runners" / "new_backend.py").write_text("TIME_MS = 3.0\n")

    assert _git_tree_stamp(repo) == f"{_head(repo)}-dirty"


def test_edit_outside_profiling_does_not_mark_rows_dirty(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("edited, cannot affect a profile number\n")

    assert _git_tree_stamp(repo) == _head(repo)


def test_profile_database_outputs_do_not_mark_profiler_code_dirty(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    database = repo / "profiling" / "profile.db"
    database.write_bytes(b"old database")
    _git(repo, "add", "profiling/profile.db")
    _git(repo, "commit", "-qm", "profile data")

    database.write_bytes(b"new measurements")
    (repo / "profiling" / "scratch.db-wal").write_bytes(b"sqlite sidecar")

    assert _git_tree_stamp(repo) == _head(repo)


def test_outside_a_git_repo_reports_unknown(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-repo"
    plain.mkdir()

    assert _git_tree_stamp(plain) == "unknown"
