from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

import profiling.db.provenance as provenance
from profiling.db.provenance import audit_profile_provenance


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo_with_kernel(
    tmp_path: Path, table: str, backends: list[str], *, source: str | None = None
) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "profiling" / "kernels").mkdir(parents=True)
    _git(repo.parent, "init", "-q", repo.name)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    body = source
    if body is None:
        registrations = "\n".join(
            f'register(KernelProfilerSpec(backend = "{name}"))' for name in backends
        )
        body = (
            f"from profiling.db.registry import KernelProfilerSpec, register\n\n{registrations}\n"
        )
    (repo / "profiling" / "kernels" / f"{table}.py").write_text(body)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "kernels")
    return repo, _git(repo, "rev-parse", "HEAD")


def _db(tmp_path: Path, table: str, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / "profile.db"
    quoted = '"' + table.replace('"', '""') + '"'
    with sqlite3.connect(path) as conn:
        conn.execute(
            f"""
            CREATE TABLE {quoted} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gpu_name TEXT NOT NULL,
                backend TEXT NOT NULL,
                profiler_git_hash TEXT NOT NULL,
                time_ms REAL NOT NULL
            )
            """
        )
        conn.executemany(
            f"INSERT INTO {quoted} (gpu_name, backend, profiler_git_hash, time_ms)"
            " VALUES ('NVIDIA B200', ?, ?, 1.0)",
            rows,
        )
    return path


def _stamps(path: Path, table: str) -> list[str]:
    quoted = '"' + table.replace('"', '""') + '"'
    with sqlite3.connect(path) as conn:
        return [row[0] for row in conn.execute(f"SELECT profiler_git_hash FROM {quoted}")]


def test_backend_not_directly_registered_is_a_read_only_candidate(tmp_path: Path) -> None:
    repo, head = _repo_with_kernel(tmp_path, "widget", ["torch"])
    db = _db(tmp_path, "widget", [("trtllm_fp8", head)])

    report = audit_profile_provenance(db, repo)

    assert report.candidate_rows == 1
    assert [finding.verdict for finding in report.findings] == ["backend_not_directly_registered"]
    assert _stamps(db, "widget") == [head]


def test_module_absent_at_the_stamped_commit_is_unverifiable(tmp_path: Path) -> None:
    repo, head = _repo_with_kernel(tmp_path, "widget", ["torch"])
    db = _db(tmp_path, "gadget", [("torch", head)])

    report = audit_profile_provenance(db, repo)

    assert report.candidate_rows == 0
    assert report.unverifiable_rows == 1
    assert [finding.verdict for finding in report.findings] == ["module_absent"]
    assert _stamps(db, "gadget") == [head]


def test_registered_backend_is_left_alone_even_with_spaced_syntax(tmp_path: Path) -> None:
    repo, head = _repo_with_kernel(tmp_path, "widget", ["torch", "trtllm_fp8"])
    db = _db(tmp_path, "widget", [("torch", head), ("trtllm_fp8", head)])

    report = audit_profile_provenance(db, repo)

    assert report.candidate_rows == 0
    assert report.findings == ()
    assert _stamps(db, "widget") == [head, head]


def test_unknown_commit_is_reported_and_database_stays_read_only(tmp_path: Path) -> None:
    repo, _ = _repo_with_kernel(tmp_path, "widget", ["torch"])
    foreign = "0" * 40
    db = _db(tmp_path, "widget", [("torch", foreign)])

    report = audit_profile_provenance(db, repo)

    assert report.candidate_rows == 0
    assert report.unverifiable_rows == 1
    assert [finding.verdict for finding in report.findings] == ["commit_unknown"]
    assert _stamps(db, "widget") == [foreign]


def test_unparseable_historical_module_is_not_called_dirty(tmp_path: Path) -> None:
    repo, head = _repo_with_kernel(tmp_path, "widget", [], source="not valid python !\n")
    db = _db(tmp_path, "widget", [("torch", head)])

    report = audit_profile_provenance(db, repo)

    assert report.candidate_rows == 0
    assert report.unverifiable_rows == 1
    assert [finding.verdict for finding in report.findings] == ["module_unreadable"]
    assert _stamps(db, "widget") == [head]


def test_dynamic_backend_registration_is_not_called_absent(tmp_path: Path) -> None:
    source = """
from profiling.db.registry import KernelProfilerSpec, register

register(KernelProfilerSpec(backend="literal"))
for name in ("dynamic",):
    register(KernelProfilerSpec(backend=name))
"""
    repo, head = _repo_with_kernel(tmp_path, "widget", [], source=source)
    db = _db(tmp_path, "widget", [("dynamic", head)])

    report = audit_profile_provenance(db, repo)

    assert report.candidate_rows == 0
    assert report.unverifiable_rows == 1
    assert [finding.verdict for finding in report.findings] == ["module_unreadable"]
    assert _stamps(db, "widget") == [head]


@pytest.mark.parametrize(
    "opaque_registration",
    [
        'register(build_spec("custom"))',
        'register(KernelProfilerSpec(**{"backend": "custom"}))',
        "register(IMPORTED_SPEC)",
        "alias = register\nalias(IMPORTED_SPEC)",
        (
            "import profiling.db.registry as registry\n"
            'registry.register(registry.KernelProfilerSpec(backend="custom"))'
        ),
        ("from profiling.db.registry import register as indirect\nindirect(IMPORTED_SPEC)"),
        "from helper import *",
    ],
)
def test_opaque_registration_makes_whole_module_unverifiable(
    tmp_path: Path, opaque_registration: str
) -> None:
    source = f"""
from profiling.db.registry import KernelProfilerSpec, register

register(KernelProfilerSpec(backend="literal"))
{opaque_registration}
"""
    repo, head = _repo_with_kernel(tmp_path, "widget", [], source=source)
    db = _db(tmp_path, "widget", [("custom", head)])

    report = audit_profile_provenance(db, repo)

    assert report.candidate_rows == 0
    assert report.unverifiable_rows == 1
    assert [finding.verdict for finding in report.findings] == ["module_unreadable"]
    assert _stamps(db, "widget") == [head]


def test_helper_side_effect_is_only_a_read_only_candidate(tmp_path: Path) -> None:
    source = """
from profiling.db.registry import KernelProfilerSpec, register
from helper import register_custom

register(KernelProfilerSpec(backend="torch"))
register_custom()
"""
    repo, head = _repo_with_kernel(tmp_path, "widget", [], source=source)
    db = _db(tmp_path, "widget", [("custom", head)])

    report = audit_profile_provenance(db, repo)

    assert report.candidate_rows == 1
    assert [finding.verdict for finding in report.findings] == ["backend_not_directly_registered"]
    assert _stamps(db, "widget") == [head]


def test_git_show_failure_is_unverifiable_and_database_stays_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, head = _repo_with_kernel(tmp_path, "widget", ["torch"])
    db = _db(tmp_path, "widget", [("other", head)])
    real_run = subprocess.run

    def fail_show(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = args[0]
        if isinstance(command, list) and len(command) > 1 and command[1] == "show":
            return subprocess.CompletedProcess(command, 128, "", "simulated git failure")
        return real_run(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(provenance.subprocess, "run", fail_show)

    report = audit_profile_provenance(db, repo)

    assert report.candidate_rows == 0
    assert report.unverifiable_rows == 1
    assert [finding.verdict for finding in report.findings] == ["module_unreadable"]
    assert _stamps(db, "widget") == [head]


def test_already_dirty_and_unknown_stamps_are_not_changed(tmp_path: Path) -> None:
    repo, head = _repo_with_kernel(tmp_path, "widget", ["torch"])
    db = _db(tmp_path, "widget", [("other", f"{head}-dirty"), ("torch", "unknown")])

    report = audit_profile_provenance(db, repo)

    assert report.findings == ()
    assert sorted(_stamps(db, "widget")) == sorted([f"{head}-dirty", "unknown"])


def test_audit_is_always_read_only(tmp_path: Path) -> None:
    repo, head = _repo_with_kernel(tmp_path, "widget", ["torch"])
    db = _db(tmp_path, "widget", [("other", head)])

    report = audit_profile_provenance(db, repo)

    assert report.candidate_rows == 1
    assert _stamps(db, "widget") == [head]


def test_repeated_audits_are_deterministic_and_read_only(tmp_path: Path) -> None:
    repo, head = _repo_with_kernel(tmp_path, "widget", ["torch"])
    db = _db(tmp_path, "widget", [("other", head)])

    first = audit_profile_provenance(db, repo)
    second = audit_profile_provenance(db, repo)

    assert first == second
    assert second.candidate_rows == 1
    assert _stamps(db, "widget") == [head]


def test_missing_database_is_an_error(tmp_path: Path) -> None:
    repo, _ = _repo_with_kernel(tmp_path, "widget", ["torch"])

    with pytest.raises(FileNotFoundError, match="profile DB not found"):
        audit_profile_provenance(tmp_path / "absent.db", repo)


def test_cli_json_audit_is_read_only(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from profiling.cli import main

    _repo, head = _repo_with_kernel(tmp_path, "widget", ["torch"])
    db = _db(tmp_path, "widget", [("other", head)])

    # The CLI audits against the real checkout. This fixture commit is foreign
    # there, but the important operator contract is JSON success without a write.
    assert main(["audit-provenance", "--db", str(db), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["candidate_rows"] == 0
    assert _stamps(db, "widget") == [head]
