"""Audit reproducibility stamps already stored in a profile database.

Profile rows historically recorded bare ``HEAD`` even when their profiler code
was uncommitted. This module reports rows whose backend is not among the direct
literal registrations visible in the historical per-kind module. Unknown
commits, missing historical paths, and modules with opaque registration syntax
are reported separately.

This is deliberately a read-only diagnostic. Import and helper calls are
arbitrary Python and can register additional backends as side effects, so a
static scan cannot safely prove that a historical stamp is false. New writes
are protected at their source by ``table._git_tree_stamp`` instead.
"""

from __future__ import annotations

import ast
import re
import sqlite3
import subprocess
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DIRTY_SUFFIX = "-dirty"
_UNKNOWN_HASH = "unknown"
_COMMIT_RE = re.compile(r"[0-9a-fA-F]{7,64}\Z")
_TABLE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_CANDIDATE_VERDICTS = frozenset({"backend_not_directly_registered"})
_UNVERIFIABLE_VERDICTS = frozenset({"commit_unknown", "module_absent", "module_unreadable"})


@dataclass(frozen=True)
class ProvenanceFinding:
    """One grouped diagnostic candidate or unverifiable stamp."""

    table: str
    backend: str
    profiler_git_hash: str
    rows: int
    verdict: str


@dataclass(frozen=True)
class ProvenanceReport:
    total_rows: int
    candidate_rows: int
    unverifiable_rows: int
    findings: tuple[ProvenanceFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["findings"] = [asdict(finding) for finding in self.findings]
        return payload


@dataclass(frozen=True)
class _HistoricalModule:
    status: str  # "present" | "absent" | "unreadable"
    source: str | None = None


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _tables_with_provenance(conn: sqlite3.Connection) -> list[str]:
    names = [
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        if not str(row[0]).startswith("_")
    ]
    tables: list[str] = []
    for name in names:
        columns = {
            str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote_identifier(name)})")
        }
        if {"backend", "profiler_git_hash"} <= columns:
            tables.append(name)
    return sorted(tables)


def _commit_exists(repo_root: Path, commit: str) -> bool:
    # Database contents must not become git command-line options or revisions.
    if _COMMIT_RE.fullmatch(commit) is None:
        return False
    result = subprocess.run(
        ["git", "cat-file", "-t", commit],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "commit"


def _kernel_module_at(repo_root: Path, commit: str, table: str) -> _HistoricalModule:
    """Read a historical module, distinguishing absence from Git failure."""
    module_path = f"profiling/kernels/{table}.py"
    lookup = subprocess.run(
        ["git", "ls-tree", "--name-only", commit, "--", module_path],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if lookup.returncode != 0:
        return _HistoricalModule("unreadable")
    if module_path not in lookup.stdout.splitlines():
        return _HistoricalModule("absent")

    result = subprocess.run(
        ["git", "show", f"{commit}:profiling/kernels/{table}.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return _HistoricalModule("unreadable")
    return _HistoricalModule("present", result.stdout)


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.List, ast.Tuple)):
        return {name for element in target.elts for name in _target_names(element)}
    return set()


def _assigned_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        for target in targets:
            names.update(_target_names(target))
    return names


def _has_exact_registry_imports(tree: ast.AST) -> bool:
    """Require unaliased direct bindings for both audited call names."""
    required = {"register", "KernelProfilerSpec"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "profiling.db.registry":
                    return False
                bound = alias.asname or alias.name.split(".", maxsplit=1)[0]
                if bound in required:
                    return False
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    return False
                bound = alias.asname or alias.name
                if alias.name in required:
                    if (
                        node.module != "profiling.db.registry"
                        or alias.asname is not None
                        or alias.name in imported
                    ):
                        return False
                    imported.add(alias.name)
                elif bound in required:
                    return False
    return imported == required


def _backend_literals(source: str) -> set[str] | None:
    """Extract direct literal registrations when their syntax is auditable.

    An imported spec, ``**kwargs``, dynamic backend, registry alias/rebinding, or
    unattached ``KernelProfilerSpec`` makes the direct-registration syntax
    unreadable. Unrelated helper calls can still have arbitrary side effects,
    which is why a missing literal becomes only a read-only candidate.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if not _has_exact_registry_imports(tree):
        return None
    if {"register", "KernelProfilerSpec"} & _assigned_names(tree):
        return None

    register_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "register"
    ]
    spec_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "KernelProfilerSpec"
    ]
    if not register_calls:
        return None

    # Every reference to the registry primitives must belong to the exact
    # direct calls audited below. Attribute calls and aliases could otherwise
    # hide additional registrations and make absence impossible to prove.
    allowed_name_nodes = {id(call.func) for call in [*register_calls, *spec_calls]}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {
            "register",
            "KernelProfilerSpec",
        }:
            return None
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in {"register", "KernelProfilerSpec"}
            and id(node) not in allowed_name_nodes
        ):
            return None

    registered_spec_ids: set[int] = set()
    backends: set[str] = set()
    for call in register_calls:
        if len(call.args) != 1 or call.keywords:
            return None
        spec = call.args[0]
        if (
            not isinstance(spec, ast.Call)
            or not isinstance(spec.func, ast.Name)
            or spec.func.id != "KernelProfilerSpec"
        ):
            return None
        if any(keyword.arg is None for keyword in spec.keywords):
            return None
        backend_keywords = [keyword for keyword in spec.keywords if keyword.arg == "backend"]
        if len(backend_keywords) != 1:
            return None
        value = backend_keywords[0].value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return None
        registered_spec_ids.add(id(spec))
        backends.add(value.value)

    if {id(call) for call in spec_calls} != registered_spec_ids:
        return None
    return backends


def _classify(
    repo_root: Path,
    table: str,
    backend: str,
    commit: str,
    module_cache: dict[tuple[str, str], _HistoricalModule],
    commit_cache: dict[str, bool],
) -> str:
    if commit not in commit_cache:
        commit_cache[commit] = _commit_exists(repo_root, commit)
    if not commit_cache[commit]:
        return "commit_unknown"
    if _TABLE_RE.fullmatch(table) is None:
        return "module_unreadable"

    key = (table, commit)
    if key not in module_cache:
        module_cache[key] = _kernel_module_at(repo_root, commit, table)
    module = module_cache[key]
    if module.status == "absent":
        return "module_absent"
    if module.status != "present" or module.source is None:
        return "module_unreadable"
    registered = _backend_literals(module.source)
    if registered is None:
        return "module_unreadable"
    return "consistent" if backend in registered else "backend_not_directly_registered"


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=30.0)


def audit_profile_provenance(
    db_path: Path,
    repo_root: Path,
) -> ProvenanceReport:
    """Classify stamped rows without modifying the profile database."""
    if not db_path.is_file():
        raise FileNotFoundError(f"profile DB not found: {db_path}")

    module_cache: dict[tuple[str, str], _HistoricalModule] = {}
    commit_cache: dict[str, bool] = {}
    findings: list[ProvenanceFinding] = []
    total_rows = 0
    candidate_rows = 0
    unverifiable_rows = 0

    with closing(_connect(db_path)) as conn:
        for table in _tables_with_provenance(conn):
            quoted_table = _quote_identifier(table)
            groups = conn.execute(
                f"""
                SELECT backend, profiler_git_hash, COUNT(*)
                FROM {quoted_table}
                WHERE profiler_git_hash IS NOT NULL AND profiler_git_hash != ''
                GROUP BY backend, profiler_git_hash
                """
            ).fetchall()
            for backend_value, stamp_value, row_count_value in groups:
                backend = str(backend_value)
                stamp = str(stamp_value)
                row_count = int(row_count_value)
                total_rows += row_count
                if stamp.endswith(_DIRTY_SUFFIX) or stamp == _UNKNOWN_HASH:
                    continue

                verdict = _classify(
                    repo_root,
                    table,
                    backend,
                    stamp,
                    module_cache,
                    commit_cache,
                )
                if verdict == "consistent":
                    continue
                findings.append(
                    ProvenanceFinding(
                        table=table,
                        backend=backend,
                        profiler_git_hash=stamp,
                        rows=row_count,
                        verdict=verdict,
                    )
                )
                if verdict in _CANDIDATE_VERDICTS:
                    candidate_rows += row_count
                elif verdict in _UNVERIFIABLE_VERDICTS:
                    unverifiable_rows += row_count
                else:  # Guard report semantics when a classifier verdict is added.
                    raise AssertionError(f"unhandled provenance verdict: {verdict}")
    findings.sort(
        key=lambda finding: (
            -finding.rows,
            finding.table,
            finding.backend,
            finding.profiler_git_hash,
        )
    )
    return ProvenanceReport(
        total_rows=total_rows,
        candidate_rows=candidate_rows,
        unverifiable_rows=unverifiable_rows,
        findings=tuple(findings),
    )
