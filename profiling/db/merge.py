"""Semantic merge support for L1 profile databases.

Profile databases are generated caches, but their measured rows and provenance
must survive branch integration.  This module merges rows by each table's
declared semantic unique key instead of resolving the SQLite file as a binary
git conflict.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from profiling.db.migrate import SCHEMA_HASH, SCHEMA_VERSION

_METADATA_TABLE = "_db_metadata"
_IGNORED_EQUALITY_COLUMNS = frozenset({"created_at"})


@dataclass(frozen=True)
class TableMergeStats:
    table: str
    left_rows: int
    right_rows: int
    inserted_rows: int
    duplicate_rows: int
    conflict_rows: int


@dataclass(frozen=True)
class MergeConflict:
    table: str
    key: dict[str, Any]
    differing_columns: tuple[str, ...]
    left: dict[str, Any]
    right: dict[str, Any]


@dataclass(frozen=True)
class ProfileDbMergeReport:
    left: dict[str, Any]
    right: dict[str, Any]
    output: dict[str, Any]
    tables: tuple[TableMergeStats, ...]
    conflicts: tuple[MergeConflict, ...]
    report_schema_version: int = 1

    @property
    def published(self) -> bool:
        return bool(self.output["published"])

    def to_dict(self) -> dict[str, Any]:
        table_rows = [asdict(table) for table in self.tables]
        conflict_rows = [asdict(conflict) for conflict in self.conflicts]
        return {
            "report_schema_version": self.report_schema_version,
            "left": self.left,
            "right": self.right,
            "output": self.output,
            "summary": {
                "table_count": len(table_rows),
                "left_rows": sum(table["left_rows"] for table in table_rows),
                "right_rows": sum(table["right_rows"] for table in table_rows),
                "inserted_rows": sum(table["inserted_rows"] for table in table_rows),
                "duplicate_rows": sum(table["duplicate_rows"] for table in table_rows),
                "conflict_rows": len(conflict_rows),
            },
            "tables": table_rows,
            "conflicts": conflict_rows,
        }


@dataclass(frozen=True)
class _TableContract:
    name: str
    columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    semantic_key: tuple[str, ...]
    column_schema: tuple[tuple[Any, ...], ...]

    @property
    def insert_columns(self) -> tuple[str, ...]:
        return tuple(column for column in self.columns if column not in self.primary_key)

    @property
    def equality_columns(self) -> tuple[str, ...]:
        return tuple(
            column for column in self.insert_columns if column not in _IGNORED_EQUALITY_COLUMNS
        )


def merge_profile_databases(
    left_path: Path | str,
    right_path: Path | str,
    output_path: Path | str,
    *,
    report_path: Path | str | None = None,
) -> ProfileDbMergeReport:
    """Merge two profile DBs without modifying either input.

    The output is published atomically only when database schemas agree and no
    semantic key maps to differing measurement or provenance data.  Conflicts
    are returned and written to the report sidecar for explicit resolution.
    """

    left = _existing_file(left_path, "left input")
    right = _existing_file(right_path, "right input")
    output = Path(output_path).expanduser().resolve()
    report = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else output.with_name(f"{output.name}.merge-report.json")
    )
    _validate_distinct_paths(left, right, output, report)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)

    try:
        with (
            closing(_connect_read_only(left)) as left_conn,
            closing(_connect_read_only(right)) as right_conn,
        ):
            _validate_database(left_conn, left)
            _validate_database(right_conn, right)
            left_metadata = _read_metadata(left_conn, left)
            right_metadata = _read_metadata(right_conn, right)
            _validate_compatible_metadata(left_metadata, right_metadata)

            with closing(sqlite3.connect(temp_path)) as output_conn:
                output_conn.row_factory = sqlite3.Row
                left_conn.backup(output_conn)
                table_stats, conflicts = _merge_tables(left_conn, right_conn, output_conn)
                output_conn.commit()
                _validate_quick_check(output_conn, temp_path)

        published = not conflicts
        output_sha256 = None
        if published:
            os.link(temp_path, output)
            output_sha256 = _sha256(output)

        merge_report = ProfileDbMergeReport(
            left=_input_description(left, left_metadata),
            right=_input_description(right, right_metadata),
            output={
                "path": str(output),
                "published": published,
                "sha256": output_sha256,
            },
            tables=tuple(table_stats),
            conflicts=tuple(conflicts),
        )
        _write_json_atomically(report, merge_report.to_dict())
        return merge_report
    finally:
        temp_path.unlink(missing_ok=True)


def _merge_tables(
    left_conn: sqlite3.Connection,
    right_conn: sqlite3.Connection,
    output_conn: sqlite3.Connection,
) -> tuple[list[TableMergeStats], list[MergeConflict]]:
    left_tables = _profile_tables(left_conn)
    right_tables = _profile_tables(right_conn)
    stats: list[TableMergeStats] = []
    conflicts: list[MergeConflict] = []

    for table_name in sorted(left_tables | right_tables):
        if table_name not in left_tables:
            stats.append(_copy_right_only_table(right_conn, output_conn, table_name))
            continue
        if table_name not in right_tables:
            contract = _table_contract(left_conn, table_name)
            stats.append(
                TableMergeStats(
                    table=table_name,
                    left_rows=_row_count(left_conn, contract),
                    right_rows=0,
                    inserted_rows=0,
                    duplicate_rows=0,
                    conflict_rows=0,
                )
            )
            continue

        left_contract = _table_contract(left_conn, table_name)
        right_contract = _table_contract(right_conn, table_name)
        _validate_matching_contracts(left_contract, right_contract)
        table_stats, table_conflicts = _merge_common_table(right_conn, output_conn, left_contract)
        stats.append(table_stats)
        conflicts.extend(table_conflicts)

    return stats, conflicts


def _merge_common_table(
    right_conn: sqlite3.Connection,
    output_conn: sqlite3.Connection,
    contract: _TableContract,
) -> tuple[TableMergeStats, list[MergeConflict]]:
    table = _quote_identifier(contract.name)
    order_by = ", ".join(_quote_identifier(column) for column in contract.semantic_key)
    right_count = _row_count(right_conn, contract)
    left_count = _row_count(output_conn, contract)
    inserted = 0
    duplicates = 0
    conflicts: list[MergeConflict] = []

    for right_row in right_conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}"):
        existing = _find_by_key(output_conn, contract, right_row)
        if existing is None:
            _insert_row(output_conn, contract, right_row)
            inserted += 1
            continue

        differing = tuple(
            column for column in contract.equality_columns if existing[column] != right_row[column]
        )
        if not differing:
            duplicates += 1
            continue
        conflicts.append(
            MergeConflict(
                table=contract.name,
                key={column: _json_value(right_row[column]) for column in contract.semantic_key},
                differing_columns=differing,
                left=_report_row(existing, contract),
                right=_report_row(right_row, contract),
            )
        )

    return (
        TableMergeStats(
            table=contract.name,
            left_rows=left_count,
            right_rows=right_count,
            inserted_rows=inserted,
            duplicate_rows=duplicates,
            conflict_rows=len(conflicts),
        ),
        conflicts,
    )


def _copy_right_only_table(
    right_conn: sqlite3.Connection,
    output_conn: sqlite3.Connection,
    table_name: str,
) -> TableMergeStats:
    contract = _table_contract(right_conn, table_name)
    schema_row = right_conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    ).fetchone()
    if schema_row is None or schema_row["sql"] is None:
        raise ValueError(f"cannot copy table {table_name!r}: CREATE TABLE SQL is unavailable")
    output_conn.execute(schema_row["sql"])

    table = _quote_identifier(table_name)
    order_by = ", ".join(_quote_identifier(column) for column in contract.semantic_key)
    right_count = _row_count(right_conn, contract)
    for row in right_conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}"):
        _insert_row(output_conn, contract, row)

    index_rows = right_conn.execute(
        """
        SELECT name, sql
        FROM sqlite_master
        WHERE type = 'index' AND tbl_name = ? AND sql IS NOT NULL
        ORDER BY name
        """,
        (table_name,),
    ).fetchall()
    for index_row in index_rows:
        output_conn.execute(index_row["sql"])

    return TableMergeStats(
        table=table_name,
        left_rows=0,
        right_rows=right_count,
        inserted_rows=right_count,
        duplicate_rows=0,
        conflict_rows=0,
    )


def _table_contract(conn: sqlite3.Connection, table_name: str) -> _TableContract:
    table = _quote_identifier(table_name)
    rows = conn.execute(f"PRAGMA table_xinfo({table})").fetchall()
    if not rows:
        raise ValueError(f"profile table {table_name!r} has no columns")

    columns = tuple(str(row["name"]) for row in rows)
    primary_key = tuple(
        str(row["name"])
        for row in sorted(rows, key=lambda row: int(row["pk"]))
        if int(row["pk"]) > 0
    )
    if primary_key != ("id",):
        raise ValueError(
            f"profile table {table_name!r} must have the standard surrogate primary key 'id'"
        )

    semantic_key = _semantic_key(conn, table_name)
    column_by_name = {str(row["name"]): row for row in rows}
    nullable_key_columns = [
        column for column in semantic_key if not int(column_by_name[column]["notnull"])
    ]
    if nullable_key_columns:
        raise ValueError(
            f"profile table {table_name!r} has nullable semantic key columns: "
            f"{', '.join(nullable_key_columns)}"
        )

    column_schema = tuple(
        (
            str(row["name"]),
            str(row["type"]),
            int(row["notnull"]),
            row["dflt_value"],
            int(row["pk"]),
            int(row["hidden"]),
        )
        for row in rows
    )
    return _TableContract(
        name=table_name,
        columns=columns,
        primary_key=primary_key,
        semantic_key=semantic_key,
        column_schema=column_schema,
    )


def _semantic_key(conn: sqlite3.Connection, table_name: str) -> tuple[str, ...]:
    table = _quote_identifier(table_name)
    candidates: list[tuple[str, ...]] = []
    for index_row in conn.execute(f"PRAGMA index_list({table})").fetchall():
        if not int(index_row["unique"]) or int(index_row["partial"]):
            continue
        index = _quote_identifier(str(index_row["name"]))
        columns = tuple(
            str(row["name"])
            for row in conn.execute(f"PRAGMA index_info({index})").fetchall()
            if row["name"] is not None
        )
        if columns[:2] == ("gpu_name", "backend"):
            candidates.append(columns)
    if len(candidates) != 1:
        raise ValueError(
            f"profile table {table_name!r} must declare exactly one unique semantic key "
            "beginning with (gpu_name, backend)"
        )
    return candidates[0]


def _validate_matching_contracts(left: _TableContract, right: _TableContract) -> None:
    if left.column_schema != right.column_schema:
        raise ValueError(f"profile table {left.name!r} has incompatible column schemas")
    if left.semantic_key != right.semantic_key:
        raise ValueError(
            f"profile table {left.name!r} has incompatible semantic keys: "
            f"{left.semantic_key!r} != {right.semantic_key!r}"
        )


def _find_by_key(
    conn: sqlite3.Connection, contract: _TableContract, row: sqlite3.Row
) -> sqlite3.Row | None:
    predicates = " AND ".join(
        f"{_quote_identifier(column)} = ?" for column in contract.semantic_key
    )
    values = tuple(row[column] for column in contract.semantic_key)
    return conn.execute(
        f"SELECT * FROM {_quote_identifier(contract.name)} WHERE {predicates}", values
    ).fetchone()


def _insert_row(conn: sqlite3.Connection, contract: _TableContract, row: sqlite3.Row) -> None:
    columns = contract.insert_columns
    column_sql = ", ".join(_quote_identifier(column) for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(
        f"INSERT INTO {_quote_identifier(contract.name)} ({column_sql}) VALUES ({placeholders})",
        tuple(row[column] for column in columns),
    )


def _report_row(row: sqlite3.Row, contract: _TableContract) -> dict[str, Any]:
    return {column: _json_value(row[column]) for column in contract.insert_columns}


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    return value


def _profile_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name != ?
        """,
        (_METADATA_TABLE,),
    ).fetchall()
    return {str(row["name"]) for row in rows}


def _row_count(conn: sqlite3.Connection, contract: _TableContract) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) AS row_count FROM {_quote_identifier(contract.name)}"
    ).fetchone()
    return int(row["row_count"])


def _read_metadata(conn: sqlite3.Connection, path: Path) -> dict[str, str]:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (_METADATA_TABLE,),
    ).fetchone()
    if exists is None:
        raise ValueError(f"profile DB has no {_METADATA_TABLE} table: {path}")
    return {
        str(row["key"]): str(row["value"])
        for row in conn.execute(f"SELECT key, value FROM {_METADATA_TABLE}").fetchall()
    }


def _validate_compatible_metadata(
    left_metadata: dict[str, str], right_metadata: dict[str, str]
) -> None:
    expected = {
        "schema_version": str(SCHEMA_VERSION),
        "schema_hash": SCHEMA_HASH,
    }
    for side, metadata in (("left", left_metadata), ("right", right_metadata)):
        actual = {key: metadata.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"{side} profile DB schema metadata is incompatible: "
                f"expected {expected!r}, got {actual!r}"
            )


def _validate_database(conn: sqlite3.Connection, path: Path) -> None:
    conn.execute("PRAGMA query_only = ON")
    _validate_quick_check(conn, path)


def _validate_quick_check(conn: sqlite3.Connection, path: Path) -> None:
    rows = conn.execute("PRAGMA quick_check").fetchall()
    results = [str(row[0]) for row in rows]
    if results != ["ok"]:
        raise ValueError(f"SQLite quick_check failed for {path}: {results!r}")


def _connect_read_only(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _existing_file(path: Path | str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _validate_distinct_paths(left: Path, right: Path, output: Path, report: Path) -> None:
    paths = {left, right, output, report}
    if len(paths) != 4:
        raise ValueError("left input, right input, output, and report paths must be distinct")


def _input_description(path: Path, metadata: dict[str, str]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "metadata": dict(sorted(metadata.items())),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
