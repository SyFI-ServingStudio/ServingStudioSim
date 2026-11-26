"""Split a single-file profile DB into the per-kind directory (``profiling.db.layout``).

Each non-empty kind table becomes ``<output>/<kind>.db`` holding:

- the table exactly as the legacy file declares it (its ``CREATE`` SQL and any
  index), with every row, ``id`` included;
- the ``_profile_run`` rows those rows reference (``run_key``);
- the file's ``_db_metadata``.

Nothing is repaired on the way. A row whose ``run_key`` has no ``_profile_run``
row stays as it is and is listed in the report (``orphan_run_keys``); an empty
table gets no file; a run no row references is not copied. Before publishing,
every output file is compared with its source table: the same rows, column for
column, and the same referenced runs.

The output directory is built beside its final path and renamed into place, so
a failed split leaves nothing behind.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from profiling.db import layout
from profiling.db.migrate import RETIRED_TABLES, require_current
from profiling.db.storage import RUN_TABLE, quote

_METADATA_TABLE = "_db_metadata"


@dataclass(frozen=True)
class SplitTable:
    table: str
    rows: int
    runs: int
    # run_keys the table's rows reference that the legacy file has no run for.
    orphan_run_keys: tuple[int, ...] = ()
    orphan_rows: int = 0


@dataclass(frozen=True)
class SplitReport:
    source: str
    output: str
    tables: tuple[SplitTable, ...]
    empty_tables: tuple[str, ...]
    unreferenced_runs: int
    summary: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_profile_db(source: Path | str, output: Path | str) -> SplitReport:
    """Split the legacy single-file DB ``source`` into the directory ``output``."""
    source = Path(source)
    output = Path(output)
    if not layout.is_legacy_file(source):
        raise ValueError(f"{source} is not a single-file profile DB")
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as src:
        src.execute("PRAGMA query_only = ON")
        require_current(src, str(source))
        tables = _tables(src)
        retired = sorted(set(tables) & set(RETIRED_TABLES))
        if retired:
            raise ValueError(
                f"{source} still holds retired tables {retired}; run "
                f"`python -m launcher kernel-profile migrate-db {source}` first"
            )
        kinds = [name for name in tables if not name.startswith("_")]
        all_runs = (
            {int(row[0]) for row in src.execute(f"SELECT run_key FROM {RUN_TABLE}")}
            if RUN_TABLE in tables
            else set()
        )

        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.split-", dir=output.parent))
        try:
            split: list[SplitTable] = []
            empty: list[str] = []
            referenced: set[int] = set()
            for kind in kinds:
                rows = src.execute(f"SELECT count(*) FROM {quote(kind)}").fetchone()[0]
                if rows == 0:
                    empty.append(kind)
                    continue
                stats = _write_kind(src, source, kind, layout.kind_file(staging, kind), all_runs)
                referenced.update(stats.pop("run_keys"))
                split.append(SplitTable(table=kind, rows=rows, **stats))
            staging.rename(output)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    orphan_rows = sum(table.orphan_rows for table in split)
    return SplitReport(
        source=str(source),
        output=str(output),
        tables=tuple(split),
        empty_tables=tuple(empty),
        unreferenced_runs=len(all_runs - referenced),
        summary={
            "files": len(split),
            "rows": sum(table.rows for table in split),
            "empty_tables": len(empty),
            "orphan_rows": orphan_rows,
            "orphan_run_keys": len({key for table in split for key in table.orphan_run_keys}),
            "unreferenced_runs": len(all_runs - referenced),
        },
    )


def _write_kind(
    src: sqlite3.Connection, source: Path, kind: str, path: Path, all_runs: set[int]
) -> dict[str, Any]:
    """Write ``kind``'s file and check it against the source; its stats and run keys."""
    # A URI connection, so the read-only ATTACH below may name its file by URI.
    with closing(sqlite3.connect(path.resolve().as_uri(), uri=True)) as out:
        out.execute("ATTACH DATABASE ? AS src", (f"{source.resolve().as_uri()}?mode=ro",))
        tables = set(_tables(src))
        for name in (_METADATA_TABLE, RUN_TABLE, kind):
            if name not in tables:
                continue
            out.execute(_create_sql(src, name))
        for (index_sql,) in src.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND tbl_name = ? "
            "AND sql IS NOT NULL",
            (kind,),
        ):
            out.execute(index_sql)
        if _METADATA_TABLE in tables:
            out.execute(f"INSERT INTO main.{_METADATA_TABLE} SELECT * FROM src.{_METADATA_TABLE}")
        out.execute(f"INSERT INTO main.{quote(kind)} SELECT * FROM src.{quote(kind)}")
        run_keys = {
            int(row[0])
            for row in out.execute(f"SELECT DISTINCT run_key FROM main.{quote(kind)}")
            if row[0] is not None
        }
        if RUN_TABLE in tables:
            out.execute(
                f"INSERT INTO main.{RUN_TABLE} SELECT * FROM src.{RUN_TABLE} "
                f"WHERE run_key IN (SELECT DISTINCT run_key FROM main.{quote(kind)})"
            )
        out.commit()
        _check_same(out, kind)
        orphans = sorted(run_keys - all_runs)
        orphan_rows = (
            out.execute(
                f"SELECT count(*) FROM main.{quote(kind)} WHERE run_key IN "
                f"({','.join('?' * len(orphans))})",
                orphans,
            ).fetchone()[0]
            if orphans
            else 0
        )
        runs = (
            out.execute(f"SELECT count(*) FROM main.{RUN_TABLE}").fetchone()[0]
            if (RUN_TABLE in tables)
            else 0
        )
        out.execute("DETACH DATABASE src")
        out.execute("VACUUM")
    return {
        "runs": runs,
        "orphan_run_keys": tuple(orphans),
        "orphan_rows": orphan_rows,
        "run_keys": run_keys,
    }


def _check_same(out: sqlite3.Connection, kind: str) -> None:
    """Refuse a kind file whose rows or referenced runs differ from the source's."""
    checks = [
        (f"main.{quote(kind)}", f"src.{quote(kind)}"),
    ]
    tables = {
        row[0] for row in out.execute("SELECT name FROM src.sqlite_master WHERE type='table'")
    }
    if RUN_TABLE in tables:
        checks.append(
            (
                f"main.{RUN_TABLE}",
                f"(SELECT * FROM src.{RUN_TABLE} WHERE run_key IN "
                f"(SELECT run_key FROM src.{quote(kind)}))",
            )
        )
    for ours, theirs in checks:
        for left, right in ((ours, theirs), (theirs, ours)):
            missing = out.execute(
                f"SELECT count(*) FROM (SELECT * FROM {left} EXCEPT SELECT * FROM {right})"
            ).fetchone()[0]
            if missing:
                raise RuntimeError(f"split of {kind}: {missing} rows of {left} are not in {right}")
    counts = [
        out.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in (f"main.{quote(kind)}", f"src.{quote(kind)}")
    ]
    if counts[0] != counts[1]:
        raise RuntimeError(f"split of {kind}: {counts[0]} rows written, {counts[1]} in the source")


def _create_sql(conn: sqlite3.Connection, table: str) -> str:
    (sql,) = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return str(sql)


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]
