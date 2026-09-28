"""Kernel-config registry: which simulator kernel configs asked for which rows.

A kind table stores measured rows keyed by ``(gpu_name, backend, args)``. The
args are what the simulator's ``KernelSpec::enumerate`` produced for one grid
cell of one kernel config, but nothing in the row says which config or which
cell. That mapping only runs forward (config -> grid -> args), so it cannot be
recovered from the rows. This registry keeps it, written by the same builds
that ask for the rows.

Keyed by content hashes so that ``profiling.db.merge`` can merge them between
databases the way it merges kind tables (``profiling.db.storage`` has the
stored layout: the use table names configs, sources and roles by content key,
and large identity arrays live once in ``_kernel_config_blob``):

- ``_kernel_config``: one config's grid on one GPU. ``identity`` is the Rust
  ``KernelConfig::identity`` (no ``gpu_name`` or ``backends``; those are row
  columns). ``cells`` holds, row-major over ``grid_axes``, the args of each grid
  cell without ``backend``, stored by column (:func:`pack_cells`) as
  zlib-compressed JSON: list-valued args (per-rank batches, row positions) make
  a grid's cells the registry's bulk, and they compress about tenfold. The grid
  is stored rather than recomputed because a corpus-routed MoE config names a
  payload file its reader may not have.
- ``_kernel_config_source``: what built configs, as the launcher describes it
  (preset or timing-predict config, pool, GPU and arch block).
- ``_kernel_config_use``: which role of which source built which config.
- ``_kernel_config_role`` and ``_kernel_config_blob``: each role path and each
  large identity array, stored once.

The simulator writes the records (``--kernel-configs-out``) and the launcher
registers them with :func:`register_kernel_configs`. Registration writes only
when something is new, so re-registering a known config leaves the file alone.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profiling.db import storage
from profiling.db.batch import coerce_args
from profiling.db.migrate import migrate_connection, require_current
from profiling.db.registry import KernelProfilerSpec, iter_kernel_profiler_specs
from profiling.db.storage import (
    BLOB_TABLE,
    CONFIG_TABLE,
    ROLE_TABLE,
    SOURCE_TABLE,
    USE_TABLE,
    canonical_json,
)
from profiling.db.table import Table

# The registry's use rows with their references resolved: what readers join.
USES_SQL = f"""
    SELECT c.kind, c.config_hash, c.gpu_name, c.profile_kind, s.source_hash, s.source,
        u.pool, r.role, u.id
    FROM {USE_TABLE} u
    JOIN {CONFIG_TABLE} c ON c.config_key = u.config_key
    JOIN {SOURCE_TABLE} s ON s.source_key = u.source_key
    JOIN {ROLE_TABLE} r ON r.role_key = u.role_key
"""
# Version of the document `simulator ... --kernel-configs-out` writes
# (`CONFIG_RECORDS_SCHEMA_VERSION` in simulator/src/timing/bridge/core.rs).
RECORDS_SCHEMA_VERSION = 1
_WRITE_LOCK_TIMEOUT_S = 120.0


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class ConfigGrid:
    cache_coords: tuple[str, ...]
    axes: tuple[tuple[float, ...], ...]
    # Row-major over `axes`: the profile-table args of each cell, without backend.
    cells: tuple[dict[str, Any], ...]
    infeasible: frozenset[int]

    def coords(self, cell: int) -> tuple[float, ...]:
        """The cache coordinates of row-major cell index `cell`."""
        out = []
        for axis in reversed(self.axes):
            cell, i = divmod(cell, len(axis))
            out.append(axis[i])
        return tuple(reversed(out))


@dataclass(frozen=True)
class ConfigUse:
    source: dict[str, Any]
    pool: str
    role: str


@dataclass(frozen=True)
class RegisteredConfig:
    kind: str
    profile_kind: str
    config_hash: str
    gpu_name: str
    identity: dict[str, Any]
    grid: ConfigGrid
    uses: tuple[ConfigUse, ...]


@dataclass(frozen=True)
class RegisterReport:
    configs: int
    # Configs skipped by `measured_only`: none of their cells has a row.
    configs_unmeasured: int
    configs_added: int
    # Known configs whose grid changed (the simulator's enumerate did); the
    # stored grid is replaced by the new one.
    configs_regridded: int
    sources_added: int
    uses_added: int

    @property
    def written(self) -> bool:
        return bool(
            self.configs_added or self.configs_regridded or self.sources_added or self.uses_added
        )


def register_kernel_configs(
    db_path: Path | str,
    document: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    *,
    measured_only: bool = False,
) -> RegisterReport:
    """Register a ``--kernel-configs-out`` document in ``db_path``.

    ``sources`` describes what built the configs, per pool name: every use in
    the document names a pool, and each pool it names must have a source.
    Nothing is written unless some config, grid, source or use is new.

    ``measured_only`` skips configs none of whose cells has a row on their GPU
    (under any backend). Registering rows measured before the registry existed
    uses it, so a config that never asked for those rows is not recorded as if
    it had.
    """
    version = document.get("schema_version")
    if version != RECORDS_SCHEMA_VERSION:
        raise ValueError(
            f"kernel config records have schema_version {version!r}; "
            f"this reader understands {RECORDS_SCHEMA_VERSION}"
        )
    records = [_Record.parse(raw) for raw in document["configs"]]
    pools = {use["pool"] for record in records for use in record.uses}
    missing = sorted(pools - set(sources))
    if missing:
        raise ValueError(f"no source given for pool(s) {missing}; have {sorted(sources)}")
    source_rows = {
        pool: (content_hash(source), canonical_json(source))
        for pool, source in sources.items()
        if pool in pools
    }

    path = Path(db_path)
    with _read_only(path) as conn:
        kept = records
        if measured_only:
            kept = [record for record in records if _has_measured_cell(conn, path, record)]
            used = {use["pool"] for record in kept for use in record.uses}
            source_rows = {pool: row for pool, row in source_rows.items() if pool in used}
        plan = _plan(conn, kept, source_rows)
    report = RegisterReport(
        configs=len(records),
        configs_unmeasured=len(records) - len(kept),
        configs_added=len(plan.new_configs),
        configs_regridded=len(plan.regridded),
        sources_added=len(plan.new_sources),
        uses_added=len(plan.new_uses),
    )
    if not report.written:
        return report

    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=_WRITE_LOCK_TIMEOUT_S)) as conn, conn:
        migrate_connection(conn)
        ensure_schema(conn)
        conn.executemany(
            f"""
            INSERT INTO {CONFIG_TABLE}
                (config_key, kind, config_hash, gpu_name, profile_kind, identity, cache_coords,
                 grid_axes, cells, infeasible)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(kind, config_hash, gpu_name) DO UPDATE SET
                profile_kind = excluded.profile_kind,
                cache_coords = excluded.cache_coords,
                grid_axes = excluded.grid_axes,
                cells = excluded.cells,
                infeasible = excluded.infeasible,
                created_at = unixepoch()
            """,
            [record.config_row(conn) for record in (*plan.new_configs, *plan.regridded)],
        )
        conn.executemany(
            f"""
            INSERT OR IGNORE INTO {SOURCE_TABLE} (source_key, source_hash, source)
            VALUES (?, ?, ?)
            """,
            [(storage.source_key(h), h, source) for h, source in plan.new_sources],
        )
        conn.executemany(
            f"""
            INSERT OR IGNORE INTO {USE_TABLE} (config_key, source_key, pool, role_key)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    storage.config_key(kind, config_hash, gpu_name),
                    storage.source_key(source_hash),
                    pool,
                    storage.ensure_role(conn, role),
                )
                for kind, config_hash, gpu_name, source_hash, pool, role in plan.new_uses
            ],
        )
    return report


def ensure_schema(conn: sqlite3.Connection) -> None:
    for statement in storage.REGISTRY_SCHEMA:
        conn.execute(statement)


def registered_configs(
    conn: sqlite3.Connection,
    profile_kind: str,
    *,
    gpu_name: str | None = None,
) -> list[RegisteredConfig]:
    """Every registered config whose grid reads rows of ``profile_kind``."""
    if not _has_registry(conn):
        return []
    require_current(conn, "profile DB")
    where = "profile_kind = ?"
    values: list[Any] = [profile_kind]
    if gpu_name is not None:
        where += " AND gpu_name = ?"
        values.append(gpu_name)
    rows = conn.execute(
        f"""
        SELECT kind, profile_kind, config_hash, gpu_name, identity, cache_coords, grid_axes,
            cells, infeasible
        FROM {CONFIG_TABLE} WHERE {where} ORDER BY kind, gpu_name, id
        """,
        values,
    ).fetchall()
    uses = _uses(conn, profile_kind)
    blobs = storage.load_blobs(conn)
    return [
        RegisteredConfig(
            kind=kind,
            profile_kind=table,
            config_hash=config_hash,
            gpu_name=gpu,
            identity=storage.unpack_identity(identity, blobs),
            grid=ConfigGrid(
                cache_coords=tuple(json.loads(cache_coords)),
                axes=tuple(tuple(axis) for axis in json.loads(grid_axes)),
                cells=tuple(_decode_cells(cells)),
                infeasible=frozenset(json.loads(infeasible)),
            ),
            uses=tuple(uses.get((kind, config_hash, gpu), ())),
        )
        for (
            kind,
            table,
            config_hash,
            gpu,
            identity,
            cache_coords,
            grid_axes,
            cells,
            infeasible,
        ) in rows
    ]


def pack_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cells by column: ``fixed`` holds each column that has one value in every
    cell, ``swept`` the per-cell values of the others. A grid repeats most of
    its columns in every cell, and profile.db is tracked in git."""
    columns = list(cells[0]) if cells else []
    for cell in cells:
        if list(cell) != columns:
            raise ValueError(f"grid cells disagree on their columns: {columns} vs {list(cell)}")
    fixed: dict[str, Any] = {}
    swept: dict[str, list[Any]] = {}
    for column in columns:
        values = [cell[column] for cell in cells]
        if all(value == values[0] for value in values):
            fixed[column] = values[0]
        else:
            swept[column] = values
    return {"count": len(cells), "fixed": fixed, "swept": swept}


def _encode_cells(cells: Sequence[Mapping[str, Any]]) -> bytes:
    return zlib.compress(json.dumps(pack_cells(cells), separators=(",", ":")).encode(), 9)


def _decode_cells(blob: bytes) -> list[dict[str, Any]]:
    return unpack_cells(json.loads(zlib.decompress(blob)))


def unpack_cells(packed: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Inverse of :func:`pack_cells`."""
    fixed, swept = packed["fixed"], packed["swept"]
    return [
        {**fixed, **{column: values[i] for column, values in swept.items()}}
        for i in range(packed["count"])
    ]


def profile_table(profile_kind: str, db_path: Path | str) -> Table:
    """The kind table a grid's cells are rows of."""
    return Table(_profiler_spec(profile_kind), db_path)


def cell_keys(table: Table, cells: Iterable[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    """Each cell's args as the table stores them, in ``table.args_columns`` order."""
    schema = table.profiler_spec.args_schema
    return [table.db_key(coerce_args(schema, dict(cell))) for cell in cells]


@dataclass(frozen=True)
class Coverage:
    profile_kind: str
    rows: int
    covered: int
    # Up to `sample` uncovered rows: gpu_name, backend and args columns.
    uncovered_sample: tuple[dict[str, Any], ...]


def cell_row_ids(
    conn: sqlite3.Connection,
    table: Table,
    gpu_name: str,
    cells: Iterable[Mapping[str, Any]],
) -> list[list[int]]:
    """The ids of the rows (any backend) each cell's args match on ``gpu_name``.

    The args columns are compared in SQL, not in Python: a column's declared
    type converts what is stored (a bool arg in a TEXT column is stored as
    ``'0'``), and SQLite applies the same conversion when it compares. Joining
    through the GPU's backends lets the ``(gpu_name, backend, args_hash)``
    unique index answer each cell with one lookup per backend.
    """
    backends = [
        row[0]
        for row in conn.execute(
            f"SELECT DISTINCT backend FROM {table.name} WHERE gpu_name = ?", (gpu_name,)
        )
    ]
    query = (
        f"SELECT id FROM {table.name} WHERE gpu_name = ? AND backend = ? "
        f"AND {table.args_match_sql()}"
    )
    out: list[list[int]] = []
    for key in cell_keys(table, cells):
        bound = (table.args_hash(key), *key)
        out.append(
            [
                row[0]
                for backend in backends
                for row in conn.execute(query, (gpu_name, backend, *bound))
            ]
        )
    return out


def coverage(
    conn: sqlite3.Connection, db_path: Path | str, profile_kind: str, *, sample: int = 5
) -> Coverage:
    """How many rows of ``profile_kind`` some registered grid cell reads."""
    table = profile_table(profile_kind, db_path)
    reached: set[int] = set()
    for config in registered_configs(conn, profile_kind):
        for ids in cell_row_ids(conn, table, config.gpu_name, config.grid.cells):
            reached.update(ids)
    columns = ["id", "gpu_name", "backend", *table.args_columns]
    rows = conn.execute(f"SELECT {', '.join(columns)} FROM {table.name} ORDER BY id").fetchall()
    uncovered = [
        dict(zip(columns[1:], row[1:], strict=True)) for row in rows if row[0] not in reached
    ]
    return Coverage(profile_kind, len(rows), len(rows) - len(uncovered), tuple(uncovered[:sample]))


# ── internals ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Record:
    kind: str
    profile_kind: str
    gpu_name: str
    identity: dict[str, Any]
    config_hash: str
    cache_coords: list[str]
    axes: list[list[float]]
    cells: list[dict[str, Any]]
    infeasible: list[int]
    uses: list[dict[str, str]]

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> _Record:
        grid = raw["grid"]
        cells = list(grid["cells"])
        size = 1
        for axis in grid["axes"]:
            size *= len(axis)
        if len(cells) != size:
            raise ValueError(f"{raw['kind']} config has {len(cells)} cells for a grid of {size}")
        return cls(
            kind=raw["kind"],
            profile_kind=raw["profile_kind"],
            gpu_name=raw["gpu_name"],
            identity=raw["identity"],
            config_hash=content_hash(raw["identity"]),
            cache_coords=list(grid["cache_coords"]),
            axes=[list(axis) for axis in grid["axes"]],
            cells=cells,
            infeasible=sorted(grid["infeasible"]),
            uses=[{"pool": use["pool"], "role": use["role"]} for use in raw["uses"]],
        )

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.kind, self.config_hash, self.gpu_name)

    def grid_columns(self) -> tuple[str, str, bytes, str]:
        return (
            json.dumps(self.cache_coords, separators=(",", ":")),
            json.dumps(self.axes, separators=(",", ":")),
            _encode_cells(self.cells),
            json.dumps(self.infeasible, separators=(",", ":")),
        )

    def same_grid(self, stored: Sequence[Any]) -> bool:
        """Whether a stored ``(profile_kind, cache_coords, grid_axes, cells,
        infeasible)`` row is this record's grid. Cells compare decoded, so a
        different zlib build's bytes do not count as a new grid."""
        profile_kind, cache_coords, axes, cells, infeasible = stored
        return (
            profile_kind == self.profile_kind
            and json.loads(cache_coords) == self.cache_coords
            and json.loads(axes) == self.axes
            and json.loads(infeasible) == self.infeasible
            and _decode_cells(cells) == self.cells
        )

    def config_row(self, conn: sqlite3.Connection) -> tuple[Any, ...]:
        return (
            storage.config_key(*self.key),
            self.kind,
            self.config_hash,
            self.gpu_name,
            self.profile_kind,
            storage.pack_identity(conn, self.identity),
            *self.grid_columns(),
        )


@dataclass
class _Plan:
    new_configs: list[_Record]
    regridded: list[_Record]
    new_sources: list[tuple[str, str]]
    new_uses: list[tuple[str, str, str, str, str, str]]


def _plan(
    conn: sqlite3.Connection | None,
    records: Sequence[_Record],
    source_rows: Mapping[str, tuple[str, str]],
) -> _Plan:
    plan = _Plan([], [], [], [])
    registry = conn is not None and _has_registry(conn)
    if registry:
        require_current(conn, "profile DB")
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        if record.key in seen:
            raise ValueError(f"config {record.key} appears twice in one document")
        seen.add(record.key)
        stored = (
            conn.execute(
                f"""
                SELECT profile_kind, cache_coords, grid_axes, cells, infeasible FROM {CONFIG_TABLE}
                WHERE kind = ? AND config_hash = ? AND gpu_name = ?
                """,
                record.key,
            ).fetchone()
            if registry
            else None
        )
        if stored is None:
            plan.new_configs.append(record)
        elif not record.same_grid(stored):
            plan.regridded.append(record)
        for use in record.uses:
            source_hash = source_rows[use["pool"]][0]
            row = (*record.key, source_hash, use["pool"], use["role"])
            known = (
                registry
                and conn.execute(
                    f"""
                SELECT 1 FROM {USE_TABLE}
                WHERE config_key = ? AND source_key = ? AND pool = ? AND role_key = ?
                """,
                    (
                        storage.config_key(*record.key),
                        storage.source_key(source_hash),
                        use["pool"],
                        storage.role_key(use["role"]),
                    ),
                ).fetchone()
            )
            if not known:
                plan.new_uses.append(row)
    for source_hash, source in source_rows.values():
        known = (
            registry
            and conn.execute(
                f"SELECT 1 FROM {SOURCE_TABLE} WHERE source_hash = ?", (source_hash,)
            ).fetchone()
        )
        if not known and (source_hash, source) not in plan.new_sources:
            plan.new_sources.append((source_hash, source))
    return plan


def _uses(
    conn: sqlite3.Connection, profile_kind: str
) -> dict[tuple[str, str, str], list[ConfigUse]]:
    rows = conn.execute(
        f"{USES_SQL} WHERE c.profile_kind = ? ORDER BY u.id", (profile_kind,)
    ).fetchall()
    out: dict[tuple[str, str, str], list[ConfigUse]] = {}
    for kind, config_hash, gpu_name, _, _, source, pool, role, _ in rows:
        out.setdefault((kind, config_hash, gpu_name), []).append(
            ConfigUse(source=json.loads(source), pool=pool, role=role)
        )
    return out


def _has_measured_cell(conn: sqlite3.Connection | None, path: Path, record: _Record) -> bool:
    if conn is None:
        return False
    table = profile_table(record.profile_kind, path)
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table.name,)
    ).fetchone()
    if exists is None:
        return False
    return any(cell_row_ids(conn, table, record.gpu_name, record.cells))


def _has_registry(conn: sqlite3.Connection) -> bool:
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?, ?, ?)",
            (CONFIG_TABLE, SOURCE_TABLE, ROLE_TABLE, USE_TABLE, BLOB_TABLE),
        )
    }
    return len(names) == 5


def _profiler_spec(profile_kind: str) -> KernelProfilerSpec:
    for spec in iter_kernel_profiler_specs():
        if spec.table_name == profile_kind:
            return spec
    raise KeyError(f"no profiler spec writes table {profile_kind!r}")


@contextmanager
def _read_only(path: Path) -> Iterator[sqlite3.Connection | None]:
    """A physically read-only connection, or None when the DB does not exist."""
    if not path.is_file():
        yield None
        return
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        yield conn
    finally:
        conn.close()
