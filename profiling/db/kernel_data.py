"""Kernel data: a registered kernel config's grid with the rows measured at it.

A config document is what the simulator fits one kernel's caches from when it
reads kernel data instead of asking perf_api per args (``KernelData`` in
``simulator/src/timing/bridge/kernel_data.rs``): the config's identity, its grid
axes, and per cell, row-major, whether the kernel can run it and the row each
backend measured there. The registry's grid and the kind table's rows are the
only inputs, so nothing here re-derives a grid or reads a corpus payload.

The public kernel API serves the same points (``/kernels/{kind}/configs/{hash}``)
with more around them; ``simulator kernel-query`` builds its kernels from
:func:`config_document` through perf_api.
"""

from __future__ import annotations

import sqlite3
from dataclasses import fields
from typing import Any

from profiling.db.kernel_config import RegisteredConfig, cell_keys, registered_configs
from profiling.db.registry import MetricFamily, iter_kernel_profiler_specs
from profiling.db.table import Table
from profiling.runners.metrics import CommMetrics, ComputeMetrics

METRICS = {
    MetricFamily.COMPUTE: [f.name for f in fields(ComputeMetrics)],
    MetricFamily.COMM: [f.name for f in fields(CommMetrics)],
}
# Bound values per statement when matching grid cells to rows; SQLite allows 32766.
_SQL_VARIABLES = 30000


def _quoted(columns: list[str]) -> str:
    return ", ".join(f'"{c}"' for c in columns)


def _spec(kind: str, profile_kind: str):
    return next(
        s
        for s in iter_kernel_profiler_specs()
        if s.kernel_kind == kind and s.table_name == profile_kind
    )


def metrics(kind: str) -> list[str]:
    """The metric columns ``kind``'s rows carry."""

    spec = next(s for s in iter_kernel_profiler_specs() if s.kernel_kind == kind)
    return METRICS[spec.metric_family]


def cell_rows(
    conn: sqlite3.Connection, db_path: Any, kind: str, config: RegisteredConfig
) -> list[dict[str, dict]]:
    """Per cell, the row each backend measured for it: ``{backend: row}``, with
    the kind's metrics and ``is_outlier``.

    The same match as ``kernel_config.cell_row_ids`` (in SQL, so each column's
    declared type converts what it compares), one statement per chunk of
    cells instead of one per cell: the cells are a ``values`` table joined to
    the kind's rows on the config's GPU."""

    table = Table(_spec(kind, config.profile_kind), db_path)
    args = list(table.args_columns)
    keys = cell_keys(table, config.grid.cells)
    columns = ["backend", *metrics(kind), "is_outlier"]
    out: list[dict[str, dict]] = [{} for _ in keys]
    width = 2 + len(args)
    chunk = max(1, (_SQL_VARIABLES - 2) // width)
    cell_columns = _quoted(["_cell", "_hash", *args])
    # Joining through the GPU's backends lets the (gpu_name, backend,
    # args_hash) unique index answer each cell with one lookup per backend;
    # the args columns then confirm the row.
    backends = f'b(backend) as (select distinct backend from "{table.name}" where gpu_name = ?)'
    on = " and ".join(
        [
            "t.gpu_name = ?",
            "t.backend = b.backend",
            't.args_hash = c."_hash"',
            *(f't."{a}" = c."{a}"' for a in args),
        ]
    )
    select = ", ".join(f't."{m}"' for m in columns)
    for start in range(0, len(keys), chunk):
        part = keys[start : start + chunk]
        values = ", ".join([f"({', '.join('?' * width)})"] * len(part))
        query = (
            f"with c({cell_columns}) as (values {values}), {backends} "
            f'select c."_cell", {select} from c cross join b join "{table.name}" t on {on}'
        )
        bound = [v for i, key in enumerate(part, start) for v in (i, table.args_hash(key), *key)]
        gpu = config.gpu_name
        for cell, *record in conn.execute(query, [*bound, gpu, gpu]):
            row = dict(zip(columns, record))
            out[cell][row.pop("backend")] = row
    return out


def points(
    conn: sqlite3.Connection, db_path: Any, kind: str, config: RegisteredConfig
) -> list[dict]:
    """Every cell of ``config``'s grid, row-major: its cache coordinates, its
    profile.db args, whether the kernel can run it, and each backend's measured
    metrics there (absent where nothing was), with an ``outlier`` flag."""

    grid = config.grid
    names = metrics(kind)
    return [
        {
            "coords": list(grid.coords(i)),
            "feasible": i not in grid.infeasible,
            "args": cell,
            "measured": {
                backend: {**{m: row[m] for m in names}, "outlier": bool(row["is_outlier"])}
                for backend, row in measured.items()
            },
        }
        for i, (cell, measured) in enumerate(
            zip(grid.cells, cell_rows(conn, db_path, kind, config), strict=True)
        )
    ]


def config_document(
    conn: sqlite3.Connection, db_path: Any, kind: str, config_hash: str, gpu_name: str
) -> dict | None:
    """The config document of ``(kind, config_hash)`` on ``gpu_name``, or None
    when the registry holds no such config."""

    tables = dict.fromkeys(
        s.table_name for s in iter_kernel_profiler_specs() if s.kernel_kind == kind
    )
    matches = [
        config
        for table in tables
        for config in registered_configs(conn, table, gpu_name=gpu_name, config_hash=config_hash)
        if config.kind == kind
    ]
    if not matches:
        return None
    [config] = matches
    return {
        "kind": kind,
        "gpu": config.gpu_name,
        "config_hash": config.config_hash,
        "identity": config.identity,
        "axes": [list(axis) for axis in config.grid.axes],
        "points": points(conn, db_path, kind, config),
    }
