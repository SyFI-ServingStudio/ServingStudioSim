"""Reusable rocprof(v3)-based kernel profiler for ROCm/MI300X workloads.

This is the ROCm counterpart of ``cupti_kernel_profiler`` and it sits in the
same place in the stack: ``Timer.rocprof`` depends on it for kernel-only
timings, while runner files stay small and only choose the timer method.

Why a different mechanism than CUPTI. On NVIDIA, ``cupti_kernel_profiler`` is a
JIT-built C++ extension that opens a CUPTI activity window *in process* around
repeated callable launches and reads each launch's kernel duration. ROCm has no
CUPTI; the supported, documented path is rocprofv3 / the ROCprofiler-SDK, whose
kernel-dispatch records carry ``Start_Timestamp`` / ``End_Timestamp`` in
nanoseconds per dispatch. rocprofv3's native output is a **rocpd** SQLite
database. So the measured quantity here is the kernel-dispatch duration
(end - start), summed per logical launch exactly as the CUPTI path sums a
launch's kernel durations, and the source of truth is the rocpd database.

Two layers, so the load-bearing logic is GPU-independent and testable:

- ``kernel_dispatch_durations_from_rocpd`` / ``summarize_rocpd`` parse a rocpd
  SQLite file into per-dispatch millisecond durations and a summary. This is
  pure SQL + arithmetic; it runs and is unit-tested on CPU with a synthetic
  database, and it is the piece the brief calls out ("kernel-dispatch durations
  from rocpd").
- ``profile_kernel`` is the in-process collector that ``Timer.rocprof`` calls.
  It runs warmup launches, then records ``num_iter`` launches under a
  ROCprofiler-SDK dispatch trace that writes a rocpd database, and feeds that
  database to the parser. The SDK session is only reachable on a ROCm host with
  ``rocprofiler-sdk`` installed; absent it, the collector raises
  ``ProfilerNotImplemented`` rather than fabricating a number.

rocpd schema note. rocprofv3's rocpd schema has shifted across ROCm releases, so
the parser does not hard-code one table/column layout: it discovers the dispatch
table and the name/string table from ``sqlite_master`` and resolves the
start/end timestamp columns by known aliases. The names it targets first are the
ROCm 7.x shape (a ``*kernel_dispatch*`` table with ``start``/``end`` ns columns,
joined to a ``*kernel_symbol*`` / ``*string*`` table for the kernel name). The
resolved names travel back in the summary so a schema surprise on the real GPU
is a debuggable value rather than a silent miss.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import fmean, median
from typing import Any

from profiling.runners.exceptions import ProfilerNotImplemented

_NS_PER_MS = 1_000_000.0

# Candidate rocpd table names, most-specific first. Discovery prefers a name
# containing "kernel_dispatch", then "dispatch", then the older generic op table.
_DISPATCH_TABLE_HINTS = ("kernel_dispatch", "dispatch", "rocpd_op", "_op")
# Candidate name/string tables carrying the kernel symbol text.
_NAME_TABLE_HINTS = ("kernel_symbol", "kernel_name", "string")
# Known start/end timestamp column aliases, checked in order.
_START_COLUMN_HINTS = ("start", "start_timestamp", "begin_ns", "start_ns", "begin")
_END_COLUMN_HINTS = ("end", "end_timestamp", "end_ns", "finish_ns", "finish")
# Known kernel-name column aliases on either the dispatch row or the name table.
_NAME_COLUMN_HINTS = (
    "kernel_name",
    "formatted_kernel_name",
    "display_name",
    "demangled_name",
    "name",
    "string",
    "value",
)


@dataclass(frozen=True)
class RocpdResolvedSchema:
    """Which rocpd table/column names the parser resolved for one database.

    Carried in the summary so a schema the parser did not expect is a concrete,
    reportable value instead of an empty result.
    """

    dispatch_table: str
    start_column: str
    end_column: str
    name_table: str | None = None
    name_column: str | None = None


@dataclass(frozen=True)
class RocprofKernelSummary:
    """Mirror of ``cupti_kernel_profiler.KernelProfileSummary`` for the rocpd path.

    ``mean_ms`` / ``per_iter_ms`` are the fields ``Timer.rocprof`` reads, kept
    name-identical with the CUPTI summary so a reader comparing the two paths
    does not have to translate.
    """

    matched_kernel_names: list[str]
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    num_warmup: int
    num_iter: int
    per_iter_ms: list[float]
    schema: RocpdResolvedSchema | None = field(default=None)


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return [row[0] for row in rows]


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    # PRAGMA table_info returns (cid, name, type, notnull, dflt, pk).
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _first_table_matching(tables: Sequence[str], hints: Sequence[str]) -> str | None:
    lowered = {table: table.lower() for table in tables}
    for hint in hints:
        for table in tables:
            if hint in lowered[table]:
                return table
    return None


def _first_column_matching(columns: Sequence[str], hints: Sequence[str]) -> str | None:
    lowered = {column: column.lower() for column in columns}
    # Exact (case-insensitive) match wins over substring so "start" does not
    # grab "start_ns" when a plain "start" column also exists.
    for hint in hints:
        for column in columns:
            if lowered[column] == hint:
                return column
    for hint in hints:
        for column in columns:
            if hint in lowered[column]:
                return column
    return None


def resolve_rocpd_schema(conn: sqlite3.Connection) -> RocpdResolvedSchema:
    """Discover the dispatch table + timestamp/name columns in a rocpd database.

    Raises ``ValueError`` when no table carries a resolvable start/end timestamp
    pair -- that is a real "this is not a kernel-dispatch rocpd database" signal,
    not something to paper over with an empty duration list.
    """

    tables = _table_names(conn)
    dispatch_table = _first_table_matching(tables, _DISPATCH_TABLE_HINTS)
    if dispatch_table is None:
        raise ValueError(
            f"no kernel-dispatch table found in rocpd database; tables={sorted(tables)}"
        )
    columns = _column_names(conn, dispatch_table)
    start_column = _first_column_matching(columns, _START_COLUMN_HINTS)
    end_column = _first_column_matching(columns, _END_COLUMN_HINTS)
    if start_column is None or end_column is None:
        raise ValueError(
            f"rocpd dispatch table {dispatch_table!r} has no resolvable start/end "
            f"timestamp columns; columns={columns}"
        )

    # Kernel name may live on the dispatch row itself (CSV-equivalent shape) or
    # on a joined symbol/string table keyed by a *_id column.
    name_table: str | None = None
    name_column = _first_column_matching(columns, _NAME_COLUMN_HINTS)
    if name_column is None:
        name_table = _first_table_matching(tables, _NAME_TABLE_HINTS)
        if name_table is not None:
            name_column = _first_column_matching(
                _column_names(conn, name_table), _NAME_COLUMN_HINTS
            )
    return RocpdResolvedSchema(
        dispatch_table=dispatch_table,
        start_column=start_column,
        end_column=end_column,
        name_table=name_table if name_column is not None else None,
        name_column=name_column,
    )


def _dispatch_rows(
    conn: sqlite3.Connection, schema: RocpdResolvedSchema
) -> list[tuple[float, str | None]]:
    """Return ``(duration_ns, kernel_name)`` per dispatch, ordered by start time.

    When the name lives on a separate symbol/string table we join it in; when it
    is absent entirely, ``kernel_name`` is ``None`` and name filtering is a no-op.
    """

    dispatch = schema.dispatch_table
    start = schema.start_column
    end = schema.end_column
    if schema.name_table is not None and schema.name_column is not None:
        # Join the dispatch's *_id column to the name table's id. rocpd uses a
        # ``*kernel*id`` foreign key; discover it rather than hard-coding.
        dispatch_columns = _column_names(conn, dispatch)
        fk = _first_column_matching(
            dispatch_columns, ("kernel_id", "symbol_id", "name_id", "string_id")
        )
        name_columns = _column_names(conn, schema.name_table)
        pk = _first_column_matching(name_columns, ("id", "kernel_id", "symbol_id"))
        if fk is not None and pk is not None:
            sql = (
                f'SELECT (d."{end}" - d."{start}") AS dur, n."{schema.name_column}" '
                f'FROM "{dispatch}" d '
                f'LEFT JOIN "{schema.name_table}" n ON d."{fk}" = n."{pk}" '
                f'ORDER BY d."{start}"'
            )
            return [(float(row[0]), row[1]) for row in conn.execute(sql).fetchall()]

    name_select = f'"{schema.name_column}"' if schema.name_column else "NULL"
    sql = (
        f'SELECT ("{end}" - "{start}") AS dur, {name_select} '
        f'FROM "{dispatch}" ORDER BY "{start}"'
    )
    return [(float(row[0]), row[1]) for row in conn.execute(sql).fetchall()]


def kernel_dispatch_durations_from_rocpd(
    db_path: str,
    kernel_name_contains: str | None = None,
) -> list[float]:
    """Per-dispatch kernel durations in milliseconds, in dispatch order.

    This is the rocpd equivalent of reading CUPTI kernel records: one entry per
    GPU kernel dispatch, duration = ``End_Timestamp - Start_Timestamp`` (ns)
    converted to ms. ``kernel_name_contains`` keeps only dispatches whose kernel
    name contains the substring (matching ``match_kernel_records`` on the CUPTI
    side); ``None`` keeps every dispatch, which is what a compound callable with
    a fixed launch sequence wants.
    """

    conn = sqlite3.connect(db_path)
    try:
        schema = resolve_rocpd_schema(conn)
        rows = _dispatch_rows(conn, schema)
    finally:
        conn.close()
    durations_ms: list[float] = []
    for duration_ns, name in rows:
        if kernel_name_contains is not None:
            if name is None or kernel_name_contains not in name:
                continue
        durations_ms.append(duration_ns / _NS_PER_MS)
    if kernel_name_contains is not None and not durations_ms:
        captured = sorted({name for _dur, name in rows if name})
        raise ValueError(
            f"no dispatch matched substring {kernel_name_contains!r}; "
            f"captured kernels: {captured}"
        )
    return durations_ms


def summarize_rocpd(
    db_path: str,
    *,
    num_iter: int,
    num_warmup: int = 0,
    kernel_name_contains: str | None = None,
) -> RocprofKernelSummary:
    """Summarize a rocpd database into the CUPTI-parallel summary shape.

    ``num_iter`` is the number of logical callable launches the trace recorded.
    When a single logical launch issues several GPU dispatches, the dispatch
    durations are folded into ``num_iter`` groups in order and summed per group,
    exactly as the CUPTI path sums a launch's kernel durations.
    """

    conn = sqlite3.connect(db_path)
    try:
        schema = resolve_rocpd_schema(conn)
    finally:
        conn.close()
    durations_ms = kernel_dispatch_durations_from_rocpd(db_path, kernel_name_contains)
    per_iter_ms = _fold_launches(durations_ms, num_iter)
    if not per_iter_ms:
        raise ValueError("rocpd database recorded no kernel dispatches")
    return RocprofKernelSummary(
        matched_kernel_names=sorted(
            {name for name in _captured_names(db_path) if name}
        )
        if kernel_name_contains is None
        else [kernel_name_contains],
        mean_ms=fmean(per_iter_ms),
        median_ms=median(per_iter_ms),
        min_ms=min(per_iter_ms),
        max_ms=max(per_iter_ms),
        num_warmup=num_warmup,
        num_iter=len(per_iter_ms),
        per_iter_ms=per_iter_ms,
        schema=schema,
    )


def _captured_names(db_path: str) -> list[str | None]:
    conn = sqlite3.connect(db_path)
    try:
        schema = resolve_rocpd_schema(conn)
        return [name for _dur, name in _dispatch_rows(conn, schema)]
    finally:
        conn.close()


def _fold_launches(durations_ms: list[float], num_iter: int) -> list[float]:
    """Fold per-dispatch durations into ``num_iter`` per-launch sums, in order.

    The common case is one dispatch per launch (``len == num_iter``), returned
    as-is. When a launch issues k dispatches the count is a clean multiple and
    each consecutive group of k is summed. A non-multiple means the launch
    boundary is ambiguous, so the raw per-dispatch series is returned rather than
    guessing a split.
    """

    if num_iter <= 0 or not durations_ms:
        return list(durations_ms)
    if len(durations_ms) == num_iter:
        return list(durations_ms)
    if len(durations_ms) % num_iter == 0:
        per_launch = len(durations_ms) // num_iter
        return [
            sum(durations_ms[i * per_launch : (i + 1) * per_launch])
            for i in range(num_iter)
        ]
    return list(durations_ms)


def profile_kernel(
    fn: Callable[[], object],
    *,
    num_warmup: int = 10,
    num_iter: int = 50,
    kernel_name_contains: str | None = None,
    **_ignored: Any,
) -> RocprofKernelSummary:
    """Record ``num_iter`` launches of ``fn`` under a rocprof dispatch trace.

    Intended contract (parallel to ``cupti_kernel_profiler.profile_kernel``): run
    ``num_warmup`` untraced launches, then record exactly ``num_iter`` launches
    under a kernel-dispatch trace that writes a rocpd database, synchronize, and
    hand the database to ``summarize_rocpd``. Extra keyword arguments
    (``clear_l2_*``, ``interval_union``) are accepted for call-site parity and
    ignored: the rocpd timestamps already price GPU-active time per dispatch.

    In-process capture needs either the ROCprofiler-SDK Python session API or the
    worker launched under ``rocprofv3 --kernel-trace --output-format rocpd``.
    Neither is verified on an MI300X yet, so this raises rather than returning a
    fabricated time. The real-GPU step wires one of those sources and then feeds
    its rocpd database to ``summarize_rocpd`` -- the already-tested parser. Offline
    attribution of an existing rocpd database is available now without a GPU via
    ``summarize_rocpd`` / ``kernel_dispatch_durations_from_rocpd``.
    """

    try:
        import rocprofiler_sdk  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - requires a ROCm host
        raise ProfilerNotImplemented(
            "Timer.rocprof requires a ROCm host with rocprofiler-sdk; the rocpd "
            "parser (summarize_rocpd) is available without it for offline traces"
        ) from exc
    raise ProfilerNotImplemented(
        "Timer.rocprof in-process capture is not yet wired to a GPU-verified "
        "ROCprofiler-SDK session; summarize a rocprofv3 --output-format rocpd "
        "database with summarize_rocpd(...) until the MI300X run confirms it"
    )
