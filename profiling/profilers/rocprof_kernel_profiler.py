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

import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median
from typing import Any

from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

_NS_PER_MS = 1_000_000.0

# Candidate rocpd table names, most-specific first. Discovery prefers a name
# containing "kernel_dispatch", then "dispatch", then the older generic op table.
_DISPATCH_TABLE_HINTS = ("kernel_dispatch", "dispatch", "rocpd_op", "_op")
# Candidate name/string tables carrying the kernel symbol text. The kernel
# symbol table is preferred over the generic string table because rocpd links a
# dispatch to it directly by kernel_id (verified on ROCm 7.2 / rocprofv3 1.3.2).
_NAME_TABLE_HINTS = ("kernel_symbol", "kernel_name", "string")
# Known start/end timestamp column aliases, checked in order.
_START_COLUMN_HINTS = ("start", "start_timestamp", "begin_ns", "start_ns", "begin")
_END_COLUMN_HINTS = ("end", "end_timestamp", "end_ns", "finish_ns", "finish")
# Known kernel-name column aliases. display_name (demangled) is preferred over
# the mangled kernel_name so a substring filter matches human-readable text.
_NAME_COLUMN_HINTS = (
    "display_name",
    "formatted_kernel_name",
    "demangled_name",
    "kernel_name",
    "name",
    "string",
    "value",
)
# Foreign-key column on the dispatch row pointing at the kernel-symbol table.
_NAME_FK_HINTS = ("kernel_id", "symbol_id", "name_id", "string_id")
# Primary-key column on the kernel-symbol / string table.
_NAME_PK_HINTS = ("id", "kernel_id", "symbol_id")

# Columns a full dispatch row carries beyond start/end/name. These feed the
# offline alignment producer (``alignment/rocpd``), not the per-kernel timer, so
# they are resolved by a separate discovery that leaves the timing path above
# untouched. All are optional: a column the schema omits resolves to ``None``
# and the reader fills a neutral default (and synthesizes correlation ids).
# The GPU stream the dispatch ran on. rocpd names it ``stream_id``; older
# shapes only have the hardware ``queue_id``, which is the same ownership key.
_STREAM_COLUMN_HINTS = ("stream_id", "queue_id")
# A per-launch identity. rocpd's ``dispatch_id`` is the monotonic launch counter
# that plays the role CUPTI's ``correlationId`` does; a kernel-only trace may
# carry neither, in which case the reader synthesizes one (see below).
_CORRELATION_COLUMN_HINTS = ("correlation_id", "dispatch_id")
# The GPU the dispatch ran on. rocpd keys it as ``agent_id`` (an FK into
# ``rocpd_info_agent``); the integer value itself is the device identity we need.
_DEVICE_COLUMN_HINTS = ("agent_id", "device_id", "gpu_id")
# The OS process that launched the dispatch.
_PID_COLUMN_HINTS = ("pid",)
# The OS thread that opened a roctx region, matched against the launching
# thread so a dispatch is attributed to the region its own thread was inside.
_TID_COLUMN_HINTS = ("tid", "thread_id", "global_tid")
# The launch grid dimensions rocpd records per dispatch (ROCm 7.x shape:
# ``grid_size_x``/``_y``/``_z``). The alignment producer uses them to recognize
# the ``vibesim_sentinel`` marker kernels the roctx shim launches at each forward
# boundary and to decode the iteration ordinal those sentinels carry in the
# y-grid dimension (see ``alignment/rocpd/evidence.py`` and the Option-B sentinel
# contract in ``alignment/profiler/roctx_shim.py``). They are optional: a schema
# that omits them resolves to ``None`` and the reader fills 0 (no grid signal).
_GRID_X_COLUMN_HINTS = ("grid_size_x",)
_GRID_Y_COLUMN_HINTS = ("grid_size_y",)
_GRID_Z_COLUMN_HINTS = ("grid_size_z",)

# roctx region table discovery. rocpd records every roctx push/pop range in a
# ``rocpd_region`` table. Its ``name_id`` FK resolves (via the string table) only
# to the API op name (``roctxThreadRangeA``); the range's real message text is on
# the joined event row's ``extdata`` JSON (see ``_EVENT_TABLE_HINTS`` below).
_REGION_TABLE_HINTS = ("rocpd_region", "roctx_region", "region")
_REGION_START_HINTS = ("start", "start_timestamp", "start_ns", "begin")
_REGION_END_HINTS = ("end", "end_timestamp", "end_ns", "finish")
# FK on the region row to the string table carrying the roctx message text.
_REGION_NAME_FK_HINTS = ("name_id", "region_name_id", "string_id")
# The shared interned-string table (``rocpd_string``: id -> string).
_STRING_TABLE_HINTS = ("rocpd_string", "string")
_STRING_PK_HINTS = ("id", "string_id")
_STRING_VALUE_HINTS = ("string", "value", "name")

# The real roctx label does NOT live in the region's ``name_id`` string: on a
# rocprofv3 1.3.2 / rocprofiler-sdk capture that column resolves to the API op
# name ``roctxThreadRangeA`` (category ``MARKER_CORE_RANGE_API``), the same for
# every roctx range. The instrumented label text (``vllm_iteration(N): forward``,
# ``VibeSimAlignmentIteration {json}``) is carried as JSON in the *event* row the
# region points at via ``event_id``: ``rocpd_event.extdata`` = ``{"message": ...}``
# (also surfaced as the ``extdata`` column of rocprofv3's ``regions`` view). So
# the reader joins region -> event and reads the ``message`` field, falling back
# to the ``name_id`` string only when no extdata message is present (older
# captures, or a synthetic db that interns the label directly as the region name).
_EVENT_TABLE_HINTS = ("rocpd_event", "event")
# FK on the region row pointing at the event table (``rocpd_region.event_id``).
_REGION_EVENT_FK_HINTS = ("event_id",)
_EVENT_PK_HINTS = ("id", "event_id")
# The JSON blob column on the event row carrying ``{"message": "<roctx label>"}``.
_EVENT_EXTDATA_HINTS = ("extdata",)


def _roctx_message_from_extdata(extdata: object) -> str | None:
    """Pull the roctx label out of a rocpd event's ``extdata`` JSON blob.

    rocprofv3 stores the roctx range's actual message as ``{"message": "..."}``
    in ``rocpd_event.extdata``; an unannotated range carries ``{}`` (or NULL).
    Returns the message string when present and non-empty, else ``None`` so the
    caller can fall back to the region's interned ``name_id`` string.
    """
    if not extdata:
        return None
    try:
        payload = json.loads(extdata)
    except (TypeError, ValueError):
        return None
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message:
            return message
    return None


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

    # Kernel name: real rocpd keeps it on a joined kernel-symbol table addressed
    # by the dispatch's kernel_id, so prefer that join. Only fall back to a name
    # column on the dispatch row itself when no symbol table + FK exists -- and
    # when doing so, ignore *_id columns, which are integer foreign keys (e.g.
    # rocpd's ``region_name_id``), not the kernel name text.
    name_table = _first_table_matching(tables, _NAME_TABLE_HINTS)
    fk_column = _first_column_matching(columns, _NAME_FK_HINTS)
    if name_table is not None and fk_column is not None:
        name_column = _first_column_matching(
            _column_names(conn, name_table), _NAME_COLUMN_HINTS
        )
        if name_column is not None:
            return RocpdResolvedSchema(
                dispatch_table=dispatch_table,
                start_column=start_column,
                end_column=end_column,
                name_table=name_table,
                name_column=name_column,
            )

    text_columns = [column for column in columns if not column.lower().endswith("_id")]
    direct_name = _first_column_matching(text_columns, _NAME_COLUMN_HINTS)
    return RocpdResolvedSchema(
        dispatch_table=dispatch_table,
        start_column=start_column,
        end_column=end_column,
        name_table=None,
        name_column=direct_name,
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
        fk = _first_column_matching(dispatch_columns, _NAME_FK_HINTS)
        name_columns = _column_names(conn, schema.name_table)
        pk = _first_column_matching(name_columns, _NAME_PK_HINTS)
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


@dataclass(frozen=True)
class RocpdDispatch:
    """One kernel dispatch with the full set of fields alignment attribution needs.

    This is the richer sibling of the ``(duration_ns, name)`` row the timing path
    reads: the timer only needs how long a launch took, while the offline
    alignment producer must also know WHICH stream, process, device, and launch a
    dispatch belongs to so it can place it in a roctx iteration window and emit
    the normalized ``parsed.kernels.parquet`` row. ``correlation_id`` is the
    monotonic launch identity (rocpd's ``dispatch_id`` when present, else a
    synthesized 1-based counter in start order); ``stream_id``/``device_id``/
    ``pid`` default to 0 when a reduced capture omits the column.
    """

    start_ns: int
    end_ns: int
    name: str | None
    stream_id: int
    correlation_id: int
    device_id: int
    pid: int
    #: The launch grid dimensions (``grid_size_x``/``_y``/``_z``). 0 when a
    #: reduced capture omits the columns. The alignment producer reads these to
    #: identify ``vibesim_sentinel`` marker kernels and decode the iteration
    #: ordinal they carry in the y-grid dimension (Option-B sentinel capture).
    grid_size_x: int = 0
    grid_size_y: int = 0
    grid_size_z: int = 0

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class RoctxRegion:
    """One roctx push/pop range: an instrumented marker like ``vllm_iteration(3): forward``.

    rocpd stores every roctx range in its region table; the message text lives
    on the joined event row's ``extdata`` JSON (the region's own ``name_id`` only
    gives the API op name). These are the windows the alignment ownership join
    tests a dispatch's launch time against. ``pid``/``tid`` are the
    process and thread that opened the range (``None`` when the schema omits them).
    """

    start_ns: int
    end_ns: int
    tid: int | None
    pid: int | None
    name: str


def kernel_dispatch_records_from_rocpd(db_path: str) -> list[RocpdDispatch]:
    """Read every kernel dispatch as a full :class:`RocpdDispatch`, in start order.

    The timing path (:func:`kernel_dispatch_durations_from_rocpd`) reuses
    :func:`resolve_rocpd_schema` for the dispatch table + start/end + name join;
    this adds a separate discovery for the stream/correlation/device/pid columns
    so that path is left byte-for-byte unchanged. Any of those columns the schema
    omits resolves to ``None`` and is filled with a neutral default. When the
    capture carries no per-launch id at all (a pure kernel trace), a monotonic
    1-based ``correlation_id`` is synthesized in start order so the downstream
    non-null ``correlation_id`` parquet column is always satisfiable.
    """
    conn = sqlite3.connect(db_path)
    try:
        schema = resolve_rocpd_schema(conn)
        dispatch = schema.dispatch_table
        columns = _column_names(conn, dispatch)
        stream_col = _first_column_matching(columns, _STREAM_COLUMN_HINTS)
        correlation_col = _first_column_matching(columns, _CORRELATION_COLUMN_HINTS)
        device_col = _first_column_matching(columns, _DEVICE_COLUMN_HINTS)
        pid_col = _first_column_matching(columns, _PID_COLUMN_HINTS)
        grid_x_col = _first_column_matching(columns, _GRID_X_COLUMN_HINTS)
        grid_y_col = _first_column_matching(columns, _GRID_Y_COLUMN_HINTS)
        grid_z_col = _first_column_matching(columns, _GRID_Z_COLUMN_HINTS)

        def qualified(column: str | None) -> str:
            return f'd."{column}"' if column else "NULL"

        join = ""
        if schema.name_table is not None and schema.name_column is not None:
            fk = _first_column_matching(columns, _NAME_FK_HINTS)
            pk = _first_column_matching(_column_names(conn, schema.name_table), _NAME_PK_HINTS)
            if fk is not None and pk is not None:
                join = f'LEFT JOIN "{schema.name_table}" n ON d."{fk}" = n."{pk}"'
                name_select = f'n."{schema.name_column}"'
            else:
                name_select = "NULL"
        elif schema.name_column is not None:
            name_select = f'd."{schema.name_column}"'
        else:
            name_select = "NULL"

        sql = (
            f'SELECT d."{schema.start_column}", d."{schema.end_column}", {name_select}, '
            f"{qualified(stream_col)}, {qualified(correlation_col)}, "
            f"{qualified(device_col)}, {qualified(pid_col)}, "
            f"{qualified(grid_x_col)}, {qualified(grid_y_col)}, {qualified(grid_z_col)} "
            f'FROM "{dispatch}" d {join} ORDER BY d."{schema.start_column}"'
        )
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    records: list[RocpdDispatch] = []
    for ordinal, (
        start,
        end,
        name,
        stream,
        correlation,
        device,
        pid,
        grid_x,
        grid_y,
        grid_z,
    ) in enumerate(rows, start=1):
        records.append(
            RocpdDispatch(
                start_ns=int(start),
                end_ns=int(end),
                name=None if name is None else str(name),
                stream_id=int(stream) if stream is not None else 0,
                correlation_id=int(correlation) if correlation is not None else ordinal,
                device_id=int(device) if device is not None else 0,
                pid=int(pid) if pid is not None else 0,
                grid_size_x=int(grid_x) if grid_x is not None else 0,
                grid_size_y=int(grid_y) if grid_y is not None else 0,
                grid_size_z=int(grid_z) if grid_z is not None else 0,
            )
        )
    return records


def roctx_regions_from_rocpd(db_path: str) -> list[RoctxRegion]:
    """Read every roctx range as a :class:`RoctxRegion`, in start order.

    Returns an empty list when the capture has no region table (a kernel-only
    trace, such as the upstream vLLM MI210 capture) -- absence of markers is a
    real state the ownership join handles, not an error. Raises ``ValueError``
    only when a region table exists but lacks a resolvable start/end/name.
    """
    conn = sqlite3.connect(db_path)
    try:
        tables = _table_names(conn)
        region_table = _first_table_matching(tables, _REGION_TABLE_HINTS)
        if region_table is None:
            return []
        columns = _column_names(conn, region_table)
        start_col = _first_column_matching(columns, _REGION_START_HINTS)
        end_col = _first_column_matching(columns, _REGION_END_HINTS)
        name_fk = _first_column_matching(columns, _REGION_NAME_FK_HINTS)
        if start_col is None or end_col is None or name_fk is None:
            raise ValueError(
                f"rocpd region table {region_table!r} lacks a resolvable "
                f"start/end/name column; columns={columns}"
            )
        tid_col = _first_column_matching(columns, _TID_COLUMN_HINTS)
        pid_col = _first_column_matching(columns, _PID_COLUMN_HINTS)

        def qualified(column: str | None) -> str:
            return f'r."{column}"' if column else "NULL"

        # Fallback label source: the region's interned ``name_id`` string. On a
        # real rocprofv3 capture this is only ``roctxThreadRangeA`` (the API op
        # name), so it is the fallback; the true label comes from the event join
        # below. A synthetic db that interns the label directly still resolves
        # here, which is why this is kept.
        string_table = _first_table_matching(tables, _STRING_TABLE_HINTS)
        join = ""
        name_select = f'r."{name_fk}"'
        if string_table is not None:
            string_columns = _column_names(conn, string_table)
            string_pk = _first_column_matching(string_columns, _STRING_PK_HINTS)
            string_value = _first_column_matching(string_columns, _STRING_VALUE_HINTS)
            if string_pk is not None and string_value is not None:
                join = f'LEFT JOIN "{string_table}" s ON r."{name_fk}" = s."{string_pk}"'
                name_select = f's."{string_value}"'

        # Primary label source: the event row the region points at via
        # ``event_id``, whose ``extdata`` JSON carries ``{"message": "<label>"}``.
        event_table = _first_table_matching(tables, _EVENT_TABLE_HINTS)
        event_fk = _first_column_matching(columns, _REGION_EVENT_FK_HINTS)
        event_join = ""
        extdata_select = "NULL"
        if event_table is not None and event_fk is not None:
            event_columns = _column_names(conn, event_table)
            event_pk = _first_column_matching(event_columns, _EVENT_PK_HINTS)
            event_extdata = _first_column_matching(event_columns, _EVENT_EXTDATA_HINTS)
            if event_pk is not None and event_extdata is not None:
                event_join = (
                    f'LEFT JOIN "{event_table}" e ON r."{event_fk}" = e."{event_pk}"'
                )
                extdata_select = f'e."{event_extdata}"'

        sql = (
            f'SELECT r."{start_col}", r."{end_col}", {name_select}, {extdata_select}, '
            f"{qualified(tid_col)}, {qualified(pid_col)} "
            f'FROM "{region_table}" r {join} {event_join} ORDER BY r."{start_col}"'
        )
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    regions: list[RoctxRegion] = []
    for start, end, fallback_name, extdata, tid, pid in rows:
        # Prefer the real roctx label from the event's extdata JSON; only when it
        # is absent does the interned region name (``roctxThreadRangeA`` on a real
        # capture) stand in. A range with neither is not an instrumented marker.
        label = _roctx_message_from_extdata(extdata)
        if label is None and fallback_name is not None:
            label = str(fallback_name)
        if label is None:
            continue
        regions.append(
            RoctxRegion(
                start_ns=int(start),
                end_ns=int(end),
                tid=None if tid is None else int(tid),
                pid=None if pid is None else int(pid),
                name=label,
            )
        )
    return regions


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


def _find_rocprofv3() -> str:
    """Locate the rocprofv3 executable, honoring ROCM_HOME / ROCM_PATH.

    rocprofv3 ships with ROCm; inside the vLLM-ROCm image it is on PATH at
    /opt/rocm/bin. Raise cleanly when absent so a non-ROCm host reports a missing
    tracer rather than a confusing subprocess error.
    """

    found = shutil.which("rocprofv3")
    if found:
        return found
    for root_env in ("ROCM_HOME", "ROCM_PATH"):
        root = os.environ.get(root_env)
        if root and (Path(root) / "bin" / "rocprofv3").exists():
            return str(Path(root) / "bin" / "rocprofv3")
    default = Path("/opt/rocm/bin/rocprofv3")
    if default.exists():
        return str(default)
    raise ProfilerNotImplemented(
        "rocprofv3 not found (set ROCM_HOME or run inside the ROCm image)"
    )


def measure_registered_via_rocprofv3(
    *,
    kind: str,
    backend: str,
    spec: dict[str, Any],
    kernel_name_contains: str | None,
    warmup: int,
    rep: int,
    fold_per_launch: bool = False,
    dispatches_per_launch: int | None = None,
) -> float:
    """Measure a registered kernel's per-launch ms via a whole-process rocprofv3 capture.

    rocprofv3's rocpd output is finalized at process exit, so a kernel cannot be
    traced by a synchronous in-process timer the way CUPTI is. Instead this spawns
    the launch driver ``profiling.profilers.rocprof_run`` -- which rebuilds the
    identical kernel from ``(kind, backend, spec)`` and runs ``warmup + rep``
    launches -- under ``rocprofv3 --kernel-trace --output-format rocpd``, then
    reads the rocpd, filters to ``kernel_name_contains``, drops the ``warmup``
    launches, and returns the mean of the remaining ``rep`` kernel-dispatch
    durations. This runs inside the ROCm image where the worker already executes,
    so no nested container is involved.

    ``fold_per_launch`` handles a compound call that launches several GPU
    dispatches per logical launch (e.g. KDA's four input copies plus the
    recurrent kernel). The driver must then build its operands so setup emits no
    kernel dispatches (host tensors moved to the device with ``.to()``), so the
    only dispatches captured are the ``warmup + rep`` launches' own, a constant
    ``D`` per launch. The per-launch time is the sum of that launch's ``D``
    dispatches (matching the CUPTI path's ``kernel_name=None`` per-launch sum),
    and the returned value is the mean over the ``rep`` launches after the
    ``warmup`` ones are dropped. ``kernel_name_contains`` must be ``None`` in this
    mode, since every one of the call's dispatches is counted.

    ``dispatches_per_launch`` is the autotune-robust variant of ``fold_per_launch``
    for a compound call whose kernels ``@autotune`` on the first launch (e.g. KDA
    chunked prefill's FLA chunk kernels). The autotuner benchmarks candidate
    configs on that first call -- a variable burst of extra dispatches -- then
    caches its winner in-process, so every later launch issues a constant ``D``
    dispatches. Rather than recover ``D`` by dividing the whole stream (which the
    burst would corrupt), the caller passes the known ``D`` and this folds only
    the *trailing* ``rep`` launches: the last ``rep*D`` dispatches, which are all
    steady-state. That skips the leading device-init + autotune-burst prefix
    whatever its size. It is mutually exclusive with ``fold_per_launch``. It may
    be paired with ``kernel_name_contains``, in which case ``D`` is the number of
    *matching* dispatches per launch (e.g. ``D=1`` for a call whose single named
    Triton kernel ``@autotune``s on the first launch): the trailing fold then
    isolates the ``rep`` steady-state launches of that kernel past the autotune
    burst, which shares the kernel's name and so would otherwise inflate the
    front-dropping mean.
    """
    if fold_per_launch and kernel_name_contains is not None:
        raise ValueError("fold_per_launch counts every dispatch; kernel_name_contains must be None")
    if dispatches_per_launch is not None:
        if fold_per_launch:
            raise ValueError("dispatches_per_launch and fold_per_launch are mutually exclusive")
        if dispatches_per_launch < 1:
            raise ValueError("dispatches_per_launch must be >= 1")

    rocprofv3 = _find_rocprofv3()
    with tempfile.TemporaryDirectory(prefix="vibesim-rocprof-") as tmp:
        rocpd_dir = Path(tmp) / "rocpd"
        driver = [
            "python3",
            "-m",
            "profiling.profilers.rocprof_run",
            "--kind",
            kind,
            "--backend",
            backend,
            "--spec",
            json.dumps(spec),
            "--warmup",
            str(warmup),
            "--rep",
            str(rep),
        ]
        cmd = [
            rocprofv3,
            "--kernel-trace",
            "--output-format",
            "rocpd",
            "-d",
            str(rocpd_dir),
            "-o",
            "run",
            "--",
            *driver,
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "")[-1500:]
            raise KernelLaunchFailed(
                f"rocprofv3 capture of {kind}:{backend} exited {completed.returncode}: {tail}"
            )
        db_files = sorted(rocpd_dir.rglob("*.db"))
        if not db_files:
            raise KernelLaunchFailed(
                f"rocprofv3 produced no rocpd database under {rocpd_dir}; "
                f"stdout tail: {(completed.stdout or '')[-800:]}"
            )
        durations_ms = kernel_dispatch_durations_from_rocpd(
            str(db_files[0]), kernel_name_contains=kernel_name_contains
        )
    if dispatches_per_launch is not None:
        return _fold_trailing_mean(durations_ms, rep=rep, per_launch=dispatches_per_launch)
    if fold_per_launch:
        return _fold_per_launch_mean(durations_ms, warmup=warmup, rep=rep)
    if len(durations_ms) < warmup + 1:
        raise KernelLaunchFailed(
            f"captured {len(durations_ms)} '{kernel_name_contains}' dispatches, "
            f"need more than warmup={warmup}"
        )
    measured = durations_ms[warmup:]
    return fmean(measured)


def _fold_per_launch_mean(durations_ms: list[float], *, warmup: int, rep: int) -> float:
    """Mean per-launch ms when each launch issues a constant ``D`` dispatches.

    The call launches a fixed ``D`` dispatches (e.g. KDA's four input copies plus
    the recurrent kernel), repeated ``warmup + rep`` times. A real capture also
    carries a small fixed prefix of one-time device-init dispatches before the
    first launch (seen: 4 on the pinned MI300X image), so the stream is
    ``[prefix] + (warmup + rep) * D``. ``D`` is recovered as ``total //
    (warmup + rep)``, which is exact as long as the prefix is smaller than one
    sweep of launches (``prefix < warmup + rep``); the leading ``prefix``
    dispatches are then dropped, the ``warmup`` launches after them dropped, and
    the trailing ``rep`` launches folded into groups of ``D`` and summed per group
    (matching the CUPTI path's ``kernel_name=None`` per-launch sum). A ``D < 1``
    (fewer dispatches than launches) is an honest failure, not a guessed split.
    """
    launches = warmup + rep
    total = len(durations_ms)
    if launches <= 0 or total == 0:
        raise KernelLaunchFailed("rocprofv3 captured no dispatches to fold")
    per_launch = total // launches
    if per_launch < 1:
        raise KernelLaunchFailed(
            f"captured {total} dispatches for {launches} launches (warmup={warmup}, "
            f"rep={rep}); fewer than one dispatch per launch, cannot fold"
        )
    prefix = total - launches * per_launch  # one-time device-init dispatches
    uniform = durations_ms[prefix:]
    measured = uniform[warmup * per_launch :]
    return fmean(
        [sum(measured[i * per_launch : (i + 1) * per_launch]) for i in range(rep)]
    )


def _fold_trailing_mean(durations_ms: list[float], *, rep: int, per_launch: int) -> float:
    """Mean per-launch ms over the trailing ``rep`` launches of a constant-``D`` call.

    For a compound call whose kernels ``@autotune`` on their first launch, the
    stream is ``[device-init + autotune burst] + ... + rep * D`` with a leading
    prefix of unknown, variable size. The trailing ``rep`` launches are always
    steady-state (the autotuner has cached its winner by then), so the last
    ``rep * D`` dispatches are exactly those launches. Fold them into ``rep``
    groups of ``D`` and sum each group, matching the CUPTI path's
    ``kernel_name=None`` per-launch sum. A capture with fewer than ``rep * D``
    dispatches is an honest failure (``D`` wrong or the autotuner never warmed),
    not a guessed split.
    """
    need = rep * per_launch
    total = len(durations_ms)
    if rep <= 0 or total == 0:
        raise KernelLaunchFailed("rocprofv3 captured no dispatches to fold")
    if total < need:
        raise KernelLaunchFailed(
            f"captured {total} dispatches, need at least rep*dispatches_per_launch="
            f"{rep}*{per_launch}={need}; dispatches_per_launch may be wrong or the "
            "autotuner never warmed"
        )
    tail = durations_ms[total - need :]
    return fmean(
        [sum(tail[i * per_launch : (i + 1) * per_launch]) for i in range(rep)]
    )


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
