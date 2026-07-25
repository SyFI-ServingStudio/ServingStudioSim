#!/usr/bin/env python3
"""Aggregate CUDA work inside fixed-label NVTX ranges from an Nsight SQLite export."""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Nsight global thread IDs store the OS thread ID in the low 24 bits. PROCESSES
# stores the process namespace base in globalPid.
_THREAD_ID_SPACE = 1 << 24

_REQUIRED_COLUMNS = {
    "PROCESSES": {"globalPid", "pid", "name"},
    "NVTX_EVENTS": {"start", "end", "text", "globalTid"},
    "CUPTI_ACTIVITY_KIND_RUNTIME": {
        "start",
        "end",
        "globalTid",
        "correlationId",
    },
    "CUPTI_ACTIVITY_KIND_KERNEL": {
        "start",
        "end",
        "globalPid",
        "correlationId",
    },
}


class SchemaError(RuntimeError):
    """Raised when an export lacks a table or column required for attribution."""


@dataclass(frozen=True)
class ProcessInfo:
    global_pid: int
    pid: int
    name: str


@dataclass(frozen=True)
class ThreadResolution:
    process: ProcessInfo | None
    os_tid: int | None
    is_main_thread: bool
    reason: str | None

    @property
    def status(self) -> str:
        if self.process is None:
            return "unmapped"
        if self.is_main_thread:
            return "mapped_main_thread"
        return "mapped_non_main_thread"


@dataclass(frozen=True)
class NvtxRange:
    label: str
    start: int
    end: int
    global_tid: int


@dataclass(frozen=True)
class RuntimeCall:
    start: int
    end: int
    correlation_id: int | None


@dataclass(frozen=True)
class Kernel:
    start: int
    end: int


def _open_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if exists is None:
        raise SchemaError(f"required table is absent: {table}")
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _validate_schema(connection: sqlite3.Connection) -> dict[str, list[str]]:
    schema: dict[str, list[str]] = {}
    for table, required in _REQUIRED_COLUMNS.items():
        columns = _table_columns(connection, table)
        missing = sorted(required - columns)
        if missing:
            raise SchemaError(f"{table} is missing required columns: {', '.join(missing)}")
        schema[table] = sorted(columns)
    return schema


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _load_ranges(
    connection: sqlite3.Connection,
    range_prefix: str,
) -> list[NvtxRange]:
    pattern = f"{_escape_like(range_prefix)}%"
    rows = connection.execute(
        """
        SELECT text, start, end, globalTid
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL
          AND text IS NOT NULL
          AND text LIKE ? ESCAPE '\\'
        ORDER BY text, start, end, globalTid
        """,
        (pattern,),
    )
    return [
        NvtxRange(
            label=str(label),
            start=int(start),
            end=int(end),
            global_tid=int(global_tid),
        )
        for label, start, end, global_tid in rows
        if int(end) >= int(start)
    ]


def _load_processes(connection: sqlite3.Connection) -> list[ProcessInfo]:
    return [
        ProcessInfo(global_pid=int(global_pid), pid=int(pid), name=str(name))
        for global_pid, pid, name in connection.execute(
            "SELECT globalPid, pid, name FROM PROCESSES ORDER BY globalPid, pid"
        )
    ]


def _resolve_thread(global_tid: int, processes: list[ProcessInfo]) -> ThreadResolution:
    candidates: list[tuple[ProcessInfo, int]] = []
    for process in processes:
        os_tid = global_tid - process.global_pid
        if 0 <= os_tid < _THREAD_ID_SPACE:
            candidates.append((process, os_tid))

    if not candidates:
        return ThreadResolution(
            process=None,
            os_tid=None,
            is_main_thread=False,
            reason="no PROCESSES namespace contains globalTid",
        )
    if len(candidates) != 1:
        return ThreadResolution(
            process=None,
            os_tid=None,
            is_main_thread=False,
            reason="multiple PROCESSES namespaces contain globalTid",
        )

    process, os_tid = candidates[0]
    return ThreadResolution(
        process=process,
        os_tid=os_tid,
        is_main_thread=global_tid == process.global_pid + process.pid,
        reason=None,
    )


def _load_runtime_calls(
    connection: sqlite3.Connection,
    relevant_tids: set[int],
) -> dict[int, list[RuntimeCall]]:
    calls: dict[int, list[RuntimeCall]] = defaultdict(list)
    if not relevant_tids:
        return calls

    for start, end, global_tid, correlation_id in connection.execute(
        """
        SELECT start, end, globalTid, correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        ORDER BY globalTid, start, end, correlationId
        """
    ):
        tid = int(global_tid)
        if tid not in relevant_tids:
            continue
        calls[tid].append(
            RuntimeCall(
                start=int(start),
                end=int(end),
                correlation_id=None if correlation_id is None else int(correlation_id),
            )
        )
    return calls


def _calls_in_range(
    runtime_calls: list[RuntimeCall],
    starts: list[int],
    nvtx_range: NvtxRange,
) -> list[RuntimeCall]:
    begin = bisect.bisect_left(starts, nvtx_range.start)
    end = bisect.bisect_left(starts, nvtx_range.end)
    return runtime_calls[begin:end]


def _load_kernels(
    connection: sqlite3.Connection,
    required_keys: set[tuple[int, int]],
) -> dict[tuple[int, int], list[Kernel]]:
    kernels: dict[tuple[int, int], list[Kernel]] = defaultdict(list)
    if not required_keys:
        return kernels

    for start, end, global_pid, correlation_id in connection.execute(
        """
        SELECT start, end, globalPid, correlationId
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY globalPid, correlationId, start, end
        """
    ):
        if correlation_id is None:
            continue
        key = (int(global_pid), int(correlation_id))
        if key in required_keys:
            kernels[key].append(Kernel(start=int(start), end=int(end)))
    return kernels


def _union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _empty_label_metrics() -> dict[str, int]:
    return {
        "occurrence_count": 0,
        "runtime_call_count": 0,
        "kernel_count": 0,
        "host_wall_time_ns": 0,
        "kernel_work_ns": 0,
        "gpu_busy_time_ns": 0,
        "mapped_main_thread_occurrences": 0,
        "mapped_non_main_thread_occurrences": 0,
        "unmapped_occurrences": 0,
    }


def aggregate(database: Path, range_prefix: str) -> dict[str, Any]:
    """Return deterministic fixed-label NVTX/CUDA attribution from a read-only export."""
    if not range_prefix:
        raise ValueError("range_prefix must be non-empty")

    with closing(_open_read_only(database)) as connection:
        schema = _validate_schema(connection)
        ranges = _load_ranges(connection, range_prefix)
        processes = _load_processes(connection)
        resolutions = {
            global_tid: _resolve_thread(global_tid, processes)
            for global_tid in {nvtx_range.global_tid for nvtx_range in ranges}
        }
        runtime_by_tid = _load_runtime_calls(
            connection,
            {nvtx_range.global_tid for nvtx_range in ranges},
        )
        runtime_starts = {
            tid: [call.start for call in calls] for tid, calls in runtime_by_tid.items()
        }

        occurrence_calls: list[list[RuntimeCall]] = []
        required_kernel_keys: set[tuple[int, int]] = set()
        for nvtx_range in ranges:
            calls = _calls_in_range(
                runtime_by_tid.get(nvtx_range.global_tid, []),
                runtime_starts.get(nvtx_range.global_tid, []),
                nvtx_range,
            )
            occurrence_calls.append(calls)
            process = resolutions[nvtx_range.global_tid].process
            if process is not None:
                required_kernel_keys.update(
                    (process.global_pid, call.correlation_id)
                    for call in calls
                    if call.correlation_id is not None
                )

        kernels_by_key = _load_kernels(connection, required_kernel_keys)

    metrics_by_label: dict[str, dict[str, int]] = defaultdict(_empty_label_metrics)
    diagnostics: dict[tuple[int, str], dict[str, Any]] = {}

    for nvtx_range, calls in zip(ranges, occurrence_calls, strict=True):
        resolution = resolutions[nvtx_range.global_tid]
        metrics = metrics_by_label[nvtx_range.label]
        metrics["occurrence_count"] += 1
        metrics["runtime_call_count"] += len(calls)
        metrics["host_wall_time_ns"] += nvtx_range.end - nvtx_range.start
        metrics[f"{resolution.status}_occurrences"] += 1

        matched_kernels: list[Kernel] = []
        if resolution.process is not None:
            correlation_keys = {
                (resolution.process.global_pid, call.correlation_id)
                for call in calls
                if call.correlation_id is not None
            }
            for key in sorted(correlation_keys):
                matched_kernels.extend(kernels_by_key.get(key, []))

        metrics["kernel_count"] += len(matched_kernels)
        metrics["kernel_work_ns"] += sum(
            max(kernel.end - kernel.start, 0) for kernel in matched_kernels
        )
        clipped_intervals = [
            (max(kernel.start, nvtx_range.start), min(kernel.end, nvtx_range.end))
            for kernel in matched_kernels
            if min(kernel.end, nvtx_range.end) > max(kernel.start, nvtx_range.start)
        ]
        metrics["gpu_busy_time_ns"] += _union_length(clipped_intervals)

        if calls:
            diagnostic_key = (nvtx_range.global_tid, resolution.status)
            diagnostic = diagnostics.setdefault(
                diagnostic_key,
                {
                    "global_tid": nvtx_range.global_tid,
                    "status": resolution.status,
                    "resolved_global_pid": (
                        None if resolution.process is None else resolution.process.global_pid
                    ),
                    "resolved_pid": (
                        None if resolution.process is None else resolution.process.pid
                    ),
                    "process_name": (
                        None if resolution.process is None else resolution.process.name
                    ),
                    "os_tid": resolution.os_tid,
                    "reason": resolution.reason,
                    "range_occurrence_count": 0,
                    "runtime_call_count": 0,
                    "labels": set(),
                },
            )
            diagnostic["range_occurrence_count"] += 1
            diagnostic["runtime_call_count"] += len(calls)
            diagnostic["labels"].add(nvtx_range.label)

    label_results: list[dict[str, Any]] = []
    for label in sorted(metrics_by_label):
        metrics = metrics_by_label[label]
        host_wall = metrics["host_wall_time_ns"]
        gpu_busy = metrics["gpu_busy_time_ns"]
        label_results.append(
            {
                "label": label,
                **metrics,
                "uncovered_host_time_ns": max(host_wall - gpu_busy, 0),
                "gpu_busy_fraction": 0.0 if host_wall == 0 else gpu_busy / host_wall,
            }
        )

    thread_diagnostics = []
    for diagnostic in sorted(
        diagnostics.values(),
        key=lambda item: (item["global_tid"], item["status"]),
    ):
        thread_diagnostics.append(
            {
                **diagnostic,
                "labels": sorted(diagnostic["labels"]),
            }
        )

    warnings = [
        (
            "NVTX labels may be nested; do not sum metrics across nesting levels. "
            "Each label is aggregated independently by occurrence."
        )
    ]
    if any(item["status"] == "mapped_non_main_thread" for item in thread_diagnostics):
        warnings.append(
            "CUDA-owning ranges were mapped from non-main threads; inspect "
            "thread_diagnostics before accepting attribution."
        )
    if any(item["status"] == "unmapped" for item in thread_diagnostics):
        warnings.append(
            "CUDA-owning ranges include unmapped threads; their runtime calls are counted "
            "but kernels are intentionally left unattributed."
        )

    return {
        "database": str(database.resolve()),
        "read_only": True,
        "range_prefix": range_prefix,
        "time_unit": "nanoseconds",
        "schema": {
            "tables": schema,
            "runtime_has_global_pid": "globalPid" in schema["CUPTI_ACTIVITY_KIND_RUNTIME"],
        },
        "labels": label_results,
        "thread_diagnostics": thread_diagnostics,
        "warnings": warnings,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate process-qualified CUDA work inside fixed-label NVTX ranges "
            "without modifying the SQLite export."
        )
    )
    parser.add_argument("sqlite", type=Path, help="Nsight Systems SQLite export")
    parser.add_argument(
        "--range-prefix",
        required=True,
        help="stable NVTX label prefix to include, for example vibeserve.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = aggregate(args.sqlite, args.range_prefix)
    except (FileNotFoundError, SchemaError, sqlite3.Error, ValueError) as error:
        print(f"aggregate_nsys.py: error: {error}", file=sys.stderr)
        return 2
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
