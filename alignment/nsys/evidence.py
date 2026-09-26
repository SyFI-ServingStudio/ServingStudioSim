"""Framework-agnostic evidence extraction from an Nsight SQLite export.

This module is the shared, read-only capture boundary for both alignment
directions. Serving profiles and real-framework optimization probes may select
different range labels, but process ownership, CUDA correlation, kernel
classification, and interval arithmetic must not fork with the comparison policy.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from functools import cache
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
    pid: int | None
    name: str | None
    source: str


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
        if self.process.source == "kernel_namespace":
            return "mapped_kernel_namespace"
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
    device_id: int
    correlation_id: int
    name: str
    category: str


def owning_global_pid(global_tid: int) -> int:
    """Return the process namespace encoded in an Nsight global thread id."""
    return global_tid - (global_tid % _THREAD_ID_SPACE)


def merge_duration_ns(intervals: list[tuple[int, int]]) -> int:
    """Return the union length of half-open nanosecond intervals."""
    if not intervals:
        return 0
    ordered_intervals = sorted(intervals)
    total = 0
    current_start, current_end = ordered_intervals[0]
    for start, end in ordered_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


@cache
def kernel_category(name: str) -> str:
    """Classify one demangled kernel name into the shared coarse taxonomy.

    Cached: a capture has ~100 distinct names but millions of launches, and the
    substring scan per launch was 9% of a 9M-kernel parse.
    """
    lowered = name.lower()
    if "multimem_all_reduce" in lowered or "cross_device_reduce" in lowered:
        return "multimem_all_reduce"
    if "nccl" in lowered:
        return "nccl_collective"
    if "fused_moe_kernel" in lowered:
        return "fused_moe"
    if "flashattn" in lowered or "flash" in lowered:
        return "attention"
    if "act_and_mul" in lowered or "silu" in lowered:
        return "activation"
    if "fillfunctor" in lowered:
        return "fill"
    if (
        "moe_align" in lowered
        or "count_and_sort_expert" in lowered
        or "moe_sum" in lowered
        or "topkgating" in lowered
    ):
        return "moe_dispatch"
    if (
        "reduce_kernel" in lowered
        or "rms" in lowered
        or "rsqrt" in lowered
        or "triton_red" in lowered
    ):
        return "norm_reduce"
    if "nvjet" in lowered or "cutlass" in lowered:
        return "gemm_or_cutlass"
    if "memcpy" in lowered or "copy" in lowered:
        return "copy_other"
    return "other"


def load_string_ids(connection: sqlite3.Connection) -> dict[int, str]:
    """Return the interned strings used throughout one Nsight export."""
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'StringIds'"
    ).fetchone()
    if exists is None:
        return {}
    return {
        int(row_id): str(value)
        for row_id, value in connection.execute("SELECT id, value FROM StringIds")
    }


def load_device_by_global_pid(connection: sqlite3.Connection) -> dict[int, int]:
    """Map every CUDA-owning process namespace to its observed device."""
    return {
        int(global_pid): int(device_id)
        for global_pid, device_id in connection.execute(
            """
            SELECT globalPid, MIN(deviceId)
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY globalPid
            """
        )
    }


def load_process_rows(connection: sqlite3.Connection) -> list[tuple[int, int, str]]:
    """Return named process rows as `(globalPid, pid, name)`."""
    return [
        (int(global_pid), int(pid), str(name))
        for global_pid, pid, name in connection.execute(
            "SELECT globalPid, pid, name FROM PROCESSES"
        )
        if name is not None
    ]


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
    string_ids_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'StringIds'"
    ).fetchone()
    if string_ids_exists is not None:
        string_columns = _table_columns(connection, "StringIds")
        missing = sorted({"id", "value"} - string_columns)
        if missing:
            raise SchemaError(f"StringIds is missing required columns: {', '.join(missing)}")
        schema["StringIds"] = sorted(string_columns)
    return schema


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _load_ranges(
    connection: sqlite3.Connection,
    range_prefix: str,
    schema: dict[str, list[str]],
) -> list[NvtxRange]:
    pattern = f"{_escape_like(range_prefix)}%"
    supports_interned_text = "StringIds" in schema and "textId" in schema["NVTX_EVENTS"]
    if supports_interned_text:
        rows = connection.execute(
            """
            SELECT COALESCE(n.text, strings.value), n.start, n.end, n.globalTid
            FROM NVTX_EVENTS n
            LEFT JOIN StringIds strings ON n.textId = strings.id
            WHERE n.end IS NOT NULL
              AND COALESCE(n.text, strings.value) IS NOT NULL
              AND COALESCE(n.text, strings.value) LIKE ? ESCAPE '\\'
            ORDER BY COALESCE(n.text, strings.value), n.start, n.end, n.globalTid
            """,
            (pattern,),
        )
    else:
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
        ProcessInfo(
            global_pid=int(global_pid),
            pid=int(pid),
            name=str(name),
            source="processes_table",
        )
        for global_pid, pid, name in connection.execute(
            "SELECT globalPid, pid, name FROM PROCESSES ORDER BY globalPid, pid"
        )
    ]


def _load_kernel_global_pids(connection: sqlite3.Connection) -> set[int]:
    return {
        int(global_pid)
        for (global_pid,) in connection.execute(
            "SELECT DISTINCT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    }


def _resolve_thread(
    global_tid: int,
    processes: list[ProcessInfo],
    kernel_global_pids: set[int],
) -> ThreadResolution:
    candidates: list[tuple[ProcessInfo, int]] = []
    for process in processes:
        os_tid = global_tid - process.global_pid
        if 0 <= os_tid < _THREAD_ID_SPACE:
            candidates.append((process, os_tid))

    if not candidates:
        namespace_global_pid = owning_global_pid(global_tid)
        if namespace_global_pid in kernel_global_pids:
            return ThreadResolution(
                process=ProcessInfo(
                    global_pid=namespace_global_pid,
                    pid=None,
                    name=None,
                    source="kernel_namespace",
                ),
                os_tid=global_tid - namespace_global_pid,
                is_main_thread=False,
                reason=(
                    "PROCESSES omits this fork child; mapped by an exact "
                    "CUPTI kernel globalPid namespace match"
                ),
            )
        return ThreadResolution(
            process=None,
            os_tid=None,
            is_main_thread=False,
            reason=(
                "no PROCESSES namespace or exact CUPTI kernel globalPid namespace "
                "contains globalTid"
            ),
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
        is_main_thread=(process.pid is not None and global_tid == process.global_pid + process.pid),
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
    string_ids: dict[int, str],
    kernel_columns: list[str],
) -> dict[tuple[int, int], list[Kernel]]:
    kernels: dict[tuple[int, int], list[Kernel]] = defaultdict(list)
    if not required_keys:
        return kernels

    device_expression = "deviceId" if "deviceId" in kernel_columns else "0"
    name_expression = "demangledName" if "demangledName" in kernel_columns else "NULL"
    query = f"""
        SELECT start, end, globalPid, correlationId,
               {device_expression}, {name_expression}
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY globalPid, correlationId, start, end
        """
    for start, end, global_pid, correlation_id, device_id, name_id in connection.execute(query):
        if correlation_id is None:
            continue
        key = (int(global_pid), int(correlation_id))
        if key in required_keys:
            name = "unknown" if name_id is None else string_ids.get(int(name_id), str(name_id))
            kernels[key].append(
                Kernel(
                    start=int(start),
                    end=int(end),
                    device_id=int(device_id),
                    correlation_id=int(correlation_id),
                    name=name,
                    category=kernel_category(name),
                )
            )
    return kernels


def _empty_label_metrics() -> dict[str, int]:
    return {
        "occurrence_count": 0,
        "runtime_call_count": 0,
        "kernel_count": 0,
        "host_wall_time_ns": 0,
        "kernel_work_ns": 0,
        "host_range_kernel_coverage_ns": 0,
        "kernel_busy_union_ns": 0,
        "kernel_span_time_ns": 0,
        "gpu_idle_within_kernel_span_ns": 0,
        "mapped_main_thread_occurrences": 0,
        "mapped_non_main_thread_occurrences": 0,
        "mapped_kernel_namespace_occurrences": 0,
        "unmapped_occurrences": 0,
    }


def aggregate(
    database: Path,
    range_prefix: str,
    *,
    include_kernel_events: bool = False,
) -> dict[str, Any]:
    """Return deterministic fixed-label NVTX/CUDA attribution from a read-only export."""
    if not range_prefix:
        raise ValueError("range_prefix must be non-empty")

    with closing(_open_read_only(database)) as connection:
        schema = _validate_schema(connection)
        ranges = _load_ranges(connection, range_prefix, schema)
        string_ids = load_string_ids(connection)
        processes = _load_processes(connection)
        kernel_global_pids = _load_kernel_global_pids(connection)
        resolutions = {
            global_tid: _resolve_thread(global_tid, processes, kernel_global_pids)
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

        kernels_by_key = _load_kernels(
            connection,
            required_kernel_keys,
            string_ids,
            schema["CUPTI_ACTIVITY_KIND_KERNEL"],
        )

    metrics_by_label: dict[str, dict[str, int]] = defaultdict(_empty_label_metrics)
    category_work_by_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    kernel_work_by_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    kernel_count_by_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    occurrence_results: list[dict[str, Any]] = []
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
        matched_kernels.sort(key=lambda kernel: (kernel.start, kernel.end, kernel.device_id))

        metrics["kernel_count"] += len(matched_kernels)
        metrics["kernel_work_ns"] += sum(
            max(kernel.end - kernel.start, 0) for kernel in matched_kernels
        )
        kernel_intervals = [
            (kernel.start, kernel.end) for kernel in matched_kernels if kernel.end > kernel.start
        ]
        kernel_busy_union = merge_duration_ns(kernel_intervals)
        kernel_span = (
            max(end for _, end in kernel_intervals) - min(start for start, _ in kernel_intervals)
            if kernel_intervals
            else 0
        )
        metrics["kernel_busy_union_ns"] += kernel_busy_union
        metrics["kernel_span_time_ns"] += kernel_span
        metrics["gpu_idle_within_kernel_span_ns"] += max(kernel_span - kernel_busy_union, 0)
        occurrence_category_work: dict[str, int] = defaultdict(int)
        for kernel in matched_kernels:
            duration = max(kernel.end - kernel.start, 0)
            category_work_by_label[nvtx_range.label][kernel.category] += duration
            kernel_work_by_label[nvtx_range.label][kernel.name] += duration
            kernel_count_by_label[nvtx_range.label][kernel.name] += 1
            occurrence_category_work[kernel.category] += duration
        clipped_intervals = [
            (max(kernel.start, nvtx_range.start), min(kernel.end, nvtx_range.end))
            for kernel in matched_kernels
            if min(kernel.end, nvtx_range.end) > max(kernel.start, nvtx_range.start)
        ]
        metrics["host_range_kernel_coverage_ns"] += merge_duration_ns(clipped_intervals)

        if include_kernel_events:
            occurrence_results.append(
                {
                    "label": nvtx_range.label,
                    "start_ns": nvtx_range.start,
                    "end_ns": nvtx_range.end,
                    "host_wall_time_ns": nvtx_range.end - nvtx_range.start,
                    "global_tid": nvtx_range.global_tid,
                    "resolved_global_pid": (
                        None if resolution.process is None else resolution.process.global_pid
                    ),
                    "device_ids": sorted({kernel.device_id for kernel in matched_kernels}),
                    "kernel_count": len(matched_kernels),
                    "kernel_work_ns": sum(
                        max(kernel.end - kernel.start, 0) for kernel in matched_kernels
                    ),
                    "kernel_busy_union_ns": kernel_busy_union,
                    "kernel_span_time_ns": kernel_span,
                    "gpu_idle_within_kernel_span_ns": max(kernel_span - kernel_busy_union, 0),
                    "kernel_work_ns_by_category": dict(sorted(occurrence_category_work.items())),
                    "kernels": [
                        {
                            "start_ns": kernel.start,
                            "end_ns": kernel.end,
                            "device_id": kernel.device_id,
                            "correlation_id": kernel.correlation_id,
                            "name": kernel.name,
                            "category": kernel.category,
                        }
                        for kernel in matched_kernels
                    ],
                }
            )

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
        host_range_kernel_coverage = metrics["host_range_kernel_coverage_ns"]
        label_results.append(
            {
                "label": label,
                **metrics,
                "uncovered_host_time_ns": max(host_wall - host_range_kernel_coverage, 0),
                # Compatibility aliases for the original skill-owned report.
                "gpu_busy_time_ns": host_range_kernel_coverage,
                "gpu_busy_fraction": (
                    0.0 if host_wall == 0 else host_range_kernel_coverage / host_wall
                ),
                "host_range_kernel_coverage_fraction": (
                    0.0 if host_wall == 0 else host_range_kernel_coverage / host_wall
                ),
                "kernel_work_ns_by_category": dict(sorted(category_work_by_label[label].items())),
                "top_kernels": [
                    {
                        "name": name,
                        "kernel_work_ns": duration,
                        "count": kernel_count_by_label[label][name],
                    }
                    for name, duration in sorted(
                        kernel_work_by_label[label].items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
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
        ),
        (
            "host_range_kernel_coverage_ns is clipped to each host range. GPU idle is "
            "gpu_idle_within_kernel_span_ns; the two are not interchangeable when "
            "launches are asynchronous."
        ),
    ]
    if any(item["status"] == "mapped_non_main_thread" for item in thread_diagnostics):
        warnings.append(
            "CUDA-owning ranges were mapped from non-main threads; inspect "
            "thread_diagnostics before accepting attribution."
        )
    if any(item["status"] == "mapped_kernel_namespace" for item in thread_diagnostics):
        warnings.append(
            "PROCESSES omitted one or more fork children; their CUDA work was mapped "
            "only where globalTid's namespace exactly matched a kernel globalPid."
        )
    if any(item["status"] == "unmapped" for item in thread_diagnostics):
        warnings.append(
            "CUDA-owning ranges include unmapped threads; their runtime calls are counted "
            "but kernels are intentionally left unattributed."
        )

    report = {
        "schema_version": 1,
        "artifact_kind": "nsys_range_evidence",
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
    if include_kernel_events:
        report["occurrences"] = occurrence_results
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract process-qualified CUDA evidence inside fixed-label NVTX ranges "
            "without modifying the SQLite export."
        )
    )
    parser.add_argument("sqlite", type=Path, help="Nsight Systems SQLite export")
    parser.add_argument(
        "--range-prefix",
        required=True,
        help="stable NVTX label prefix to include, for example vibeserve.",
    )
    parser.add_argument(
        "--kernel-events-output",
        type=Path,
        help="write ordered per-occurrence kernel events to this separate JSON artifact",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = aggregate(
            args.sqlite,
            args.range_prefix,
            include_kernel_events=args.kernel_events_output is not None,
        )
    except (FileNotFoundError, SchemaError, sqlite3.Error, ValueError) as error:
        print(f"alignment ranges: error: {error}", file=sys.stderr)
        return 2
    if args.kernel_events_output is not None:
        occurrences = report.pop("occurrences")
        args.kernel_events_output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_kind": "nsys_kernel_events",
                    "database": report["database"],
                    "range_prefix": report["range_prefix"],
                    "time_unit": report["time_unit"],
                    "occurrences": occurrences,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
