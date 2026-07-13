"""Parse an exported Nsight Systems SQLite trace into normalized kernel data.

This is the alignment ground-truth side: given an nsys `.sqlite` export of a
vLLM run (with the instrumented fork's `vllm_iteration(N): <phase>` NVTX ranges),
attribute every CUDA kernel to the iteration/phase that launched it, preserve the
complete ordered launch list, and sum kernel busy time per **category**
(attention / gemm / norm / …) per device, per iteration.

Faithfully ported from the reference harness
(`ref/.../phase2_profiling/nsys_iter_analysis.py`): `KernelEvent`, `RangeStats`,
`kernel_category`, `load_string_ids/workers/ranges`, `merge_duration_ns`, and the
**ownership-by-correlationId** attribution (`attach_kernels_by_correlation`) —
kernel → runtime `correlationId` → enclosing NVTX phase. Ownership (not
time-overlap clipping) is required for CUDA-graph decode iters, whose kernels can
execute past the `forward` NVTX range into `sample`/`bookkeep`.

The kernel taxonomy is model-agnostic; the dense (Llama3) buckets it produces are
`attention`, `gemm_or_cutlass`, `norm_reduce`, `copy_other`/`other`. The MoE/comm
categories stay in the table (harmless — they just don't appear for a tp=1 dense
run) so the parser is reused unchanged when parallel/MoE arches land.

This module stops at the nsys-source boundary: it resolves process/range/kernel
ownership and emits JSON-ready records. Cross-source statistics, reports, and
plots belong to the repository-level `analyzer/`.

SQLite tables read: `StringIds`, `PROCESSES`, `NVTX_EVENTS`,
`CUPTI_ACTIVITY_KIND_KERNEL`, `CUPTI_ACTIVITY_KIND_RUNTIME`.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .sequence import build_kernel_sequences

# Alignment traces require inline, indexed iteration markers. Stage information
# comes from the same iteration index in the metrics JSONL.
ITER_RE = re.compile(r"^(?:vllm|sglang)_iteration\((\d+)\):\s*(.+)$")

_RUNTIME_LOOKUP_INDEX = "vibesim_runtime_gtid_start_idx"
_KERNEL_LOOKUP_INDEX = "vibesim_kernel_gpid_corr_start_idx"


def iteration_kind(metric: dict | None, fallback: str = "all") -> str:
    """Classify one workload unit without collapsing mixed into prefill."""
    if metric is None:
        return fallback
    has_prefill = int(metric.get("prefill_tokens", 0)) > 0
    has_decode = int(metric.get("decode_requests", metric.get("decode_tokens_scheduled", 0))) > 0
    if has_prefill and has_decode:
        return "mixed"
    if has_prefill:
        return "prefill"
    if has_decode:
        return "decode"
    return fallback


def _parse_label(label: str) -> tuple[int, str] | None:
    """Return the indexed iteration and phase for a canonical NVTX marker."""
    m = ITER_RE.match(label)
    if m:
        return int(m.group(1)), m.group(2)
    return None


@dataclass(frozen=True)
class Worker:
    global_pid: int
    pid: int
    name: str
    device_id: int | None


@dataclass(frozen=True)
class KernelEvent:
    start: int
    end: int
    name: str
    category: str
    stream_id: int

    @property
    def duration_ns(self) -> int:
        return self.end - self.start


@dataclass
class RangeStats:
    iteration: int
    phase: str
    stage: str
    worker: Worker
    start: int
    end: int
    intervals: list[tuple[int, int]] = field(default_factory=list)
    kernel_count: int = 0
    sum_ns: int = 0
    category_ns: Counter[str] = field(default_factory=Counter)
    kernel_name_ns: Counter[str] = field(default_factory=Counter)
    kernel_name_count: Counter[str] = field(default_factory=Counter)
    kernel_events: list[KernelEvent] = field(default_factory=list)

    @property
    def window_ms(self) -> float:
        return (self.end - self.start) / 1e6

    @property
    def sum_ms(self) -> float:
        return self.sum_ns / 1e6

    @property
    def busy_ms(self) -> float:
        return merge_duration_ns(self.intervals) / 1e6

    @property
    def idle_ms(self) -> float:
        return max(0.0, self.window_ms - self.busy_ms)


def merge_duration_ns(intervals: list[tuple[int, int]]) -> int:
    """Total wall time covered by a set of [start, end) intervals (union)."""
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        elif end > cur_end:
            cur_end = end
    total += cur_end - cur_start
    return total


def kernel_category(name: str) -> str:
    """Map a demangled kernel name to a coarse op category (reference taxonomy)."""
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


def load_metrics(path: Path | None) -> dict[int, dict]:
    """iteration_index → vLLM per-iteration metrics row (from the metrics jsonl)."""
    if path is None:
        return {}
    metrics: dict[int, dict] = {}
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            iteration = row.get("iteration_index", row.get("iteration_num"))
            if iteration is None:
                continue
            metrics[int(iteration)] = row
    return metrics


def load_string_ids(con: sqlite3.Connection) -> dict[int, str]:
    return {
        int(row_id): str(value) for row_id, value in con.execute("SELECT id, value FROM StringIds")
    }


def ensure_query_indexes(con: sqlite3.Connection) -> None:
    """Persist the two indexes required by correlation-based attribution.

    NSYS exports these CUPTI tables without indexes. Attribution issues two
    selective queries for every indexed phase range, so leaving them unindexed
    turns a few thousand ranges into repeated full-table scans. The SQLite file
    is a rebuildable derivative of the immutable ``.nsys-rep`` capture; adding
    B-tree indexes changes neither source rows nor attribution semantics.
    """
    con.execute(
        f"CREATE INDEX IF NOT EXISTS {_RUNTIME_LOOKUP_INDEX} "
        "ON CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, start)"
    )
    con.execute(
        f"CREATE INDEX IF NOT EXISTS {_KERNEL_LOOKUP_INDEX} "
        "ON CUPTI_ACTIVITY_KIND_KERNEL(globalPid, correlationId, start)"
    )
    # Make the one-time acceleration reusable by standalone parse/analyze runs.
    # SQLite DDL is transactional, so an interrupted build cannot leave a
    # partially formed index behind.
    con.commit()


def load_workers(con: sqlite3.Connection) -> dict[int, Worker]:
    """Map Nsight globalTid (globalPid + pid of the Python main thread) → Worker.

    Devices come from the min CUPTI kernel deviceId per globalPid; workers are the
    `VLLM::Worker` (or sglang) processes, falling back to bare `python` procs.
    """
    device_by_pid = {
        int(global_pid): int(device_id)
        for global_pid, device_id in con.execute(
            """
            SELECT globalPid, MIN(deviceId)
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY globalPid
            """
        )
    }
    process_rows = [
        (int(global_pid), int(pid), str(name))
        for global_pid, pid, name in con.execute("SELECT globalPid, pid, name FROM PROCESSES")
        if name is not None
    ]

    workers: dict[int, Worker] = {}
    # nsys only traces CUDA for the launched process tree, so `device_by_pid` is
    # already narrowed to *this* run's kernel-launching processes even though
    # PROCESSES lists every process on the box. Prefer the named vLLM/sglang
    # workers, then python interpreters, then — since a process that launched CUDA
    # kernels in our own capture IS a worker regardless of its (setproctitle-
    # truncated, e.g. "VLLM::EngineCor") comm — any process with a device.
    on_device = [row for row in process_rows if row[0] in device_by_pid]
    candidate_rows = [
        row
        for row in on_device
        if "VLLM::Worker" in row[2]
        or "EngineCor" in row[2]
        or row[2].startswith("sglang::scheduler")
    ]
    if not candidate_rows:
        candidate_rows = [row for row in on_device if row[2] in {"python", "python3"}]
    if not candidate_rows:
        candidate_rows = on_device

    for global_pid, pid, name in candidate_rows:
        worker = Worker(
            global_pid=int(global_pid),
            pid=int(pid),
            name=str(name),
            device_id=device_by_pid.get(int(global_pid)),
        )
        # Nsight globalTid for the Python main thread is globalPid + pid.
        workers[worker.global_pid + worker.pid] = worker
    return workers


def load_ranges(
    con: sqlite3.Connection,
    workers_by_gtid: dict[int, Worker],
    metrics: dict[int, dict],
    iteration_start: int,
    iteration_end: int,
    range_mode: str,
    default_stage: str = "all",
) -> list[RangeStats]:
    """Extract per-worker NVTX iteration ranges in [iteration_start, iteration_end].

    Only inline, indexed `vllm_iteration(N): <phase>` and
    `sglang_iteration(N): <phase>` markers are valid inputs. A trace without
    this contract is rejected naturally by producing no ranges.

    `range_mode="forward"` keeps only the forward phase (the usual alignment
    target); "phases" keeps every phase; "envelope" merges an iteration's phases.
    `default_stage` tags ranges whose iteration is absent from the metrics join
    (use "decode" only for a capture window independently known to be decode).
    """
    parsed_rows = []  # [iteration, phase, worker, start, end]
    for start, end, global_tid, label in con.execute(
        """
        SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value)
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON n.textId = s.id
        WHERE COALESCE(n.text, s.value) LIKE 'vllm_iteration(%'
           OR COALESCE(n.text, s.value) LIKE 'sglang_iteration(%'
        """
    ):
        parsed = _parse_label(str(label))
        if parsed is None:
            continue
        iteration, phase = parsed
        worker = workers_by_gtid.get(int(global_tid))
        if worker is None or worker.device_id is None:
            continue
        parsed_rows.append([iteration, phase, worker, int(start), int(end)])

    if range_mode == "forward":
        parsed_rows = [r for r in parsed_rows if r[1] == "forward"]

    def resolve_stage(iteration: int) -> str:
        return iteration_kind(metrics.get(iteration), default_stage)

    windowed = [r for r in parsed_rows if iteration_start <= r[0] <= iteration_end]

    if range_mode in ("forward", "phases"):
        return [
            RangeStats(
                iteration=iteration,
                phase=phase,
                stage=resolve_stage(iteration),
                worker=worker,
                start=start,
                end=end,
            )
            for iteration, phase, worker, start, end in windowed
        ]

    grouped: dict[tuple[int, int], list] = defaultdict(list)
    for iteration, phase, worker, start, end in windowed:
        grouped[(iteration, worker.global_pid)].append((phase, worker, start, end))

    ranges = []
    for (iteration, _), items in grouped.items():
        worker = items[0][1]
        ranges.append(
            RangeStats(
                iteration=iteration,
                phase="iteration_envelope",
                stage=resolve_stage(iteration),
                worker=worker,
                start=min(item[2] for item in items),
                end=max(item[3] for item in items),
            )
        )
    return ranges


def attach_kernels_by_correlation(
    con: sqlite3.Connection,
    string_ids: dict[int, str],
    ranges: list[RangeStats],
) -> int:
    """Attach kernels owned by CUDA runtime calls launched inside each NVTX range.

    Ownership path: for each range, find runtime API calls on the worker's
    globalTid within [start, end), collect their correlationIds, then pull every
    kernel with a matching correlationId on that globalPid. Robust to CUDA-graph
    kernels that execute outside the range wall-clock.
    """
    kernel_rows = 0
    for item in ranges:
        global_tid = item.worker.global_pid + item.worker.pid
        correlation_ids = [
            int(row[0])
            for row in con.execute(
                """
                SELECT correlationId
                FROM CUPTI_ACTIVITY_KIND_RUNTIME
                WHERE globalTid = ?
                  AND start >= ?
                  AND start < ?
                  AND correlationId IS NOT NULL
                """,
                (global_tid, item.start, item.end),
            )
        ]
        if not correlation_ids:
            continue

        placeholders = ",".join("?" for _ in correlation_ids)
        params = [item.worker.global_pid, *correlation_ids]
        query = f"""
            SELECT
                k.start,
                k.end,
                k.demangledName,
                k.streamId
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            WHERE k.globalPid = ?
              AND k.correlationId IN ({placeholders})
            ORDER BY k.start
        """
        for (
            start,
            end,
            name,
            stream_id,
        ) in con.execute(query, params):
            kernel_rows += 1
            start = int(start)
            end = int(end)
            duration = end - start
            kernel_name = string_ids.get(int(name), str(name))
            category = kernel_category(kernel_name)
            item.intervals.append((start, end))
            item.kernel_count += 1
            item.sum_ns += duration
            item.category_ns[category] += duration
            item.kernel_name_ns[kernel_name] += duration
            item.kernel_name_count[kernel_name] += 1
            item.kernel_events.append(
                KernelEvent(
                    start=start,
                    end=end,
                    name=kernel_name,
                    category=category,
                    stream_id=int(stream_id),
                )
            )
    return kernel_rows


def build_iteration_details(
    ranges: list[RangeStats],
    metrics: dict[int, dict],
    kernel_name_ids: dict[str, int],
) -> list[dict]:
    """Serialize every owned kernel in launch order, grouped by iteration.

    Summary categories are convenient analyzer inputs, but this list is the
    lossless ground-truth artifact. Kernel names are never truncated here.
    """
    by_iteration: dict[int, list[RangeStats]] = defaultdict(list)
    for item in ranges:
        by_iteration[item.iteration].append(item)

    details = []
    for iteration in sorted(by_iteration):
        items = sorted(
            by_iteration[iteration],
            key=lambda item: (
                item.worker.device_id if item.worker.device_id is not None else -1,
                item.start,
                item.phase,
            ),
        )
        metric = metrics.get(iteration)
        iteration_type = iteration_kind(metric, items[0].stage)

        serialized_ranges = []
        for item in items:
            kernels = []
            for ordinal, event in enumerate(item.kernel_events, start=1):
                kernels.append(
                    {
                        "ordinal": ordinal,
                        "name_id": kernel_name_ids[event.name],
                        "category": event.category,
                        "start_ns": event.start,
                        "end_ns": event.end,
                        "stream_id": event.stream_id,
                    }
                )
            serialized_ranges.append(
                {
                    "device_id": item.worker.device_id,
                    "worker": {
                        "name": item.worker.name,
                        "pid": item.worker.pid,
                        "global_pid": item.worker.global_pid,
                    },
                    "phase": item.phase,
                    "start_ns": item.start,
                    "end_ns": item.end,
                    "duration_ns": item.end - item.start,
                    "kernel_count": item.kernel_count,
                    "kernel_sum_duration_ns": item.sum_ns,
                    "kernel_busy_union_ns": merge_duration_ns(item.intervals),
                    "kernels": kernels,
                }
            )

        details.append(
            {
                "iteration": iteration,
                "iteration_type": iteration_type,
                "stage": items[0].stage,
                "metrics": metric,
                "ranges": serialized_ranges,
            }
        )
    return details


def build_kernel_name_index(
    ranges: list[RangeStats],
) -> tuple[dict[str, int], dict[int, str]]:
    """Assign compact integer IDs to names in first-launch order."""
    name_ids: dict[str, int] = {}
    for item in sorted(
        ranges,
        key=lambda item: (
            item.iteration,
            item.worker.device_id if item.worker.device_id is not None else -1,
            item.start,
            item.phase,
        ),
    ):
        for event in item.kernel_events:
            if event.name not in name_ids:
                name_ids[event.name] = len(name_ids) + 1
    return name_ids, {name_id: name for name, name_id in name_ids.items()}


def summarize_devices(ranges: list[RangeStats], top_n: int = 12) -> dict:
    """Per-device, per-category kernel busy time averaged over the iterations.

    Returns a dict keyed by device_id string, each with `category_ms_per_iter`
    (the gap-table input), the merged `busy_ms`/`window_ms`/`idle_ms` means, the
    iteration count, and `top_kernels` (name → ms/iter) for residual naming.
    """
    out: dict[str, dict] = {}
    devices = sorted({r.worker.device_id for r in ranges if r.worker.device_id is not None})
    for device_id in devices:
        items = [r for r in ranges if r.worker.device_id == device_id]
        n_iters = max(1, len({r.iteration for r in items}))
        cat_totals: Counter[str] = Counter()
        kern_totals: Counter[str] = Counter()
        kern_counts: Counter[str] = Counter()
        for r in items:
            cat_totals.update(r.category_ns)
            kern_totals.update(r.kernel_name_ns)
            kern_counts.update(r.kernel_name_count)
        out[str(device_id)] = {
            "n_iters": n_iters,
            "n_ranges": len(items),
            "window_ms_mean": sum(r.window_ms for r in items) / len(items),
            "busy_ms_mean": sum(r.busy_ms for r in items) / len(items),
            "idle_ms_mean": sum(r.idle_ms for r in items) / len(items),
            "category_ms_per_iter": {
                cat: ns / 1e6 / n_iters for cat, ns in cat_totals.most_common()
            },
            "top_kernels": [
                {
                    "name": name[:160],
                    "ms_per_iter": ns / 1e6 / n_iters,
                    "calls_per_iter": kern_counts[name] / n_iters,
                }
                for name, ns in kern_totals.most_common(top_n)
            ],
        }
    return out


def parse_trace(
    sqlite_path: Path,
    metrics_jsonl: Path | None,
    iteration_start: int,
    iteration_end: int,
    *,
    range_mode: str = "phases",
    default_stage: str = "all",
    top_n: int = 12,
) -> dict:
    """Top-level parse: nsys sqlite → per-device per-category busy time / iter.

    The alignment ground truth. Groups the analysis window by nsys `stage`
    ("mixed"/"decode") so a prefill target and a decode target can be selected
    independently from one capture. `default_stage` tags ranges with no
    label/metrics stage (use "decode" for a pure-decode capture window).
    """
    con = sqlite3.connect(str(sqlite_path))
    try:
        ensure_query_indexes(con)
        metrics = load_metrics(metrics_jsonl)
        string_ids = load_string_ids(con)
        workers = load_workers(con)
        ranges = load_ranges(
            con,
            workers,
            metrics,
            iteration_start,
            iteration_end,
            range_mode,
            default_stage,
        )
        kernel_rows = attach_kernels_by_correlation(con, string_ids, ranges)
    finally:
        con.close()

    iterations = sorted({r.iteration for r in ranges})
    stages = sorted({r.stage for r in ranges})
    kernel_name_ids, kernel_names = build_kernel_name_index(ranges)
    by_stage = {
        stage: summarize_devices([r for r in ranges if r.stage == stage], top_n) for stage in stages
    }
    phases = sorted({r.phase for r in ranges})
    by_phase = {}
    for phase in phases:
        phase_ranges = [item for item in ranges if item.phase == phase]
        phase_stages = sorted({item.stage for item in phase_ranges})
        by_phase[phase] = {
            "by_stage": {
                stage: summarize_devices(
                    [item for item in phase_ranges if item.stage == stage], top_n
                )
                for stage in phase_stages
            },
            "all": summarize_devices(phase_ranges, top_n),
        }
    iteration_details = build_iteration_details(ranges, metrics, kernel_name_ids)
    return {
        "schema_version": 2,
        "sqlite": str(sqlite_path),
        "iteration_start": iteration_start,
        "iteration_end": iteration_end,
        "range_mode": range_mode,
        "iterations": iterations,
        "scanned_kernel_rows": kernel_rows,
        "stages": stages,
        "phases": phases,
        "kernel_names": kernel_names,
        "iteration_details": iteration_details,
        "kernel_sequences": build_kernel_sequences(iteration_details, kernel_names),
        "by_phase": by_phase,
        "by_stage": by_stage,
        "all": summarize_devices(ranges, top_n),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse nsys sqlite → per-op busy time")
    parser.add_argument("--sqlite", type=Path, required=True, help="Exported nsys SQLite database")
    parser.add_argument("--metrics", type=Path, default=None, help="vLLM iteration metrics JSONL")
    parser.add_argument("--iteration-start", type=int, required=True)
    parser.add_argument("--iteration-end", type=int, required=True)
    parser.add_argument("--range-mode", choices=["forward", "envelope", "phases"], default="phases")
    parser.add_argument(
        "--default-stage",
        default="all",
        help="stage for ranges lacking a label/metrics stage (e.g. 'decode' for a decode window)",
    )
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON here (else stdout)")
    parser.add_argument(
        "--sequences-output",
        type=Path,
        default=None,
        help="Also write the exact unique full-sequence catalog here",
    )
    return parser


def write_kernel_sequences(path: Path, parsed: dict, source_parsed: Path | None) -> None:
    """Write the folded, label-ready sequence inventory separately from parsed.json."""
    document = {
        "schema_version": 2,
        "encoding": "folded-v1",
        "source_parsed": str(source_parsed) if source_parsed is not None else None,
        "folding_policy": {
            "kind": "exact_contiguous_repeat",
            "match_fields": ["name", "suggested_category"],
            "row_identity": "sequence_id:expanded_ordinal",
        },
        "phases": parsed["kernel_sequences"],
    }
    Path(path).write_text(_format_sequence_document(document))


def _format_sequence_document(document: dict) -> str:
    """Pretty-print structure while keeping compact integer iteration lists."""
    text = json.dumps(document, indent=2)
    pattern = re.compile(r'("iterations": \[)\n((?:\s+\d+,?\n)+)\s+\]')

    def inline_iterations(match: re.Match[str]) -> str:
        values = [part.strip().rstrip(",") for part in match.group(2).splitlines()]
        return f"{match.group(1)}{', '.join(values)}]"

    return pattern.sub(inline_iterations, text)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = parse_trace(
        args.sqlite,
        args.metrics,
        args.iteration_start,
        args.iteration_end,
        range_mode=args.range_mode,
        default_stage=args.default_stage,
        top_n=args.top_n,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    if args.sequences_output:
        write_kernel_sequences(args.sequences_output, result, args.output)
        print(f"wrote {args.sequences_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
