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

`--host-timeline-output` additionally writes the **host** side of the same
capture — every NVTX range and every CUDA runtime call, on every thread that
carries either — as a separate sidecar. It is deliberately lossless and
un-interpreted: absolute nsys nanoseconds, no iteration windows, no nesting
depth, no call classification. Those are all anchoring decisions, and anchoring
belongs to the analyzer that already owns the anchor rule.

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

from .evidence import (
    kernel_category,
    load_device_by_global_pid,
    load_process_rows,
    load_string_ids,
    merge_duration_ns,
    owning_global_pid,
)
from .sequence import build_device_kernel_sequences

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
    correlation_id: int | None = None

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
    #: The thread that emitted the NVTX range, and therefore the thread whose
    #: launches this range owns. Left unset by fixtures and by the envelope
    #: path, where `launch_global_tid` falls back to the worker's main thread.
    emitting_global_tid: int | None = None
    intervals: list[tuple[int, int]] = field(default_factory=list)
    kernel_count: int = 0
    sum_ns: int = 0
    category_ns: Counter[str] = field(default_factory=Counter)
    kernel_name_ns: Counter[str] = field(default_factory=Counter)
    kernel_name_count: Counter[str] = field(default_factory=Counter)
    kernel_events: list[KernelEvent] = field(default_factory=list)

    @property
    def launch_global_tid(self) -> int:
        """The globalTid whose CUDA runtime calls this range owns.

        The recorded emitter when there is one. Otherwise the process's main
        thread, reconstructed as globalPid + pid -- which only works when the
        process was named in the trace's process table, so it is the fallback
        rather than the rule.
        """
        if self.emitting_global_tid is not None:
            return self.emitting_global_tid
        return self.worker.global_pid + self.worker.pid

    @property
    def nvtx_window_ms(self) -> float:
        """Length of the NVTX range, which is a HOST fact.

        Named for what it is, because it reads like a GPU quantity and is not.
        Under CUDA graphs the range routinely closes before its own kernels
        finish -- on this capture's decode iterations the correlated GPU busy
        time reaches 149% of it -- so it bounds nothing on the device.
        """
        return (self.end - self.start) / 1e6

    @property
    def sum_ms(self) -> float:
        return self.sum_ns / 1e6

    @property
    def busy_ms(self) -> float:
        return merge_duration_ns(self.intervals) / 1e6

    @property
    def kernel_span_ms(self) -> float:
        """First kernel start to last kernel end: the correlated GPU span.

        This, not the NVTX window, is what idle has to be measured against.
        """
        if not self.intervals:
            return 0.0
        span_ns = max(end for _, end in self.intervals) - min(start for start, _ in self.intervals)
        return span_ns / 1e6

    @property
    def idle_ms(self) -> float:
        """GPU time inside the kernel span with no kernel on it.

        Was `nvtx_window_ms - busy_ms`, where the `max(0.0, ...)` silently
        clamped to zero exactly when busy exceeded the range -- so the reported
        idle of a CUDA-graph capture was zero no matter how much the GPU stalled.
        """
        return max(0.0, self.kernel_span_ms - self.busy_ms)


def load_metrics(path: Path | None) -> dict[tuple[int, int], dict]:
    """(dp_rank, iteration_index) → vLLM per-iteration metrics row.

    Under data parallelism each rank owns an independent scheduler and numbers
    its own iterations, so several rows share one `iteration_index`. Keying on
    the rank as well keeps every rank's batch shape; a capture without rank
    provenance (`dp_rank` absent) is a single-EngineCore run and lands on rank 0.
    """
    if path is None:
        return {}
    metrics: dict[tuple[int, int], dict] = {}
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            iteration = row.get("iteration_index", row.get("iteration_num"))
            if iteration is None:
                continue
            key = (int(row.get("dp_rank", 0)), int(iteration))
            if key in metrics:
                raise ValueError(
                    f"duplicate metrics record for dp_rank {key[0]} iteration {key[1]}"
                )
            metrics[key] = row
    return metrics


def fold_rank_metrics(rows: list[dict], iteration_index: int) -> dict:
    """One replica batch shape out of the per-rank rows of ONE wall-clock step.

    The rows must already be the ranks of the same step. Which rows those are is
    not a question this function can answer: ranks number their own iterations
    (see `align_ranges_into_steps`), so the answer comes from the measured
    timeline, not from the index.
    """
    adapters = {row.get("input_adapter") for row in rows}
    if len(adapters) > 1:
        raise ValueError(f"iteration {iteration_index} mixes input adapters {sorted(adapters)}")
    return {
        "schema_version": min(int(row.get("schema_version", 1)) for row in rows),
        "input_adapter": next(iter(adapters)),
        "iteration_index": iteration_index,
        "dp_ranks": [int(row.get("dp_rank", 0)) for row in rows],
        "prefill_tokens": sum(int(row.get("prefill_tokens", 0)) for row in rows),
        "decode_requests": sum(int(row.get("decode_requests", 0)) for row in rows),
        "decode_tokens_scheduled": sum(int(row.get("decode_tokens_scheduled", 0)) for row in rows),
        "prefill_chunk_pairs": [
            pair for row in rows for pair in row.get("prefill_chunk_pairs", [])
        ],
        "decode_kv_lens": [kv_len for row in rows for kv_len in row.get("decode_kv_lens", [])],
    }


def aggregate_metrics_by_iteration(rank_metrics: dict[tuple[int, int], dict]) -> dict[int, dict]:
    """Fold the ranks that share an `iteration_index` into one batch shape.

    Index-keyed, and therefore only correct when every rank numbers its steps
    alike — a single-EngineCore run, or a capture whose ranks never diverged.
    Under data parallelism they do diverge (a rank that runs a prefill chunk
    alone advances its counter while its peers run uncounted dummy steps), and
    the wall-clock grouping in `align_ranges_into_steps` is what pairs the ranks
    there. This stays for the replica-level, timeline-free views.
    """
    grouped: dict[int, list[dict]] = defaultdict(list)
    for (_, iteration), row in sorted(rank_metrics.items()):
        grouped[iteration].append(row)
    return {iteration: fold_rank_metrics(rows, iteration) for iteration, rows in grouped.items()}


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

    Also keyed by bare globalPid, so a range can be resolved by masking its
    globalTid when the thread-level key misses -- see `worker_for_global_tid`.
    """
    device_by_pid = load_device_by_global_pid(con)
    process_rows = load_process_rows(con)

    workers: dict[int, Worker] = {}
    # Prefer the named vLLM/sglang workers, then python interpreters, then —
    # since a process that launched CUDA kernels in our own capture IS a worker
    # regardless of its (setproctitle-truncated, e.g. "VLLM::EngineCor") comm —
    # any process with a device.
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

    # A process that ran kernels but has no row in the process table is still a
    # worker: Nsight does not always record a comm for a process it only saw
    # through CUPTI, and an SGLang capture hit exactly that. Its kernels are
    # evidence enough, so it is admitted under its globalPid with the name left
    # honestly unknown rather than dropped -- dropping it empties the worker set
    # and the whole capture parses to zero ranges.
    for global_pid, device_id in device_by_pid.items():
        if any(worker.global_pid == global_pid for worker in workers.values()):
            continue
        workers[int(global_pid)] = Worker(
            global_pid=int(global_pid),
            pid=0,
            name="unknown",
            device_id=device_id,
        )
    return workers


def worker_for_global_tid(workers_by_gtid: dict[int, Worker], global_tid: int) -> Worker | None:
    """Resolve the worker that emitted an NVTX range on `global_tid`.

    The exact key works for a range emitted on the process's main thread, which
    is where both forks emit theirs. The fallback masks the globalTid down to
    its globalPid -- Nsight packs the thread id into the low 24 bits -- so a
    range still resolves when it came from another thread of the same process,
    or when that process is only known by its globalPid.
    """
    worker = workers_by_gtid.get(int(global_tid))
    if worker is not None:
        return worker
    return workers_by_gtid.get(owning_global_pid(int(global_tid)))


def window_rows_by_reference_rank(
    parsed_rows: list[list], iteration_start: int, iteration_end: int
) -> list[list]:
    """Keep the reference rank's numbered window, and its peers by overlap.

    `parsed_rows` is `[iteration, phase, worker, start_ns, end_ns, global_tid]`. The
    reference rank is the lowest device id present — the same choice
    `align_ranges_into_steps` makes, so the two agree on whose clock the capture
    is described in.
    """
    devices = {row[2].device_id for row in parsed_rows if row[2].device_id is not None}
    if not devices:
        return [row for row in parsed_rows if iteration_start <= row[0] <= iteration_end]
    reference_device_id = min(devices)
    reference_rows = [
        row
        for row in parsed_rows
        if row[2].device_id == reference_device_id and iteration_start <= row[0] <= iteration_end
    ]
    if not reference_rows:
        return []
    span_start = min(row[3] for row in reference_rows)
    span_end = max(row[4] for row in reference_rows)

    def mostly_inside(row: list) -> bool:
        # More than half of the range inside the span, not merely touching it:
        # the step before the window ends where the window begins, so a
        # touch-anywhere rule admits it on a nanosecond of overlap.
        overlap = min(row[4], span_end) - max(row[3], span_start)
        return overlap * 2 > max(row[4] - row[3], 1)

    return reference_rows + [
        row for row in parsed_rows if row[2].device_id != reference_device_id and mostly_inside(row)
    ]


def load_ranges(
    con: sqlite3.Connection,
    workers_by_gtid: dict[int, Worker],
    metrics: dict[int, dict],
    iteration_start: int,
    iteration_end: int,
    range_mode: str,
    default_stage: str = "all",
) -> list[RangeStats]:
    """Extract the NVTX iteration ranges of the window `[iteration_start, iteration_end]`.

    Only inline, indexed `vllm_iteration(N): <phase>` and
    `sglang_iteration(N): <phase>` markers are valid inputs. A trace without
    this contract is rejected naturally by producing no ranges.

    The window is an interval of the REFERENCE rank's iteration numbers, and the
    other ranks join it by measured time. Applying the same numeric interval to
    every rank looks equivalent and is not: a rank numbers its own scheduled
    steps, and in the GLM-5.2 DP8 capture the ranks that step together at
    t=15.67 s call it iteration 11, 7 and 6. Filtering each rank on its own
    number would then drop seven of the eight ranks from the first steps of the
    window and leave a step that looks like one rank running alone — which is
    what happened, and what this rule exists to prevent.

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
        worker = worker_for_global_tid(workers_by_gtid, int(global_tid))
        if worker is None or worker.device_id is None:
            continue
        parsed_rows.append([iteration, phase, worker, int(start), int(end), int(global_tid)])

    if range_mode == "forward":
        parsed_rows = [r for r in parsed_rows if r[1] == "forward"]

    def resolve_stage(iteration: int) -> str:
        return iteration_kind(metrics.get(iteration), default_stage)

    windowed = window_rows_by_reference_rank(parsed_rows, iteration_start, iteration_end)

    if range_mode in ("forward", "phases"):
        return [
            RangeStats(
                iteration=iteration,
                phase=phase,
                stage=resolve_stage(iteration),
                worker=worker,
                start=start,
                end=end,
                emitting_global_tid=global_tid,
            )
            for iteration, phase, worker, start, end, global_tid in windowed
        ]

    grouped: dict[tuple[int, int], list] = defaultdict(list)
    for iteration, phase, worker, start, end, global_tid in windowed:
        grouped[(iteration, worker.global_pid)].append((phase, worker, start, end, global_tid))

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
                emitting_global_tid=items[0][4],
            )
        )
    return ranges


def attach_kernels_by_correlation(
    con: sqlite3.Connection,
    string_ids: dict[int, str],
    ranges: list[RangeStats],
) -> int:
    """Attach kernels owned by CUDA runtime calls launched inside each NVTX range.

    Ownership path: for each range, find runtime API calls on the emitting
    globalTid within [start, end), collect their correlationIds, then pull every
    kernel with a matching correlationId on that globalPid. Robust to CUDA-graph
    kernels that execute outside the range wall-clock.
    """
    kernel_rows = 0
    for item in ranges:
        global_tid = item.launch_global_tid
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
                k.streamId,
                k.correlationId
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
            correlation_id,
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
                    correlation_id=(None if correlation_id is None else int(correlation_id)),
                )
            )
    return kernel_rows


def resolve_dp_rank_by_device(
    workers: dict[int, Worker],
    worker_ranks: dict[int, int],
    tp_size: int,
) -> dict[int, int]:
    """CUDA device → data-parallel rank, joining nsys pids with the server log.

    nsys knows which pid ran on which device; vLLM's startup banner states which
    pid holds which global rank. A replica lays its ranks out as
    `global_rank = dp_rank * tp_size + tp_rank`, so the DP rank is the global
    rank divided by the TP degree. With no banner (single-process capture) every
    observed device belongs to rank 0.
    """
    if not worker_ranks:
        return {worker.device_id: 0 for worker in workers.values() if worker.device_id is not None}
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")

    dp_rank_by_device: dict[int, int] = {}
    for worker in workers.values():
        if worker.device_id is None:
            continue
        global_rank = worker_ranks.get(worker.pid)
        if global_rank is None:
            raise ValueError(
                f"nsys worker pid {worker.pid} (device {worker.device_id}) has no rank banner "
                "in the server log; the capture and the log do not describe the same run"
            )
        dp_rank = global_rank // tp_size
        existing = dp_rank_by_device.setdefault(worker.device_id, dp_rank)
        if existing != dp_rank:
            raise ValueError(f"device {worker.device_id} maps to DP ranks {existing} and {dp_rank}")
    return dp_rank_by_device


@dataclass
class DeviceStepWindow:
    """One device's own step: every phase it ran under one of ITS iteration ids."""

    device_id: int | None
    iteration: int
    start: int
    end: int
    items: list[RangeStats]


@dataclass
class AlignedStep:
    """One wall-clock step, and which of each device's own iterations it is.

    `iteration` is the reference device's index and remains the step's identity
    downstream — a case index, a labeled sequence occurrence and an analyzer row
    all address a step by it. `index_by_device` is the provenance that makes the
    identity honest: under data parallelism the same step is iteration 153 on
    device 0 and 149 on device 1.
    """

    iteration: int
    reference_device_id: int | None
    windows: list[DeviceStepWindow]

    @property
    def index_by_device(self) -> dict[int | None, int]:
        return {window.device_id: window.iteration for window in self.windows}


def _device_step_windows(ranges: list[RangeStats]) -> dict[int | None, list[DeviceStepWindow]]:
    """Each device's own iterations, in time order, phases folded into one span."""
    grouped: dict[tuple[int | None, int], list[RangeStats]] = defaultdict(list)
    for item in ranges:
        grouped[(item.worker.device_id, item.iteration)].append(item)
    by_device: dict[int | None, list[DeviceStepWindow]] = defaultdict(list)
    for (device_id, iteration), items in grouped.items():
        by_device[device_id].append(
            DeviceStepWindow(
                device_id=device_id,
                iteration=iteration,
                start=min(item.start for item in items),
                end=max(item.end for item in items),
                items=sorted(items, key=lambda item: (item.start, item.phase)),
            )
        )
    for windows in by_device.values():
        windows.sort(key=lambda window: (window.start, window.iteration))
    return by_device


def _pair_by_overlap(
    reference: list[DeviceStepWindow], peer: list[DeviceStepWindow]
) -> tuple[dict[int, DeviceStepWindow], list[DeviceStepWindow]]:
    """Match a peer device's steps to the reference's, one to one, by overlap.

    A merge join rather than nearest-index: the two lists are each in time order
    and internally non-overlapping, so one forward pass assigns every peer window
    to the reference window it shares the most time with. Peer windows that match
    nothing are returned rather than dropped — they are steps the reference was
    not running, which is a fact about the capture, not noise to swallow.
    """
    matched: dict[int, DeviceStepWindow] = {}
    unpaired: list[DeviceStepWindow] = []
    reference_index = 0
    peer_index = 0

    def overlap(left: DeviceStepWindow, right: DeviceStepWindow) -> int:
        return min(left.end, right.end) - max(left.start, right.start)

    while reference_index < len(reference) and peer_index < len(peer):
        reference_window = reference[reference_index]
        peer_window = peer[peer_index]
        if peer_window.end <= reference_window.start:
            unpaired.append(peer_window)
            peer_index += 1
            continue
        if reference_window.end <= peer_window.start:
            reference_index += 1
            continue
        # They overlap. Take the next peer instead when it overlaps this
        # reference step more, so a peer step straddling two reference steps
        # lands on the one it actually shares its time with.
        if peer_index + 1 < len(peer) and overlap(reference_window, peer[peer_index + 1]) > overlap(
            reference_window, peer_window
        ):
            unpaired.append(peer_window)
            peer_index += 1
            continue
        matched[reference_window.iteration] = peer_window
        reference_index += 1
        peer_index += 1
    unpaired.extend(peer[peer_index:])
    return matched, unpaired


def align_ranges_into_steps(ranges: list[RangeStats]) -> tuple[list[AlignedStep], dict[int, int]]:
    """Group the devices' ranges into wall-clock steps, NOT by iteration index.

    Every rank's `iteration_index` counts that rank's own scheduled steps. Under
    data parallelism the counters diverge and stay diverged: in the GLM-5.2 DP8
    capture device 0 ran four prefill chunks alone at the head of the window
    while its peers ran dummy batches that participate in the expert-parallel
    collectives but are not scheduled iterations, so from then on device 0 was
    permanently four to five ahead. Index 9 is a 1,927 ms prefill on device 0 and
    a 20 ms decode on device 1, six seconds later.

    Grouping by index therefore reduces kernels from different steps together —
    a per-position `max` across ranks over events that never coexisted. This
    groups by measured time instead: the lowest device is the reference clock and
    each peer's step joins the reference step it overlaps most.

    Returns the steps plus, per device, how many of its steps found no reference
    step to join (reported in provenance; never silently dropped).
    """
    by_device = _device_step_windows(ranges)
    if not by_device:
        return [], {}
    reference_device_id = sorted(by_device, key=lambda device: (device is None, device))[0]
    reference = by_device[reference_device_id]

    joined: dict[int, list[DeviceStepWindow]] = {window.iteration: [window] for window in reference}
    unpaired_by_device: dict[int, int] = {}
    for device_id, windows in by_device.items():
        if device_id == reference_device_id:
            continue
        matched, unpaired = _pair_by_overlap(reference, windows)
        for iteration, window in matched.items():
            joined[iteration].append(window)
        if unpaired:
            unpaired_by_device[device_id] = len(unpaired)

    steps = [
        AlignedStep(
            iteration=window.iteration,
            reference_device_id=reference_device_id,
            windows=sorted(
                joined[window.iteration],
                key=lambda joined_window: (
                    joined_window.device_id if joined_window.device_id is not None else -1
                ),
            ),
        )
        for window in reference
    ]
    steps.sort(key=lambda step: step.iteration)
    return steps, unpaired_by_device


def build_iteration_details(
    ranges: list[RangeStats],
    metrics: dict[int, dict],
    kernel_name_ids: dict[str, int],
    rank_metrics: dict[tuple[int, int], dict] | None = None,
    dp_rank_by_device: dict[int, int] | None = None,
) -> list[dict]:
    """Serialize every owned kernel in launch order, grouped by wall-clock step.

    Summary categories are convenient analyzer inputs, but this list is the
    lossless ground-truth artifact. Kernel names are never truncated here.

    A step is the set of per-device ranges that overlap in measured time, not
    the set that shares an `iteration_index` — see `align_ranges_into_steps` for
    why those are different things under data parallelism. Each serialized range
    therefore carries `iteration_index`: the id its OWN rank gave the step, which
    is the key its metrics row is under. A rank that scheduled nothing emits no
    record, so its range metrics are `None` — an expected dummy step, never an
    error — and a rank with no overlapping range at all is named in
    `devices_absent` rather than left to be inferred from a short `ranges` list.

    `metrics` is the index-keyed replica aggregate, used only as the fallback for
    a capture with no per-rank rows; when those rows exist the step's shape is
    folded from the ranks this step actually joined.
    """
    rank_metrics = rank_metrics or {}
    dp_rank_by_device = dp_rank_by_device or {}
    steps, _ = align_ranges_into_steps(ranges)
    all_devices = sorted(
        {item.worker.device_id for item in ranges if item.worker.device_id is not None}
    )

    details = []
    for step in steps:
        items = [item for window in step.windows for item in window.items]
        items.sort(
            key=lambda item: (
                item.worker.device_id if item.worker.device_id is not None else -1,
                item.start,
                item.phase,
            )
        )
        index_by_device = step.index_by_device
        step_rows = [
            rank_metrics[(dp_rank_by_device[device_id], iteration)]
            for device_id, iteration in sorted(
                ((device, index) for device, index in index_by_device.items() if device is not None)
            )
            if (dp_rank_by_device.get(device_id), iteration) in rank_metrics
        ]
        metric = (
            fold_rank_metrics(step_rows, step.iteration)
            if step_rows
            else metrics.get(step.iteration)
        )
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
                        "correlation_id": event.correlation_id,
                    }
                )
            range_dp_rank = dp_rank_by_device.get(item.worker.device_id)
            serialized_ranges.append(
                {
                    "device_id": item.worker.device_id,
                    "dp_rank": range_dp_rank,
                    "iteration_index": item.iteration,
                    "metrics": (
                        rank_metrics.get((range_dp_rank, item.iteration))
                        if range_dp_rank is not None
                        else None
                    ),
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

        present_devices = {device for device in index_by_device if device is not None}
        details.append(
            {
                "iteration": step.iteration,
                "iteration_type": iteration_type,
                "stage": items[0].stage,
                "reference_device_id": step.reference_device_id,
                "iteration_index_by_device": {
                    str(device): index
                    for device, index in sorted(index_by_device.items())
                    if device is not None
                },
                "devices_absent": [
                    device for device in all_devices if device not in present_devices
                ],
                "metrics": metric,
                "metrics_by_dp_rank": {
                    str(dp_rank_by_device[device_id]): rank_metrics[
                        (dp_rank_by_device[device_id], iteration)
                    ]
                    for device_id, iteration in sorted(
                        (device, index)
                        for device, index in index_by_device.items()
                        if device is not None
                    )
                    if (dp_rank_by_device.get(device_id), iteration) in rank_metrics
                },
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
    (the gap-table input), the merged `busy_ms`/`kernel_span_ms`/`idle_ms` means
    (plus `nvtx_window_ms_mean`, kept separately because it is a host fact), the
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
            "nvtx_window_ms_mean": sum(r.nvtx_window_ms for r in items) / len(items),
            "kernel_span_ms_mean": sum(r.kernel_span_ms for r in items) / len(items),
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


# ---------------------------------------------------------------------------
# Host (CPU) side
#
# Everything above answers "which kernel belongs to which iteration". This
# section answers "what was the CPU doing meanwhile", and deliberately stops
# there: it emits absolute nanoseconds on named threads and leaves iteration
# windows, nesting depth, and call classification to the analyzer that already
# owns the anchor rule. Interpreting here would fork that rule in two places.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostThread:
    global_tid: int
    global_pid: int
    device_id: int | None
    process: str
    role: str
    is_main: bool


def load_host_threads(con: sqlite3.Connection) -> list[HostThread]:
    """Every thread that carries an NVTX mark or a CUDA runtime call.

    Three roles, and the third is the reason this is worth emitting: a thread
    owning no device at all is the engine core's scheduler, whose host time is
    invisible in any device-side view.
    """
    device_by_global_pid = load_device_by_global_pid(con)
    process_rows = load_process_rows(con)
    name_by_global_pid = {global_pid: name for global_pid, _pid, name in process_rows}
    main_thread_owner = {global_pid + pid: global_pid for global_pid, pid, _name in process_rows}

    threads = []
    for (global_tid,) in con.execute(
        """
        SELECT DISTINCT globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME
        UNION
        SELECT DISTINCT globalTid FROM NVTX_EVENTS
        """
    ):
        global_tid = int(global_tid)
        global_pid = owning_global_pid(global_tid)
        device_id = device_by_global_pid.get(global_pid)
        is_main = main_thread_owner.get(global_tid) == global_pid
        if device_id is None:
            role = "scheduler thread"
        elif is_main:
            role = "worker main thread"
        else:
            role = "worker helper thread"
        threads.append(
            HostThread(
                global_tid=global_tid,
                global_pid=global_pid,
                device_id=device_id,
                process=name_by_global_pid.get(global_pid, "unknown"),
                role=role,
                is_main=is_main,
            )
        )
    # Scheduler first, then each device's main thread ahead of its helpers. This
    # is the lane order a host timeline reads top-to-bottom in.
    threads.sort(
        key=lambda thread: (
            thread.device_id is not None,
            thread.device_id if thread.device_id is not None else -1,
            not thread.is_main,
            thread.global_tid,
        )
    )
    return threads


def parsed_window_ns(parsed: dict) -> tuple[int, int]:
    """The absolute [start, end) the parsed iteration ranges cover.

    The host sidecar accompanies one parse, so it carries the same window: a
    capture's warmup and shutdown are not part of the iterations being aligned.
    """
    starts = [
        range_row["start_ns"]
        for detail in parsed["iteration_details"]
        for range_row in detail["ranges"]
    ]
    ends = [
        range_row["end_ns"]
        for detail in parsed["iteration_details"]
        for range_row in detail["ranges"]
    ]
    if not starts:
        return (0, 0)
    return (min(starts), max(ends))


def build_host_timeline(
    con: sqlite3.Connection,
    sqlite_path: Path,
    window_start_ns: int,
    window_end_ns: int,
) -> dict:
    """Collect the host events overlapping the window, interned and unsorted-free.

    Rows reference threads by index into `threads`, so a consumer cannot read an
    event without also reading whose thread it was on.
    """
    string_ids = load_string_ids(con)
    threads = load_host_threads(con)
    thread_index = {thread.global_tid: index for index, thread in enumerate(threads)}

    string_pool: list[str] = []
    string_index: dict[str, int] = {}

    def intern(value: str) -> int:
        if value not in string_index:
            string_index[value] = len(string_pool)
            string_pool.append(value)
        return string_index[value]

    nvtx_ranges: list[list[int]] = []
    unclosed_nvtx = 0
    for global_tid, start, end, label in con.execute(
        """
        SELECT n.globalTid, n.start, n.end, COALESCE(n.text, s.value)
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON n.textId = s.id
        WHERE n.start < ?
        """,
        (window_end_ns,),
    ):
        index = thread_index.get(int(global_tid))
        if index is None or label is None:
            continue
        if end is None:
            # An instantaneous mark or a range the capture cut off. Counted
            # rather than dropped silently, so a shrunken lane is explainable.
            if int(start) >= window_start_ns:
                unclosed_nvtx += 1
            continue
        if int(end) < window_start_ns:
            continue
        nvtx_ranges.append([index, int(start), int(end), intern(str(label))])

    api_calls: list[list[int | None]] = []
    for global_tid, start, end, name_id, correlation_id in con.execute(
        """
        SELECT globalTid, start, end, nameId, correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE start < ? AND end >= ?
        """,
        (window_end_ns, window_start_ns),
    ):
        index = thread_index.get(int(global_tid))
        if index is None:
            continue
        name = string_ids.get(int(name_id), str(name_id))
        api_calls.append(
            [
                index,
                int(start),
                int(end),
                intern(name),
                None if correlation_id is None else int(correlation_id),
            ]
        )

    nvtx_ranges.sort()
    api_calls.sort()
    return {
        "schema_version": 1,
        "sqlite": str(sqlite_path),
        "window_start_ns": window_start_ns,
        "window_end_ns": window_end_ns,
        "time_base": "absolute nsys nanoseconds, as stored in the capture",
        "row_format": {
            "nvtx_ranges": ["thread_index", "start_ns", "end_ns", "string_id"],
            "api_calls": [
                "thread_index",
                "start_ns",
                "end_ns",
                "string_id",
                "correlation_id",
            ],
        },
        "unclosed_nvtx_marks": unclosed_nvtx,
        "threads": [
            {
                "global_tid": thread.global_tid,
                "global_pid": thread.global_pid,
                "device_id": thread.device_id,
                "process": thread.process,
                "role": thread.role,
                "main": thread.is_main,
            }
            for thread in threads
        ],
        "strings": string_pool,
        "nvtx_ranges": nvtx_ranges,
        "api_calls": api_calls,
    }


def parse_host_timeline(sqlite_path: Path, window_start_ns: int, window_end_ns: int) -> dict:
    """Open the capture and extract the host sidecar for one parsed window."""
    con = sqlite3.connect(str(sqlite_path))
    try:
        return build_host_timeline(con, sqlite_path, window_start_ns, window_end_ns)
    finally:
        con.close()


def parse_trace(
    sqlite_path: Path,
    metrics_jsonl: Path | None,
    iteration_start: int,
    iteration_end: int,
    *,
    range_mode: str = "phases",
    default_stage: str = "all",
    top_n: int = 12,
    worker_ranks: dict[int, int] | None = None,
    tp_size: int = 1,
    dp_rank_by_device: dict[int, int] | None = None,
) -> dict:
    """Top-level parse: nsys sqlite → per-device per-category busy time / iter.

    The alignment ground truth. Groups the analysis window by nsys `stage`
    ("mixed"/"decode") so a prefill target and a decode target can be selected
    independently from one capture. `default_stage` tags ranges with no
    label/metrics stage (use "decode" for a pure-decode capture window).

    `worker_ranks` (worker pid → global rank, from the server log) and `tp_size`
    resolve which DP rank ran on which device. Without them the capture is read
    as a single-rank run.

    `dp_rank_by_device` short-circuits that: an engine whose workers state their
    own device and rank has already answered the question, so its answer is used
    directly rather than rederived from pids. It is checked against the devices
    the capture actually shows — a mapping that does not cover them describes a
    different run.
    """
    con = sqlite3.connect(str(sqlite_path))
    try:
        ensure_query_indexes(con)
        rank_metrics = load_metrics(metrics_jsonl)
        metrics = aggregate_metrics_by_iteration(rank_metrics)
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

    if dp_rank_by_device is None:
        dp_rank_by_device = resolve_dp_rank_by_device(workers, worker_ranks or {}, tp_size)
    else:
        captured_devices = {
            worker.device_id for worker in workers.values() if worker.device_id is not None
        }
        unstated = sorted(captured_devices - set(dp_rank_by_device))
        if unstated:
            raise ValueError(
                f"the capture ran kernels on device(s) {unstated}, which no worker record "
                f"claims; stated devices are {sorted(dp_rank_by_device)}"
            )

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
    iteration_details = build_iteration_details(
        ranges, metrics, kernel_name_ids, rank_metrics, dp_rank_by_device
    )
    aligned_steps, unpaired_by_device = align_ranges_into_steps(ranges)
    kernel_sequences, device_ids = build_device_kernel_sequences(iteration_details, kernel_names)
    return {
        "schema_version": 4,
        # How the devices' ranges were grouped into steps. Recorded because the
        # obvious rule — same `iteration_index` — is wrong under data
        # parallelism, and a reader of this file has no way to tell which rule
        # produced it otherwise.
        "step_alignment": {
            "rule": "wall_clock_overlap",
            "reference_device_id": (
                aligned_steps[0].reference_device_id if aligned_steps else None
            ),
            "steps": len(aligned_steps),
            "unpaired_steps_by_device": {
                str(device): count for device, count in sorted(unpaired_by_device.items())
            },
        },
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
        "device_ids": device_ids,
        "dp_rank_by_device": {
            str(device): rank for device, rank in sorted(dp_rank_by_device.items())
        },
        "tp_size": tp_size,
        "kernel_sequences": kernel_sequences,
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
    parser.add_argument(
        "--server-log",
        type=Path,
        default=None,
        help="vLLM server log; supplies the worker pid → rank banner that maps devices to DP ranks",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="tensor-parallel degree, used to fold global ranks into DP ranks",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON here (else stdout)")
    parser.add_argument(
        "--sequences-output",
        type=Path,
        default=None,
        help="Also write the exact unique full-sequence catalog here",
    )
    parser.add_argument(
        "--host-timeline-output",
        type=Path,
        default=None,
        help="Also write the host-side NVTX ranges and CUDA runtime calls here",
    )
    return parser


def write_kernel_sequences(path: Path, parsed: dict, source_parsed: Path | None) -> None:
    """Write the folded, label-ready sequence inventory separately from parsed.json."""
    document = {
        "schema_version": 4,
        "encoding": "folded-v1",
        "source_parsed": str(source_parsed) if source_parsed is not None else None,
        "device_ids": parsed["device_ids"],
        "folding_policy": {
            "kind": "exact_contiguous_repeat",
            "match_fields": ["name", "suggested_category"],
            "row_identity": "sequence_id:expanded_ordinal",
            "rank_policy": (
                "union across devices; each sequence lists its (device, iteration) occurrences"
            ),
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
    # Imported lazily: this module is the nsys-source boundary and must stay
    # usable without the vLLM launcher package present.
    if args.server_log is not None:
        from ..profiler.vllm_server import extract_worker_device_ranks

        worker_ranks = extract_worker_device_ranks(args.server_log)
    else:
        worker_ranks = {}
    result = parse_trace(
        args.sqlite,
        args.metrics,
        args.iteration_start,
        args.iteration_end,
        range_mode=args.range_mode,
        default_stage=args.default_stage,
        top_n=args.top_n,
        worker_ranks=worker_ranks,
        tp_size=args.tp_size,
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
    if args.host_timeline_output:
        window_start_ns, window_end_ns = parsed_window_ns(result)
        host = parse_host_timeline(args.sqlite, window_start_ns, window_end_ns)
        Path(args.host_timeline_output).write_text(json.dumps(host, separators=(",", ":")))
        print(
            f"wrote {args.host_timeline_output} "
            f"({len(host['threads'])} threads, {len(host['nvtx_ranges'])} nvtx, "
            f"{len(host['api_calls'])} api)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
