"""Reading one simulated Text2SQL streaming cell (one rollout at one B).

Shared by align_streaming.py and plot_t2s_vs_oracle.py. Times are simulated
seconds; this deployment's clock is used as is (no end-to-end factor K).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

GPUS = 8
GPUS_PER_BLOCK = 2
PENDING, SUSPENDED = 0, 4


def _intervals(table) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(table["wall_start_ms"], dtype=float)
    return start, start + np.asarray(table["total_time_ms"], dtype=float)


def _union_length(start: np.ndarray, end: np.ndarray) -> float:
    order = np.argsort(start)
    total, cur_s, cur_e = 0.0, None, None
    for s, e in zip(start[order], end[order]):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def training_window(cell: Path) -> tuple[float, float, int]:
    """(first chunk start, last chunk end, chunks) in seconds, from the per-block train cost_log."""
    starts, ends, chunks = [], [], 0
    for path in sorted((cell / "raw" / "cost_log").glob("worker_main.train_*.parquet")):
        table = pq.read_table(path, columns=["wall_start_ms", "total_time_ms"])
        if table.num_rows == 0:
            continue
        s, e = _intervals(table)
        starts.append(s.min())
        ends.append(e.max())
        chunks += table.num_rows
    if not starts:
        raise SystemExit(f"{cell}: no train rows")
    return min(starts) / 1e3, max(ends) / 1e3, chunks


def inference_end(cell: Path) -> float:
    t = pq.read_table(cell / "raw" / "request_slo.parquet", columns=["finish_decode_time_ms"])
    return float(np.nanmax(np.asarray(t["finish_decode_time_ms"], dtype=float))) / 1e3


def observation(cell: Path) -> dict[str, float]:
    """slime's `collect_observation` idle split (interior idle ratio), as replay_cubic.observation."""
    log = cell / "raw" / "cost_log"
    inference = [_intervals(pq.read_table(log / f"worker_main_{g}.parquet",
                                          columns=["wall_start_ms", "total_time_ms"]))
                 for g in range(GPUS)]
    blocks = GPUS // GPUS_PER_BLOCK
    chunks = [(np.zeros(0), np.zeros(0))] * blocks
    for path in log.glob("worker_main.train_*.parquet"):
        chunks[int(path.stem.rsplit("_", 1)[1])] = _intervals(
            pq.read_table(path, columns=["wall_start_ms", "total_time_ms"]))
    training_end = max(end.max() for _, end in chunks if len(end))
    span = busy = interior = trailing = 0.0
    for block in range(blocks):
        gpus = range(block * GPUS_PER_BLOCK, (block + 1) * GPUS_PER_BLOCK)
        flip = max(inference[g][1].max() for g in gpus)
        start, end = chunks[block]
        length = training_end - flip
        if not len(start):
            span += length * len(gpus)
            interior += length * len(gpus)
            continue
        start = np.maximum(start, flip)
        trained = _union_length(start, end)
        tail = max(0.0, training_end - end.max())
        span += length * len(gpus)
        busy += trained * len(gpus)
        trailing += tail * len(gpus)
        interior += max(0.0, length - trained - tail) * len(gpus)
    return {"span": span / 1e3, "busy": busy / 1e3, "interior": interior / 1e3,
            "trailing": trailing / 1e3, "ratio": interior / span, "end": training_end / 1e3}


def _peak(events: list[tuple[float, int]]) -> int:
    level = peak = 0
    for _, step in sorted(events, key=lambda e: (e[0], e[1])):
        level += step
        peak = max(peak, level)
    return peak


def engine_queues(cell: Path) -> np.ndarray:
    """Peak queue depth on the worst engine: [any re-entry, migrated in, retracted].

    A request's first Pending stay is its arrival (each round is its own request,
    released when its predecessor and tool wait are done); a later one is a
    migration arrival (previous stamp Suspended) or a KV retraction.
    """
    table = pq.read_table(cell / "raw" / "request_slo.parquet",
                          columns=["stage_times_ms", "stage_codes", "stage_worker_ids"])
    queued = [[[] for _ in range(GPUS)] for _ in range(3)]
    for times, codes, workers in zip(*(table[c].to_pylist() for c in table.column_names)):
        for k in range(1, len(codes)):
            if codes[k] != PENDING:
                continue
            leave = times[k + 1] if k + 1 < len(times) else np.inf
            span = [(times[k], 1), (leave, -1)]
            queued[0][workers[k]] += span
            queued[1 if codes[k - 1] == SUSPENDED else 2][workers[k]] += span
    return np.array([max(map(_peak, kind)) for kind in queued])
