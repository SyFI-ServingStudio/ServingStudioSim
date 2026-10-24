"""Compare per-round timing-predict totals with a TensorFold nsys capture.

Inputs:

1. the sidecar ``alignment.timing_predict_input.tensorfold_iter_records`` wrote
   beside the cases (one entry per case: level, record timestamps, measured wall,
   stream and prefill counts);
2. the timing-predict ``log_dir`` run on those cases. Per-case totals come from
   ``raw/cost_log/worker_predict_0.parquet`` (``iter_id`` = case index, every row
   of a case summed). ``payloads/iter_breakdown.json`` is not used: it holds only
   the Analyzer's render window (32 cases by default);
3. per level, the nsys sqlite of that level's capture window (``--nsys 2=path``;
   ``all=path`` for a capture without ``phases.tsv`` levels).

Two comparisons per level:

- **kernel**: predicted total vs measured kernel busy of the round. A round is
  the NVTX range it ran inside (``tensorfold.engine.iteration``, or
  ``tensorfold.prefill.chunk`` for a single-stream prefill chunk); busy is the
  union of the kernels launched from that range's thread while it was open, as
  ``alignment.nsys.evidence`` counts it. Records and ranges are joined in
  timestamp order: the record clock (``time.monotonic_ns``) and the nsys session
  clock differ by an offset that is anchored on the first ranges and then
  re-taken at every match, which follows the small drift between the two clocks.
- **wall**: predicted total plus a fitted host term vs the record's wall
  (``end_ns - start_ns``). The host term ``a + b * tree_streams (+ c *
  prefill_pieces)`` is fitted by least squares to ``wall - busy`` on the fit
  levels (default 2, 4) and evaluated held out on the others (default 1, 8). It
  is reported apart from the kernel prediction; it is not part of the CostTree.

A matched range whose length disagrees with its record's wall by more than
``--max-wall-mismatch-us`` was cut by the capture edge; it is left out and
counted. Records outside the nsys window are counted as unmatched.

Usage::

    uv run python tools/tensorfold_compare/tensorfold_compare.py \\
        --sidecar CASES_DIR/tensorfold_round_sidecar.json --log-dir PREDICT_LOG_DIR \\
        --nsys 1=tf27b_n1.sqlite --nsys 2=tf27b_n2.sqlite ... --out-dir OUT
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the checkout root, for `alignment`
from alignment.nsys.evidence import merge_duration_ns  # noqa: E402

ANCHOR_RANGES = 20
ANCHOR_TOLERANCE_NS = 200_000
WALK_TOLERANCE_NS = 2_000_000


@dataclass(frozen=True)
class Range:
    start: int
    end: int
    global_tid: int


@dataclass
class Round:
    """One sidecar case joined to its measurement and prediction."""

    case: dict[str, Any]
    predicted_ms: float
    busy_ms: float | None = None
    nvtx_ms: float | None = None
    status: str = "unmatched"
    host_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def wall_ms(self) -> float:
        return self.case["wall_ns"] / 1e6


def level_key(level: Any) -> str:
    return "all" if level is None else str(level)


def load_predicted(log_dir: Path, case_count: int) -> list[float]:
    """Per-case predicted totals (ms), checked against the snapshotted cases."""
    snapshot = log_dir / "prediction.cases.json"
    if snapshot.exists():
        predicted_cases = json.loads(snapshot.read_text())
        if len(predicted_cases) != case_count:
            raise ValueError(
                f"{snapshot} holds {len(predicted_cases)} cases; the sidecar has {case_count}"
            )
    table = pq.read_table(
        log_dir / "raw" / "cost_log" / "worker_predict_0.parquet",
        columns=["iter_id", "total_time_ms"],
    )
    totals = [0.0] * case_count
    seen = [False] * case_count
    for iter_id, total in zip(
        table.column("iter_id").to_pylist(), table.column("total_time_ms").to_pylist(), strict=True
    ):
        if not 0 <= iter_id < case_count:
            raise ValueError(f"cost_log iter_id {iter_id} is not a sidecar case")
        totals[iter_id] += float(total)
        seen[iter_id] = True
    missing = [index for index, ok in enumerate(seen) if not ok]
    if missing:
        raise ValueError(f"cost_log has no row for {len(missing)} case(s), first {missing[:5]}")
    return totals


def load_ranges(connection: sqlite3.Connection, name: str) -> list[Range]:
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    if "StringIds" in tables:
        query = """
            SELECT n.start, n.end, n.globalTid FROM NVTX_EVENTS n
            LEFT JOIN StringIds s ON n.textId = s.id
            WHERE COALESCE(n.text, s.value) = ? AND n.end IS NOT NULL ORDER BY n.start
        """
    else:
        query = """
            SELECT start, end, globalTid FROM NVTX_EVENTS
            WHERE text = ? AND end IS NOT NULL ORDER BY start
        """
    return [Range(int(s), int(e), int(t)) for s, e, t in connection.execute(query, (name,))]


def load_launches(
    connection: sqlite3.Connection, tids: set[int]
) -> dict[int, tuple[list[int], list[tuple[int, int]]]]:
    """Per launching thread: sorted launch-call starts and their kernels' intervals."""
    out: dict[int, tuple[list[int], list[tuple[int, int]]]] = {}
    for tid in tids:
        rows = connection.execute(
            """
            SELECT r.start, k.start, k.end
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN CUPTI_ACTIVITY_KIND_KERNEL k
              ON k.correlationId = r.correlationId
             AND k.globalPid = r.globalTid - (r.globalTid % 16777216)
            WHERE r.globalTid = ?
            ORDER BY r.start
            """,
            (tid,),
        ).fetchall()
        out[tid] = ([int(r[0]) for r in rows], [(int(r[1]), int(r[2])) for r in rows])
    return out


def busy_ns(launches: dict[int, tuple[list[int], list[tuple[int, int]]]], span: Range) -> int:
    starts, kernels = launches[span.global_tid]
    first = bisect.bisect_left(starts, span.start)
    last = bisect.bisect_left(starts, span.end)
    return merge_duration_ns(kernels[first:last])


def _nearest(starts: list[int], target: int, tolerance: int) -> int | None:
    index = bisect.bisect_left(starts, target - tolerance)
    best = None
    while index < len(starts) and starts[index] <= target + tolerance:
        if best is None or abs(starts[index] - target) < abs(starts[best] - target):
            best = index
        index += 1
    return best


def match_ranges(record_starts: list[int], ranges: list[Range]) -> dict[int, int]:
    """Map range index → record index, in timestamp order, following clock drift.

    The offset (record clock minus nsys clock) is anchored where most of the first
    ``ANCHOR_RANGES`` ranges land on a record start, then re-taken at each match.
    """
    if not ranges or not record_starts:
        return {}
    head = ranges[:ANCHOR_RANGES]
    best: tuple[int, int] | None = None
    for anchor in ranges[:5]:
        for start in record_starts:
            offset = start - anchor.start
            hits = sum(
                _nearest(record_starts, r.start + offset, ANCHOR_TOLERANCE_NS) is not None
                for r in head
            )
            if best is None or hits > best[0]:
                best = (hits, offset)
    hits, offset = best
    if hits * 2 < len(head):
        raise ValueError(
            f"cannot anchor the record clock to the nsys clock ({hits}/{len(head)} ranges)"
        )
    matched: dict[int, int] = {}
    used: set[int] = set()
    for index, span in enumerate(ranges):
        record = _nearest(record_starts, span.start + offset, WALK_TOLERANCE_NS)
        if record is None or record in used:
            continue
        matched[index] = record
        used.add(record)
        offset = record_starts[record] - span.start
    return matched


def measure_level(rounds: list[Round], sqlite_path: Path, max_mismatch_ns: int) -> None:
    """Fill busy / nvtx length / status on one level's rounds from its capture."""
    by_range: dict[str, list[Round]] = defaultdict(list)
    for item in rounds:
        by_range[item.case.get("nvtx_range", "tensorfold.engine.iteration")].append(item)
    with closing(sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)) as connection:
        for name, items in by_range.items():
            items.sort(key=lambda item: item.case["start_ns"])
            ranges = load_ranges(connection, name)
            matched = match_ranges([item.case["start_ns"] for item in items], ranges)
            launches = load_launches(connection, {ranges[i].global_tid for i in matched})
            for range_index, record_index in matched.items():
                item, span = items[record_index], ranges[range_index]
                item.nvtx_ms = (span.end - span.start) / 1e6
                if abs(item.case["wall_ns"] - (span.end - span.start)) > max_mismatch_ns:
                    item.status = "cut_by_capture_edge"
                    continue
                item.busy_ms = busy_ns(launches, span) / 1e6
                item.status = "matched"


def _features(case: dict[str, Any], prefill_term: bool) -> list[float]:
    row = [1.0, float(case["tree_streams"])]
    if prefill_term:
        row.append(float(case["prefill_pieces"]))
    return row


def fit_host(rounds: list[Round], prefill_term: bool) -> dict[str, Any]:
    """Least-squares host term on matched rounds: wall - busy (ms)."""
    usable = [item for item in rounds if item.status == "matched"]
    if not usable:
        raise ValueError("no matched round on the fit levels")
    if prefill_term and not any(item.case["prefill_pieces"] for item in usable):
        prefill_term = False
    x = np.array([_features(item.case, prefill_term) for item in usable])
    y = np.array([item.wall_ms - item.busy_ms for item in usable])
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    names = ["a_ms", "b_ms_per_tree_stream"] + (["c_ms_per_prefill_piece"] if prefill_term else [])
    residual = y - x @ coefficients
    return {
        "coefficients": dict(zip(names, (float(c) for c in coefficients), strict=True)),
        "prefill_term": prefill_term,
        "rank": int(rank),
        "rounds": len(usable),
        "residual_ms_median_abs": float(np.median(np.abs(residual))),
    }


def host_ms(fit: dict[str, Any], case: dict[str, Any]) -> float:
    c = fit["coefficients"]
    value = c["a_ms"] + c["b_ms_per_tree_stream"] * case["tree_streams"]
    if fit["prefill_term"]:
        value += c["c_ms_per_prefill_piece"] * case["prefill_pieces"]
    return value


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]


def _errors(predicted: list[float], measured: list[float]) -> dict[str, float]:
    rel = [100.0 * (p - m) / m for p, m in zip(predicted, measured, strict=True) if m > 0]
    signed_ms = [p - m for p, m in zip(predicted, measured, strict=True)]
    return {
        "measured_median_ms": statistics.median(measured),
        "predicted_median_ms": statistics.median(predicted),
        "err_median_pct": statistics.median(rel),
        "err_mean_pct": statistics.fmean(rel),
        "abs_err_p90_pct": _p90([abs(v) for v in rel]),
        "err_median_ms": statistics.median(signed_ms),
        "sum_err_pct": 100.0 * (sum(predicted) - sum(measured)) / sum(measured),
    }


def summarize(
    rounds_by_level: dict[str, list[Round]], fit_levels: set[str], fit: dict[str, Any] | None
) -> list[dict[str, Any]]:
    rows = []
    for level, rounds in rounds_by_level.items():
        matched = [item for item in rounds if item.status == "matched"]
        row: dict[str, Any] = {
            "level": level,
            "role": "fit" if level in fit_levels else "holdout",
            "cases": len(rounds),
            "matched": len(matched),
            "cut_by_capture_edge": sum(item.status == "cut_by_capture_edge" for item in rounds),
            "unmatched": sum(item.status == "unmatched" for item in rounds),
            "no_capture": sum(item.status == "no_capture" for item in rounds),
        }
        if matched:
            kernel = _errors([i.predicted_ms for i in matched], [i.busy_ms for i in matched])
            row.update({f"kernel_{k}": v for k, v in kernel.items()})
            if fit is not None:
                wall = _errors(
                    [i.predicted_ms + i.host_ms for i in matched], [i.wall_ms for i in matched]
                )
                row.update({f"wall_{k}": v for k, v in wall.items()})
                row["measured_host_median_ms"] = statistics.median(
                    i.wall_ms - i.busy_ms for i in matched
                )
                row["fitted_host_median_ms"] = statistics.median(i.host_ms for i in matched)
        rows.append(row)
    return rows


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:+.{digits}f}" if digits else f"{value:.0f}"
    return str(value)


def render_markdown(summary: list[dict[str, Any]], fit: dict[str, Any] | None) -> str:
    lines = [
        "## Kernel: predicted total vs measured kernel busy per round",
        "",
        "| level | role | rounds | busy med ms | pred med ms | err med % | err mean % "
        "| p90 abs err % | sum err % |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if "kernel_err_median_pct" not in row:
            lines.append(f"| {row['level']} | {row['role']} | 0 | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {row['level']} | {row['role']} | {row['matched']} "
            f"| {row['kernel_measured_median_ms']:.1f} | {row['kernel_predicted_median_ms']:.1f} "
            f"| {_fmt(row['kernel_err_median_pct'])} | {_fmt(row['kernel_err_mean_pct'])} "
            f"| {row['kernel_abs_err_p90_pct']:.1f} | {_fmt(row['kernel_sum_err_pct'])} |"
        )
    lines += ["", "## Wall: predicted total + fitted host vs measured round wall", ""]
    if fit is None:
        lines.append("No host fit: no fit level has matched rounds.")
    else:
        terms = ", ".join(f"{k} = {v:.3f}" for k, v in fit["coefficients"].items())
        lines += [
            f"Host term ({fit['rounds']} fit rounds; median |residual| "
            f"{fit['residual_ms_median_abs']:.2f} ms): {terms}",
            "",
            "| level | role | rounds | wall med ms | pred+host med ms | host meas med ms "
            "| host fit med ms | err med % | err mean % | p90 abs err % | sum err % |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary:
            if "wall_err_median_pct" not in row:
                continue
            lines.append(
                f"| {row['level']} | {row['role']} | {row['matched']} "
                f"| {row['wall_measured_median_ms']:.1f} | {row['wall_predicted_median_ms']:.1f} "
                f"| {row['measured_host_median_ms']:.1f} | {row['fitted_host_median_ms']:.1f} "
                f"| {_fmt(row['wall_err_median_pct'])} | {_fmt(row['wall_err_mean_pct'])} "
                f"| {row['wall_abs_err_p90_pct']:.1f} | {_fmt(row['wall_sum_err_pct'])} |"
            )
    lines += [
        "",
        "## Coverage",
        "",
        "| level | cases | matched | cut by capture edge | outside nsys window "
        "| no capture given |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['level']} | {row['cases']} | {row['matched']} | {row['cut_by_capture_edge']} "
            f"| {row['unmatched']} | {row['no_capture']} |"
        )
    return "\n".join(lines) + "\n"


def compare(
    sidecar_path: Path,
    log_dir: Path,
    nsys: dict[str, Path],
    *,
    fit_levels: set[str],
    prefill_term: bool = True,
    max_mismatch_ns: int = 500_000,
) -> tuple[dict[str, list[Round]], dict[str, Any] | None, list[dict[str, Any]]]:
    sidecar = json.loads(sidecar_path.read_text())
    cases = sidecar["cases"]
    predicted = load_predicted(log_dir, len(cases))
    rounds_by_level: dict[str, list[Round]] = defaultdict(list)
    for case in cases:
        rounds_by_level[level_key(case["level"])].append(
            Round(case=case, predicted_ms=predicted[case["case_index"]])
        )
    for level, rounds in rounds_by_level.items():
        if level in nsys:
            measure_level(rounds, nsys[level], max_mismatch_ns)
        else:
            for item in rounds:
                item.status = "no_capture"
    fit_rounds = [item for level in fit_levels for item in rounds_by_level.get(level, [])]
    fit = None
    if any(item.status == "matched" for item in fit_rounds):
        fit = fit_host(fit_rounds, prefill_term)
        for rounds in rounds_by_level.values():
            for item in rounds:
                item.host_ms = host_ms(fit, item.case)
    ordered = dict(
        sorted(
            rounds_by_level.items(),
            key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0),
        )
    )
    return ordered, fit, summarize(ordered, fit_levels, fit)


ROUND_COLUMNS = [
    "case_index", "level", "round_index", "kind", "status", "start_ns", "end_ns",
    "streams", "tree_streams", "verify_rows", "prefill_pieces", "prefill_rows",
    "predicted_kernel_ms", "measured_busy_ms", "nvtx_ms", "measured_wall_ms",
    "fitted_host_ms", "predicted_wall_ms",
]  # fmt: skip


def write_outputs(
    out_dir: Path,
    rounds_by_level: dict[str, list[Round]],
    fit: dict[str, Any] | None,
    summary: list[dict[str, Any]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rounds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ROUND_COLUMNS)
        writer.writeheader()
        for level, rounds in rounds_by_level.items():
            for item in rounds:
                case = item.case
                writer.writerow(
                    {
                        **{k: case.get(k) for k in ROUND_COLUMNS if k in case},
                        "level": level,
                        "status": item.status,
                        "predicted_kernel_ms": item.predicted_ms,
                        "measured_busy_ms": item.busy_ms,
                        "nvtx_ms": item.nvtx_ms,
                        "measured_wall_ms": item.wall_ms,
                        "fitted_host_ms": item.host_ms,
                        "predicted_wall_ms": (
                            None if item.host_ms is None else item.predicted_ms + item.host_ms
                        ),
                    }
                )
    columns = sorted({key for row in summary for key in row}, key=lambda k: (k != "level", k))
    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "host_fit.json").write_text(json.dumps(fit, indent=1))
    markdown = out_dir / "compare.md"
    markdown.write_text(render_markdown(summary, fit))
    return markdown


def _nsys_arg(value: str) -> tuple[str, Path]:
    level, sep, path = value.partition("=")
    if not sep or not level or not path:
        raise argparse.ArgumentTypeError(f"expected LEVEL=PATH, got {value!r}")
    return level, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True, help="timing-predict log_dir")
    parser.add_argument("--nsys", type=_nsys_arg, action="append", default=[], help="LEVEL=sqlite")
    parser.add_argument("--fit-levels", default="2,4", help="levels the host term is fitted on")
    parser.add_argument("--no-prefill-term", action="store_true", help="fit a + b*tree_streams")
    parser.add_argument("--max-wall-mismatch-us", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    rounds_by_level, fit, summary = compare(
        args.sidecar,
        args.log_dir,
        dict(args.nsys),
        fit_levels={level.strip() for level in args.fit_levels.split(",") if level.strip()},
        prefill_term=not args.no_prefill_term,
        max_mismatch_ns=args.max_wall_mismatch_us * 1000,
    )
    markdown = write_outputs(args.out_dir, rounds_by_level, fit, summary)
    print(markdown.read_text())
    print(f"wrote {markdown}, rounds.csv, summary.csv, host_fit.json under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
