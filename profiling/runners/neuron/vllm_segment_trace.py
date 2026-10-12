"""Cut the stock forward's native device timeline at its reduction collectives.

The unchanged production NEFF runs one TP4/LNC2 forward per execution. Each
rank's device profile records the execution bounds (ExecutionInfo), its
collective operations (CcOp) and every instruction on both physical cores
(Instruction). The forward's sublayers end at cross-rank reductions:

* decode reduces with AllReduce; prefill (sequence parallel) with ReduceScatter,
  and each prefill sublayer also starts with one AllGather;
* reductions r0..r64 close the embedding, then attention and MLP of layers 0..31;
* the head (final norm, SP gather, lm_head, logit gathers, sampling) follows r64
  and owns the remaining AllGathers (decode 3, prefill 4).

Segments are wall time between those sync points, measured per rank:
``embedding = (start, r0]``, ``attention_block[i] = (r2i, r2i+1]``,
``mlp_block[i] = (r2i+1, r2i+2]``, ``head = (r64, end]``. They partition the
execution span exactly. Nothing is fitted or scaled; any structural surprise,
incomplete core trace, lost notification or inhomogeneous layer stack fails.
"""

from __future__ import annotations

import collections
import json
import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path

import numpy as np

SEGMENTS = ("embedding", "attention_block", "mlp_block", "head")
LAYERS = 32
REDUCTION = {"decode": "AllReduce", "prefill": "ReduceScatter"}
# AllGathers per sublayer interval (prefill sequence-parallel gather) and in the head.
SUBLAYER_ALLGATHERS = {"decode": 0, "prefill": 1}
HEAD_ALLGATHERS = {"decode": 3, "prefill": 4}
HOMOGENEITY_TOLERANCE = 0.10
# Each physical core's instructions must span this fraction of its execution.
# A capture that lost one core's tail is otherwise warning-free (REPORT history).
MIN_CORE_SPAN_FRACTION = 0.95
# Trailing control instructions (NOTIFY, EVENT_SEMAPHORE, COMPARE_BRANCH) were
# observed ending up to 5.7 us past the recorded execution end. 10 us is 0.1% of
# the shortest forward (decode1, ~10 ms); a longer spill is real overlap. Each
# forward's spill is kept in its record.
BOUNDARY_TOLERANCE_NS = 10_000
PHYSICAL_CORES = frozenset({0, 1})
# Tolerated "loss" messages: DMA descriptors, which no segment uses, and the
# exporter's known schema omission of a TimelineAnnotation key (not runtime loss).
TOLERATED_LOSS_WARNINGS = frozenset(
    {
        "Notification from block DMA of type DMA were dropped. This can cause "
        "corresponding data to be incorrect.",
        "Schema validation: Unexpected key 'subgraph' cannot be saved to table "
        "'TimelineAnnotation' because no schema.yaml entry exists or no key name "
        "remapping. This key and value will be dropped and not found in parquet output.",
    }
)
_LOSS = re.compile(r"drop(?:ped|ping)?|lost|overflow|truncat", re.I)
_GRAPH = re.compile(r"graph_([0-9a-f]{32})\.neff")


class SegmentTraceError(ValueError):
    """The native trace cannot be segmented under the documented rule."""


# --- One forward ------------------------------------------------------------------


def segment_forward(cc_ops: list[dict], start: int, end: int, phase: str) -> dict:
    """Segment one execution of the stock forward from its own CcOp rows (ns)."""
    if phase not in REDUCTION:
        raise SegmentTraceError(f"unknown phase {phase!r}")
    reduction = REDUCTION[phase]
    kinds = collections.Counter(op["operation"] for op in cc_ops)
    sublayer_gathers = SUBLAYER_ALLGATHERS[phase] * 2 * LAYERS
    expected = {reduction: 2 * LAYERS + 1, "AllGather": sublayer_gathers + HEAD_ALLGATHERS[phase]}
    if dict(kinds) != expected:
        raise SegmentTraceError(f"{phase} collectives {dict(kinds)} differ from {expected}")
    reductions = sorted(
        (op for op in cc_ops if op["operation"] == reduction),
        key=lambda op: (op["start_ts"], op["end_ts"]),
    )
    ends = [op["end_ts"] for op in reductions]
    if not (start < ends[0] and all(a < b for a, b in zip(ends, ends[1:])) and ends[-1] < end):
        raise SegmentTraceError("reductions are not strictly ordered inside the execution")
    boundaries = [start, *ends, end]
    # Every AllGather must start inside the interval the documented layout gives it.
    gathers = collections.Counter(
        int(np.searchsorted(boundaries, op["start_ts"], side="right")) - 1
        for op in cc_ops
        if op["operation"] == "AllGather"
    )
    want = {k: SUBLAYER_ALLGATHERS[phase] for k in range(1, 2 * LAYERS + 1)}
    want = {k: v for k, v in want.items() if v} | {2 * LAYERS + 1: HEAD_ALLGATHERS[phase]}
    if dict(gathers) != want:
        raise SegmentTraceError(f"{phase} AllGather placement differs from the layer layout")
    durations = [b - a for a, b in zip(boundaries, boundaries[1:])]
    forward = {
        "embedding_ns": durations[0],
        "attention_ns": durations[1 : 2 * LAYERS : 2],
        "mlp_ns": durations[2 : 2 * LAYERS + 1 : 2],
        "head_ns": durations[-1],
        "span_ns": end - start,
    }
    total = (
        forward["embedding_ns"]
        + sum(forward["attention_ns"])
        + sum(forward["mlp_ns"])
        + forward["head_ns"]
    )
    if total != forward["span_ns"] or min(durations) <= 0:
        raise SegmentTraceError("segments do not partition the execution span")
    return forward


def layer_homogeneity(values, tolerance: float = HOMOGENEITY_TOLERANCE) -> dict:
    """Diagnostic spread of one forward's 32 block times; the gate is ``homogeneity_verdict``."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    deviation = float(np.abs(values / mean - 1).max())
    return {
        "mean_ns": mean,
        "min_ns": float(values.min()),
        "max_ns": float(values.max()),
        "cv": float(values.std() / mean),
        "max_relative_deviation": deviation,
        "passed": deviation <= tolerance,
    }


# --- One rank session -------------------------------------------------------------


def check_warnings(warnings: list[dict]) -> list[str]:
    """Reject any lost/dropped notification except DMA descriptors."""
    lost = [row["message"] for row in warnings if _LOSS.search(row["message"])]
    fatal = [message for message in lost if message not in TOLERATED_LOSS_WARNINGS]
    if fatal:
        raise SegmentTraceError(f"device trace lost notifications: {fatal}")
    return lost


def boundary_overrun(instructions: dict[str, np.ndarray], start: int, end: int) -> dict:
    """Nanoseconds by which traced instructions spill past the execution bounds."""
    starts, ends = instructions["start_ts"], instructions["end_ts"]
    lower, upper = np.searchsorted(starts, [start, end], side="left")
    before = int(instructions["prefix_max_end"][lower - 1] - start) if lower else 0
    after = int(ends[lower:upper].max() - end) if upper > lower else 0
    return {"start": max(before, 0), "end": max(after, 0)}


def core_coverage(instructions: dict[str, np.ndarray], start: int, end: int) -> dict:
    """Both physical cores must trace instructions across the whole execution.

    ``instructions`` holds ``start_ts``/``end_ts``/``pcore_idx`` sorted by start.
    Rows crossing the execution boundary by more than ``BOUNDARY_TOLERANCE_NS``
    make attribution ambiguous and fail; the largest crossing is reported.
    """
    starts, ends, cores = (instructions[k] for k in ("start_ts", "end_ts", "pcore_idx"))
    lower, upper = np.searchsorted(starts, [start, end], side="left")
    overrun = boundary_overrun(instructions, start, end)
    if overrun["start"] > BOUNDARY_TOLERANCE_NS:
        raise SegmentTraceError("an earlier instruction crosses the execution start")
    if overrun["end"] > BOUNDARY_TOLERANCE_NS:
        raise SegmentTraceError("an instruction crosses the execution end")
    span = end - start
    result = {}
    for core in sorted(PHYSICAL_CORES):
        mask = cores[lower:upper] == core
        if not mask.any():
            raise SegmentTraceError(f"physical core {core} traced no instruction")
        first = int(starts[lower:upper][mask].min())
        last = int(ends[lower:upper][mask].max())
        result[str(core)] = {
            "rows": int(mask.sum()),
            "first_offset_ns": first - start,
            "last_gap_ns": end - last,
            "span_fraction": (last - first) / span,
        }
    if set(np.unique(cores[lower:upper]).tolist()) != PHYSICAL_CORES:
        raise SegmentTraceError("execution traced an unexpected physical core")
    short = [core for core, row in result.items() if row["span_fraction"] < MIN_CORE_SPAN_FRACTION]
    if short:
        raise SegmentTraceError(f"physical core(s) {short} lack part of the execution trace")
    return result


def analyze_session(directory: Path, graph_shapes: dict[str, tuple[str, int]]) -> dict:
    """Segment every stock-graph execution of one rank's exported device session.

    ``graph_shapes`` maps compile-cache keys of the validated stock NEFFs to
    (phase, token_bucket). Runtime barrier NEFFs are recorded and skipped; any
    other graph fails.
    """
    from profiling.runners.neuron.parquet_columns import read_int_arrays, read_rows

    directory = Path(directory)
    lost = check_warnings(read_rows(directory / "Warning.parquet"))
    executions = sorted(
        read_rows(directory / "ExecutionInfo.parquet"),
        key=lambda row: (row["execution_start_ts"] is None, row["execution_start_ts"] or 0),
    )
    cc_ops = read_rows(directory / "CcOp.parquet", ["start_ts", "end_ts", "operation"])
    instructions = read_int_arrays(
        directory / "Instruction.parquet", ["start_ts", "end_ts", "pcore_idx"]
    )
    order = np.argsort(instructions["start_ts"], kind="stable")
    instructions = {name: values[order] for name, values in instructions.items()}
    instructions["prefix_max_end"] = np.maximum.accumulate(instructions["end_ts"])
    forwards, skipped = [], []
    for execution in executions:
        name = execution["neff_name"]
        match = _GRAPH.fullmatch(name or "")
        if match is None and (name or "").startswith("barrier_"):
            skipped.append(name)
            continue
        if match is None or match[1] not in graph_shapes:
            raise SegmentTraceError(f"device trace executed an unvalidated graph {name!r}")
        start, end = execution["execution_start_ts"], execution["execution_end_ts"]
        if start is None or end is None or end <= start:
            raise SegmentTraceError(f"execution {execution['execution_index']} has no bounds")
        phase, bucket = graph_shapes[match[1]]
        crossing = [
            op
            for op in cc_ops
            if op["start_ts"] < end
            and op["end_ts"] > start
            and not (start <= op["start_ts"] and op["end_ts"] <= end)
        ]
        if crossing:
            raise SegmentTraceError("a collective crosses the execution boundary")
        own = [op for op in cc_ops if start <= op["start_ts"] and op["end_ts"] <= end]
        forward = segment_forward(own, start, end, phase)
        attention = layer_homogeneity(forward["attention_ns"])
        mlp = layer_homogeneity(forward["mlp_ns"])
        forwards.append(
            {
                "execution_index": execution["execution_index"],
                "graph": match[1],
                "phase": phase,
                "token_bucket": bucket,
                "start_ns": start,
                **forward,
                "attention_homogeneity": attention,
                "mlp_homogeneity": mlp,
                "core_coverage": core_coverage(instructions, start, end),
                "boundary_overrun_ns": boundary_overrun(instructions, start, end),
            }
        )
    if not forwards:
        raise SegmentTraceError(f"{directory} has no stock forward execution")
    return {
        "session": directory.name,
        "tolerated_loss_warnings": lost,
        "skipped_executions": skipped,
        "instruction_rows": int(len(instructions["start_ts"])),
        "forwards": forwards,
    }


# --- Across ranks and forwards ------------------------------------------------------


def homogeneity_verdict(sessions: list[dict]) -> dict[str, dict]:
    """Per shape and rank, interior layers must share one block time.

    Rows store the 32-layer mean, so 32 x mean reproduces every forward exactly;
    the gate guards that the mean describes a repeated layer. Each layer's
    offset is its median over forwards of (layer / that forward's layer mean),
    which ignores one-off spikes. The first and last layers carry systematic
    compiler-partition offsets (decode1 layer-0 attention, prefill layer-31 MLP
    sharing the head's partition); they are reported, not gated.
    """
    stacked: dict[tuple[str, str], dict[str, list]] = {}
    for session in sessions:
        for forward in session["forwards"]:
            key = (f"{forward['phase']}:{forward['token_bucket']}", session["session"])
            for block in ("attention", "mlp"):
                stacked.setdefault(key, {}).setdefault(block, []).append(forward[f"{block}_ns"])
    verdict: dict[str, dict] = {}
    for (shape, session), blocks in sorted(stacked.items()):
        row = verdict.setdefault(
            shape, {"passed": True, "tolerance": HOMOGENEITY_TOLERANCE, "failures": [], "ranks": {}}
        )
        for block, values in blocks.items():
            values = np.asarray(values, dtype=float)
            offsets = np.median(values / values.mean(axis=1, keepdims=True), axis=0) - 1
            interior = np.abs(offsets[1:-1])
            stats = {
                "forwards": len(values),
                "first_layer_offset": float(offsets[0]),
                "last_layer_offset": float(offsets[-1]),
                "worst_interior_offset": float(interior.max()),
                "worst_interior_layer": int(interior.argmax()) + 1,
                "worst_single_forward_deviation": float(
                    np.abs(values / values.mean(axis=1, keepdims=True) - 1).max()
                ),
            }
            row["ranks"].setdefault(session, {})[block] = stats
            if stats["worst_interior_offset"] > HOMOGENEITY_TOLERANCE:
                row["passed"] = False
                row["failures"].append({"session": session, "block": block, **stats})
    return verdict


def combine_ranks(sessions: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """Per shape, pair the k-th forward of every rank and average its segments.

    Ranks run the same forward sequence, so each must hold the same number of
    executions per shape. Averaging preserves the exact per-rank partition.
    """
    by_shape = collections.defaultdict(list)
    for session in sessions:
        grouped = collections.defaultdict(list)
        for forward in sorted(session["forwards"], key=lambda row: row["start_ns"]):
            grouped[(forward["phase"], forward["token_bucket"])].append(forward)
        for shape, forwards in grouped.items():
            by_shape[shape].append(forwards)
    combined = {}
    for shape, ranks in by_shape.items():
        if len(ranks) != len(sessions) or len({len(rank) for rank in ranks}) != 1:
            raise SegmentTraceError(f"ranks captured different {shape} forward counts")
        rows = []
        for forwards in zip(*ranks):
            row = {
                "embedding": statistics.fmean(f["embedding_ns"] for f in forwards) / 1e6,
                "attention_block": statistics.fmean(
                    statistics.fmean(f["attention_ns"]) for f in forwards
                )
                / 1e6,
                "mlp_block": statistics.fmean(statistics.fmean(f["mlp_ns"]) for f in forwards)
                / 1e6,
                "head": statistics.fmean(f["head_ns"] for f in forwards) / 1e6,
                "span": statistics.fmean(f["span_ns"] for f in forwards) / 1e6,
            }
            composed = composed_forward_ms(row)
            if not np.isclose(composed, row["span"], rtol=1e-12, atol=1e-9):
                raise SegmentTraceError("rank-averaged segments do not sum to the span")
            rows.append(row)
        combined[shape] = rows
    return combined


def composed_forward_ms(segments: dict) -> float:
    """The simulator's fold: embedding + 32 x (attention + MLP) + head."""
    return (
        segments["embedding"]
        + LAYERS * (segments["attention_block"] + segments["mlp_block"])
        + segments["head"]
    )


def segment_times(combined) -> dict[tuple[str, str, int], list[float]]:
    """``{(segment, phase, bucket): [ms per captured forward]}``."""
    return {
        (segment, *shape): [row[segment] for row in rows]
        for shape, rows in combined.items()
        for segment in SEGMENTS
    }


def timing_verdict(times, stock_times, shapes, tolerance: float = 0.05) -> dict:
    """Per shape: the composed segment medians against the same-group stock median."""
    verdict = {}
    for phase, bucket in sorted(shapes):
        medians = {s: statistics.median(times[(s, phase, bucket)]) for s in SEGMENTS}
        composed = composed_forward_ms(medians)
        stock = statistics.median(stock_times[(phase, bucket)])
        error = composed / stock - 1
        verdict[f"{phase}:{bucket}"] = {
            **{f"{s}_median_ms": medians[s] for s in SEGMENTS},
            "composed_ms": composed,
            "stock_whole_median_ms": stock,
            "error_pct": 100 * error,
            "forwards": len(times[("head", phase, bucket)]),
            "stock_forwards": len(stock_times[(phase, bucket)]),
            "passed": abs(error) <= tolerance,
        }
    return verdict


def layer_statistics(sessions: list[dict], shape: tuple[str, int]) -> dict:
    """Provenance summary of the per-layer distributions behind the 32x fold."""
    out = {}
    for key in ("attention", "mlp"):
        forwards = [
            f for s in sessions for f in s["forwards"] if (f["phase"], f["token_bucket"]) == shape
        ]
        stack = np.array([f[f"{key}_ns"] for f in forwards], dtype=float)
        per_layer = np.median(stack, axis=0)
        out[key] = {
            "per_layer_median_min_ms": float(per_layer.min() / 1e6),
            "per_layer_median_max_ms": float(per_layer.max() / 1e6),
            "worst_layer_deviation": max(
                f[f"{key}_homogeneity"]["max_relative_deviation"] for f in forwards
            ),
            "median_cv": float(statistics.median(f[f"{key}_homogeneity"]["cv"] for f in forwards)),
        }
    return out


def rank_spread(sessions: list[dict], shape: tuple[str, int]) -> dict:
    """Per segment: each rank's median, and the max-min spread relative to the mean."""
    spread = {}
    for segment, field in (
        ("embedding", "embedding_ns"),
        ("attention_block", "attention_ns"),
        ("mlp_block", "mlp_ns"),
        ("head", "head_ns"),
    ):
        medians = []
        for session in sessions:
            values = [
                statistics.fmean(f[field]) if isinstance(f[field], list) else f[field]
                for f in session["forwards"]
                if (f["phase"], f["token_bucket"]) == shape
            ]
            medians.append(statistics.median(values) / 1e6)
        spread[segment] = {
            "rank_medians_ms": medians,
            "spread_pct": 100 * (max(medians) - min(medians)) / statistics.fmean(medians),
        }
    return spread


# --- Export -------------------------------------------------------------------------


def _stage_rank(profiles: Path, rank_dir: Path, stage: Path) -> None:
    """Hard-link one rank's session and the shared NEFF copies into ``stage``."""
    for source in (rank_dir, profiles / "neffs"):
        for path in source.rglob("*"):
            target = stage / source.name / path.relative_to(source)
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, target)


def export_device_sessions(profiles: Path, output: Path) -> list[Path]:
    """Export each rank's device session under ``profiles`` to Parquet, one per rank.

    A multi-rank directory can export every session into one flat directory,
    each overwriting the last, so every rank is staged and exported alone.
    Ranks run one after another: each export ingests a multi-GB raw trace, and
    four concurrent exports exhausted host memory. Event and DMA traces are not
    used by the rule and are skipped; this leaves ExecutionInfo, CcOp, Warning
    and Instruction rows unchanged.
    """
    ranks = sorted(path for path in profiles.glob("*_pid_*") if path.is_dir())
    if not ranks:
        raise SegmentTraceError(f"no rank device profile under {profiles}")
    sessions = []
    for rank in ranks:
        stage = output / "staging" / rank.name
        _stage_rank(profiles, rank, stage)
        parquet = output / rank.name
        status = output / f"{rank.name}.status.json"
        command = [
            "/opt/aws/neuron/bin/neuron-explorer", "view", "-d", str(stage),
            "--output-format", "parquet", "--output-file", str(parquet), "--ingest-only",
            "--disable-ui", "--force", "--ignore-system-profile", "--ignore-event-trace",
            "--ignore-dma-trace", "--ignore-nc-buf-usage", "--ignore-instruction-hierarchy",
            "--processing-status-file", str(status),
        ]  # fmt: skip
        with (output / f"{rank.name}.log").open("w") as log:
            code = subprocess.run(command, stdout=log, stderr=log).returncode
        report = json.loads(status.read_text()) if status.exists() else {"status_code": code}
        profiles_ok = [p.get("status") == "success" for p in report.get("profiles", [])]
        if code or report["status_code"] or profiles_ok != [True]:
            raise SegmentTraceError(f"device profile export failed; see {status}")
        found = [path.parent for path in parquet.rglob("ExecutionInfo.parquet")]
        if len(found) != 1:
            raise SegmentTraceError(f"{parquet} must hold exactly one exported session")
        sessions.append(found[0])
    for stage in (output / "staging").iterdir():
        shutil.rmtree(stage)
    return sessions


def main(argv=None) -> None:
    """Offline replay: ``python -m ...vllm_segment_trace PARQUET_ROOT KEY=phase:bucket ...``."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("parquet_root", type=Path)
    parser.add_argument("graphs", nargs="+", help="compile key=phase:token_bucket")
    args = parser.parse_args(argv)
    shapes = {}
    for item in args.graphs:
        key, value = item.split("=")
        phase, bucket = value.split(":")
        shapes[key] = (phase, int(bucket))
    sessions = [
        analyze_session(path.parent, shapes)
        for path in sorted(args.parquet_root.glob("*/ExecutionInfo.parquet"))
    ]
    combined = combine_ranks(sessions)
    report = {}
    for shape, rows in combined.items():
        report[f"{shape[0]}:{shape[1]}"] = {
            "forwards": len(rows),
            "segments_median_ms": {s: statistics.median(r[s] for r in rows) for s in SEGMENTS},
            "span_median_ms": statistics.median(r["span"] for r in rows),
            "layers": layer_statistics(sessions, shape),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
