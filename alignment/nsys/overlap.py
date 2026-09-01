"""Diagnose same-stream and multi-stream overlap in a bounded NSYS window.

The output is interval arithmetic, not a performance counterfactual.  In
particular, a same-stream overlap is evidence of programmatic dependent launch
(PDL), but the overlapped consumer residency can include time spent waiting in
``griddepcontrol.wait``.  Therefore ``*_trace_reduction_ns`` means only "the
amount removed when raw kernel durations are reduced to a busy union"; it is
not automatically reclaimable wall time and must not be copied into a CostTree
as a discount.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .evidence import merge_duration_ns
from .parse import parse_trace


@dataclass(frozen=True)
class KernelInterval:
    """One physical launch on one CUDA stream."""

    start_ns: int
    end_ns: int
    global_pid: int
    stream_id: int
    name: str
    category: str
    stable_order: int

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    @property
    def stream_key(self) -> tuple[int, int]:
        """CUDA stream identity within the process/context that owns it."""
        return (self.global_pid, self.stream_id)


@dataclass
class PairReduction:
    """Trace-union reduction attributed to one producer/consumer name pair."""

    producer_category: str
    consumer_category: str
    trace_reduction_ns: int = 0
    occurrences: int = 0
    by_stage_ns: Counter[str] = field(default_factory=Counter)


def _name_index(parsed: dict) -> dict[int, str]:
    return {int(name_id): str(name) for name_id, name in parsed["kernel_names"].items()}


def _events_for_device(
    detail: dict,
    device_id: int,
    names: dict[int, str],
    phases: set[str] | None,
) -> list[KernelInterval]:
    events: list[KernelInterval] = []
    stable_order = 0
    for item in detail["ranges"]:
        if int(item["device_id"]) != device_id:
            continue
        phase = str(item["phase"])
        if phases is not None and phase not in phases:
            continue
        global_pid = int(item["worker"]["global_pid"])
        for kernel in item["kernels"]:
            start_ns = int(kernel["start_ns"])
            end_ns = int(kernel["end_ns"])
            if end_ns < start_ns:
                raise ValueError(
                    f"iteration {detail['iteration']} device {device_id} has a kernel "
                    f"with end_ns {end_ns} before start_ns {start_ns}"
                )
            name_id = int(kernel["name_id"])
            if name_id not in names:
                raise ValueError(f"kernel name_id {name_id} is absent from kernel_names")
            events.append(
                KernelInterval(
                    start_ns=start_ns,
                    end_ns=end_ns,
                    global_pid=global_pid,
                    stream_id=int(kernel["stream_id"]),
                    name=names[name_id],
                    category=str(kernel["category"]),
                    stable_order=stable_order,
                )
            )
            stable_order += 1
    return events


def _pair_reductions_on_stream(
    events: Iterable[KernelInterval],
) -> dict[tuple[str, str], PairReduction]:
    """Attribute each same-stream duplicate nanosecond to one earlier launch.

    Events are inserted by start time, with a wider interval winning an exact
    start tie.  For each atomic segment of the current launch already covered
    by prior launches, the latest-starting prior launch is called the producer.
    This is deterministic and sums exactly to ``raw - per-stream union``.  It
    is a trace attribution only: it does not prove the chosen launch supplied
    the consumer's actual PDL dependency.
    """

    ordered = sorted(
        events,
        key=lambda event: (
            event.start_ns,
            -event.end_ns,
            event.stable_order,
        ),
    )
    active: list[KernelInterval] = []
    reductions: dict[tuple[str, str], PairReduction] = {}
    for consumer in ordered:
        active = [producer for producer in active if producer.end_ns > consumer.start_ns]
        candidates = [
            producer
            for producer in active
            if producer.start_ns < consumer.end_ns and producer.end_ns > consumer.start_ns
        ]
        boundaries = {consumer.start_ns, consumer.end_ns}
        for producer in candidates:
            boundaries.add(max(consumer.start_ns, producer.start_ns))
            boundaries.add(min(consumer.end_ns, producer.end_ns))
        points = sorted(boundaries)
        attributed_to_consumer: Counter[tuple[str, str]] = Counter()
        producer_by_pair: dict[tuple[str, str], KernelInterval] = {}
        for start_ns, end_ns in zip(points, points[1:], strict=False):
            if end_ns <= start_ns:
                continue
            covering = [
                producer
                for producer in candidates
                if producer.start_ns <= start_ns and producer.end_ns >= end_ns
            ]
            if not covering:
                continue
            producer = max(
                covering,
                key=lambda event: (
                    event.start_ns,
                    event.end_ns,
                    -event.stable_order,
                ),
            )
            pair = (producer.name, consumer.name)
            attributed_to_consumer[pair] += end_ns - start_ns
            producer_by_pair[pair] = producer
        for pair, reduction_ns in attributed_to_consumer.items():
            producer = producer_by_pair[pair]
            row = reductions.setdefault(
                pair,
                PairReduction(
                    producer_category=producer.category,
                    consumer_category=consumer.category,
                ),
            )
            row.trace_reduction_ns += reduction_ns
            row.occurrences += 1
        active.append(consumer)
    return reductions


def _merge_pair_reductions(
    target: dict[tuple[str, str], PairReduction],
    source: dict[tuple[str, str], PairReduction],
    stage: str,
) -> None:
    for pair, source_row in source.items():
        target_row = target.setdefault(
            pair,
            PairReduction(
                producer_category=source_row.producer_category,
                consumer_category=source_row.consumer_category,
            ),
        )
        target_row.trace_reduction_ns += source_row.trace_reduction_ns
        target_row.occurrences += source_row.occurrences
        target_row.by_stage_ns[stage] += source_row.trace_reduction_ns


def _diagnose_events(events: list[KernelInterval]) -> dict[str, int]:
    intervals = [(event.start_ns, event.end_ns) for event in events]
    raw_ns = sum(event.duration_ns for event in events)
    busy_union_ns = merge_duration_ns(intervals)
    intervals_by_stream: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for event in events:
        intervals_by_stream[event.stream_key].append((event.start_ns, event.end_ns))
    per_stream_union_ns = sum(
        merge_duration_ns(stream_intervals) for stream_intervals in intervals_by_stream.values()
    )
    pdl_reduction_ns = raw_ns - per_stream_union_ns
    multistream_reduction_ns = per_stream_union_ns - busy_union_ns
    return {
        "kernel_launches": len(events),
        "stream_count": len(intervals_by_stream),
        "raw_kernel_ns": raw_ns,
        "per_stream_busy_union_ns": per_stream_union_ns,
        "busy_union_ns": busy_union_ns,
        "pdl_same_stream_trace_reduction_ns": pdl_reduction_ns,
        "multistream_trace_reduction_ns": multistream_reduction_ns,
        "invariant_residual_ns": (
            raw_ns - pdl_reduction_ns - multistream_reduction_ns - busy_union_ns
        ),
    }


def _sum_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    summed_fields = (
        "kernel_launches",
        "raw_kernel_ns",
        "per_stream_busy_union_ns",
        "busy_union_ns",
        "pdl_same_stream_trace_reduction_ns",
        "multistream_trace_reduction_ns",
        "invariant_residual_ns",
    )
    result = {field: sum(int(row[field]) for row in rows) for field in summed_fields}
    result["iteration_device_rows"] = len(rows)
    raw_ns = result["raw_kernel_ns"]
    result["busy_union_pct_of_raw"] = 100.0 * result["busy_union_ns"] / raw_ns if raw_ns else 0.0
    result["pdl_same_stream_pct_of_raw"] = (
        100.0 * result["pdl_same_stream_trace_reduction_ns"] / raw_ns if raw_ns else 0.0
    )
    result["multistream_pct_of_raw"] = (
        100.0 * result["multistream_trace_reduction_ns"] / raw_ns if raw_ns else 0.0
    )
    return result


def build_overlap_diagnostics(
    parsed: dict,
    *,
    device_ids: set[int] | None = None,
    all_devices: bool = False,
    stages: set[str] | None = None,
    phases: set[str] | None = None,
    top_pairs: int | None = 50,
    source_overrides: dict | None = None,
) -> dict:
    """Build an exact overlap-decomposition report from normalized NSYS data."""
    if device_ids is not None and all_devices:
        raise ValueError("device_ids and all_devices are mutually exclusive")
    if top_pairs is not None and top_pairs < 0:
        raise ValueError("top_pairs must be non-negative or None")
    names = _name_index(parsed)
    available_stages = {str(detail["stage"]) for detail in parsed["iteration_details"]}
    available_phases = {
        str(item["phase"]) for detail in parsed["iteration_details"] for item in detail["ranges"]
    }
    available_devices = {
        int(item["device_id"])
        for detail in parsed["iteration_details"]
        for item in detail["ranges"]
    }
    for label, selected, available in (
        ("stage", stages, available_stages),
        ("phase", phases, available_phases),
        ("device", device_ids, available_devices),
    ):
        unknown = set() if selected is None else selected - available
        if unknown:
            raise ValueError(
                f"unknown {label} selection {sorted(unknown)}; available values are "
                f"{sorted(available)}"
            )
    rows: list[dict] = []
    pairs: dict[tuple[str, str], PairReduction] = {}
    for detail in parsed["iteration_details"]:
        stage = str(detail["stage"])
        if stages is not None and stage not in stages:
            continue
        detail_devices = sorted(
            {
                int(item["device_id"])
                for item in detail["ranges"]
                if phases is None or str(item["phase"]) in phases
            }
        )
        if all_devices:
            selected_devices = detail_devices
        elif device_ids is not None:
            selected_devices = [device for device in detail_devices if device in device_ids]
        else:
            reference = detail.get("reference_device_id")
            selected_devices = [] if reference is None else [int(reference)]
        for device_id in selected_devices:
            events = _events_for_device(detail, device_id, names, phases)
            if not events:
                continue
            row = {
                "iteration": int(detail["iteration"]),
                "stage": stage,
                "device_id": device_id,
                **_diagnose_events(events),
            }
            rows.append(row)
            by_stream: dict[tuple[int, int], list[KernelInterval]] = defaultdict(list)
            for event in events:
                by_stream[event.stream_key].append(event)
            for stream_events in by_stream.values():
                _merge_pair_reductions(
                    pairs,
                    _pair_reductions_on_stream(stream_events),
                    stage,
                )

    if not rows:
        raise ValueError(
            "overlap selection matched no iteration/device rows; check the joint "
            "device, stage, phase, and parse-window selection"
        )

    overall = _sum_rows(rows)
    by_stage = {
        stage: _sum_rows(row for row in rows if row["stage"] == stage)
        for stage in sorted({str(row["stage"]) for row in rows})
    }
    pdl_total_ns = overall["pdl_same_stream_trace_reduction_ns"]
    all_pair_rows = [
        {
            "producer_kernel": producer,
            "consumer_kernel": consumer,
            "producer_category": reduction.producer_category,
            "consumer_category": reduction.consumer_category,
            "trace_reduction_ns": reduction.trace_reduction_ns,
            "occurrences": reduction.occurrences,
            "mean_trace_reduction_ns": (
                reduction.trace_reduction_ns / reduction.occurrences
                if reduction.occurrences
                else 0.0
            ),
            "pct_of_pdl_trace_reduction": (
                100.0 * reduction.trace_reduction_ns / pdl_total_ns if pdl_total_ns else 0.0
            ),
            "by_stage_ns": dict(sorted(reduction.by_stage_ns.items())),
        }
        for (producer, consumer), reduction in sorted(
            pairs.items(),
            key=lambda item: (-item[1].trace_reduction_ns, item[0]),
        )
    ]
    pair_rows = all_pair_rows if top_pairs is None else all_pair_rows[:top_pairs]
    pair_sum_ns = sum(reduction.trace_reduction_ns for reduction in pairs.values())
    returned_pair_sum_ns = sum(int(row["trace_reduction_ns"]) for row in pair_rows)
    source = {
        "parsed_schema_version": parsed.get("schema_version"),
        "sqlite": parsed.get("sqlite"),
        "iteration_start": parsed.get("iteration_start"),
        "iteration_end": parsed.get("iteration_end"),
        "range_mode": parsed.get("range_mode"),
        "tp_size": parsed.get("tp_size"),
    }
    if source_overrides is not None:
        source.update(source_overrides)
    return {
        "schema_version": 1,
        "source": source,
        "selection": {
            "device_policy": (
                "all" if all_devices else "explicit" if device_ids is not None else "reference"
            ),
            "device_ids": sorted(device_ids) if device_ids is not None else None,
            "stages": sorted(stages) if stages is not None else None,
            "phases": sorted(phases) if phases is not None else None,
            "top_pairs": top_pairs,
        },
        "overall": overall,
        "by_stage": by_stage,
        "by_iteration_device": rows,
        "pdl_kernel_pairs": pair_rows,
        "checks": {
            "raw_equals_busy_plus_pdl_plus_multistream": all(
                row["invariant_residual_ns"] == 0 for row in rows
            ),
            "pdl_pairs_sum_to_pdl_reduction": pair_sum_ns == pdl_total_ns,
            "pdl_pair_sum_ns": pair_sum_ns,
            "pdl_pair_rows_total": len(all_pair_rows),
            "pdl_pair_rows_returned": len(pair_rows),
            "returned_pdl_pair_sum_ns": returned_pair_sum_ns,
            "omitted_pdl_pair_sum_ns": pair_sum_ns - returned_pair_sum_ns,
        },
        "definitions": {
            "pdl_same_stream_trace_reduction_ns": (
                "raw kernel duration minus the union within each CUDA stream; same-stream "
                "overlap is PDL evidence, but can include consumer griddepcontrol wait"
            ),
            "multistream_trace_reduction_ns": (
                "sum of per-stream busy unions minus the device-wide busy union"
            ),
            "pdl_kernel_pairs": (
                "each duplicate same-stream nanosecond is charged once to the latest-starting "
                "earlier kernel covering that segment; deterministic attribution, not proof of "
                "the launch dependency; checks describe the complete attribution before the "
                "optional top-pairs display limit"
            ),
            "not_reclaimable_savings": (
                "trace reductions are interval-union accounting, not an estimate of speedup or "
                "a CostTree discount; measuring savings requires an otherwise-identical PDL "
                "on/off pair experiment"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose PDL/same-stream and multi-stream NSYS overlap"
    )
    parser.add_argument("--sqlite", type=Path, required=True, help="Exported nsys SQLite database")
    parser.add_argument("--metrics", type=Path, default=None, help="iteration metrics JSONL")
    parser.add_argument("--iteration-start", type=int, required=True)
    parser.add_argument("--iteration-end", type=int, required=True)
    parser.add_argument("--range-mode", choices=["forward", "envelope", "phases"], default="phases")
    parser.add_argument("--default-stage", default="all")
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--device", type=int, action="append", dest="devices")
    parser.add_argument("--all-devices", action="store_true")
    parser.add_argument("--stage", action="append", dest="stages")
    parser.add_argument("--phase", action="append", dest="phases")
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=50,
        help="maximum PDL pair rows; use 0 to retain every pair",
    )
    parser.add_argument("--output", type=Path, default=None, help="write JSON here (else stdout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.devices and args.all_devices:
        raise SystemExit("--device and --all-devices are mutually exclusive")
    if args.iteration_end < args.iteration_start:
        raise SystemExit("--iteration-end must be >= --iteration-start")
    if args.top_pairs < 0:
        raise SystemExit("--top-pairs must be >= 0")
    if args.server_log is not None:
        from ..profiler.vllm_server import extract_worker_device_ranks

        worker_ranks = extract_worker_device_ranks(args.server_log)
    else:
        worker_ranks = {}
    parsed = parse_trace(
        args.sqlite,
        args.metrics,
        args.iteration_start,
        args.iteration_end,
        range_mode=args.range_mode,
        default_stage=args.default_stage,
        worker_ranks=worker_ranks,
        tp_size=args.tp_size,
    )
    try:
        report = build_overlap_diagnostics(
            parsed,
            device_ids=set(args.devices) if args.devices else None,
            all_devices=args.all_devices,
            stages=set(args.stages) if args.stages else None,
            phases=set(args.phases) if args.phases else None,
            top_pairs=None if args.top_pairs == 0 else args.top_pairs,
            source_overrides={
                "metrics": str(args.metrics) if args.metrics is not None else None,
                "default_stage": args.default_stage,
                "server_log": str(args.server_log) if args.server_log is not None else None,
            },
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    text = json.dumps(report, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
