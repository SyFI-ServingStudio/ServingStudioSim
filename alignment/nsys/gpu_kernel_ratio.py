"""Derive an experiment-specific GPU-time multiplier from one NSYS profile.

The simulator predicts kernel workload time.  This module measures how much
first-kernel-to-next-first-kernel GPU cycle time surrounds that workload in one
captured vLLM run.  It deliberately reports both attributed and global kernel
busy unions: the former audits parser ownership, while the latter is the
denominator used for ``gpu_time_multiplier``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _IterationBoundary:
    device_id: int
    iteration: int
    iteration_type: str
    first_kernel_start_ns: int
    kernel_duration_sum_ns: int
    attributed_busy_intervals: tuple[tuple[int, int], ...]


@dataclass
class _GpuCycle:
    device_id: int
    iteration: int
    iteration_type: str
    start_ns: int
    end_ns: int
    kernel_duration_sum_ns: int
    attributed_kernel_busy_ns: int
    attributed_kernel_busy_in_cycle_ns: int
    global_kernel_busy_in_cycle_ns: int = 0


def _merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    ordered = sorted(intervals)
    if not ordered:
        return ()
    merged: list[tuple[int, int]] = []
    current_start, current_end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))
    return tuple(merged)


def _clipped_union_duration_ns(
    merged_intervals: Iterable[tuple[int, int]], start_ns: int, end_ns: int
) -> int:
    return sum(
        min(interval_end_ns, end_ns) - max(interval_start_ns, start_ns)
        for interval_start_ns, interval_end_ns in merged_intervals
        if interval_end_ns > start_ns and interval_start_ns < end_ns
    )


def _load_iteration_boundaries(parsed_path: Path) -> dict[int, list[_IterationBoundary]]:
    parsed = json.loads(parsed_path.read_text())
    boundaries_by_device: dict[int, list[_IterationBoundary]] = defaultdict(list)

    for iteration_detail in parsed["iteration_details"]:
        intervals_by_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
        duration_sum_by_device: dict[int, int] = defaultdict(int)
        for measured_range in iteration_detail["ranges"]:
            device_id = int(measured_range["device_id"])
            for kernel in measured_range["kernels"]:
                start_ns = int(kernel["start_ns"])
                end_ns = int(kernel["end_ns"])
                if end_ns <= start_ns:
                    raise ValueError(
                        f"iteration {iteration_detail['iteration']} has a non-positive "
                        f"kernel interval [{start_ns}, {end_ns})"
                    )
                intervals_by_device[device_id].append((start_ns, end_ns))
                duration_sum_by_device[device_id] += end_ns - start_ns

        for device_id, intervals in intervals_by_device.items():
            attributed_busy_intervals = _merge_intervals(intervals)
            if not attributed_busy_intervals:
                continue
            boundaries_by_device[device_id].append(
                _IterationBoundary(
                    device_id=device_id,
                    iteration=int(iteration_detail["iteration"]),
                    iteration_type=str(iteration_detail["iteration_type"]),
                    first_kernel_start_ns=attributed_busy_intervals[0][0],
                    kernel_duration_sum_ns=duration_sum_by_device[device_id],
                    attributed_busy_intervals=attributed_busy_intervals,
                )
            )

    return boundaries_by_device


def _build_gpu_cycles(
    boundaries_by_device: dict[int, list[_IterationBoundary]],
) -> dict[int, list[_GpuCycle]]:
    cycles_by_device: dict[int, list[_GpuCycle]] = {}
    for device_id, boundaries in boundaries_by_device.items():
        boundaries.sort(key=lambda boundary: boundary.first_kernel_start_ns)
        cycles: list[_GpuCycle] = []
        for current, following in zip(boundaries, boundaries[1:]):
            if following.first_kernel_start_ns <= current.first_kernel_start_ns:
                raise ValueError(
                    f"device {device_id} has non-increasing first-kernel timestamps at "
                    f"iterations {current.iteration} and {following.iteration}"
                )
            attributed_busy_ns = sum(
                end_ns - start_ns
                for start_ns, end_ns in current.attributed_busy_intervals
            )
            cycles.append(
                _GpuCycle(
                    device_id=device_id,
                    iteration=current.iteration,
                    iteration_type=current.iteration_type,
                    start_ns=current.first_kernel_start_ns,
                    end_ns=following.first_kernel_start_ns,
                    kernel_duration_sum_ns=current.kernel_duration_sum_ns,
                    attributed_kernel_busy_ns=attributed_busy_ns,
                    attributed_kernel_busy_in_cycle_ns=_clipped_union_duration_ns(
                        current.attributed_busy_intervals,
                        current.first_kernel_start_ns,
                        following.first_kernel_start_ns,
                    ),
                )
            )
        if cycles:
            cycles_by_device[device_id] = cycles
    return cycles_by_device


def _assign_global_busy_interval(
    cycles: list[_GpuCycle], cycle_index: int, start_ns: int, end_ns: int
) -> int:
    while cycle_index < len(cycles) and cycles[cycle_index].end_ns <= start_ns:
        cycle_index += 1
    overlapping_index = cycle_index
    while overlapping_index < len(cycles) and cycles[overlapping_index].start_ns < end_ns:
        cycle = cycles[overlapping_index]
        overlap_ns = min(end_ns, cycle.end_ns) - max(start_ns, cycle.start_ns)
        if overlap_ns > 0:
            cycle.global_kernel_busy_in_cycle_ns += overlap_ns
        overlapping_index += 1
    return cycle_index


def _populate_global_kernel_busy(
    connection: sqlite3.Connection, device_id: int, cycles: list[_GpuCycle]
) -> None:
    first_cycle_start_ns = cycles[0].start_ns
    last_cycle_end_ns = cycles[-1].end_ns
    rows = connection.execute(
        """
        SELECT start, end
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId = ? AND end > ? AND start < ?
        ORDER BY start, end
        """,
        (device_id, first_cycle_start_ns, last_cycle_end_ns),
    )

    merged_start_ns: int | None = None
    merged_end_ns: int | None = None
    cycle_index = 0
    for raw_start_ns, raw_end_ns in rows:
        start_ns = int(raw_start_ns)
        end_ns = int(raw_end_ns)
        if merged_start_ns is None:
            merged_start_ns, merged_end_ns = start_ns, end_ns
        elif start_ns <= merged_end_ns:
            merged_end_ns = max(merged_end_ns, end_ns)
        else:
            cycle_index = _assign_global_busy_interval(
                cycles, cycle_index, merged_start_ns, merged_end_ns
            )
            merged_start_ns, merged_end_ns = start_ns, end_ns

    if merged_start_ns is not None and merged_end_ns is not None:
        _assign_global_busy_interval(
            cycles, cycle_index, merged_start_ns, merged_end_ns
        )


def _summarize_cycles(cycles: list[_GpuCycle]) -> dict[str, Any]:
    gpu_cycle_ns = sum(cycle.end_ns - cycle.start_ns for cycle in cycles)
    kernel_duration_sum_ns = sum(cycle.kernel_duration_sum_ns for cycle in cycles)
    attributed_busy_ns = sum(cycle.attributed_kernel_busy_ns for cycle in cycles)
    attributed_busy_in_cycle_ns = sum(
        cycle.attributed_kernel_busy_in_cycle_ns for cycle in cycles
    )
    global_busy_ns = sum(cycle.global_kernel_busy_in_cycle_ns for cycle in cycles)

    for cycle in cycles:
        if cycle.attributed_kernel_busy_in_cycle_ns > cycle.global_kernel_busy_in_cycle_ns:
            raise ValueError(
                f"iteration {cycle.iteration} device {cycle.device_id} has more attributed "
                "busy time than the global CUPTI busy union in the same GPU cycle"
            )
    if global_busy_ns <= 0 or gpu_cycle_ns <= 0:
        raise ValueError("GPU cycle population has no positive cycle or kernel-busy time")

    kernel_gpu_fraction = global_busy_ns / gpu_cycle_ns
    gpu_time_multiplier = gpu_cycle_ns / global_busy_ns
    unattributed_busy_ns = global_busy_ns - attributed_busy_in_cycle_ns
    no_kernel_gap_ns = gpu_cycle_ns - global_busy_ns
    return {
        "cycles": len(cycles),
        "kernel_duration_sum_ms": kernel_duration_sum_ns / 1e6,
        "attributed_kernel_busy_ms": attributed_busy_ns / 1e6,
        "attributed_kernel_busy_in_cycle_ms": attributed_busy_in_cycle_ns / 1e6,
        "global_kernel_busy_in_cycle_ms": global_busy_ns / 1e6,
        "unattributed_kernel_busy_in_cycle_ms": unattributed_busy_ns / 1e6,
        "gpu_no_kernel_gap_ms": no_kernel_gap_ns / 1e6,
        "gpu_cycle_ms": gpu_cycle_ns / 1e6,
        "kernel_gpu_fraction": kernel_gpu_fraction,
        "gpu_time_multiplier": gpu_time_multiplier,
        "cycles_with_unattributed_kernel_busy": sum(
            cycle.global_kernel_busy_in_cycle_ns
            > cycle.attributed_kernel_busy_in_cycle_ns
            for cycle in cycles
        ),
        "cycles_with_attributed_kernel_outside_cycle": sum(
            cycle.attributed_kernel_busy_ns
            > cycle.attributed_kernel_busy_in_cycle_ns
            for cycle in cycles
        ),
    }


def compute_gpu_kernel_ratio(parsed_path: Path, sqlite_path: Path) -> dict[str, Any]:
    """Compute pooled GPU-cycle/kernel-busy statistics for one parsed capture."""

    parsed_path = parsed_path.resolve()
    sqlite_path = sqlite_path.resolve()
    boundaries_by_device = _load_iteration_boundaries(parsed_path)
    cycles_by_device = _build_gpu_cycles(boundaries_by_device)
    if not cycles_by_device:
        raise ValueError("parsed capture has fewer than two kernel-bearing iterations")

    with sqlite3.connect(sqlite_path) as connection:
        for device_id, cycles in cycles_by_device.items():
            _populate_global_kernel_busy(connection, device_id, cycles)

    all_cycles = [
        cycle
        for device_cycles in cycles_by_device.values()
        for cycle in device_cycles
    ]
    iteration_types = sorted({cycle.iteration_type for cycle in all_cycles})
    return {
        "schema_version": 1,
        "sources": {
            "parsed_json": str(parsed_path),
            "nsys_sqlite": str(sqlite_path),
        },
        "definition": {
            "gpu_cycle": (
                "first attributed kernel start of iteration i to the first attributed "
                "kernel start of the next kernel-bearing iteration on the same device"
            ),
            "kernel_duration_sum": (
                "additive duration sum of kernels attributed to iteration i"
            ),
            "attributed_kernel_busy": (
                "union of kernels attributed to iteration i by indexed phase ranges"
            ),
            "global_kernel_busy_in_cycle": (
                "union of every CUPTI kernel interval on the device, clipped to the GPU cycle"
            ),
            "population": (
                "all kernel-bearing parsed iterations except the final iteration on each device"
            ),
            "gpu_time_multiplier": "pooled GPU cycle / pooled global kernel busy",
        },
        "overall": _summarize_cycles(all_cycles),
        "by_iteration_type": {
            iteration_type: _summarize_cycles(
                [
                    cycle
                    for cycle in all_cycles
                    if cycle.iteration_type == iteration_type
                ]
            )
            for iteration_type in iteration_types
        },
        "by_device": {
            str(device_id): {
                "overall": _summarize_cycles(cycles),
                "by_iteration_type": {
                    iteration_type: _summarize_cycles(
                        [
                            cycle
                            for cycle in cycles
                            if cycle.iteration_type == iteration_type
                        ]
                    )
                    for iteration_type in sorted(
                        {cycle.iteration_type for cycle in cycles}
                    )
                },
            }
            for device_id, cycles in sorted(cycles_by_device.items())
        },
    }


def compute_profile_gpu_kernel_ratio(profile_dir: Path) -> dict[str, Any]:
    """Resolve the normalized JSON and SQLite owned by a profile artifact root."""

    profile_dir = profile_dir.resolve()
    profile_result_path = profile_dir / "profile_result.json"
    profile_result = json.loads(profile_result_path.read_text())
    result = compute_gpu_kernel_ratio(
        Path(profile_result["parsed_nsys"]), Path(profile_result["sqlite"])
    )
    result["sources"]["profile_dir"] = str(profile_dir)
    result["sources"]["profile_result"] = str(profile_result_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure pooled kernel/GPU utilization and derive gpu_time_multiplier "
            "from one completed alignment profile"
        )
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        required=True,
        help="completed alignment profile artifact root containing profile_result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON destination for the measured ratio and audit totals",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compute_profile_gpu_kernel_ratio(args.profile_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    overall = result["overall"]
    print(
        f"kernel/GPU={overall['kernel_gpu_fraction']:.6f}; "
        f"gpu_time_multiplier={overall['gpu_time_multiplier']:.6f}; "
        f"cycles={overall['cycles']}; output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
