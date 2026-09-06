"""Inspect startup NVTX phases, CUDA API calls and kernel residency per rank."""

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


def union_ms(intervals):
    total, end = 0, None
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total / 1e6


def inspect(profile, count):
    result = json.loads((profile / "profile_result.json").read_text())
    host = json.loads(Path(result["host_timeline"]).read_text())
    strings, threads = host["strings"], host["threads"]
    phases = defaultdict(list)
    for thread, start, end, name_id in host["nvtx_ranges"]:
        match = re.fullmatch(r"vllm_iteration\((\d+)\): (.+)", strings[name_id])
        device = threads[thread]["device_id"]
        if match and device is not None and int(match[1]) < count:
            phases[int(match[1]), device].append((start, end, match[2]))
    if not phases:
        raise ValueError("capture has no requested startup iteration ranges")
    low = min(a for values in phases.values() for a, _, _ in values)
    high = max(b for values in phases.values() for _, b, _ in values)
    with sqlite3.connect(Path(result["sqlite"]).resolve().as_uri() + "?mode=ro", uri=True) as db:
        kernels = db.execute(
            'SELECT start, end, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL '
            'WHERE start < ? AND end > ? ORDER BY start', (high, low),
        ).fetchall()
    records = []
    for (iteration, device), ranges in sorted(phases.items()):
        start, end = min(r[0] for r in ranges), max(r[1] for r in ranges)
        api = defaultdict(list)
        for thread, a, b, name_id, _ in host["api_calls"]:
            if threads[thread]["device_id"] == device and a < end and b > start:
                api[strings[name_id]].append((max(a, start), min(b, end)))
        phase_times = defaultdict(float)
        for a, b, phase in ranges:
            phase_times[phase] += (b - a) / 1e6
        # Draft ranges use an unindexed name; associate only within this rank's window.
        for thread, a, b, name_id in host["nvtx_ranges"]:
            if (threads[thread]["device_id"] == device
                    and strings[name_id] == "gpu_model_runner: draft"
                    and start <= a and b <= end):
                phase_times["draft"] += (b - a) / 1e6
        residency = union_ms((max(a, start), min(b, end))
                             for a, b, gpu in kernels if gpu == device and a < end and b > start)
        records.append({
            "iteration": iteration, "device": device, "worker_span_ms": (end - start) / 1e6,
            "kernel_residency_union_ms": residency, "phases_ms": dict(phase_times),
            "module_load_union_ms": union_ms(
                interval for name, intervals in api.items() if "ModuleLoad" in name
                for interval in intervals),
            "graph_launch_count": sum(len(v) for k, v in api.items() if "GraphLaunch" in k),
            "forward_graph_launch_count": sum(
                1 for name, intervals in api.items() if "GraphLaunch" in name
                for a, b in intervals for x, y, phase in ranges
                if phase == "forward" and x <= a and b <= y),
            "largest_cuda_api": sorted([
                {"name": name, "count": len(v), "union_ms": union_ms(v)}
                for name, v in api.items()
            ], key=lambda row: -row["union_ms"])[:8],
        })
    return {
        "profile": str(profile.resolve()), "rows": records,
        "limitations": "Profiled diagnostic, not unprofiled E2E. GPU residency includes "
                       "collective waiting; API durations can overlap GPU work and phases. "
                       "These columns must not be added or treated as exclusive causal costs.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.profile, args.iterations)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["rows"]:
        print(row["iteration"], row["device"], "span", round(row["worker_span_ms"], 2),
              "module", round(row["module_load_union_ms"], 2), "graphs", row["graph_launch_count"],
              "phases", {k: round(v, 2) for k, v in row["phases_ms"].items()})
