"""Native timing for synchronous, rank-local libtorch-neuronx-lite calls.

The stock 2.11.0.1.0.1284 runtime exports complete epoch-ns intervals through
Explorer. Each call must contain exactly one execution on both PNCs of an LNC2
unit. Host brackets identify invocations; only the native interval union counts.
"""

from __future__ import annotations

import collections
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

_DROP_WARNING = re.compile(
    r"events? were dropped|dropped due to full ring|incomplete trace|"
    r"(?:events?|notifications?|trace|profil(?:e|ing)|ring(?: buffer)?)[^\n]{0,120}"
    r"(?:dropped|lost|overflow|truncated)|"
    r"(?:dropped|lost|overflow|truncated)[^\n]{0,120}"
    r"(?:events?|notifications?|trace|profil(?:e|ing)|ring(?: buffer)?)",
    re.IGNORECASE,
)


def verify_no_event_drops(*sources: str) -> None:
    if any(_DROP_WARNING.search(source) for source in sources):
        raise ValueError("native trace reported dropped events; timing is rejected")


def execution_unions(trace: dict, brackets: list[dict]) -> list[dict]:
    """Require complete one-to-one synchronous calls, preserving device gaps."""
    verify_no_event_drops(json.dumps(trace))
    if not brackets or [row["iteration"] for row in brackets] != list(range(len(brackets))):
        raise ValueError("native timing requires ordered invocation brackets")
    for index, row in enumerate(brackets):
        if row["stop_epoch_ns"] <= row["start_epoch_ns"] or (
            index and row["start_epoch_ns"] < brackets[index - 1]["stop_epoch_ns"]
        ):
            raise ValueError("invocation brackets overlap or have nonpositive duration")
    groups = collections.defaultdict(list)
    for event in trace["trace_event"]:
        if event["name"] != "nc_exec_running":
            continue
        if (
            event.get("timestamp_unit") != "ns"
            or any(type(event[name]) is not int for name in ("timestamp", "duration", "exec_id"))
            or event["duration"] <= 0
        ):
            raise ValueError("unsupported native execution timestamp schema")
        matches = [
            row["iteration"]
            for row in brackets
            if row["start_epoch_ns"] <= event["timestamp"]
            and event["timestamp"] + event["duration"] <= row["stop_epoch_ns"]
        ]
        if len(matches) != 1:
            raise ValueError("native execution lacks one unique invocation bracket")
        groups[matches[0]].append(event)
    if set(groups) != set(range(len(brackets))):
        raise ValueError("native trace is missing an invocation")
    records, identities, executions = [], set(), set()
    for iteration in range(len(brackets)):
        rows = groups[iteration]
        cores = {row["device_core_idx"] for row in rows}
        identity = (
            frozenset(cores),
            frozenset(row["process_id"] for row in rows),
            frozenset(row["lnc_idx"] for row in rows),
            frozenset(row["nc_idx"] for row in rows),
            frozenset(row.get("model_name", "") for row in rows),
        )
        if len(rows) != 2 or len(cores) != 2 or any(len(part) != 1 for part in identity[1:]):
            raise ValueError("native invocation is not one complete two-core LNC2 execution")
        if min(cores) % 2 or max(cores) != min(cores) + 1:
            raise ValueError("native cores do not form a physical LNC2 pair")
        exec_ids = {row["exec_id"] for row in rows}
        if len(exec_ids) != 1 or next(iter(exec_ids)) in executions:
            raise ValueError("native execution IDs are duplicated or disagree across cores")
        executions.update(exec_ids)
        identities.add(identity)
        (start, stop), (other_start, other_stop) = sorted(
            (row["timestamp"], row["timestamp"] + row["duration"]) for row in rows
        )
        duration = (
            stop - start + other_stop - other_start - max(0, min(stop, other_stop) - other_start)
        )
        records.append(
            {
                "iteration": iteration,
                "exec_id": next(iter(exec_ids)),
                "physical_cores": sorted(cores),
                "lnc_idx": rows[0]["lnc_idx"],
                "process_id": rows[0]["process_id"],
                "time_ms": duration / 1e6,
            }
        )
    if len(identities) != 1:
        raise ValueError("native timing changed physical cores, process, LNC or graph identity")
    return records


def measure_lite(invoke, runtime, root: Path, compile_cache: Path, *, warmup=5, iterations=20):
    """`invoke` returns a synchronized CPU output; transfers are never timed."""
    for _ in range(warmup):
        invoke()
    runtime.start_profiling(
        str(root / "profiles"), ["system_profile"], None, 4_000_000, str(compile_cache)
    )
    brackets = []
    try:
        for iteration in range(iterations):
            start = time.time_ns()
            invoke()
            brackets.append(
                {
                    "iteration": iteration,
                    "start_epoch_ns": start,
                    "stop_epoch_ns": time.time_ns(),
                }
            )
    finally:
        runtime.stop_profiling()
        (root / "invocations.json").write_text(json.dumps(brackets, indent=2))
    command = [
        "/opt/aws/neuron/bin/neuron-explorer",
        "view",
        "-d",
        str(root / "profiles"),
        "--output-format",
        "json",
        "--ignore-device-profile",
        "--system-trace-filter-event-type",
        "nc_exec_running,nrt_profile_add_node_info",
        "--output-file",
        str(root / "system-trace.json"),
        "--disable-ui",
        "--force",
    ]
    (root / "export-command.json").write_text(json.dumps(command, indent=2))
    with (root / "export.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    trace = json.loads((root / "system-trace.json").read_text())
    verify_no_event_drops((root / "export.log").read_text())
    records = execution_unions(trace, brackets)
    return statistics.median(row["time_ms"] for row in records), records, trace
