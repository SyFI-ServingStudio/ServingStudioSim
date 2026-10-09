"""Join stock async submissions to NRT execution IDs, then union all TP4 cores.

Explorer's system JSON uses complete epoch-ns intervals (not the NxDI paired
event schema). Submission belongs to a worker call; execution may finish later.
Ordered submit/pre joins require identical model sequences and exact counts.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from alignment.neuron.normalize import union_intervals
from alignment.nsys.sequence import build_device_kernel_sequences


def _model_hash(name: str) -> str:
    matched = re.search(r"/compile_cache/([0-9a-f]+)/", name)
    if matched is None:
        raise ValueError("native model is not a stock compiled graph")
    return matched[1]


def verify_no_event_drops(trace: dict, logs: dict[str, str]) -> dict:
    """NRT reports ring overflow in both runtime warnings and exported trace messages.

    Exact submit/exec counts are a separate gate: absence of a warning alone cannot
    establish complete iteration coverage.
    """
    warning = re.compile(r"events? were dropped|dropped due to full ring|incomplete trace", re.I)
    sources = {**logs, "exported_trace": json.dumps(trace)}
    for name, value in sources.items():
        if warning.search(value):
            raise ValueError(f"native event drops reported in {name}; recapture required")
    return {"passed": True, "checked_sources": sorted(sources), "overflow_warning_found": False}


def normalize(trace: dict, records: dict, model_info: dict[str, tuple[str, int]]) -> dict:
    if records.get("engine") != "vllm_neuron" or records.get("tp_size") != 4:
        raise ValueError("stock native records must declare vllm_neuron TP4")
    events = trace.get("trace_event")
    if not isinstance(events, list):
        raise ValueError("expected Explorer complete-event system trace")
    metrics = sorted(records["iterations"], key=lambda r: r["iteration_index"])
    fingerprints = [r["fingerprint"] for r in metrics]
    if not metrics or len(set(fingerprints)) != len(metrics):
        raise ValueError("iteration fingerprints must be nonempty and unique")
    if len({m["iteration_index"] for m in metrics}) != len(metrics):
        raise ValueError("duplicate scheduler iteration")
    by_rank = defaultdict(list)
    for row in records["forwards"]:
        by_rank[row["rank"]].append(row)
    if set(by_rank) != set(range(4)):
        raise ValueError("missing TP4 worker observations")
    observed_pids = set()
    native_by_fingerprint = defaultdict(list)
    physical_by_rank = {}
    for rank, rows in by_rank.items():
        rows.sort(key=lambda r: r["worker_sequence"])
        pids = {str(r["pid"]) for r in rows}
        if len(pids) != 1 or pids & observed_pids:
            raise ValueError("worker PID/rank identity is ambiguous")
        observed_pids |= pids
        pid = pids.pop()
        if [r["fingerprint"] for r in rows] != fingerprints:
            raise ValueError("worker and scheduler iteration sequences disagree")
        if any(b["worker_sequence"] != a["worker_sequence"] + 1 for a, b in zip(rows, rows[1:])):
            raise ValueError("missing or duplicate worker sequence")
        if any(b["start_epoch_ns"] < a["stop_epoch_ns"] for a, b in zip(rows, rows[1:])):
            raise ValueError("worker execute calls overlap; submit order cannot be proven")
        native = [
            e
            for e in events
            if str(e.get("process_id")) == pid
            and e.get("name") in {"nrt_model_submit", "kbl_exec_pre", "nc_exec_running"}
        ]
        for e in native:
            if (
                e.get("timestamp_unit") != "ns"
                or type(e.get("timestamp")) is not int
                or type(e.get("duration")) is not int
                or e["duration"] <= 0
            ):
                raise ValueError("native events require positive integer epoch-ns intervals")
        submits = sorted(
            (e for e in native if e["name"] == "nrt_model_submit"), key=lambda e: e["timestamp"]
        )
        pres = sorted(
            (e for e in native if e["name"] == "kbl_exec_pre"), key=lambda e: e["timestamp"]
        )
        if len(submits) != len(rows) or len(pres) != len(rows):
            raise ValueError("native submit/pre counts differ from observed forwards")
        execs = defaultdict(list)
        for e in native:
            if e["name"] == "nc_exec_running":
                execs[(e["exec_id"], e["model_name"])].append(e)
        if len(execs) != len(rows):
            raise ValueError("native execution count differs from worker observations")
        used = set()
        for row, metric, submit, pre in zip(rows, metrics, submits, pres, strict=True):
            if not (
                row["start_epoch_ns"] <= submit["timestamp"]
                and submit["timestamp"] + submit["duration"] <= row["stop_epoch_ns"]
            ):
                raise ValueError("ordered native submit is outside its worker call")
            if submit["model_name"] != pre["model_name"]:
                raise ValueError("ordered submit/pre model sequences disagree")
            if pre["timestamp"] < submit["timestamp"]:
                raise ValueError("native execution precedes its ordered submission")
            key = (pre["exec_id"], pre["model_name"])
            if key in used or key not in execs:
                raise ValueError("missing or duplicate native exec identity")
            used.add(key)
            group = execs[key]
            cores = {e["device_core_idx"] for e in group}
            if len(group) != 2 or len(cores) != 2:
                raise ValueError("LNC2 forward must cover exactly two physical cores")
            if rank in physical_by_rank and physical_by_rank[rank] != cores:
                raise ValueError("worker physical core assignment changed")
            physical_by_rank[rank] = cores
            model_hash = _model_hash(pre["model_name"])
            expected = (metric["phase"], metric["compiled_shapes"]["token_bucket"])
            if model_info.get(model_hash) != expected:
                raise ValueError("native model metadata disagrees with scheduler compiled geometry")
            bucket = expected[1]
            if row["input_ids_shape"] != [bucket]:
                raise ValueError("worker actual token input shape disagrees with compiled bucket")
            table = [1 if expected[0] == "prefill" else bucket, 16]
            if row["block_table_shapes"] != [table]:
                raise ValueError("worker block table geometry differs from C512")
            native_by_fingerprint[row["fingerprint"]].append(
                {
                    "rank": rank,
                    "pid": pid,
                    "exec_id": key[0],
                    "model_hash": model_hash,
                    "submit": submit,
                    "pre": pre,
                    "cores": group,
                }
            )
    physical = set().union(*physical_by_rank.values())
    if len(physical) != 8:
        raise ValueError("one Trainium2 TP4 chip must cover eight disjoint physical cores")
    unexpected = {
        str(e.get("process_id")) for e in events if e.get("name") == "nc_exec_running"
    } - observed_pids
    if unexpected:
        raise ValueError("native trace includes an unobserved worker process")
    details, names = [], {}
    for metric in metrics:
        group = native_by_fingerprint[metric["fingerprint"]]
        if len(group) != 4 or len({r["model_hash"] for r in group}) != 1:
            raise ValueError("TP ranks did not execute one shared whole-forward graph")
        intervals = union_intervals(
            [(e["timestamp"], e["timestamp"] + e["duration"]) for r in group for e in r["cores"]]
        )
        name = (
            f"Stock vLLM Neuron Llama3.1-8B TP4 {metric['phase']} "
            f"B{metric['compiled_shapes']['token_bucket']} C512"
        )
        if name not in names.values():
            names[len(names)] = name
        name_id = next(i for i, value in names.items() if value == name)
        kernels = [
            {
                "ordinal": i + 1,
                "name_id": name_id,
                "category": "compiled_graph",
                "start_ns": a,
                "end_ns": b,
                "stream_id": 0,
                "correlation_id": metric["iteration_index"],
                "track_index": 0,
            }
            for i, (a, b) in enumerate(intervals)
        ]
        details.append(
            {
                "iteration": metric["iteration_index"],
                "iteration_type": "mixed" if metric["phase"] == "prefill" else "decode",
                "stage": metric["phase"],
                "metrics": metric,
                "metrics_by_dp_rank": {"0": metric},
                "ranges": [
                    {
                        "device_id": 0,
                        "dp_rank": 0,
                        "phase": "forward",
                        "start_ns": intervals[0][0],
                        "end_ns": intervals[-1][1],
                        "kernel_count": len(kernels),
                        "kernels": kernels,
                        "native_rank_executions": group,
                    }
                ],
            }
        )
    sequences, devices = build_device_kernel_sequences(details, names)
    return {
        "schema_version": 5,
        "producer_kind": "framework_capture",
        "engine": "vllm_neuron",
        "trace_provider": "neuron_system_trace",
        "measurement_granularity": "whole_compiled_forward",
        "timestamp_clock": "Explorer synchronized epoch-ns",
        "device_ids": devices,
        "dp_rank_by_device": {"0": 0},
        "physical_core_ids": sorted(physical),
        "logical_nc_config": 2,
        "tp_size": 4,
        "phases": ["forward"],
        "iterations": [m["iteration_index"] for m in metrics],
        "kernel_names": names,
        "iteration_details": details,
        "kernel_sequences": sequences,
        "scanned_kernel_rows": sum(len(d["ranges"][0]["kernels"]) for d in details),
    }


def write_normalized(log_dir: Path, records: dict, model_info: dict) -> dict:
    from alignment.nsys.parsed_io import kernel_rows_path, write_parsed

    parsed = normalize(json.loads((log_dir / "system-trace.json").read_text()), records, model_info)
    path = log_dir / "parsed.json"
    write_parsed(path, parsed)
    inventory = log_dir / "kernel_sequences.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "encoding": "folded-v2",
                "source_parsed": str(path),
                "device_ids": parsed["device_ids"],
                "folding_policy": {
                    "kind": "exact_contiguous_repeat",
                    "match_fields": ["name", "suggested_category"],
                    "row_identity": "sequence_id:expanded_ordinal",
                    "rank_policy": "TP4 physical execution intervals unioned once on chip lane0",
                    "track_policy": (
                        "one native whole-chip execution track; disjoint busy intervals retained"
                    ),
                },
                "phases": parsed["kernel_sequences"],
            }
        )
        + "\n"
    )
    return {
        "parsed_trace": str(path),
        "parsed_kernel_rows": str(kernel_rows_path(path)),
        "kernel_sequences": str(inventory),
    }
