"""Validated NRT whole-forward timing; rejects partial rank/core coverage."""

from __future__ import annotations

import collections
import json
import re
import subprocess


def union_duration(spans):
    merged = []
    for start, stop in sorted(spans):
        if stop <= start:
            raise ValueError("native execution interval must be positive")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return sum(stop - start for start, stop in merged)


def model_geometry(metadata, expected_context=None):
    token_match = re.search(r"Input 0:\s+Shape: \((\d+),\)", metadata)
    if token_match is None:
        raise ValueError("unrecognized full-model token input")
    slot = re.search(r"Input 5:(.*?)(?:Input 6:)", metadata, re.S)
    if slot is None:
        raise ValueError("missing full-model cache/block-table input")
    if "Dtype: int32" in slot[1] and re.search(r"Shape: \(\d+, \d+\)", slot[1]):
        phase = "decode"
        if expected_context is not None:
            shape = re.search(r"Shape: \((\d+), (\d+)\)", slot[1])
            if (int(shape[1]), int(shape[2])) != (int(token_match[1]), expected_context // 32):
                raise ValueError("compiled decode block-table width differs from declared context")
            if not re.search(r"Shape: \(6782, 2, 32, 128\)\s+Dtype: bfloat16", metadata):
                raise ValueError("compiled decode KV pool or dtype differs from declared layout")
    elif "Dtype: bfloat16" in slot[1] and "6782, 2, 32, 128" in slot[1]:
        phase = "prefill"
    else:
        raise ValueError("full-model cache/block-table layout changed")
    return phase, int(token_match[1])


def measure_trace(events, requests, model_info, physical_cores):
    grouped = collections.defaultdict(list)
    for event in events:
        matches = [
            i
            for i, request in enumerate(requests)
            if request["start_epoch_ns"] <= event["timestamp"]
            and event["timestamp"] + event["duration"] <= request["stop_epoch_ns"]
        ]
        if len(matches) != 1:
            raise ValueError("native interval has no unique public-request bracket")
        grouped[(matches[0], event["exec_id"], event["model_id"])].append(event)
    records, measured = [], collections.defaultdict(list)
    for (request_index, exec_id, model_id), group in grouped.items():
        counts = collections.Counter(event["process_id"] for event in group)
        if (
            len(group) != 8
            or {event["device_core_idx"] for event in group} != physical_cores
            or len(counts) != 4
            or set(counts.values()) != {2}
        ):
            raise ValueError(
                "native forward is missing a rank/physical core or contains duplicates"
            )
        hashes = {re.search(r"/compile_cache/([^/]+)/", event["model_name"])[1] for event in group}
        if len(hashes) != 1:
            raise ValueError("ranks executed different compiled graph identities")
        model_hash = hashes.pop()
        phase, bucket = model_info[model_hash]
        duration = (
            union_duration(
                [(event["timestamp"], event["timestamp"] + event["duration"]) for event in group]
            )
            / 1e6
        )
        measured[(phase, bucket)].append(duration)
        records.append(
            {
                "request_index": request_index,
                "exec_id": exec_id,
                "model_id": model_id,
                "model_hash": model_hash,
                "phase": phase,
                "token_bucket": bucket,
                "time_ms": duration,
            }
        )
    for index, request in enumerate(requests):
        selected = [record for record in records if record["request_index"] == index]
        if sum(record["phase"] == "prefill" for record in selected) != request["batch"]:
            raise ValueError("native trace is missing a prompt forward")
        if sum(record["phase"] == "decode" for record in selected) < 7:
            raise ValueError("native trace is missing output decode forwards")
        if (
            sum(
                record["phase"] == "decode" and record["token_bucket"] == request["bucket"]
                for record in selected
            )
            < 7
        ):
            raise ValueError(
                "scheduler did not execute the requested compiled decode bucket seven times"
            )
    return dict(measured), records


def export_and_measure(root, compile_cache):
    plan = json.loads((root / "plan.json").read_text())
    output = root / "system-trace.json"
    filters = (
        "nc_exec_running,nrt_profile_add_node_info,nrt_model_submit,nrta_execute_schedule,"
        "kbl_exec_pre,kbl_exec_post,kbl_exec_wait,nc_model_switch,cc_exec_barrier"
    )
    command = [
        "/opt/aws/neuron/bin/neuron-explorer",
        "view",
        "-d",
        str(root / "profiles"),
        "--output-format",
        "json",
        "--ignore-device-profile",
        "--system-trace-filter-event-type",
        filters,
        "--output-file",
        str(output),
        "--disable-ui",
        "--force",
    ]
    with (root / "export.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    events = [
        event
        for event in json.loads(output.read_text())["trace_event"]
        if event["name"] == "nc_exec_running"
    ]
    if not events:
        raise ValueError("NRT trace has no device executions")
    accuracy_log = (root / "accuracy.log").read_text()
    models = {}
    for event in events:
        match = re.search(r"/compile_cache/([^/]+)/", event["model_name"])
        if match is None:
            raise ValueError("native model identity has no compiler cache key")
        model_hash = match[1]
        if model_hash in models:
            continue
        if model_hash not in accuracy_log:
            raise ValueError(
                "production compiled graph was not observed during numerical validation"
            )
        files = list((compile_cache / model_hash).rglob("example_inputs.txt"))
        if not files:
            raise ValueError("cannot recover compiled input geometry")
        geometries = {model_geometry(path.read_text(), plan["context"]) for path in files}
        if len(geometries) != 1:
            raise ValueError("compiled graph rank geometries disagree")
        models[model_hash] = geometries.pop()
    requests = json.loads((root / "profiled-outputs.json").read_text())
    # The worker reservation controls visibility; infer absolute physical IDs
    # from the reserved LNC2 logical span, never from the captured event set.
    import os

    logical = os.environ["NEURON_VISIBLE_DEVICES"]
    start, end = (int(value) for value in logical.split("-"))
    if end - start != 3:
        raise ValueError("whole forward requires four reserved logical LNC2 cores")
    measured, records = measure_trace(
        events, requests, models, set(range(start * 2, (end + 1) * 2))
    )
    (root / "execution-records.json").write_text(json.dumps(records, indent=2))
    (root / "timing.json").write_text(
        json.dumps(
            {f"{phase}:{bucket}": values for (phase, bucket), values in measured.items()}, indent=2
        )
    )
    return measured
