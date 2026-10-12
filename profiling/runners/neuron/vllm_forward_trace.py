"""Validated NRT whole-forward timing; rejects partial rank/core coverage."""

from __future__ import annotations

import collections
import json
import re
import subprocess

from profiling.runners.neuron.stock_contract import BLOCK_SIZE, KV_BLOCKS

# The per-rank paged KV pool operand: pages, the rank's two KV heads, page
# tokens and head width.
_KV_POOL_SHAPE = f"{KV_BLOCKS}, 2, {BLOCK_SIZE}, 128"


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
            if (int(shape[1]), int(shape[2])) != (
                int(token_match[1]),
                expected_context // BLOCK_SIZE,
            ):
                raise ValueError("compiled decode block-table width differs from declared context")
            if not re.search(rf"Shape: \({_KV_POOL_SHAPE}\)\s+Dtype: bfloat16", metadata):
                raise ValueError("compiled decode KV pool or dtype differs from declared layout")
    elif "Dtype: bfloat16" in slot[1] and _KV_POOL_SHAPE in slot[1]:
        phase = "prefill"
    else:
        raise ValueError("full-model cache/block-table layout changed")
    return phase, int(token_match[1])


def compiled_inputs(metadata):
    """``example_inputs.txt`` as {index: (shape, dtype)}."""
    inputs = {}
    for index, shape, dtype in re.findall(
        r"Input (\d+):\s+Shape: \(([^)]*)\)\s+Dtype: (\w+)", metadata
    ):
        inputs[int(index)] = (tuple(int(n) for n in shape.replace(",", " ").split()), dtype)
    return inputs


KV_POOL = ((KV_BLOCKS, 2, BLOCK_SIZE, 128), "bfloat16")


def segmented_model_geometry(metadata, context, segment_size):
    """Geometry of a stock segmented-prefill engine's graphs.

    The prefill graph runs one sequence's S-token segment against its cached
    history: S token ids, a (1, 1) cached-length operand (input 5), a
    (1, context/32) block table and the 64 per-layer K/V pools. Decode graphs are
    the stock ones (``model_geometry``); an unsegmented prefill graph is refused.
    """
    inputs = compiled_inputs(metadata)
    if inputs.get(5) == ((1, 1), "int32"):
        if inputs.get(0) != ((segment_size,), "int64"):
            raise ValueError("compiled prefill segment differs from declared segment size")
        if ((1, context // BLOCK_SIZE), "int32") not in inputs.values():
            raise ValueError("compiled segment block-table width differs from declared context")
        if sum(value == KV_POOL for value in inputs.values()) != 64:
            raise ValueError("compiled segment KV pool or dtype differs from declared layout")
        return "prefill", segment_size
    phase, bucket = model_geometry(metadata, context)
    if phase != "decode":
        raise ValueError("segmented engine compiled an unsegmented prefill graph")
    return phase, bucket


def plan_geometry(metadata, plan):
    """The (phase, bucket) of a compiled graph under ``plan``'s engine."""
    if "segment_size" in plan:
        return segmented_model_geometry(metadata, plan["context"], plan["segment_size"])
    return model_geometry(metadata, plan["context"])


def segment_count(context, segment_size):
    """Prefill executions of one (context-8)-token canonical prompt."""
    return -(-(context - 8) // segment_size)


def group_executions(events, requests, physical_cores):
    """Join rank intervals into whole-chip executions inside one public request.

    One execution is the eight physical-core intervals (four ranks, two cores
    each) that share an NRT exec_id and model_id and run one compiled graph.
    """
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
    executions = []
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
        executions.append(
            {
                "request_index": request_index,
                "exec_id": exec_id,
                "model_id": model_id,
                "model_hash": hashes.pop(),
                "spans": sorted(
                    (event["timestamp"], event["timestamp"] + event["duration"]) for event in group
                ),
            }
        )
    return executions


# Stock async scheduling may plan one decode of the first prompt before the next
# prefill is admitted. That step runs at a smaller bucket and is never a timing
# sample for the requested bucket (user-approved rule, 2026-10-12).
EARLY_DECODES_ALLOWED = 1


def check_forward_coverage(records, requests):
    """Every public request must show its prompt forwards and seven output decodes,
    all but one of them at the requested compiled bucket."""
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
            < 7 - EARLY_DECODES_ALLOWED
        ):
            raise ValueError(
                "scheduler did not execute the requested compiled decode bucket six times"
            )


def measure_trace(events, requests, model_info, physical_cores):
    records, measured = [], collections.defaultdict(list)
    for execution in group_executions(events, requests, physical_cores):
        phase, bucket = model_info[execution["model_hash"]]
        duration = union_duration(execution["spans"]) / 1e6
        measured[(phase, bucket)].append(duration)
        records.append(
            {
                "request_index": execution["request_index"],
                "exec_id": execution["exec_id"],
                "model_id": execution["model_id"],
                "model_hash": execution["model_hash"],
                "phase": phase,
                "token_bucket": bucket,
                "time_ms": duration,
            }
        )
    check_forward_coverage(records, requests)
    return dict(measured), records


def measure_segmented_trace(events, requests, model_info, physical_cores, segments):
    """Time a segmented-prefill run; keys are (phase, bucket, prior_tokens).

    Every call must show ``segments`` prefill executions per request. Prefill
    timing comes only from batch-1 ``segment_timing`` calls: the lone request is
    the only sequence the scheduler can prefill, so its k-th prefill execution
    in device order is segment k (prior = k x S), and all of them precede its
    first decode. Decode keys use prior 0. The unchanged whole-forward coverage
    check then sees one prefill per request (the last segment, which samples).
    """
    records, measured, coverage = [], collections.defaultdict(list), []
    by_call = collections.defaultdict(list)
    for execution in group_executions(events, requests, physical_cores):
        phase, bucket = model_info[execution["model_hash"]]
        record = {
            "request_index": execution["request_index"],
            "exec_id": execution["exec_id"],
            "model_id": execution["model_id"],
            "model_hash": execution["model_hash"],
            "phase": phase,
            "token_bucket": bucket,
            "prior_tokens": 0 if phase == "decode" else None,
            "start_ns": execution["spans"][0][0],
            "time_ms": union_duration(execution["spans"]) / 1e6,
        }
        records.append(record)
        by_call[execution["request_index"]].append(record)
    for index, request in enumerate(requests):
        ordered = sorted(by_call[index], key=lambda record: record["start_ns"])
        prefills = [record for record in ordered if record["phase"] == "prefill"]
        if len(prefills) != request["batch"] * segments:
            raise ValueError("native trace is missing a prefill segment")
        if request.get("segment_timing"):
            if request["batch"] != 1:
                raise ValueError("segment timing requires one request per call")
            if any(record["phase"] == "decode" for record in ordered[:segments]):
                raise ValueError("a decode ran before the lone request finished its prefill")
            for ordinal, record in enumerate(prefills):
                record["prior_tokens"] = ordinal * record["token_bucket"]
        coverage += [record for record in ordered if record["phase"] == "decode"]
        coverage += prefills[segments - 1 :: segments]
    for record in records:
        if record["prior_tokens"] is not None:
            measured[(record["phase"], record["token_bucket"], record["prior_tokens"])].append(
                record["time_ms"]
            )
    check_forward_coverage(coverage, requests)
    return dict(measured), records


def export_events(root):
    """Export the NRT system trace under ``root/profiles``; return device executions."""
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
    return events


def reserved_physical_cores():
    """Absolute physical cores of the reserved LNC2 span, never the captured event set."""
    import os

    logical = os.environ["NEURON_VISIBLE_DEVICES"]
    start, end = (int(value) for value in logical.split("-"))
    if end - start != 3:
        raise ValueError("whole forward requires four reserved logical LNC2 cores")
    return set(range(start * 2, (end + 1) * 2))


def export_and_measure(root, compile_cache):
    plan = json.loads((root / "plan.json").read_text())
    events = export_events(root)
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
        geometries = {plan_geometry(path.read_text(), plan) for path in files}
        if len(geometries) != 1:
            raise ValueError("compiled graph rank geometries disagree")
        models[model_hash] = geometries.pop()
    requests = json.loads((root / "profiled-outputs.json").read_text())
    if "segment_size" in plan:
        segments = segment_count(plan["context"], plan["segment_size"])
        measured, records = measure_segmented_trace(
            events, requests, models, reserved_physical_cores(), segments
        )
    else:
        measured, records = measure_trace(events, requests, models, reserved_physical_cores())
    (root / "execution-records.json").write_text(json.dumps(records, indent=2))
    (root / "timing.json").write_text(
        json.dumps({":".join(map(str, key)): values for key, values in measured.items()}, indent=2)
    )
    return measured
