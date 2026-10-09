"""Normalize genuine NRT invocations without inventing compiler subkernels.

One LNC2 invocation cooperates across two physical cores. Preserve the raw
trace, then expose their synchronized busy UNION as one logical compiled-graph
execution. A disjoint union retains separate busy segments; gaps stay idle.
Host epoch brackets associate native calls with recorded public NxDI forwards.
"""

import json
from collections import defaultdict
from pathlib import Path

from alignment.nsys.sequence import build_device_kernel_sequences


def union_intervals(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(spans):
        if stop <= start:
            raise ValueError("nonpositive native execution interval")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def iteration_metric(row: dict) -> dict:
    prefill = row["phase"] == "prefill"
    q, kv = row["q_tokens"], row["kv_len_before"]
    if (
        type(q) is not int
        or type(kv) is not int
        or (prefill and not (1 <= q <= 128 and kv == 0))
        or (not prefill and not (row["phase"] == "decode" and q == 1 and 1 <= kv < 512))
    ):
        raise ValueError("invalid NxDI logical forward geometry")
    if (
        type(row["start_monotonic_ns"]) is not int
        or type(row["stop_monotonic_ns"]) is not int
        or row["stop_monotonic_ns"] <= row["start_monotonic_ns"]
    ):
        raise ValueError("invalid NxDI host monotonic observation")
    return {
        "schema_version": 2,
        "input_adapter": "nxdi_text",
        "dp_rank": 0,
        "iteration_index": row["iteration_id"],
        "prefill_tokens": q if prefill else 0,
        "decode_requests": 0 if prefill else 1,
        "decode_tokens_scheduled": 0 if prefill else 1,
        "prefill_chunk_pairs": [[0, q]] if prefill else [],
        # Predictor K includes the current input. The recorded position counts
        # only the resident prefix before that input updates the aliased cache.
        "decode_kv_lens": [] if prefill else [kv + q],
        "observed_start_monotonic_ns": row["start_monotonic_ns"],
        "observed_end_monotonic_ns": row["stop_monotonic_ns"],
        "observed_elapsed_ms": (row["stop_monotonic_ns"] - row["start_monotonic_ns"]) / 1e6,
        "compiled_shapes": {
            "context_bucket": 128,
            "kv_bucket": 512,
            "query_tokens": 128 if prefill else 1,
        },
        "request_id": row["request_id"],
    }


def normalize(trace: dict, records: dict, *, request_ids: set[str] | None = None) -> dict:
    if trace.get("data_version") != 2 or not isinstance(trace.get("events"), list):
        raise ValueError("unsupported NRT trace schema; expected data_version 2")
    if records.get("engine") != "nxdi" or records.get("producer_kind") != "framework_capture":
        raise ValueError("native records require explicit NxDI framework_capture provenance")
    config = records.get("config", {})
    neuron = config.get("neuron_config", {})
    model_shape = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "model_type": "llama",
    }
    if (
        any(config.get(key) != value for key, value in model_shape.items())
        or neuron.get("layer_boundary_markers") is not False
        or neuron.get("context_encoding_buckets") != [128]
        or neuron.get("token_generation_buckets") != [512]
        or neuron.get("logical_nc_config") != 2
        or neuron.get("tp_degree") != 1
        or neuron.get("batch_size") != 1
        or neuron.get("seq_len") != 512
        or neuron.get("torch_dtype") != "bfloat16"
    ):
        raise ValueError(
            "native capture must declare the full32-layer TP1/LNC2 no-marker CTE128/TKG512 config"
        )
    starts, calls, cores, execution_names = {}, [], defaultdict(list), {}
    for event in trace["events"]:
        kind = event.get("event_type")
        if kind not in {"nc_exec_running", "nrt_execute", "kbl_exec_pre"}:
            continue
        key = (kind, event["nc_idx"], event["tracking_id"])
        stamp, data = event["timestamp_ns"], event["data"]
        if type(stamp) is not int:
            raise ValueError("native timestamp_ns must be an integer")
        if event["phase"] == "start":
            if key in starts:
                raise ValueError("duplicate NRT start identity")
            starts[key] = (stamp, data)
            if kind == "kbl_exec_pre":
                execution_names[(event["nc_idx"], data["exec_id"])] = (
                    data["model_id"],
                    data["model_name"],
                )
        elif event["phase"] == "stop":
            if key not in starts:
                raise ValueError("NRT stop has no matching start")
            begin, opening = starts.pop(key)
            if stamp <= begin:
                raise ValueError("nonpositive NRT event duration")
            if kind == "nrt_execute":
                calls.append((begin, stamp, opening["model_id"], opening["model_name"]))
            elif kind == "nc_exec_running":
                cores[(event["nc_idx"], opening["exec_id"])].append(
                    (begin, stamp, opening["device_core_idx"])
                )
        else:
            raise ValueError("unsupported paired NRT event phase")
    if starts:
        raise ValueError("incomplete native trace: unmatched event starts")
    if not calls or not cores:
        raise ValueError("native trace contains no complete device invocations")
    invocations = []
    physical = set()
    for identity, spans in cores.items():
        pair = {core for _, _, core in spans}
        if len(pair) != 2:
            raise ValueError("each LNC2 execution must include both physical cores")
        physical.add(tuple(sorted(pair)))
        if identity not in execution_names:
            raise ValueError("native execution lacks a model identity")
        model_id, name = execution_names[identity]
        intervals = union_intervals([(start, end) for start, end, _ in spans])
        matches = [
            call
            for call in calls
            if call[2] == model_id and call[0] <= intervals[0][0] and intervals[-1][1] <= call[1]
        ]
        if len(matches) != 1:
            raise ValueError("native execution has no unique enclosing nrt_execute")
        invocations.append((matches[0], identity, name, intervals))
    if len(physical) != 1:
        raise ValueError("capture mixes physical LNC2 units")
    details, names, observed = [], {}, set()
    selected = [
        r for r in records["iterations"] if request_ids is None or r["request_id"] in request_ids
    ]
    for row in selected:
        iteration = row["iteration_id"]
        if type(iteration) is not int or iteration < 0 or iteration in observed:
            raise ValueError("invalid or duplicate NxDI iteration identity")
        observed.add(iteration)
        metric = iteration_metric(row)
        matches = [
            entry
            for entry in invocations
            if row["start_realtime_ns"] <= entry[0][0] and entry[0][1] <= row["stop_realtime_ns"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"forward {iteration} does not own exactly one whole-model native execution"
            )
        _, identity, model_name, spans = matches[0]
        component = (
            "context_encoding_model" if row["phase"] == "prefill" else "token_generation_model"
        )
        if component not in Path(model_name).parts:
            raise ValueError("recorded forward phase disagrees with its native compiled model")
        stable_name = f"NxDI Llama3.1-8B whole forward {row['phase']} CTE128 TKG512"
        if stable_name not in names.values():
            names[len(names)] = stable_name
        name_id = next(key for key, value in names.items() if value == stable_name)
        kernels = [
            {
                "ordinal": i + 1,
                "name_id": name_id,
                "category": "compiled_graph",
                "start_ns": start,
                "end_ns": stop,
                "stream_id": 0,
                "correlation_id": identity[1],
                "track_index": 0,
            }
            for i, (start, stop) in enumerate(spans)
        ]
        details.append(
            {
                "iteration": iteration,
                "iteration_type": "mixed" if row["phase"] == "prefill" else "decode",
                "stage": row["phase"],
                "metrics": metric,
                "metrics_by_dp_rank": {"0": metric},
                "ranges": [
                    {
                        "device_id": 0,
                        "dp_rank": 0,
                        "phase": "forward",
                        "start_ns": row["start_realtime_ns"],
                        "end_ns": row["stop_realtime_ns"],
                        "kernel_count": len(kernels),
                        "kernels": kernels,
                        "native_model_name": model_name,
                        "native_exec_id": identity[1],
                    }
                ],
            }
        )
    if not details:
        raise ValueError("no measured native forwards selected")
    sequences, device_ids = build_device_kernel_sequences(details, names)
    return {
        "schema_version": 5,
        "producer_kind": "framework_capture",
        "engine": "nxdi",
        "trace_provider": "neuron_system_trace",
        "measurement_granularity": "whole_compiled_forward",
        "timestamp_clock": "NRT synchronized timestamp_ns",
        "device_ids": device_ids,
        "dp_rank_by_device": {"0": 0},
        "physical_core_ids": list(next(iter(physical))),
        "logical_nc_config": 2,
        "phases": ["forward"],
        "iterations": [r["iteration"] for r in details],
        "kernel_names": names,
        "iteration_details": details,
        "kernel_sequences": sequences,
        "scanned_kernel_rows": sum(len(r["ranges"][0]["kernels"]) for r in details),
    }


def write_normalized(
    log_dir: Path, trace_path: Path, records_path: Path, *, request_ids: set[str] | None = None
) -> dict:
    # Parquet belongs to the controller; the isolated Neuron server only emits records.
    from alignment.nsys.parsed_io import kernel_rows_path, write_parsed

    parsed = normalize(
        json.loads(trace_path.read_text()),
        json.loads(records_path.read_text()),
        request_ids=request_ids,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
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
                "trace_provider": parsed["trace_provider"],
                "measurement_granularity": parsed["measurement_granularity"],
                "phases": parsed["kernel_sequences"],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "parsed_trace": str(path),
        "parsed_kernel_rows": str(kernel_rows_path(path)),
        "kernel_sequences": str(inventory),
        "trace_provider": parsed["trace_provider"],
        "parsed_device_ids": parsed["device_ids"],
        "parsed_dp_rank_by_device": {"0": 0},
        "physical_core_ids": parsed["physical_core_ids"],
    }
