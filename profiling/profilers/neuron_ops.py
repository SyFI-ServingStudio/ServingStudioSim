"""Measured per-operation attribution of unmodified Trainium2 NEFF executions.

Active instruction intervals, overlapping operations, engine waits, and DMA
completion latency are separate measurements. Only ``exclusive + shared + idle``
partitions a physical core's execution. Per-op windows and per-engine waits must
not be added to predict a forward. The input is a versioned capture manifest;
capture/SDK invocation and Analyzer resource registration live in the driver.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from profiling.profilers.neuron_ops_dependencies import observed_dependencies
from profiling.profilers.neuron_ops_labels import source_operation

_ENGINES = {"PE": "Tensor", "Activation": "Scalar", "DVE": "Vector", "Pool": "GpSimd", "SP": "Sync"}
_LOSS = re.compile(r"drop(?:ped|ping)?|lost|overflow|truncat", re.I)
_ANNOTATION_WARNING = (
    "Schema validation: Unexpected key 'subgraph' cannot be saved to table "
    "'TimelineAnnotation' because no schema.yaml entry exists or no key name "
    "remapping. This key and value will be dropped and not found in parquet output."
)
_INSTRUCTION_COLUMNS = (
    "id",
    "pcore_idx",
    "engine",
    "timeline_track",
    "execution_unit",
    "compiler_pc",
    "pc",
    "start_ts",
    "end_ts",
    "opcode",
    "operands",
    "nki_source_location",
    "instruction_chain",
)
_UPDATE_COLUMNS = (
    "timestamp",
    "id",
    "value",
    "update_source",
    "update_operation_type",
    "pcore_idx",
)


class NeuronOpsTraceError(ValueError):
    """An incomplete or ambiguous capture cannot establish measured attribution."""


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _table(directory: Path, name: str, *, required: bool = True) -> list[str]:
    """Explorer may split any large table across numbered parquet shards."""
    pattern = re.compile(rf"{re.escape(name)}(?:_\d+)?\.parquet")
    paths = sorted(str(path) for path in directory.iterdir() if pattern.fullmatch(path.name))
    if required and not paths:
        raise NeuronOpsTraceError(f"Capture is missing required table {name}")
    return paths


def _operation(kernel: str | None, source: str | None) -> str | None:
    """Recognize named production operations; preserve unknown kernel names."""
    text = f"{kernel or ''} {source or ''}".lower()
    for needle, operation in (
        ("mlp_tkg_gate_up", "mlp_gate_up"),
        ("mlp_tkg_down", "mlp_down"),
        ("qkv", "qkv_projection"),
        ("output_projection", "output_projection"),
        ("attention_cte", "attention"),
        ("gen_mask", "attention_mask"),
        ("rmsnorm", "rmsnorm"),
        ("rms_norm", "rmsnorm"),
        ("norm_tkg", "rmsnorm"),
        ("embedding", "embedding"),
        ("rotational_topk", "sampling"),
        ("sampling", "sampling"),
        ("cumsum", "sampling"),
        ("/mlp/", "mlp"),
        ("mlp_lnc", "mlp"),
        ("_functional_collectives", "collective"),
        ("device_communicators", "collective"),
    ):
        if needle in text:
            return operation
    if kernel and kernel.strip():
        return kernel.strip()
    return None


def _static_labels(path: Path | None, source_hashes: dict | None = None) -> dict:
    """Keep all parent labels; disagreeing parents never choose an arbitrary op."""
    if path is None:
        return {}
    data = json.loads(path.read_text())
    labels = {}
    source_hashes = source_hashes or {}
    if "instructions" in data:
        for row in data["instructions"]:
            parents = row.get("parents", [])
            debug_parents = row.get("matched_debug_parents", [])
            debug_evidence = None
            if debug_parents:
                proof = row.get("debug_match", {})
                # The mapper verifies unique same-function instruction context.
                # Its receipt must retain the exact matching instruction bytes.
                if (
                    not row.get("instruction_sha256")
                    or proof.get("instruction_sha256") != row["instruction_sha256"]
                    or proof.get("method")
                    not in ("identical_function_stream", "unique_binary_context")
                    or not proof.get("debug_neff_sha256")
                ):
                    raise NeuronOpsTraceError("Debug label transfer lacks equal instruction hashes")
                if proof["method"] == "identical_function_stream":
                    if (
                        proof.get("label_confidence") != "exact_function"
                        or not proof.get("stock_function_stream_sha256")
                        or proof["stock_function_stream_sha256"]
                        != proof.get("debug_function_stream_sha256")
                    ):
                        raise NeuronOpsTraceError(
                            "Debug function transfer lacks equal stream hashes"
                        )
                    debug_evidence = "exact_function_debug_transfer"
                else:
                    if (
                        proof.get("label_confidence") != "inferred_context"
                        or not isinstance(proof.get("context_radius"), int)
                        or proof["context_radius"] < 8
                        or not proof.get("homogeneous_context_owner")
                    ):
                        raise NeuronOpsTraceError(
                            "Debug context transfer lacks a strong semantic context"
                        )
                    debug_evidence = "inferred_context_debug_transfer"

            def parent_evidence(records):
                owners, sources = [], set()
                for parent in records:
                    source = f"{parent.get('kernel_file') or ''}:{parent.get('kernel_line') or ''}"
                    owner = (
                        None
                        if parent.get("missing")
                        else (
                            source_operation(source, source_hashes)
                            or _operation(parent.get("kernel_name"), source)
                        )
                    )
                    owners.append(owner)
                    if owner:
                        sources.add(source)
                return {owner for owner in owners if owner}, sources, bool(owners) and all(owners)

            stock_owners, stock_sources, stock_complete = parent_evidence(parents)
            debug_owners, debug_sources, debug_complete = parent_evidence(debug_parents)
            owners = stock_owners | debug_owners
            # A complete debug parent set may restore stripped stock metadata.
            # Unknown debug candidates cannot supply that proof; a complete
            # stock set remains usable independently. Any known conflict wins.
            operation = (
                next(iter(owners))
                if len(owners) == 1 and (stock_complete or debug_complete)
                else None
            )
            ambiguous = operation is None and (
                bool(owners) or any(parent.get("missing") for parent in parents + debug_parents)
            )
            primitive = (
                "transpose"
                if row.get("compiler_operation") == "transpose"
                and row.get("compiler_operation_evidence") == "unanimous_backend_instruction_type"
                and parents
                and all(
                    not parent.get("missing") and parent.get("instruction_type") == "TRANSPOSE"
                    for parent in parents
                )
                else None
            )
            key = row["pcore_idx"], row["engine"], row["compiler_pc"]
            labels[key] = {
                "function": row.get("function") or "unknown",
                "operation": operation,
                "evidence": (debug_evidence if debug_complete else "static_debug")
                if operation
                else "unresolved",
                "source": " | ".join(sorted(stock_sources | debug_sources)),
                "opcode_byte": row.get("opcode_byte"),
                "ambiguous": ambiguous,
                "compiler_operation": primitive,
            }
    else:
        for core, engines in data.items():
            if not re.fullmatch(r"sg\d+", core):
                continue
            for engine, values in engines.items():
                if engine not in _ENGINES:
                    continue
                for pc, (opcode, function, _, kernel, source) in enumerate(values["instr"]):
                    operation = source_operation(source, source_hashes) or _operation(
                        kernel, source
                    )
                    labels[int(core[2:]), _ENGINES[engine], pc] = {
                        "function": function,
                        "operation": operation,
                        "evidence": "static_debug" if operation else "unresolved",
                        "source": source,
                        "opcode_byte": opcode,
                        "ambiguous": False,
                    }
    # Label-stripped instructions between the same named kernel remain an
    # inference. The same function and engine are required; calls are excluded.
    streams = defaultdict(list)
    for key in labels:
        streams[key[:2]].append(key)
    for keys in streams.values():
        keys.sort(key=lambda key: key[2])
        previous = None
        pending = []
        for key in keys:
            row = labels[key]
            if row["opcode_byte"] in (209, 210, 211):
                previous, pending = None, []
                continue
            if row["operation"]:
                marker = row["function"], row["operation"]
                if marker == previous:
                    for missing in pending:
                        target = labels[missing]
                        if (
                            target["function"] == marker[0]
                            and not target["ambiguous"]
                            and target["opcode_byte"] not in (209, 210, 211)
                        ):
                            target["operation"] = marker[1]
                            target["evidence"] = "same_kernel_sandwich"
                previous, pending = marker, []
            else:
                pending.append(key)
    return labels


def interval_accounting(instructions: list[dict], start: int, end: int) -> dict:
    """Union active operations and partition elapsed time without double counts."""
    events = []
    counts = Counter()
    for row in instructions:
        if row["execution_unit"] != "compute":
            continue
        lower, upper = max(start, row["start_ts"]), min(end, row["end_ts"])
        if upper > lower:
            events.extend(((lower, row["operation"], 1), (upper, row["operation"], -1)))
    events.sort()
    active, exclusive = Counter(), Counter()
    shared_ns = active_ns = 0
    previous = start
    for timestamp, operation, delta in events:
        duration = timestamp - previous
        owners = [owner for owner, count in counts.items() if count > 0]
        if owners:
            active_ns += duration
            for owner in owners:
                active[owner] += duration
            if len(owners) == 1:
                exclusive[owners[0]] += duration
            else:
                shared_ns += duration
        counts[operation] += delta
        previous = timestamp
    elapsed = end - start
    assert sum(exclusive.values()) + shared_ns == active_ns
    return {
        "active_ns": active_ns,
        "idle_ns": elapsed - active_ns,
        "shared_ns": shared_ns,
        "exclusive_ns": dict(exclusive),
        "per_op_active_ns": dict(active),
        "elapsed_ns": elapsed,
    }


def _rows(
    connection, paths: list[str], columns: tuple[str, ...] | None = None, where="", parameters=None
):
    select = ",".join(columns) if columns else "*"
    cursor = connection.execute(
        f"select {select} from read_parquet(?) {where}", [paths, *(parameters or [])]
    )
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _append_parquet(writers, name: str, rows: list[dict], output: Path):
    if not rows:
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Per-file schemas are explicit in the first batch. Cast subsequent null-only
    # columns to that schema rather than allowing pyarrow's inferred null type.
    schema = writers[name].schema if name in writers else None
    if schema is None:
        fields = []
        for key in rows[0]:
            numeric = key in {
                "session",
                "rank",
                "execution_index",
                "token_bucket",
                "id",
                "pcore_idx",
                "compiler_pc",
                "pc",
                "start_ts",
                "end_ts",
                "gap_ns",
                "explained_ns",
                "residual_ns",
                "waiter_id",
                "source_id",
                "blocking_instruction_id",
                "semaphore",
                "trigger_id",
                "reset_ts",
                "completion_counter",
                "trigger_start_ts",
                "trigger_end_ts",
                "completion_ts",
                "counter_crossing_ts",
                "latency_ns",
            }
            typ = pa.int64() if numeric else pa.string()
            fields.append(pa.field(key, typ))
        schema = pa.schema(fields)
        writers[name] = pq.ParquetWriter(output / f"{name}.parquet", schema, compression="zstd")
    for begin in range(0, len(rows), 50_000):
        writers[name].write_table(pa.Table.from_pylist(rows[begin : begin + 50_000], schema=schema))


def _neff_for_execution(neffs: list[dict], name: str) -> dict:
    graph = name.removeprefix("graph_").removesuffix(".neff")
    matches = [
        item
        for item in neffs
        if item.get("graph_id") in (name, graph) or Path(item.get("path", "")).name == name
    ]
    if len(matches) != 1:
        raise NeuronOpsTraceError(f"Execution {name!r} needs exactly one NEFF manifest entry")
    return matches[0]


def _execution_roles(session: dict, executions: list[dict], shape: dict) -> dict[int, str]:
    """Validate bounded native-execution patterns before selecting one prefill."""
    by_index = {row["execution_index"]: row for row in executions}
    if len(by_index) != len(executions):
        raise NeuronOpsTraceError("Duplicate execution indices")
    selection = session.get("execution_selection")
    if selection is None:
        for execution in executions:
            neff = _neff_for_execution(session["neffs"], execution["neff_name"])
            if any(
                neff.get(key) != shape[key] for key in ("phase", "token_bucket") if key in shape
            ):
                raise NeuronOpsTraceError(
                    "Every execution must match the requested geometry without explicit selection"
                )
        return dict.fromkeys(by_index, "selected")
    if (
        not isinstance(selection, dict)
        or selection.get("policy")
        not in {
            "prefill_with_trailing_decode_guard_v1",
            "prefill_with_trailing_native_guard_v2",
            "prefill_with_decode_tail_v3",
        }
        or shape.get("phase") != "prefill"
    ):
        raise NeuronOpsTraceError("Unsupported execution selection policy or requested phase")
    chosen, guards = selection.get("selected_indices"), selection.get("guard_indices")
    guard_count = 2 if selection["policy"] == "prefill_with_decode_tail_v3" else 1
    if (
        not isinstance(chosen, list)
        or not isinstance(guards, list)
        or len(chosen) != 1
        or len(guards) != guard_count
        or any(type(index) is not int for index in chosen + guards)
        or len(set(chosen + guards)) != 1 + guard_count
        or set(chosen + guards) != set(by_index)
    ):
        raise NeuronOpsTraceError(
            f"Execution selection must partition one prefill and {guard_count} guards"
        )
    if selection["policy"] == "prefill_with_decode_tail_v3":
        if any(
            row.get(key) is None
            for row in executions
            for key in ("execution_start_ts", "execution_end_ts")
        ):
            raise NeuronOpsTraceError("Decode-tail selection requires complete execution bounds")
        ordered = sorted(executions, key=lambda row: row["execution_start_ts"])
        if any(
            right["execution_start_ts"] < left["execution_end_ts"]
            for left, right in zip(ordered, ordered[1:])
        ):
            raise NeuronOpsTraceError("Decode-tail executions must not overlap")
        neffs = [_neff_for_execution(session["neffs"], row["neff_name"]) for row in ordered]
        pattern = tuple(neff.get("phase") for neff in neffs)
        if pattern not in (("prefill", "prefill", "decode"), ("prefill", "decode", "decode")):
            raise NeuronOpsTraceError("Decode-tail policy requires P,P,D16 or P,D16,D16")
        last_prefill = 1 if pattern[1] == "prefill" else 0
        if chosen[0] != ordered[last_prefill]["execution_index"]:
            raise NeuronOpsTraceError("Decode-tail selection must choose the last prefill")
        for phase in ("prefill", "decode"):
            group = [neff for neff in neffs if neff["phase"] == phase]
            expected_bucket = shape.get("token_bucket") if phase == "prefill" else 16
            if (
                not group[0].get("graph_id")
                or not group[0].get("sha256")
                or any(neff.get("token_bucket") != expected_bucket for neff in group)
                or any(
                    neff.get(key) != group[0].get(key)
                    for neff in group
                    for key in ("graph_id", "sha256")
                )
            ):
                raise NeuronOpsTraceError(
                    "Decode-tail repeated phases require identical graph, hash and bucket"
                )
        return {index: "selected" if index == chosen[0] else "guard" for index in by_index}
    first, guard = by_index[chosen[0]], by_index[guards[0]]
    first_neff = _neff_for_execution(session["neffs"], first["neff_name"])
    guard_neff = _neff_for_execution(session["neffs"], guard["neff_name"])
    same_prefill_guard = (
        selection["policy"] == "prefill_with_trailing_native_guard_v2"
        and guard_neff.get("phase") == "prefill"
        and bool(first_neff.get("graph_id"))
        and bool(first_neff.get("sha256"))
        and all(
            guard_neff.get(key) == first_neff.get(key)
            for key in ("graph_id", "sha256", "token_bucket")
        )
    )
    decode_guard = guard_neff.get("phase") == "decode" and guard_neff.get("token_bucket") == 16
    if (first_neff.get("phase"), first_neff.get("token_bucket")) != (
        shape.get("phase"),
        shape.get("token_bucket"),
    ) or not (same_prefill_guard or decode_guard):
        raise NeuronOpsTraceError(
            "Selected/guard execution geometry differs from the declared policy"
        )
    if (
        first.get("execution_end_ts") is None
        or guard.get("execution_start_ts") is None
        or guard["execution_start_ts"] < first["execution_end_ts"]
    ):
        raise NeuronOpsTraceError("Guard must follow the selected prefill execution")
    return {chosen[0]: "selected", guards[0]: "guard"}


def analyze_capture(manifest_path: Path, output_dir: Path) -> dict:
    """Analyze every exported execution; write an auditable measured resource."""
    import duckdb

    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("sessions"):
        raise NeuronOpsTraceError("Expected a schema_version 1 manifest with sessions")
    hardware_name = manifest.get("hardware", {}).get("name")
    if not isinstance(hardware_name, str) or not hardware_name.strip():
        raise NeuronOpsTraceError("Capture manifest lacks observed hardware identity")
    base = manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("set memory_limit='3GB'; set threads=2")
    writers, operations, executions, selection_receipts = {}, [], [], []
    quality = {
        "schema_version": 1,
        "scope": "measured_rank_local_attribution",
        "warnings": [],
        "unseen_shape_prediction_supported": False,
        "instruction_label_counts": Counter(),
        "dma_triggers": 0,
        "dma_completion_matches": 0,
        "engine_gap_ns": 0,
        "explained_engine_gap_ns": 0,
        "timestamp_inversion_matches": 0,
        "input_instruction_count": 0,
        "attributed_instruction_count": 0,
        "excluded_execution_instruction_count": 0,
        "excluded_control_spill_count": 0,
        "active_instruction_ns": 0,
        "labeled_active_ns": 0,
        "unresolved_active_ns": 0,
        "compiler_primitive_active_ns": 0,
        "per_engine_active_coverage": {},
        "missing_optional_tables": [],
    }
    maps = {}
    source_receipt = manifest.get("semantic_source_identity", {})
    if isinstance(source_receipt, str):
        source_receipt = json.loads(_path(base, source_receipt).read_text())
    source_hashes = source_receipt.get("source_hashes", {})
    quality["semantic_source_hashes"] = source_hashes
    try:
        for session_index, session in enumerate(manifest["sessions"]):
            directory = _path(base, session["session_dir"])
            instruction_paths = _table(directory, "Instruction")
            execution_paths = _table(directory, "ExecutionInfo")
            total, unique, invalid = connection.execute(
                "select count(*),count(distinct (pcore_idx,timeline_track,id)),"
                "count(*) filter (where start_ts is null or end_ts is null "
                "or end_ts < start_ts) from read_parquet(?)",
                [instruction_paths],
            ).fetchone()
            if total != unique or invalid:
                raise NeuronOpsTraceError(
                    "Duplicate composite instruction identity or invalid instruction interval"
                )
            quality["input_instruction_count"] += total
            warning_paths = _table(directory, "Warning", required=False)
            warnings = _rows(connection, warning_paths) if warning_paths else []
            for warning in warnings:
                message = str(warning.get("message", warning))
                if _LOSS.search(message) and message != _ANNOTATION_WARNING:
                    raise NeuronOpsTraceError(f"Incomplete instruction/DMA capture: {message}")
                quality["warnings"].append(message)
            execution_rows = _rows(connection, execution_paths)
            roles = _execution_roles(session, execution_rows, manifest.get("shape", {}))
            bounds = sorted(
                (row["execution_start_ts"], row["execution_end_ts"])
                for row in execution_rows
                if row.get("execution_start_ts") is not None
                and row.get("execution_end_ts") is not None
            )
            if any(left[1] > right[0] for left, right in zip(bounds, bounds[1:])):
                raise NeuronOpsTraceError(
                    "Overlapping execution bounds cannot partition instruction rows"
                )
            for execution in execution_rows:
                start, end = execution.get("execution_start_ts"), execution.get("execution_end_ts")
                if start is None or end is None:
                    raise NeuronOpsTraceError("An execution is missing start/end timestamps")
                if end <= start:
                    raise NeuronOpsTraceError("An execution has a nonpositive duration")
                neff = _neff_for_execution(session["neffs"], execution["neff_name"])
                for field in ("sha256", "static_instructions_sha256"):
                    if not re.fullmatch(r"[0-9a-f]{64}", neff.get(field, "")):
                        raise NeuronOpsTraceError(f"NEFF entry requires a valid {field} hash")
                if not neff.get("path") or not neff.get("static_instructions"):
                    raise NeuronOpsTraceError(
                        "NEFF entry requires binary and static instruction map paths"
                    )
                static_path = _path(base, neff["static_instructions"])
                map_key = str(static_path), neff["static_instructions_sha256"], neff["sha256"]
                if map_key not in maps:
                    if (
                        neff.get("static_instructions_sha256")
                        and _digest(static_path) != neff["static_instructions_sha256"]
                    ):
                        raise NeuronOpsTraceError(
                            "Static instruction map hash differs from the manifest"
                        )
                    decoded = json.loads(static_path.read_text())
                    if neff["sha256"] != decoded.get("neff_sha256"):
                        raise NeuronOpsTraceError(
                            "Static instruction map belongs to a different NEFF"
                        )
                    maps[map_key] = _static_labels(static_path, source_hashes)
                labels = maps[map_key]
                if neff.get("sha256"):
                    if not neff.get("path"):
                        raise NeuronOpsTraceError(
                            "Hashed NEFF manifest entry is missing its binary path"
                        )
                    neff_path = _path(base, neff["path"])
                    if not neff_path.is_file() or _digest(neff_path) != neff["sha256"]:
                        raise NeuronOpsTraceError("Captured NEFF hash differs from the manifest")
                identity = {
                    "session": session_index,
                    "rank": session["rank"],
                    "execution_index": execution["execution_index"],
                    "phase": neff.get("phase", "unknown"),
                    "token_bucket": neff.get("token_bucket", 0),
                    "execution_role": roles[execution["execution_index"]],
                }
                is_selected = identity["execution_role"] == "selected"
                instructions = _rows(
                    connection,
                    instruction_paths,
                    _INSTRUCTION_COLUMNS,
                    "where start_ts >= ? and start_ts < ? order by start_ts,id",
                    [start, end],
                )
                if not instructions:
                    raise NeuronOpsTraceError("Execution contains no instructions")
                if max(inst["end_ts"] for inst in instructions) > end + 10_000:
                    raise NeuronOpsTraceError(
                        "Instruction end extends more than 10us beyond execution"
                    )
                quality[
                    "attributed_instruction_count"
                    if is_selected
                    else "excluded_execution_instruction_count"
                ] += len(instructions)
                attribution = []
                for inst in instructions:
                    label = labels.get((inst["pcore_idx"], inst["engine"], inst["compiler_pc"]), {})
                    operation = label.get("operation")
                    evidence = label.get("evidence", "unresolved")
                    if inst.get("instruction_chain"):
                        operation, evidence = "collective", "trace_instruction_chain"
                    elif not operation and not label.get("ambiguous"):
                        operation = source_operation(
                            inst.get("nki_source_location") or "", source_hashes
                        ) or _operation(None, inst.get("nki_source_location"))
                        if operation:
                            evidence = "trace_source"
                    category = "model_semantic" if operation else "unresolved"
                    if not operation and label.get("compiler_operation"):
                        operation = f"compiler_{label['compiler_operation']}"
                        evidence = "unanimous_backend_instruction_type"
                        category = "compiler_primitive"
                    inst["operation"] = operation or "unresolved"
                    inst["attribution_category"] = category
                    if is_selected:
                        quality["instruction_label_counts"][evidence] += 1
                    attribution.append(
                        identity
                        | {
                            key: inst[key]
                            for key in (
                                "id",
                                "pcore_idx",
                                "engine",
                                "timeline_track",
                                "execution_unit",
                                "compiler_pc",
                                "pc",
                                "start_ts",
                                "end_ts",
                                "opcode",
                                "operation",
                                "attribution_category",
                            )
                        }
                        | {
                            "function": label.get("function", "unknown"),
                            "label_evidence": evidence,
                            "source": label.get("source") or inst.get("nki_source_location") or "",
                        }
                    )
                cores = sorted({inst["pcore_idx"] for inst in instructions})
                expected = manifest.get("hardware", {}).get("physical_cores_per_rank", [0, 1])
                if cores != sorted(expected):
                    raise NeuronOpsTraceError(
                        f"Execution physical cores {cores} differ from {expected}"
                    )
                selection_receipts.append(
                    identity
                    | {
                        "policy": (session.get("execution_selection") or {}).get(
                            "policy", "all_executions"
                        ),
                        "neff_name": execution["neff_name"],
                        "neff_sha256": neff["sha256"],
                        "start_ts": start,
                        "end_ts": end,
                        "per_core_instruction_counts": dict(
                            Counter(inst["pcore_idx"] for inst in instructions)
                        ),
                        "guard_position": (
                            None
                            if is_selected
                            else "before_selected"
                            if start
                            < next(
                                row["execution_start_ts"]
                                for row in execution_rows
                                if roles[row["execution_index"]] == "selected"
                            )
                            else "after_selected"
                        ),
                        "exclusion_reason": (
                            None
                            if is_selected
                            else "decode_tail_guard"
                            if session["execution_selection"]["policy"]
                            == "prefill_with_decode_tail_v3"
                            else "trailing_native_guard"
                            if session["execution_selection"]["policy"]
                            == "prefill_with_trailing_native_guard_v2"
                            else "trailing_decode_guard"
                        ),
                    }
                )
                update_paths = _table(directory, "SemaphoreUpdate", required=False)
                # Padding may recover control updates in an idle gap, but must
                # not borrow writers from another native execution whose DMA
                # triggers are absent from this dependency analysis. Timestamps
                # are integer ns; exclude shared execution boundary timestamps.
                previous_end = max((stop for _, stop in bounds if stop <= start), default=None)
                next_start = min((begin for begin, _ in bounds if begin >= end), default=None)
                update_start = (
                    max(start - 10_000, previous_end + 1)
                    if previous_end is not None
                    else start - 10_000
                )
                update_end = (
                    min(end + 10_001, next_start) if next_start is not None else end + 10_001
                )
                if not update_paths:
                    quality["missing_optional_tables"].append(
                        f"session {session_index}: SemaphoreUpdate"
                    )
                updates = (
                    _rows(
                        connection,
                        update_paths,
                        _UPDATE_COLUMNS,
                        "where timestamp >= ? and timestamp < ?",
                        [update_start, update_end],
                    )
                    if update_paths
                    else []
                )
                dependencies = observed_dependencies(instructions, updates)
                gaps, transfers = dependencies["gaps"], dependencies["transfers"]
                if is_selected:
                    quality["dma_triggers"] += len(transfers)
                    quality["dma_completion_matches"] += sum(
                        row["completion_ts"] is not None for row in transfers
                    )
                    quality["engine_gap_ns"] += sum(row["gap_ns"] for row in gaps)
                    quality["explained_engine_gap_ns"] += sum(row["explained_ns"] for row in gaps)
                    quality["timestamp_inversion_matches"] += sum(
                        row["match"] == "adjacent_dma_timestamp_inversion" for row in gaps
                    )
                for core in cores:
                    mine = [inst for inst in instructions if inst["pcore_idx"] == core]
                    span = max(inst["end_ts"] for inst in mine) - min(
                        inst["start_ts"] for inst in mine
                    )
                    if span / (end - start) < 0.95:
                        raise NeuronOpsTraceError(f"Core {core} instruction span is truncated")
                    accounting = interval_accounting(mine, start, end)
                    labeled = interval_accounting(
                        [inst for inst in mine if inst["attribution_category"] == "model_semantic"],
                        start,
                        end,
                    )["active_ns"]
                    primitive = interval_accounting(
                        [
                            inst
                            for inst in mine
                            if inst["attribution_category"] == "compiler_primitive"
                        ],
                        start,
                        end,
                    )["active_ns"]
                    unknown = accounting["per_op_active_ns"].get("unresolved", 0)
                    if is_selected:
                        quality["active_instruction_ns"] += accounting["active_ns"]
                        quality["labeled_active_ns"] += labeled
                        quality["unresolved_active_ns"] += unknown
                        quality["compiler_primitive_active_ns"] += primitive
                    for track in (
                        sorted({inst["timeline_track"] for inst in mine}) if is_selected else []
                    ):
                        track_rows = [inst for inst in mine if inst["timeline_track"] == track]
                        coverage = quality["per_engine_active_coverage"].setdefault(
                            track,
                            {
                                "active_ns": 0,
                                "labeled_active_ns": 0,
                                "unresolved_active_ns": 0,
                                "compiler_primitive_active_ns": 0,
                            },
                        )
                        track_accounting = interval_accounting(track_rows, start, end)
                        coverage["active_ns"] += track_accounting["active_ns"]
                        coverage["unresolved_active_ns"] += track_accounting[
                            "per_op_active_ns"
                        ].get("unresolved", 0)
                        coverage["labeled_active_ns"] += interval_accounting(
                            [
                                inst
                                for inst in track_rows
                                if inst["attribution_category"] == "model_semantic"
                            ],
                            start,
                            end,
                        )["active_ns"]
                        coverage["compiler_primitive_active_ns"] += interval_accounting(
                            [
                                inst
                                for inst in track_rows
                                if inst["attribution_category"] == "compiler_primitive"
                            ],
                            start,
                            end,
                        )["active_ns"]
                    executions.append(
                        identity
                        | {
                            "pcore_idx": core,
                            "runtime_ms": (end - start) / 1e6,
                            "active_us": accounting["active_ns"] / 1000,
                            "exclusive_us": sum(accounting["exclusive_ns"].values()) / 1000,
                            "shared_us": accounting["shared_ns"] / 1000,
                            "idle_us": accounting["idle_ns"] / 1000,
                            "labeled_active_us": labeled / 1000,
                            "unresolved_active_us": unknown / 1000,
                            "compiler_primitive_active_us": primitive / 1000,
                            "instruction_count": len(mine),
                        }
                    )
                    groups = defaultdict(list)
                    for inst in mine:
                        groups[inst["operation"]].append(inst)
                    for operation, group in sorted(groups.items()):
                        own_gaps = [
                            row
                            for row in gaps
                            if row["pcore_idx"] == core and row["consumer_operation"] == operation
                        ]
                        own_dma = [
                            row
                            for row in transfers
                            if row["pcore_idx"] == core and row["operation"] == operation
                        ]
                        active = accounting["per_op_active_ns"].get(operation, 0)
                        exclusive = accounting["exclusive_ns"].get(operation, 0)
                        operations.append(
                            identity
                            | {
                                "pcore_idx": core,
                                "operation": operation,
                                "attribution_category": group[0]["attribution_category"],
                                "instruction_count": len(group),
                                "active_us": active / 1000,
                                "exclusive_us": exclusive / 1000,
                                "overlap_us": (active - exclusive) / 1000,
                                "window_us": (
                                    max(row["end_ts"] for row in group)
                                    - min(row["start_ts"] for row in group)
                                )
                                / 1000,
                                "engine_wait_us": sum(row["explained_ns"] for row in own_gaps)
                                / 1000,
                                "unexplained_engine_gap_us": sum(
                                    row["residual_ns"] for row in own_gaps
                                )
                                / 1000,
                                "dma_completion_latency_us": (
                                    sum(row["latency_ns"] or 0 for row in own_dma) / 1000
                                    if any(row["completion_ts"] is not None for row in own_dma)
                                    else None
                                ),
                                "dma_transfer_count": len(own_dma),
                                "dma_completion_count": sum(
                                    row["completion_ts"] is not None for row in own_dma
                                ),
                                "opcode_counts": json.dumps(
                                    dict(Counter(row["opcode"] for row in group)), sort_keys=True
                                ),
                                "per_engine_active_us": json.dumps(
                                    {
                                        track: interval_accounting(
                                            [
                                                row
                                                for row in group
                                                if row["timeline_track"] == track
                                            ],
                                            start,
                                            end,
                                        )["active_ns"]
                                        / 1000
                                        for track in sorted(
                                            {row["timeline_track"] for row in group}
                                        )
                                    },
                                    sort_keys=True,
                                ),
                            }
                        )
                _append_parquet(writers, "instruction_attribution", attribution, output_dir)
                _append_parquet(
                    writers, "engine_waits", [identity | row for row in gaps], output_dir
                )
                _append_parquet(
                    writers, "dma_transfers", [identity | row for row in transfers], output_dir
                )
            outside = _rows(
                connection,
                instruction_paths,
                _INSTRUCTION_COLUMNS,
                "i where not exists (select 1 from read_parquet(?) e "
                "where i.start_ts >= e.execution_start_ts and i.start_ts < e.execution_end_ts)",
                [execution_paths],
            )
            if outside:
                control = {"EVENT_SEMAPHORE", "COMPARE_BRANCH", "NOTIFY", "NOP", "HALT", "RETURN"}
                for row in outside:
                    near_end = any(0 <= row["start_ts"] - upper <= 10_000 for _, upper in bounds)
                    if not near_end or (
                        row["execution_unit"] == "compute" and row["opcode"] not in control
                    ):
                        raise NeuronOpsTraceError(
                            "Unassigned non-control instruction outside execution bounds"
                        )
                    row["exclusion_reason"] = "control_spill_after_execution"
                quality["excluded_control_spill_count"] += len(outside)
                _append_parquet(writers, "excluded_instructions", outside, output_dir)
    finally:
        for writer in writers.values():
            writer.close()
        connection.close()
    if not executions:
        raise NeuronOpsTraceError("Capture contains no executions")
    if (
        quality["input_instruction_count"]
        != quality["attributed_instruction_count"]
        + quality["excluded_execution_instruction_count"]
        + quality["excluded_control_spill_count"]
    ):
        raise NeuronOpsTraceError("Instruction count conservation failed")
    grouped = defaultdict(list)
    populations = Counter(
        (row["phase"], row["token_bucket"])
        for row in executions
        if row["execution_role"] == "selected"
    )
    for row in operations:
        if row["execution_role"] != "selected":
            continue
        grouped[
            row["phase"], row["token_bucket"], row["operation"], row["attribution_category"]
        ].append(row)
    per_op = []
    for (phase, bucket, operation, category), rows in sorted(grouped.items()):
        per_op.append(
            {
                "phase": phase,
                "token_bucket": bucket,
                "operation": operation,
                "attribution_category": category,
                "samples": len(rows),
                "population": populations[phase, bucket],
                "dma_completion_fraction": (
                    sum(row["dma_completion_count"] for row in rows)
                    / sum(row["dma_transfer_count"] for row in rows)
                    if any(row["dma_transfer_count"] for row in rows)
                    else None
                ),
                "dma_completion_latency_us": (
                    statistics.median(
                        row["dma_completion_latency_us"]
                        for row in rows
                        if row["dma_completion_latency_us"] is not None
                    )
                    if any(row["dma_completion_count"] for row in rows)
                    else None
                ),
            }
            | {
                key: statistics.median(row[key] for row in rows)
                for key in (
                    "instruction_count",
                    "active_us",
                    "exclusive_us",
                    "overlap_us",
                    "engine_wait_us",
                    "dma_transfer_count",
                    "dma_completion_count",
                )
            }
        )
    shape = manifest.get("shape", {})
    selected = [
        row
        for row in executions
        if row["execution_role"] == "selected"
        and all(row[key] == shape[key] for key in ("phase", "token_bucket") if key in shape)
    ]
    if not selected:
        raise NeuronOpsTraceError("No exported execution matches the requested phase/token bucket")
    quality["warnings"] = sorted(set(quality["warnings"]))
    quality["missing_optional_tables"] = sorted(set(quality["missing_optional_tables"]))
    quality["instruction_label_counts"] = dict(quality["instruction_label_counts"])
    quality["dma_packet_to_transfer_mapping"] = "unavailable_without_dynamic_metadata"
    quality["labeled_active_fraction"] = quality["labeled_active_ns"] / max(
        quality["active_instruction_ns"], 1
    )
    quality["unresolved_active_fraction"] = quality["unresolved_active_ns"] / max(
        quality["active_instruction_ns"], 1
    )
    quality["compiler_primitive_active_fraction"] = quality["compiler_primitive_active_ns"] / max(
        quality["active_instruction_ns"], 1
    )
    quality["model_labeled_active_ns"] = quality["labeled_active_ns"]
    quality["semantic_coverage_note"] = (
        "Labeled means a model semantic operation. Compiler primitives have unknown model "
        "owner and origin. Each category is an active interval union; categories may overlap. "
        "Per-engine coverage avoids hiding unresolved work behind another engine."
    )
    for coverage in quality["per_engine_active_coverage"].values():
        coverage["labeled_active_fraction"] = coverage["labeled_active_ns"] / max(
            coverage["active_ns"], 1
        )
        coverage["model_labeled_active_ns"] = coverage["labeled_active_ns"]
        coverage["compiler_primitive_active_fraction"] = coverage[
            "compiler_primitive_active_ns"
        ] / max(coverage["active_ns"], 1)
        coverage["unresolved_active_fraction"] = coverage["unresolved_active_ns"] / max(
            coverage["active_ns"], 1
        )
    quality["elapsed_accounting"] = (
        "exclusive_us + shared_us + idle_us = runtime_ms * 1000 per execution/core"
    )
    quality["instruction_accounting"] = (
        "input_instruction_count = attributed_instruction_count (selected) + "
        "excluded_execution_instruction_count (guards) + excluded_control_spill_count"
    )
    quality["aggregation_scope"] = (
        "Coverage, DMA and wait aggregates use selected executions only. All raw executions, "
        "including guards, pass identity, loss, interval and per-core completeness checks. "
        "Detailed artifacts retain both execution roles."
    )
    quality["dependency_timestamp_scope"] = (
        "Semaphore updates use the execution interval plus at most 10us idle-gap padding, "
        "strictly after the previous execution end and before the next execution start. "
        "Boundary completions without isolated writer evidence remain unresolved."
    )
    quality["dma_latency_aggregation"] = (
        "Matched-completion latency subtotal per sample, including queueing; null without "
        "matches. Per-op median conditions on samples with matched completions; completion "
        "fraction uses all observed triggers for that operation."
    )
    summary = {
        "schema_version": 1,
        "runtime_ms": {
            "median": statistics.median(row["runtime_ms"] for row in selected),
            "min": min(row["runtime_ms"] for row in selected),
            "max": max(row["runtime_ms"] for row in selected),
        },
        "aggregation": (
            "Per-op medians conditional on operation attributed in execution/rank/physical core; "
            "samples/population show coverage. Missing identities remain unresolved. "
            "Phases and buckets stay separate; runtime and per-op aggregates use selected "
            "executions only. Detailed records retain guards."
        ),
        "per_op": per_op,
        "executions": executions,
        "execution_selection": selection_receipts,
        "quality": quality,
    }
    _write_csv(output_dir / "operations.csv", operations)
    (output_dir / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "execution_selection.json").write_text(
        json.dumps(selection_receipts, indent=2) + "\n"
    )
    artifacts = [
        output_dir / name
        for name in ("operations.csv", "quality.json", "summary.json", "execution_selection.json")
    ]
    artifacts += [output_dir / f"{name}.parquet" for name in writers]
    return {
        "summary": summary,
        "artifacts": [str(path) for path in artifacts],
        "plots": [],
        "observed_gpu_name": hardware_name,
        "gpu_count": 1,
    }
