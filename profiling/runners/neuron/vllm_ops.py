"""Validated stock vLLM Neuron per-op capture; never returns timing DB metrics."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path

from profiling.runners.neuron.vllm_forward import (
    precision_shape,
    run_stage,
    validate_whole_context_shape,
    verify_identity,
    write_plan,
)
from profiling.runners.neuron.vllm_identity import (
    accuracy_source_identity,
    load_accuracy_binaries,
    seal_accuracy_binaries,
    sha256,
    stock_compile_cache_dir,
    verify_cache_binaries,
)


def _save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _run_ops_stage(root: Path, stage: str, env: dict) -> None:
    # vLLM appends a UUID to this IPC base. Keep Unix sockets below sun_path's
    # 107-byte limit independently of persistent compiler/cache path lengths.
    with tempfile.TemporaryDirectory(prefix="nops-", dir="/tmp") as sockets:
        run_stage(
            root,
            stage,
            engine="profiling.runners.neuron.vllm_ops_engine",
            env={**env, "VLLM_RPC_BASE_PATH": sockets},
        )


def _validation(root, shape, cache, model, identity, validated_run):
    if validated_run is None:
        validated = root / "validation"
        validated.mkdir()
        write_plan(validated, [shape], shape["max_model_len"], model, identity, cache)
        run_stage(validated, "accuracy")
        run_stage(validated, "reference")
        sealed = seal_accuracy_binaries(validated)
        if not sealed["binary_binding_available"]:
            if not sealed["precision_passed"]:
                raise ValueError("initial numerical validation failed; capture was not attempted")
            before_load = json.loads((validated / "accuracy-binaries-before-load.json").read_text())
            missing = set(sealed["graphs"]) - set(before_load)
            coverage = sealed["load"]["loader_coverage"]["ranks_by_graph"]
            if not missing or any(
                ranks != [0, 1, 2, 3] and graph not in missing for graph, ranks in coverage.items()
            ):
                raise ValueError(
                    "numerical binary binding unavailable for a reason other than cold compilation"
                )
            # A cold pass compiles the cache but cannot prove which on-disk
            # bytes Executor loaded. Preserve it, then accept one warm pass.
            validated.rename(root / "compile-warmup")
            validated.mkdir()
            write_plan(validated, [shape], shape["max_model_len"], model, identity, cache)
            run_stage(validated, "accuracy")
            run_stage(validated, "reference")
            seal_accuracy_binaries(validated)
    else:
        validated = Path(validated_run)
    receipt = load_accuracy_binaries(validated)
    if not receipt.get("available"):
        raise ValueError("per-op capture requires a numerically validated warm stock NEFF")
    plan = json.loads((validated / "plan.json").read_text())
    if plan["identity"] != identity or plan["context"] != shape["max_model_len"]:
        raise ValueError("validated run checkpoint/runtime/context differs from capture")
    expected_source = {
        key: value for key, value in receipt["load"]["source"].items() if key != "compile_cache_dir"
    }
    current_source = {
        key: value
        for key, value in accuracy_source_identity().items()
        if key != "compile_cache_dir"
    }
    if expected_source != current_source:
        raise ValueError("validated run source/cache identity differs from current producer")
    target = precision_shape(shape, shape["max_model_len"])
    precision = json.loads((validated / "precision.json").read_text())
    if not precision["by_shape"].get(target, {}).get("passed"):
        raise ValueError(f"numerical validation did not pass for {target}")
    verify_cache_binaries(stock_compile_cache_dir(), receipt["graphs"])
    return validated, receipt, plan


def _select_executions(executions: list[dict], graphs: dict, shape: dict, request: dict):
    """Select only after checking every captured graph and the explicit guard window."""
    rows = []
    prefill = shape["phase"] == "prefill"
    for execution in executions:
        name = execution["neff_name"] or ""
        if name.startswith("barrier_") and not prefill:
            continue
        graph = re.fullmatch(r"graph_([0-9a-f]{32})\.neff", name)
        if graph is None or graph[1] not in graphs:
            raise ValueError(f"capture executed an unvalidated NEFF: {name}")
        rows.append((execution, graph[1], graphs[graph[1]]))
    if prefill:
        if request["forwards"] != 1 or request.get("guard_decode_bucket") != 16 or len(rows) != 2:
            raise ValueError("prefill capture requires exactly one selected prefill and one guard")
        for execution, _, _ in rows:
            start, end = execution.get("execution_start_ts"), execution.get("execution_end_ts")
            if start is None or end is None or end <= start:
                raise ValueError("guard selection requires complete positive execution bounds")
            if type(execution.get("execution_index")) is not int:
                raise ValueError("guard selection requires integer execution indices")
        rows.sort(key=lambda row: row[0]["execution_start_ts"])
        target, decode = ("prefill", shape["token_bucket"]), ("decode", 16)
        geometry = tuple((row[2]["phase"], row[2]["token_bucket"]) for row in rows)
        if (
            geometry != (target, decode)
            or any(right[0]["execution_start_ts"] < left[0]["execution_end_ts"]
                   for left, right in zip(rows, rows[1:]))
            or len({row[0]["execution_index"] for row in rows}) != 2
        ):
            raise ValueError(
                "prefill guard requires two distinct complete executions in P,D16 order"
            )
        selection = {
            "policy": "prefill_with_trailing_decode_guard_v1",
            "selected_indices": [rows[0][0]["execution_index"]],
            "guard_indices": [rows[1][0]["execution_index"]],
        }
    else:
        if any((row[2]["phase"], row[2]["token_bucket"])
               != (shape["phase"], shape["token_bucket"]) for row in rows):
            raise ValueError("bounded capture included a different phase or compiled bucket")
        if len(rows) != request["forwards"]:
            raise ValueError(f"captured {len(rows)} forwards, expected {request['forwards']}")
        selection = None
    return {row[1] for row in rows}, selection


def _capture_sessions(root: Path, graphs: dict, shape: dict, request: dict) -> list[dict]:
    from profiling.runners.neuron.parquet_columns import read_rows
    from profiling.runners.neuron.vllm_segment_trace import export_device_sessions

    directories = export_device_sessions(
        root / "device-profiles", root / "device-parquet", include_ops=True
    )
    sessions = []
    for directory in directories:
        match = re.search(r"_nc_(\d+)_session_", directory.name)
        if match is None or int(match[1]) not in request["ranks"]:
            raise ValueError(f"exported session has no requested rank identity: {directory}")
        rank = int(match[1])
        split_tables = [
            path.name
            for path in directory.iterdir()
            if re.fullmatch(r"(?:ExecutionInfo|Warning)_\d+\.parquet", path.name)
        ]
        if split_tables:
            raise ValueError(
                f"capture validation does not support split metadata tables: {split_tables}"
            )
        warnings = (
            read_rows(directory / "Warning.parquet")
            if (directory / "Warning.parquet").exists()
            else []
        )
        fatal = [
            row["message"]
            for row in warnings
            if re.search(r"drop(?:ped|ping)?|lost|overflow|truncat", row["message"], re.I)
            and not row["message"].startswith("Schema validation: Unexpected key 'subgraph'")
        ]
        if fatal:
            raise ValueError(f"per-op trace lost notifications: {fatal}")
        executions = read_rows(directory / "ExecutionInfo.parquet")
        traced, selection = _select_executions(executions, graphs, shape, request)
        neffs = []
        for graph_id in sorted(traced):
            neff = root / "device-profiles" / "neffs" / graph_id / f"graph_{graph_id}.neff"
            expected = graphs[graph_id]["binary"]["neff_sha256"]
            if sha256(neff) != expected:
                raise ValueError("captured NEFF differs from numerically sealed stock bytes")
            static_path = root / f"static-{graph_id}.json"
            if not static_path.exists():
                from profiling.profilers.neuron_ops_labels import export_static_map

                _save(static_path, export_static_map(neff))
            neffs.append(
                {
                    "graph_id": graph_id,
                    "path": str(neff),
                    "sha256": expected,
                    "phase": graphs[graph_id]["phase"],
                    "token_bucket": graphs[graph_id]["token_bucket"],
                    "static_instructions": str(static_path),
                    "static_instructions_sha256": sha256(static_path),
                }
            )
        session = {"rank": rank, "session_dir": str(directory), "neffs": neffs}
        if selection is not None:
            session["execution_selection"] = selection
        sessions.append(session)
    if sorted(row["rank"] for row in sessions) != sorted(request["ranks"]):
        raise ValueError("exported sessions do not cover every requested rank exactly once")
    return sessions


def _debug_labels(root: Path, ops_plan: dict, sessions: list[dict], source_identity: dict) -> None:
    """Compile labels separately, then transfer only verified instruction identities."""
    from profiling.profilers.neuron_ops_labels import export_static_map, transfer_debug_labels
    from profiling.runners.neuron.vllm_forward_trace import plan_geometry

    label_root = root / "debug-labels"
    label_root.mkdir()
    _save(label_root / "ops-plan.json", ops_plan)
    cache = label_root / "cache"
    for name in ("tmp", "artifacts", "nki-cache"):
        (label_root / name).mkdir()
    _run_ops_stage(
        label_root,
        "labels",
        env={
            "VLLM_CACHE_ROOT": str(cache),
            "XDG_CACHE_HOME": str(cache),
            "NEURON_LIBTORCH_CACHE_ROOT": str(cache),
            "NKI_COMPILE_CACHE_URL": str(label_root / "nki-cache"),
            "TMPDIR": str(label_root / "tmp"),
            "NKI_ARTIFACTS_DIR": str(label_root / "artifacts"),
            "XLA_IR_DEBUG": "1",
            "XLA_HLO_DEBUG": "1",
            "NEURON_FRAMEWORK_DEBUG": "1",
            "NKI_DEBUG_INFO": "1",
        },
    )
    snapshot = json.loads((label_root / "label-compile-cache.json").read_text())
    wanted = (ops_plan["shape"]["phase"], ops_plan["shape"]["token_bucket"])
    matches = []
    geometry_plan = {"context": ops_plan["shape"]["max_model_len"], "specs": [ops_plan["shape"]]}
    for graph_id, binary in snapshot["graphs"].items():
        try:
            geometry = plan_geometry(binary["input_metadata"], geometry_plan)
        except ValueError:
            continue
        if geometry == wanted:
            matches.append((graph_id, binary))
    if len(matches) != 1:
        raise ValueError("debug-label compile has ambiguous or missing selected graph geometry")
    graph_id, binary = matches[0]
    neff = Path(snapshot["compile_cache_dir"]) / graph_id / f"graph_{graph_id}.neff"
    if sha256(neff) != binary["neff_sha256"]:
        raise ValueError("debug NEFF changed after its checked public execution")
    debug_map = export_static_map(neff)
    _save(label_root / "static-debug.json", debug_map)
    paths = set()
    for session in sessions:
        for entry in session["neffs"]:
            if (entry["phase"], entry["token_bucket"]) != wanted:
                continue  # Guard executions keep their exact stock-only static map.
            path = Path(entry["static_instructions"])
            if path not in paths:
                stock_map = json.loads(path.read_text())
                mapped = transfer_debug_labels(
                    stock_map,
                    debug_map,
                    source_identity=source_identity,
                    expected_stock_neff_sha256=entry["sha256"],
                    expected_debug_neff_sha256=binary["neff_sha256"],
                )
                mapped_path = root / f"mapped-{entry['graph_id']}.json"
                _save(mapped_path, mapped)
                paths.add(path)
            entry["static_instructions"] = str(root / f"mapped-{entry['graph_id']}.json")
            entry["static_instructions_sha256"] = sha256(Path(entry["static_instructions"]))


def capture_ops(shape: dict, request: dict) -> dict:
    from profiling.exec.neuron import current_neuron_name
    from profiling.profilers.neuron_ops_labels import collect_source_identity
    from profiling.runners.neuron.vllm_forward_engine import engine_config

    validate_whole_context_shape(**shape)
    cache = Path(os.environ["SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR"])
    model = Path(os.environ["SERVINGSTUDIO_VLLM_NEURON_MODEL_DIR"])
    identity = verify_identity(model)
    root = cache / "ops-runs" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        validated, receipt, plan = _validation(
            root, shape, cache, model, identity, request.get("validated_run")
        )
        cases = json.loads((validated / "cases.json").read_text())
        target = precision_shape(shape, shape["max_model_len"])
        case = next(case for case in cases if case["shape_id"] == target)
        if shape["phase"] == "prefill":
            request = {**request, "guard_decode_bucket": case["bucket"]}
        # A validated cache may have been mounted at another path. Checkpoint
        # bytes were verified above; only the local pathname is translated.
        profile_config = engine_config({**plan, "model": str(model)}, cases, root, "profile")
        ops_plan = {
            "shape": shape,
            "request": request,
            "validated_run": str(validated),
            "case": case,
            "profile_config": profile_config,
        }
        _save(root / "ops-plan.json", ops_plan)
        _run_ops_stage(
            root,
            "capture",
            env={
                "NEURON_RT_PROFILE_BUF_INSTRUCTION_MB": str(request["instruction_buffer_mb"]),
                "NEURON_RT_PROFILE_BUF_EVENT_MB": "64",
                "NEURON_RT_PROFILE_BUF_DMA_MB": str(request["dma_buffer_mb"]),
            },
        )
        control_receipt = None
        if shape["phase"] == "prefill":
            from profiling.runners.neuron.vllm_ops_control import collect_control_receipt

            control_receipt = collect_control_receipt(root, shape, request["ranks"])
        graphs = receipt["graphs"]
        sessions = _capture_sessions(root, graphs, shape, request)
        source_identity = collect_source_identity()
        if request["debug_labels"]:
            _debug_labels(root, ops_plan, sessions, source_identity)
        after = verify_cache_binaries(stock_compile_cache_dir(), graphs)
        _save(
            root / "identity-gate.json",
            {"passed": True, "graphs": after, "binary_receipt_sha256": receipt["sha256"]},
        )
        profiler_window = json.loads((root / "capture-config.json").read_text())["profiler_config"]
        manifest = {
            "schema_version": 1,
            "producer": "neuron_ops",
            "kernel": {"kind": "neuron_llama_forward", "backend": "vllm_neuron"},
            "shape": shape,
            "sampling": {
                "window": ("geometry_drained_prefill_decode_v1"
                           if shape["phase"] == "prefill" else "steady_decode"),
                "canonical_requests": len(case["prompts"]),
                "requested_forwards": request["forwards"],
                "guard_forwards": 1 if shape["phase"] == "prefill" else 0,
                "worker_delay_iterations": profiler_window["delay_iterations"],
                "profiler_max_iterations": profiler_window["max_iterations"],
            },
            "sessions": sessions,
            "semantic_source_identity": source_identity,
            "hardware": {
                "name": current_neuron_name(),
                "captured_ranks": request["ranks"],
                "physical_cores_per_rank": [0, 1],
                "gpu_count": 1,
            },
            "execution": {
                "capture_root": str(root),
                "validated_run": str(validated),
                "accuracy_receipt": str(validated / "precision.json"),
                "accuracy_receipt_sha256": sha256(validated / "precision.json"),
                "binary_receipt": receipt["path"],
                "binary_receipt_sha256": receipt["sha256"],
                "identity_gate": str(root / "identity-gate.json"),
                "identity_gate_sha256": sha256(root / "identity-gate.json"),
            },
            "provenance": {
                "versions": identity["versions"],
                "source_hashes": receipt["load"]["source"],
                "capture_config_sha256": sha256(root / "capture-config.json"),
            },
        }
        if control_receipt is not None:
            manifest["execution"].update(
                capture_control_receipt=str(control_receipt),
                capture_control_receipt_sha256=sha256(control_receipt),
            )
            manifest["sampling"]["synchronization"] = (
                "previous_output_before_start; decode_output_before_stop"
            )
        _save(root / "capture-manifest.json", manifest)
        return {"manifest_path": str(root / "capture-manifest.json")}
    except Exception as error:
        _save(root / "failure.json", {"error": str(error)})
        raise RuntimeError(f"{error}; capture evidence: {root}") from error
