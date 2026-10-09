"""Shared stock HTTP replay, host whole-chip lease, and resumable native evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from alignment.load_generator import runner as frontend
from alignment.neuron.runner import check_port
from alignment.neuron.vllm_config import VllmNeuronProfileConfig
from alignment.neuron.vllm_corpus import verify_corpus
from alignment.neuron.vllm_normalize import verify_no_event_drops, write_normalized
from alignment.neuron.vllm_records import STOCK_VERSIONS, token_hash
from alignment.neuron.vllm_server import (
    SERVED_MODEL,
    docker_environment,
    docker_prefix,
    export_command,
    server_command,
)
from alignment.profiler.engine_records import VLLM_NEURON_RECORDS
from alignment.profiler.record_extraction import (
    extract_metrics_jsonl,
    extract_request_timings_jsonl,
)
from profiling.exec.neuron import LocalNeuronPool
from profiling.runners.neuron import vllm_identity
from profiling.runners.neuron.vllm_forward_trace import model_geometry
from profiling.runners.neuron.vllm_identity import (
    MODEL_HASHES,
    load_accuracy_binaries,
    verify_cache_binaries,
    verify_checkpoint,
    warm_loader_coverage,
)


def _save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _http(base: str, path: str, *, post: bool = False) -> dict:
    request = urllib.request.Request(
        base + path, data=b"{}" if post else None, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    # Stock /health and profiler endpoints may return empty successful responses.
    return json.loads(content) if content else {}


def _tagged(path: Path, tag: str) -> list[dict]:
    # The stock multiprocess logger may concatenate complete worker records on
    # one line. Decode one object after every tag, without swallowing later tags.
    text = path.read_text()
    decoder = json.JSONDecoder()
    rows = []
    for match in re.finditer(re.escape(tag + " "), text):
        row, _ = decoder.raw_decode(text[match.end() :].lstrip())
        if not isinstance(row, dict):
            raise ValueError("tagged stock observation is not a JSON object")
        rows.append(row)
    return rows


def source_provenance(config: VllmNeuronProfileConfig) -> dict:
    """Record the exact observer/client/model configuration used by this launch."""
    source = Path(__file__).parent
    model_config = Path(config.server.model_path) / "config.json"
    cfg = json.loads(model_config.read_text())
    expected = {
        "model_type": "llama",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("stock alignment requires the full Llama3.1-8B checkpoint configuration")
    binary = Path(config.server.req_frontend_binary)
    frontend_source = binary.parents[2] / "src"
    files = [
        *source.parent.rglob("*.py"),
        Path(vllm_identity.__file__),
        model_config,
        binary,
        frontend_source / "tokens.rs",
        frontend_source / "executor/independent.rs",
        frontend_source / "record.rs",
        frontend_source / "timeline.rs",
        frontend_source / "timeline/writer.rs",
    ]
    return {
        "image": config.server.image,
        "request_timing_schema": 2,
        "model_config": cfg,
        "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "checkpoint_path": config.server.model_path,
        "checkpoint_value_authority": (
            "configuration only; fresh captures separately persist streamed checkpoint hashes"
        ),
    }


def accepted_forward_evidence(path: Path) -> dict:
    """Bind existing numerical validation to its actual compiled graphs and corpus.

    This establishes graph provenance. The HTTP timing client does not persist
    generated IDs or logits, so it cannot independently repeat that validation.
    """
    binaries = load_accuracy_binaries(path)
    names = ["precision.json", "accuracy-outputs.json"]
    for name in ("execution-records.json", "profiled-outputs.json"):
        if (path / name).exists():
            names.append(name)
        elif not binaries["available"]:
            raise ValueError("legacy accepted forward artifact lacks its profile evidence")
    data = {name: json.loads((path / name).read_text()) for name in names}
    precision = data["precision.json"]
    if precision.get("passed") is not True or not precision.get("by_shape"):
        raise ValueError("declared stock forward artifact did not pass its numerical criterion")
    graphs = {}
    for row in data.get("execution-records.json", []):
        key = f"{row['phase']}:{row['token_bucket']}"
        if key in graphs and graphs[key] != row["model_hash"]:
            raise ValueError("accepted artifact has ambiguous graph hashes")
        graphs[key] = row["model_hash"]
    if binaries["available"]:
        binary_graphs = {
            f"{row['phase']}:{row['token_bucket']}": key
            for key, row in binaries["graphs"].items()
        }
        if not graphs:
            graphs = binary_graphs  # Fresh accuracy-only validation needs no timing rerun.
        if binary_graphs != graphs or len(binaries["graphs"]) != len(graphs):
            raise ValueError("numerical binary receipt differs from accepted execution graphs")
    if set(graphs) != {"prefill:512", "decode:1", "decode:16"}:
        raise ValueError("accepted artifact lacks the configured stock graphs")
    prompts = {}
    for row in data["accuracy-outputs.json"]:
        prompts.setdefault(token_hash(row["prompt_ids"]), []).append(row["token_ids"])
    return {
        "path": str(path),
        "files": {
            str(path / n): hashlib.sha256((path / n).read_bytes()).hexdigest() for n in names
        },
        "graphs": graphs,
        "criterion": precision["criterion"],
        "precision_passed": True,
        "binary_provenance": binaries,
        "subject_continuations_by_prompt_sha256": prompts,
        "http_generated_id_comparison": {
            "available": False,
            "reason": (
                "req-frontend replay and timeline persist token counts/times, "
                "not generated token IDs"
            ),
            "http_full_logit_validation": False,
        },
    }


def compare_accepted_graphs(model_info: dict, evidence: dict) -> dict:
    mismatches = {}
    for graph_hash, (phase, bucket) in model_info.items():
        expected = evidence["graphs"].get(f"{phase}:{bucket}")
        if graph_hash != expected:
            mismatches[graph_hash] = {"geometry": [phase, bucket], "expected": expected}
    return {
        "passed": not mismatches,
        "observed_graphs": model_info,
        "mismatches": mismatches,
        "authority": evidence["path"],
        "scope": "graph_keys_only; binary comparison is a separate capture receipt",
        "http_full_logit_validation": False,
    }


def capture_binary_snapshot(config: VllmNeuronProfileConfig, evidence: dict) -> dict:
    binaries = evidence.get("binary_provenance", {})
    if binaries.get("available") is not True:
        raise ValueError(
            "fresh HTTP capture requires producer-recorded binary-provenance.json; "
            "rerun canonical stock forward accuracy/reference with the unchanged cache"
        )
    return verify_cache_binaries(
        Path(config.server.cache_path) / "cache/neuron/compile_cache", binaries["graphs"]
    )


def captured_binary_provenance(root: Path, evidence: dict) -> dict:
    """Preserve the weaker historical scope when acquisition had no byte receipts."""
    paths = [root / name for name in (
        "capture-binaries-before.json", "capture-binaries-after.json", "checkpoint-provenance.json",
    )]
    if not any(path.exists() for path in paths):
        return {"available": False, "scope": "legacy_graph_keys_only"}
    if not all(path.exists() for path in paths):
        raise ValueError("incomplete HTTP binary/checkpoint provenance")
    binaries = evidence.get("binary_provenance", {})
    if binaries.get("available") is not True:
        raise ValueError("HTTP binary receipt lacks producer-recorded numerical authority")
    before, after, checkpoint = [json.loads(path.read_text()) for path in paths]
    expected = {key: row["binary"] for key, row in binaries["graphs"].items()}
    if before != expected or after != expected or checkpoint.get("model_sha256") != MODEL_HASHES:
        raise ValueError("HTTP binary/checkpoint bytes differ from numerical validation")
    coverage = warm_loader_coverage((root / "server.log").read_text(), expected)
    if not coverage["complete"]:
        raise ValueError("HTTP server lacks all four rank warm Executor loader receipts")
    return {
        "available": True,
        "scope": "cache_neff_bytes_stable_across_fresh_public_server_lifetime",
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "numerical_binary_receipt_sha256": binaries["sha256"],
        "loader_coverage": coverage,
        "checkpoint_mounted_read_only": True,
        "runtime_model_name_suffix_decoded": False,
        "http_full_logit_validation": False,
    }


def verify_server_log(text: str) -> None:
    if "WorkerProc hit an exception" in text or "Traceback (most recent call last):" in text:
        raise ValueError("stock server/observer exception; capture cannot supply timing evidence")


def validate_request_mapping(iterations: list[dict], replay_rows: list[dict]) -> dict:
    """Enforce the bounded 16x(504+8) corpus and exact engine/client ID mapping."""
    successful = {}
    for row in replay_rows:
        outcome = row["outcome"]
        rid = outcome["request_id"]
        usage = outcome.get("server_usage") or {}
        if (
            outcome["status"] != "SUCCESS"
            or rid in successful
            or outcome["output_len_actual"] != 8
            or usage.get("prompt_tokens") != 504
            or usage.get("completion_tokens") != 8
            or usage.get("cached_prompt_tokens") != 0
            or outcome.get("first_token_id_ms") is None
            or outcome.get("token_delivery_tpot_ms") is None
        ):
            raise ValueError(
                "stock client replay must contain unique uncached 504+8 token requests"
            )
        successful[rid] = outcome
    if len(successful) != 16:
        raise ValueError("initial stock capture requires exactly 16 successful requests")
    mapping = {}
    progress = {}
    for metric in sorted(iterations, key=lambda r: r["iteration_index"]):
        for req in metric["requests"]:
            rid = req["request_id"]
            client = VLLM_NEURON_RECORDS.unwrap_request_id(rid)
            if client not in successful:
                raise ValueError("scheduler request does not map to a successful client request")
            if client in mapping and mapping[client] != rid:
                raise ValueError("multiple engine requests map to one client request")
            mapping[client] = rid
            expected = progress.get(client, 0)
            q = req["q_tokens"]
            if (
                req["kv_len_before"] != expected
                or req["prompt_tokens"] != 504
                or q != (504 if expected == 0 else 1)
            ):
                raise ValueError("scheduler geometry has missing/repeated request progress")
            progress[client] = expected + q
    if set(mapping) != set(successful) or set(progress.values()) != {511}:
        raise ValueError("scheduler does not cover one prefill and seven decodes per request")
    return {
        "client_to_engine": mapping,
        "scheduled_tokens_by_request": progress,
        "request_count": 16,
        "prompt_tokens": 504,
        "output_tokens": 8,
        "client_timing_authority": "req-frontend generated-token SSE arrivals",
    }


def extract_stock_request_timings(
    measurement: Path, root: Path, mapping: dict, *, required: bool
) -> Path | None:
    """Old receipts remain explicitly unavailable; new captures require every request."""
    if "VibeSimAlignmentRequestTiming " not in measurement.read_text():
        if required:
            raise ValueError("stock capture is missing required EngineCore request timings")
        return None
    path = root / "request_timings.jsonl"
    extract_request_timings_jsonl(
        measurement,
        path,
        expected_request_ids=set(mapping["client_to_engine"]),
        records=VLLM_NEURON_RECORDS,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if any(row["num_output_tokens"] != mapping["output_tokens"] for row in rows):
        raise ValueError("EngineCore emitted token count differs from successful client evidence")
    return path


def postprocess(config: VllmNeuronProfileConfig) -> dict:
    root = Path(config.log_dir).resolve()
    verify_server_log((root / "server.log").read_text())
    drive = json.loads((root / "drive_summary.json").read_text())
    measurement = root / "measurement.log"
    measurement.write_bytes((root / "server.log").read_bytes()[drive["measurement_log_offset"] :])
    metrics_path = root / "metrics.jsonl"
    extract_metrics_jsonl(measurement, metrics_path, records=VLLM_NEURON_RECORDS)
    iterations = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    replay_rows = [json.loads(line) for line in Path(drive["log_path"]).read_text().splitlines()]
    mapping = validate_request_mapping(iterations, replay_rows)
    corpus = json.loads((root / "corpus-provenance.json").read_text())
    evidence = json.loads((root / "accepted-forward-provenance.json").read_text())
    if not set(corpus["prompt_hashes"].values()) <= set(
        evidence["subject_continuations_by_prompt_sha256"]
    ):
        raise ValueError("HTTP corpus differs from accepted forward subjects")
    frozen = json.loads((root / "source-provenance.json").read_text())
    request_timings = extract_stock_request_timings(
        measurement, root, mapping, required=frozen.get("request_timing_schema") == 2
    )
    after = json.loads((root / "capture-source-after.json").read_text())
    if after["files"] != frozen["files"]:
        raise ValueError("imported stock alignment source changed during capture")
    # Parsing can be resumed after a parser-only fix without rerunning hardware.
    # Keep its code authority separate from the before/after capture snapshot.
    current = source_provenance(config)
    for key in ("image", "model_config", "checkpoint_path"):
        if current[key] != frozen[key]:
            raise ValueError("stock postprocessing configuration differs from the capture")
    _save(root / "postprocess-source-provenance.json", current)
    current_evidence = accepted_forward_evidence(Path(config.server.accepted_forward_path))
    # Old acquisitions did not record binary_provenance. Compare their exact
    # original fields without retroactively adding proof from today's cache.
    if evidence != {key: current_evidence.get(key) for key in evidence}:
        raise ValueError("accepted forward evidence changed during capture")
    binary_proof = captured_binary_provenance(root, evidence)
    prompt_rows = _tagged(measurement, "VllmNeuronPrompt")
    actual_hashes = {}
    for row in prompt_rows:
        rid = VLLM_NEURON_RECORDS.unwrap_request_id(row["request_id"])
        if rid in actual_hashes or row["prompt_tokens"] != 504:
            raise ValueError("duplicate or malformed observed stock prompt")
        actual_hashes[rid] = row["prompt_sha256"]
    if actual_hashes != corpus["prompt_hashes"]:
        raise ValueError("observed stock prompt IDs differ from the frozen corpus")
    _save(root / "request-map.json", mapping)
    workers = _tagged(root / "server.log", "VllmNeuronWorker")
    if (
        len(workers) != 4
        or {r["rank"] for r in workers} != set(range(4))
        or len({r["pid"] for r in workers}) != 4
        or len({r["async_scheduling"] for r in workers}) != 1
        or any(r["versions"] != STOCK_VERSIONS for r in workers)
    ):
        raise ValueError("stock capture lacks unique TP4 worker/async provenance")
    _save(root / "runtime-provenance.json", workers)
    result = {
        "producer_kind": "framework_capture",
        "engine": "vllm_neuron",
        "profile_kind": config.profile_kind,
        "log_dir": str(root),
        "gpu": config.gpu,
        "server_tp_size": 4,
        "server_dp_size": 1,
        "metrics_jsonl": str(metrics_path),
        "request_timings_jsonl": str(request_timings) if request_timings else None,
        "server_log": str(root / "server.log"),
        "replay_result": drive["log_path"],
        "drive_summary": drive,
        "request_map": str(root / "request-map.json"),
        "runtime_provenance": str(root / "runtime-provenance.json"),
        "accepted_forward_provenance": str(root / "accepted-forward-provenance.json"),
        "binary_provenance": binary_proof,
        "http_generated_id_comparison": evidence["http_generated_id_comparison"],
        "compiled_shapes": {
            "context": 512,
            "token_buckets": [1, 16],
            "kv_blocks": 6782,
            "block_size": 32,
        },
    }
    if config.profile_kind == "neuron":
        trace = json.loads((root / "system-trace.json").read_text())
        drops = verify_no_event_drops(
            trace, {n: (root / n).read_text() for n in ("server.log", "export.log")}
        )
        _save(root / "native-event-completeness.json", drops)
        records = {
            "engine": "vllm_neuron",
            "producer_kind": "framework_capture",
            "tp_size": 4,
            "iterations": iterations,
            "forwards": _tagged(measurement, "VllmNeuronForward"),
        }
        _save(root / "forward-records.json", records)
        metadata_path = root / "model-metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        else:
            trace = json.loads((root / "system-trace.json").read_text())
            hashes = {
                re.search(r"/compile_cache/([0-9a-f]+)/", e["model_name"])[1]
                for e in trace["trace_event"]
                if e["name"] == "nc_exec_running"
            }
            metadata = {}
            for graph_hash in hashes:
                path = (
                    Path(config.server.cache_path)
                    / "cache/neuron/compile_cache"
                    / graph_hash
                    / "example_inputs.txt"
                )
                text = path.read_text()
                phase, bucket = model_geometry(text, 512)
                metadata[graph_hash] = {
                    "phase": phase,
                    "token_bucket": bucket,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "text": text,
                }
            _save(metadata_path, metadata)
        model_info = {}
        for key, row in metadata.items():
            if hashlib.sha256(row["text"].encode()).hexdigest() != row["sha256"]:
                raise ValueError("saved native graph metadata text hash changed")
            geometry = model_geometry(row["text"], 512)
            if geometry != (row["phase"], row["token_bucket"]):
                raise ValueError("saved native graph metadata geometry changed")
            model_info[key] = geometry
        graph_match = compare_accepted_graphs(model_info, evidence)
        _save(root / "accepted-graph-comparison.json", graph_match)
        if not graph_match["passed"]:
            raise ValueError("HTTP native graph differs from the accepted forward artifact")
        result.update(write_normalized(root, records, model_info))
        result.update(
            raw_trace=str(root / "system-trace.json"),
            forward_records=str(root / "forward-records.json"),
        )
    _save(root / "profile_result.json", result)
    return result


def run_profile(config: VllmNeuronProfileConfig, *, resume: bool = False) -> dict:
    config.validate()
    root = Path(config.log_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if resume:
        return postprocess(config)
    if (root / "server.log").exists():
        raise FileExistsError("capture already exists; use --resume or a fresh log_dir")
    check_port(config.server.host, config.server.port)
    config.server.container_env().validate()
    prepared = frontend.prepare_replay(config.workload, root)
    _save(
        root / "corpus-provenance.json",
        verify_corpus(config.workload, Path(config.server.model_path)),
    )
    _save(root / "source-provenance.json", source_provenance(config))
    evidence = accepted_forward_evidence(Path(config.server.accepted_forward_path))
    _save(root / "accepted-forward-provenance.json", evidence)
    binaries_before = capture_binary_snapshot(config, evidence)
    _save(root / "capture-binaries-before.json", binaries_before)
    # One streaming read of the eight pinned files, before public server startup.
    # The model is mounted read-only into every stock worker.
    _save(root / "checkpoint-provenance.json", {
        "model_sha256": verify_checkpoint(Path(config.server.model_path)),
        "checkpoint_path": config.server.model_path,
    })
    env = docker_environment()
    reservation = next(LocalNeuronPool([config.server.neuron_device]).acquire_chunks(1))
    container = "servingstudio-vllm-neuron-" + uuid.uuid4().hex[:12]
    try:
        command = server_command(config, reservation.device, container)
        _save(
            root / "launch.json",
            {
                "command": command,
                "image": config.server.image,
                "engine": "vllm_neuron",
                "reservation": {
                    "device_id": reservation.device.device_id,
                    "core_ids": reservation.device.core_ids,
                    "lnc": reservation.device.lnc,
                },
                "http_binding": "loopback; host networking",
                "source_paths": [str(Path(__file__).parent)],
            },
        )
        base = f"http://{config.server.host}:{config.server.port}"
        boundary, active = {}, False
        with (root / "server.log").open("w") as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + config.server.startup_timeout
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"stock server exited {proc.returncode}; inspect server.log"
                        )
                    try:
                        _http(base, "/health")
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("stock HTTP readiness timed out")
                        time.sleep(0.2)

                def ready():
                    nonlocal active
                    if _http(base, "/load")["server_load"] != 0:
                        raise RuntimeError("stock preflight requests have not drained")
                    boundary["measurement_log_offset"] = (root / "server.log").stat().st_size
                    if config.profile_kind == "neuron":
                        _http(base, "/start_profile", post=True)
                        active = True
                    boundary["replay_start_monotonic_ns"] = time.monotonic_ns()

                drive = frontend.run_replay(
                    config.workload,
                    prepared,
                    base_url=base,
                    model=SERVED_MODEL,
                    measurement_ready=ready,
                    session_runner=Path(config.server.req_frontend_binary),
                    timeline_path=root / "load_generator/timeline.parquet",
                )
                boundary["replay_end_monotonic_ns"] = time.monotonic_ns()
                if _http(base, "/load")["server_load"] != 0:
                    raise RuntimeError("stock measured requests have not drained")
                if active:
                    _http(base, "/stop_profile", post=True)
                    active = False
                _save(root / "drive_summary.json", {**drive, **boundary})
            finally:
                if active and proc.poll() is None:
                    try:
                        _http(base, "/stop_profile", post=True)
                    except (urllib.error.URLError, TimeoutError):
                        pass
                # Stop via the declared private daemon before releasing the host chip lease.
                stopped = subprocess.run(
                    [*docker_prefix(config), "stop", "--time", "30", container],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if proc.poll() is None and stopped.returncode:
                    # A failed graceful stop must not leave the chip occupied.
                    killed = subprocess.run(
                        [*docker_prefix(config), "kill", container],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if killed.returncode and proc.poll() is None:
                        raise RuntimeError(f"cannot stop reserved stock server: {stopped.stderr}")
                proc.wait(timeout=60)
    finally:
        reservation.lock.close()
    _save(root / "capture-binaries-after.json", capture_binary_snapshot(config, evidence))
    after = source_provenance(config)
    _save(root / "capture-source-after.json", after)
    if json.loads((root / "source-provenance.json").read_text()) != after:
        raise ValueError("imported stock alignment source changed during capture")
    if config.profile_kind == "neuron":
        export = export_command(config)
        _save(root / "export-command.json", export)
        with (root / "export.log").open("w") as log:
            subprocess.run(export, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return postprocess(config)
