"""Native evidence reduction, strict dispatch, and actual token SSE contracts."""

import copy
import http.client
import json
import threading
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from alignment.neuron import runner as neuron_runner
from alignment.neuron.normalize import iteration_metric, normalize, write_normalized
from alignment.neuron.server import CompletionsHandler, validate_request
from alignment.nsys.parsed_io import read_parsed
from alignment.timing_predict_input.builder import BuildRequest, EngineTextInputSpec, build_inputs
from launcher.alignment_config import load_profile_config


def capture():
    row = {
        "iteration_id": 0,
        "request_id": "a",
        "phase": "prefill",
        "q_tokens": 6,
        "kv_len_before": 0,
        "start_realtime_ns": 100,
        "stop_realtime_ns": 140,
        "start_monotonic_ns": 1000,
        "stop_monotonic_ns": 1040,
    }
    events = []

    def pair(kind, track, start, end, data):
        events.extend(
            [
                {
                    "event_type": kind,
                    "nc_idx": 0,
                    "tracking_id": track,
                    "phase": "start",
                    "timestamp_ns": start,
                    "data": data,
                },
                {
                    "event_type": kind,
                    "nc_idx": 0,
                    "tracking_id": track,
                    "phase": "stop",
                    "timestamp_ns": end,
                    "data": {},
                },
            ]
        )

    pair(
        "nrt_execute",
        1,
        105,
        135,
        {"model_id": 10, "model_name": "context_encoding_model/whole.neff"},
    )
    pair(
        "kbl_exec_pre",
        2,
        106,
        107,
        {"exec_id": 23, "model_id": 10, "model_name": "context_encoding_model/whole.neff"},
    )
    pair("nc_exec_running", 3, 110, 120, {"exec_id": 23, "device_core_idx": 4})
    pair("nc_exec_running", 4, 111, 121, {"exec_id": 23, "device_core_idx": 5})
    records = {
        "engine": "nxdi",
        "producer_kind": "framework_capture",
        "iterations": [row],
        "config": {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 128256,
            "model_type": "llama",
            "neuron_config": {
                "layer_boundary_markers": False,
                "context_encoding_buckets": [128],
                "token_generation_buckets": [512],
                "logical_nc_config": 2,
                "tp_degree": 1,
                "batch_size": 1,
                "seq_len": 512,
                "torch_dtype": "bfloat16",
            },
        },
    }
    return {"data_version": 2, "events": events}, records


def test_two_physical_cores_are_one_logical_busy_union():
    trace, records = capture()
    parsed = normalize(trace, records)
    row = parsed["iteration_details"][0]
    kernels = row["ranges"][0]["kernels"]
    assert len(kernels) == 1
    assert kernels[0]["end_ns"] - kernels[0]["start_ns"] == 11
    assert parsed["device_ids"] == [0] and parsed["physical_core_ids"] == [4, 5]
    assert row["metrics"]["prefill_chunk_pairs"] == [[0, 6]]
    assert row["metrics"]["compiled_shapes"]["query_tokens"] == 128
    assert "sqlite" not in parsed and "nsys_profiler" not in parsed


def test_disjoint_native_intervals_keep_idle_gap():
    trace, records = capture()
    trace["events"][-2]["timestamp_ns"] = 125
    trace["events"][-1]["timestamp_ns"] = 130
    kernels = normalize(trace, records)["iteration_details"][0]["ranges"][0]["kernels"]
    assert [(r["start_ns"], r["end_ns"]) for r in kernels] == [(110, 120), (125, 130)]
    assert sum(r["end_ns"] - r["start_ns"] for r in kernels) == 15


@pytest.mark.parametrize(
    "defect",
    [
        "missing_core",
        "unmatched_start",
        "unknown_version",
        "duplicate_iteration",
        "wrong_producer",
        "marker_path",
        "clock",
        "wrong_model_shape",
        "wrong_dtype",
        "wrong_compiled_phase",
    ],
)
def test_native_capture_defects_fail_closed(defect):
    trace, records = capture()
    if defect == "missing_core":
        trace["events"] = trace["events"][:-2]
    elif defect == "unmatched_start":
        trace["events"] = trace["events"][:-1]
    elif defect == "unknown_version":
        trace["data_version"] = 9
    elif defect == "duplicate_iteration":
        records["iterations"].append(copy.deepcopy(records["iterations"][0]))
    elif defect == "wrong_producer":
        records.pop("producer_kind")
    elif defect == "marker_path":
        records["config"]["neuron_config"]["layer_boundary_markers"] = True
    elif defect == "clock":
        records["iterations"][0]["stop_monotonic_ns"] = 999
    elif defect == "wrong_model_shape":
        records["config"]["hidden_size"] = 2048
    elif defect == "wrong_dtype":
        records["config"]["neuron_config"]["torch_dtype"] = "float16"
    elif defect == "wrong_compiled_phase":
        records["iterations"][0].update(phase="decode", q_tokens=1, kv_len_before=6)
    with pytest.raises(ValueError):
        normalize(trace, records)


def test_native_parse_and_inventory_feed_shared_predictor(tmp_path):
    trace, records = capture()
    (tmp_path / "system-trace.json").write_text(json.dumps(trace))
    (tmp_path / "forward-records.json").write_text(json.dumps(records))
    result = write_normalized(
        tmp_path, tmp_path / "system-trace.json", tmp_path / "forward-records.json"
    )
    parsed = read_parsed(tmp_path / "parsed.json")
    assert parsed["schema_version"] == 6
    assert len(parsed["iteration_details"][0]["ranges"][0]["kernels"]) == 1
    from pathlib import Path

    request = BuildRequest(
        simulation_preset=tmp_path / "sim.yaml",
        profile_log_dir=tmp_path,
        parsed_nsys=Path(result["parsed_trace"]),
        output_dir=tmp_path / "prediction",
        gpu="AWS Trainium2 LNC2",
        arch={"type": "llama3_nxdi", "kv_capacity": 512},
        backends={},
        input_spec=EngineTextInputSpec(),
    )
    built = build_inputs(request)
    assert json.loads(built.cases.read_text()) == [
        {"groups": [{"prefill_chunk_pairs": [[0, 6]], "decode_kv_lens": []}]}
    ]
    manifest = json.loads(built.input_manifest.read_text())
    assert "parsed_trace" in manifest and "parsed_nsys" not in manifest
    with pytest.raises(ValueError, match="whole-forward"):
        from dataclasses import replace

        build_inputs(replace(request, arch={"type": "llama3_neuron"}))


def test_decode_shapes_are_logical_not_bucket_lengths():
    _, records = capture()
    row = records["iterations"][0]
    row.update(phase="decode", q_tokens=1, kv_len_before=6)
    metric = iteration_metric(row)
    assert metric["decode_kv_lens"] == [7]
    assert metric["compiled_shapes"]["kv_bucket"] == 512


def profile_document(tmp_path):
    return {
        "schema_version": 1,
        "name": "native",
        "log_dir": "./profile",
        "gpu": "AWS Trainium2 LNC2",
        "engine": "nxdi",
        "profile_kind": "neuron",
        "fork_python": "/usr/bin/python3",
        "server": {"model_path": "./model", "compiled_path": "./compiled"},
        "workload": {
            "frontend": {"type": "independent", "path": "./trace.csv"},
            "text_file": "./prompts.txt",
            "tokenizer": "local",
            "max_concurrency": 1,
        },
    }


def test_nxdi_config_has_separate_strict_device_contract(tmp_path):
    path = tmp_path / "profile.json"
    raw = profile_document(tmp_path)
    path.write_text(json.dumps(raw))
    config = load_profile_config(path)
    assert config.server.compiled_path == str(tmp_path / "compiled")
    assert config.engine == "nxdi" and config.profile_kind == "neuron"
    raw["cuda_visible_devices"] = "0,1"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="invalid NxDI"):
        load_profile_config(path)


@pytest.mark.parametrize(
    "field,value",
    [("port", True), ("tp_size", True), ("startup_timeout", float("nan")), ("neuron_device", -1)],
)
def test_nxdi_config_rejects_invalid_resource_values(tmp_path, field, value):
    raw = profile_document(tmp_path)
    raw["server"][field] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="invalid NxDI"):
        load_profile_config(path)


@pytest.mark.parametrize("kind", ["neuron", "workload_metrics"])
def test_controller_excludes_warmup_and_clean_pass_never_starts_profiler(
    tmp_path, monkeypatch, kind
):
    raw = profile_document(tmp_path)
    raw["profile_kind"] = kind
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))
    config = load_profile_config(path)
    log_dir = Path(config.log_dir)
    prepared = neuron_runner.frontend.PreparedReplay(
        tmp_path / "trace.csv",
        tmp_path / "prompts.txt",
        "local",
        log_dir / "replay.jsonl",
        log_dir / "summary.json",
    )
    trace, records = capture()
    row = records["iterations"][0]
    metric = iteration_metric(row)
    timing = {
        "schema_version": 2,
        "request_id": "a",
        "engine_queue_wait_ms": 0.1,
        "engine_first_schedule_to_first_token_ms": 0.9,
        "engine_core_ttft_ms": 1.0,
        "engine_core_decode_ms": 0.0,
        "engine_core_tpot_ms": None,
        "num_output_tokens": 1,
    }
    calls = []
    holder = {}

    class Process:
        pid = 12345
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode

    def launch(command, *, env, stdout, **kwargs):
        assert env["PJRT_DEVICE"] == "NEURON" and env["NXD_CPU_MODE"] == "0"
        assert "alignment.neuron.server" in command
        holder["output"] = stdout
        warm = {**timing, "request_id": "warmup"}
        stdout.write("VibeSimAlignmentRequestTiming " + json.dumps(warm) + "\n")
        stdout.flush()
        return Process()

    def http(base, endpoint, *, post=False):
        calls.append(endpoint)
        if endpoint == "/stop_profile":
            (log_dir / "system-trace.json").write_text(json.dumps(trace))
            (log_dir / "forward-records.json").write_text(json.dumps(records))
        return {"server_load": 0}

    def replay(workload, prepared, *, measurement_ready, **kwargs):
        measurement_ready()
        output = holder["output"]
        output.write("VibeSimAlignmentIteration " + json.dumps(metric) + "\n")
        output.write("VibeSimAlignmentRequestTiming " + json.dumps(timing) + "\n")
        output.write('NxdiOutputEvidence {"request_id":"a"}\n')
        output.flush()
        prepared.log_path.write_text(
            json.dumps({"outcome": {"request_id": "a", "status": "SUCCESS"}})
        )
        return {}

    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    monkeypatch.setattr(neuron_runner, "check_port", lambda *args: None)
    monkeypatch.setattr(neuron_runner.frontend, "prepare_replay", lambda *args: prepared)
    monkeypatch.setattr(neuron_runner.frontend, "build_session_runner", lambda: None)
    monkeypatch.setattr(neuron_runner.frontend, "run_replay", replay)
    monkeypatch.setattr(neuron_runner.subprocess, "Popen", launch)
    monkeypatch.setattr(neuron_runner.os, "killpg", lambda *args: None)
    monkeypatch.setattr(neuron_runner, "_http", http)
    result = neuron_runner.run_profile(config)
    assert result["request_timing_count"] == 1
    assert json.loads(Path(result["request_timings_jsonl"]).read_text())["request_id"] == "a"
    assert "warmup" not in (log_dir / "measurement.log").read_text()
    if kind == "neuron":
        assert "/start_profile" in calls and "/stop_profile" in calls
        assert read_parsed(Path(result["parsed_trace"]))["trace_provider"] == "neuron_system_trace"
    else:
        assert "/start_profile" not in calls and "/stop_profile" not in calls
        assert "parsed_trace" not in result and not (log_dir / "system-trace.json").exists()


def test_native_campaign_rendering_validates_without_cuda_or_device_access(tmp_path):
    from launcher.alignment_campaign import check as campaign_check
    from launcher.alignment_campaign.pack import ProfilePass, load_pack
    from launcher.alignment_campaign.render import case_documents

    root = Path(__file__).resolve().parents[1]
    original = load_pack(root / "presets/alignment/glm52_nvfp4_b200_tp8")
    first = original.cases[0]
    variant = replace(
        original.variant_of(first),
        engine="nxdi",
        backend="openai",
        gpu="AWS Trainium2 LNC2",
        server={"compiled_checkpoint": "nxdi_full", "tp_size": 1, "dp_size": 1},
        arch={"type": "llama3_nxdi"},
        worker={"type": "barebone"},
        input_builder={"type": "engine_text"},
        profile_passes=(
            ProfilePass("neuron", "profile_neuron", "workload", True),
            ProfilePass("workload_metrics", "profile_workload", "workload", True),
        ),
    )
    case = replace(
        first,
        max_model_len=512,
        max_concurrency=1,
        chunk_size=None,
        kernel_trace=None,
        workload_trace=replace(first.workload_trace, shapes=((6, 4), (128, 2)), repeats=1),
    )
    pack = replace(original, variants={variant.name: variant}, cases=(case,))
    host = campaign_check.host_for(pack, None)
    documents = case_documents(pack, case, host, tmp_path, root)
    native = documents["profile_neuron.yaml"]
    assert native["server"]["compiled_path"].endswith("nxdi_full")
    assert native["server"]["neuron_device"] == 0
    assert native["workload"]["warmup"] is True
    assert "cuda_visible_devices" not in native and "nsys" not in native
    arch = documents["simulation.yaml"]["pools"]["main"]["groups"][0]["arch"]
    assert arch["kv_capacity"] == 512 and "max_model_len" not in arch
    assert not campaign_check._check_rendering(pack, host)


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": [1], "max_tokens": 1, "stream": True, "ignore_eos": True, "temperature": 1},
        {"prompt": [1] * 129, "max_tokens": 1, "stream": True, "ignore_eos": True},
        {"prompt": [1], "max_tokens": 512, "stream": True, "ignore_eos": True},
    ],
)
def test_server_rejects_unsupported_geometry_and_sampling(body):
    with pytest.raises(ValueError):
        validate_request(body)


def test_real_http_stream_delivers_token_ids_and_usage():
    class Runtime:
        busy = 0

        def generate(self, prompt, count, request_id, emit):
            assert request_id == "req-0"
            for token in [10, 20, 30][:count]:
                emit(token)

    server = ThreadingHTTPServer(("127.0.0.1", 0), CompletionsHandler)
    server.runtime = Runtime()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = http.client.HTTPConnection(*server.server_address, timeout=5)
        body = json.dumps(
            {
                "prompt": [128000, 1],
                "max_tokens": 3,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
            }
        )
        client.request("POST", "/v1/completions", body, {"X-Request-Id": "req-0"})
        response = client.getresponse()
        assert response.status == 200
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.read().decode().splitlines()
            if line.startswith("data: {")
        ]
        assert [e["choices"][0]["token_ids"] for e in events[:-1]] == [[10], [20], [30]]
        assert events[-1]["usage"]["completion_tokens"] == 3
        assert events[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
