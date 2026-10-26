"""Stock Neuron geometry, observational dispatch, and strict asynchronous native joins."""

import copy
import importlib
import json
import re
import sys
import types
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from alignment.load_generator.config import IndependentFrontendConfig, LoadGeneratorConfig
from alignment.neuron import vllm_runner
from alignment.neuron.vllm_config import VllmNeuronProfileConfig, VllmNeuronServerConfig
from alignment.neuron.vllm_corpus import POOL_SIZE, materialize, verify_corpus
from alignment.neuron.vllm_normalize import normalize
from alignment.neuron.vllm_records import iteration_metric, scheduler_snapshot
from alignment.neuron.vllm_runner import validate_request_mapping
from alignment.neuron.vllm_server import export_command, server_command
from alignment.timing_predict_input.engine_text import build_cases
from launcher.alignment_config import load_profile_config


def config(tmp_path):
    server = VllmNeuronServerConfig(
        model_path=str(tmp_path / "model"),
        cache_path=str(tmp_path / "cache"),
        image="sha256:" + "a" * 64,
        docker_host="unix:///tmp/docker.sock",
        req_frontend_binary=str(tmp_path / "session_runner"),
        accepted_forward_path=str(tmp_path / "accepted"),
    )
    workload = LoadGeneratorConfig(
        frontend=IndependentFrontendConfig(str(tmp_path / "requests.csv")),
        text_file=str(tmp_path / "text.txt"),
        tokenizer=server.model_path,
        max_concurrency=16,
        max_items=16,
        max_model_len=512,
        token_pool_limit=POOL_SIZE,
    )
    return VllmNeuronProfileConfig(
        "stock", str(tmp_path / "capture"), "AWS Trainium2 LNC2", server, workload
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_size", 1),
        ("max_model_len", 128),
        ("token_buckets", [1, 2, 4, 8]),
        ("max_num_seqs", 8),
        ("kv_blocks", 6781),
        ("block_size", 16),
        ("dtype", "float32"),
        ("image", "latest"),
        ("host", "0.0.0.0"),
        ("port", True),
        ("docker_host", "tcp://127.0.0.1:2375"),
    ],
)
def test_config_rejects_unmeasured_geometry_and_transport(tmp_path, field, value):
    cfg = config(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        replace(cfg, server=replace(cfg.server, **{field: value})).validate()


def test_launcher_loads_stock_separately(tmp_path):
    cfg = config(tmp_path)
    raw = asdict(cfg)
    raw["schema_version"] = 1
    raw["workload"]["frontend"]["type"] = "independent"
    raw["workload"]["backend"]["type"] = "openai"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))
    loaded = load_profile_config(path, require_python_runtime=False)
    assert isinstance(loaded, VllmNeuronProfileConfig)
    assert loaded.engine == "vllm_neuron" and loaded.server.tp_size == 4
    raw["server"]["compiled_path"] = "/unknown/field"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="invalid vllm_neuron"):
        load_profile_config(path, require_python_runtime=False)


@pytest.mark.parametrize("tokenizer_spelling", ["model", "model-alias"])
def test_stock_loader_canonicalizes_local_tokenizer_paths(tmp_path, tokenizer_spelling):
    cfg = config(tmp_path)
    (tmp_path / "model").mkdir()
    (tmp_path / "cache").mkdir()
    (tmp_path / "session_runner").touch()
    (tmp_path / "model-alias").symlink_to(tmp_path / "model", target_is_directory=True)
    raw = asdict(cfg)
    raw["schema_version"] = 1
    raw["workload"]["frontend"]["type"] = "independent"
    raw["workload"]["backend"]["type"] = "openai"
    raw["server"]["model_path"] = "model-alias"
    raw["workload"]["tokenizer"] = tokenizer_spelling
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))
    loaded = load_profile_config(path)
    assert loaded.server.model_path == loaded.workload.tokenizer == str(tmp_path / "model")
    raw["workload"]["tokenizer"] = "different-model"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="local model tokenizer"):
        load_profile_config(path)


def test_server_reserves_whole_chip_and_public_offline_engine(tmp_path):
    cfg = config(tmp_path)
    device = NS(device_id=0, lnc=2, core_ids=(0, 1, 2, 3), busy=False)
    command = server_command(cfg, device, "test-stock")
    assert "--pull=never" in command and "--network=host" in command
    assert "NEURON_VISIBLE_DEVICES=0-3" in command
    assert not any("NEURON_RT_VISIBLE_CORES" in arg for arg in command)
    assert "vllm.entrypoints.cli.main" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert command[command.index("--max-num-seqs") + 1] == "16"
    assert command.count("--shutdown-timeout") == 1
    assert command[command.index("--shutdown-timeout") + 1] == "10"
    extra = json.loads(command[command.index("--additional-config") + 1])
    assert extra["neuron_config"]["num_seqs_buckets"] == [1, 16]
    assert extra["neuron_profiler"]["neuron_cores"] == [0, 1, 2, 3]
    clean = server_command(replace(cfg, profile_kind="workload_metrics"), device, "clean")
    assert "--profiler-config" not in clean
    assert clean[clean.index("--shutdown-timeout") + 1] == "10"
    assert "--network=none" in export_command(cfg)
    assert "--device" not in export_command(cfg)
    with pytest.raises(ValueError, match="whole"):
        server_command(cfg, replace_ns(device, core_ids=(0,)), "wrong")


# The C512 [1,16] clean-pass argv from the first --env on, as before the API
# hand-off stamp existed. With `handoff_stamp` off it must stay byte-identical;
# on, it may add only its logging-config env and the public request-log flags.
PRE_HANDOFF_C512_TAIL = [
    *("--env", "PYTHONPATH=/source", "--env", "PYTHONDONTWRITEBYTECODE=1"),
    *("--env", "PYTHONUNBUFFERED=1", "--env", "HF_HUB_OFFLINE=1"),
    *("--env", "TRANSFORMERS_OFFLINE=1", "--env", "HF_HUB_DISABLE_TELEMETRY=1"),
    *("--env", "VLLM_NO_USAGE_STATS=1", "--env", "DO_NOT_TRACK=1"),
    *("--env", "VLLM_SERVER_DEV_MODE=1", "--env", "VLLM_CACHE_ROOT=/trial/cache"),
    *("--env", "XDG_CACHE_HOME=/trial/cache", "--env", "NKI_COMPILE_CACHE_URL=/trial/nki-cache"),
    *("--env", "TMPDIR=/trial/tmp", "--env", "NEURON_SKIP_EFA_AFFINITY=1"),
    *("--env", "NEURON_LOGICAL_NC_CONFIG=2", "--env", "NEURON_VISIBLE_DEVICES=0-3"),
    *("--env", "MASTER_ADDR=127.0.0.1", "--env", "VLLM_HOST_IP=127.0.0.1"),
    *("--env", "GLOO_SOCKET_IFNAME=lo", "--env", "OMP_NUM_THREADS=4"),
    *("--entrypoint", "/opt/conda/bin/python", "sha256:" + "a" * 64),
    *("-m", "vllm.entrypoints.cli.main", "serve", "/model"),
    *("--host", "127.0.0.1", "--port", "63030", "--served-model-name", "llama31-8b-neuron"),
    *("--dtype", "bfloat16", "--tensor-parallel-size", "4", "--seed", "0"),
    *("--max-model-len", "512", "--max-num-batched-tokens", "512", "--max-num-seqs", "16"),
    *("--num-gpu-blocks-override", "6782", "--no-enable-prefix-caching"),
    *("--enable-prompt-tokens-details", "--enable-server-load-tracking"),
    *("--api-server-count", "1", "--shutdown-timeout", "10"),
    *("--scheduler-cls", "alignment.neuron.vllm_observer.ObservedScheduler"),
    *("--worker-cls", "alignment.neuron.vllm_observer.ObservedWorker"),
    "--additional-config",
    '{"neuron_config":{"num_batched_tokens_buckets":[512],"num_seqs_buckets":[1,16]}}',
]


def test_handoff_stamp_off_keeps_the_c512_command_byte_identical(tmp_path):
    cfg = replace(config(tmp_path), profile_kind="workload_metrics")
    assert cfg.server.handoff_stamp is False
    device = NS(device_id=0, lnc=2, core_ids=(0, 1, 2, 3), busy=False)
    command = server_command(cfg, device, "handoff")
    assert command[command.index("--env") :] == PRE_HANDOFF_C512_TAIL


def test_handoff_stamp_on_adds_only_its_public_logging(tmp_path):
    cfg = replace(config(tmp_path), profile_kind="workload_metrics")
    cfg = replace(cfg, server=replace(cfg.server, handoff_stamp=True))
    cfg.validate()
    device = NS(device_id=0, lnc=2, core_ids=(0, 1, 2, 3), busy=False)
    command = server_command(cfg, device, "handoff")
    tail = command[command.index("--env") :]
    added = [
        "--env",
        "VLLM_LOGGING_CONFIG_PATH=/source/alignment/neuron/vllm_logging.json",
        "--enable-log-requests",
        "--max-log-len",
        "0",
    ]
    env_at = tail.index(added[1]) - 1
    assert tail[env_at : env_at + 2] == added[:2]
    flags_at = tail.index("--enable-log-requests")
    assert tail[flags_at : flags_at + 3] == added[2:]
    stripped = tail[:env_at] + tail[env_at + 2 : flags_at] + tail[flags_at + 3 :]
    assert stripped == PRE_HANDOFF_C512_TAIL
    # The config is shipped in the read-only source mount the env points into.
    assert any(arg.endswith("/alignment:/source/alignment:ro") for arg in command)
    assert (vllm_runner.Path(vllm_runner.__file__).parent / "vllm_logging.json").is_file()
    with pytest.raises(ValueError, match="handoff_stamp"):
        replace(cfg, server=replace(cfg.server, handoff_stamp=1)).validate()


def replace_ns(value, **fields):
    return NS(**{**vars(value), **fields})


def output(prefill=True, context=512):
    prompt = context - 8
    return NS(
        scheduled_new_reqs=[
            NS(req_id="cmpl-a-0", num_computed_tokens=0, prompt_token_ids=[1] * prompt)
        ]
        if prefill
        else [],
        scheduled_cached_reqs=NS(
            req_ids=[] if prefill else ["cmpl-a-0", "cmpl-b-0"],
            num_computed_tokens=[] if prefill else [prompt, prompt + 3],
        ),
        num_scheduled_tokens={"cmpl-a-0": prompt} if prefill else {"cmpl-a-0": 1, "cmpl-b-0": 1},
        num_scheduled_tokens_padded={"cmpl-a-0": context}
        if prefill
        else {"cmpl-a-0": 1, "cmpl-b-0": 1},
    )


def metric(prefill=True, index=0, context=512, buckets=(1, 16)):
    return iteration_metric(
        scheduler_snapshot(output(prefill, context)),
        prompt_lengths={"cmpl-a-0": context - 8, "cmpl-b-0": context - 8},
        iteration=index,
        start_ns=1000,
        stop_ns=2000,
        context=context,
        decode_buckets=buckets,
    )


def test_scheduler_uses_preincrement_cpu_progress_and_padding():
    row = metric(False)
    assert row["decode_kv_lens"] == [505, 508]
    assert row["compiled_shapes"]["token_bucket"] == 16
    assert metric()["prefill_chunk_pairs"] == [[0, 504]]
    broken = output()
    broken.scheduled_new_reqs.clear()
    with pytest.raises(ValueError, match="pre-increment"):
        scheduler_snapshot(broken)


def capture(context=512, buckets=(1, 16)):
    metrics = [metric(context=context, buckets=buckets), metric(False, 1, context, buckets)]
    forwards, events = [], []
    info = {
        "a" * 32: ("prefill", context),
        "b" * 32: ("decode", metrics[1]["compiled_shapes"]["token_bucket"]),
    }
    for rank in range(4):
        pid = str(20 + rank)
        for i, m in enumerate(metrics):
            bucket = m["compiled_shapes"]["token_bucket"]
            graph_hash = ("a" if i == 0 else "b") * 32
            name = f"/trial/cache/neuron/compile_cache/{graph_hash}/dev0_3.rank{rank}/g.neff"
            forwards.append(
                {
                    "pid": int(pid),
                    "rank": rank,
                    "worker_sequence": i + 30,
                    "fingerprint": m["fingerprint"],
                    "start_epoch_ns": 100 + 100 * i,
                    "stop_epoch_ns": 120 + 100 * i,
                    "input_ids_shape": [bucket],
                    "block_table_shapes": [[1 if i == 0 else bucket, context // 32]],
                }
            )
            base = {"process_id": pid, "timestamp_unit": "ns", "model_name": name}
            events.append(
                {**base, "name": "nrt_model_submit", "timestamp": 110 + 100 * i, "duration": 2}
            )
            events.append(
                {
                    **base,
                    "name": "kbl_exec_pre",
                    "timestamp": 130 + 100 * i,
                    "duration": 2,
                    "exec_id": i + 500 + 20 * rank,
                }
            )
            # Device runs after the worker call: containment would fail this valid async case.
            for core in range(2):
                events.append(
                    {
                        **base,
                        "name": "nc_exec_running",
                        "timestamp": 200 + 100 * i + rank + core,
                        "duration": 20,
                        "exec_id": i + 500 + 20 * rank,
                        "device_core_idx": rank * 2 + core,
                    }
                )
    return (
        {"trace_event": list(reversed(events))},
        {
            "engine": "vllm_neuron",
            "tp_size": 4,
            "iterations": metrics,
            "forwards": forwards,
        },
        info,
    )


def test_async_native_join_covers_tp4_and_unions_once():
    trace, records, info = capture()
    parsed = normalize(trace, records, info)
    assert parsed["physical_core_ids"] == list(range(8))
    kernels = parsed["iteration_details"][0]["ranges"][0]["kernels"]
    assert [(k["start_ns"], k["end_ns"]) for k in kernels] == [(200, 224)]
    assert len(parsed["iteration_details"][0]["ranges"][0]["native_rank_executions"]) == 4
    cases, _, excluded = build_cases(parsed, "forward")
    assert cases[0]["groups"] == [{"prefill_chunk_pairs": [[0, 504]], "decode_kv_lens": []}]
    assert cases[1]["groups"][0]["decode_kv_lens"] == [505, 508] and excluded == []


def test_native_union_keeps_a_true_idle_gap():
    trace, records, info = capture()
    for event in trace["trace_event"]:
        if event["name"] == "nc_exec_running" and event["timestamp"] < 300:
            event["timestamp"] = 200 if event["device_core_idx"] < 4 else 230
            event["duration"] = 10
    kernels = normalize(trace, records, info)["iteration_details"][0]["ranges"][0]["kernels"]
    assert [(k["start_ns"], k["end_ns"]) for k in kernels] == [(200, 210), (230, 240)]


def test_native_inventory_enters_shared_labeling_contract(tmp_path):
    from alignment.labeling.cli import main as label_main
    from alignment.neuron.vllm_normalize import write_normalized
    from launcher.alignment_config import load_labeled_kernel_sequences

    trace, records, info = capture()
    (tmp_path / "system-trace.json").write_text(json.dumps(trace))
    result = write_normalized(tmp_path, records, info)
    labeled = tmp_path / "labeled.json"
    assert label_main(["initialize", result["kernel_sequences"], str(labeled)]) == 0
    inventory = load_labeled_kernel_sequences(labeled)
    assert inventory["folding_policy"]["kind"] == "exact_contiguous_repeat"
    assert inventory["device_ids"] == [0]


def test_native_parse_feeds_only_the_stock_whole_forward_predictor(tmp_path):
    from alignment.neuron.vllm_normalize import write_normalized
    from alignment.timing_predict_input.builder import (
        BuildRequest,
        EngineTextInputSpec,
        build_inputs,
    )

    trace, records, info = capture()
    (tmp_path / "system-trace.json").write_text(json.dumps(trace))
    result = write_normalized(tmp_path, records, info)
    request = BuildRequest(
        simulation_preset=tmp_path / "sim.yaml",
        profile_log_dir=tmp_path,
        parsed_nsys=Path(result["parsed_trace"]),
        output_dir=tmp_path / "prediction",
        gpu="AWS Trainium2 LNC2",
        arch={"type": "llama3_vllm_neuron", "max_model_len": 512},
        backends={},
        input_spec=EngineTextInputSpec(),
    )
    built = build_inputs(request)
    assert json.loads(built.cases.read_text())[0]["groups"] == [
        {"prefill_chunk_pairs": [[0, 504]], "decode_kv_lens": []}
    ]
    manifest = json.loads(built.input_manifest.read_text())
    assert "parsed_trace" in manifest and "parsed_nsys" not in manifest
    with pytest.raises(ValueError, match="llama3_vllm_neuron"):
        build_inputs(replace(request, arch={"type": "llama3_dense_tp", "tp_size": 4}))


@pytest.mark.parametrize(
    "mutation",
    [
        "drop_core",
        "duplicate_core",
        "unknown_worker",
        "wrong_model",
        "wrong_bucket",
        "missing_rank",
        "wrong_sequence",
        "wrong_unit",
        "duplicate_submit",
    ],
)
def test_native_join_rejects_incomplete_or_ambiguous_evidence(mutation):
    trace, records, info = capture()
    cores = [e for e in trace["trace_event"] if e["name"] == "nc_exec_running"]
    if mutation == "drop_core":
        trace["trace_event"].remove(cores[0])
    elif mutation == "duplicate_core":
        trace["trace_event"].append(copy.deepcopy(cores[0]))
    elif mutation == "unknown_worker":
        trace["trace_event"].append({**cores[0], "process_id": "999"})
    elif mutation == "wrong_model":
        next(e for e in trace["trace_event"] if e["name"] == "kbl_exec_pre")["model_name"] = "bad"
    elif mutation == "wrong_bucket":
        info["b" * 32] = ("decode", 8)
    elif mutation == "missing_rank":
        records["forwards"] = [r for r in records["forwards"] if r["rank"] != 3]
    elif mutation == "wrong_sequence":
        records["forwards"][0]["fingerprint"] = "wrong"
    elif mutation == "wrong_unit":
        cores[0]["timestamp_unit"] = "us"
    else:
        trace["trace_event"].append(
            copy.deepcopy(next(e for e in trace["trace_event"] if e["name"] == "nrt_model_submit"))
        )
    with pytest.raises(ValueError):
        normalize(trace, records, info)


def test_request_mapping_covers_all16_prefill_and_decode_steps():
    rows, iterations = [], []
    for i in range(16):
        rows.append(
            {
                "outcome": {
                    "request_id": f"independent_{i}",
                    "status": "SUCCESS",
                    "output_len_actual": 8,
                    "first_token_id_ms": 1.0,
                    "token_delivery_tpot_ms": 2.0,
                    "server_usage": {
                        "prompt_tokens": 504,
                        "completion_tokens": 8,
                        "cached_prompt_tokens": 0,
                    },
                }
            }
        )
        for step in range(8):
            iterations.append(
                {
                    "iteration_index": len(iterations),
                    "requests": [
                        {
                            "request_id": f"cmpl-independent_{i}-0",
                            "prompt_tokens": 504,
                            "kv_len_before": 0 if step == 0 else 503 + step,
                            "q_tokens": 504 if step == 0 else 1,
                        }
                    ],
                }
            )
    assert validate_request_mapping(iterations, rows)["request_count"] == 16
    iterations[-1]["requests"][0]["kv_len_before"] -= 1
    with pytest.raises(ValueError, match="progress"):
        validate_request_mapping(iterations, rows)


def test_observers_preserve_resolved_async_class_and_stock_outputs(monkeypatch, capsys):
    class Base:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.log_stats = getattr(vllm_config, "log_stats", True)
            self.requests = {"cmpl-a-0": NS(num_prompt_tokens=504)}
            self.max_model_len = 512
            self.num_seqs_buckets = [1, 16]
            self.calls = []

        def schedule(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.stock_output

        def update_from_output(self, scheduler_output, model_output):
            self.calls.append((scheduler_output, model_output))
            return model_output

    class AsyncBase(Base):
        pass

    scheduler = types.ModuleType("vllm_neuron.vllm.core.scheduler")
    scheduler.NeuronScheduler = Base
    scheduler.NeuronAsyncScheduler = AsyncBase
    worker = types.ModuleType("vllm_neuron.vllm.worker.neuron_worker")
    worker.NeuronWorker = object
    monkeypatch.setitem(sys.modules, scheduler.__name__, scheduler)
    monkeypatch.setitem(sys.modules, worker.__name__, worker)
    sys.modules.pop("alignment.neuron.vllm_observer", None)
    observer = importlib.import_module("alignment.neuron.vllm_observer")
    monkeypatch.setattr(observer, "install_cleanup_adapter", lambda: None)
    layer = {"block_table_tensor": NS(shape=(16, 16))}
    assert observer.block_table_shapes(layer) == [(16, 16)]
    metadata = {
        "model.layers.0.self_attn": layer,
        "_cached_decode_mask": NS(shape=(1, 32, 16, 512)),
    }
    assert observer.block_table_shapes(metadata) == [(16, 16)]
    with pytest.raises(RuntimeError, match="unknown stock layer"):
        observer.block_table_shapes({**metadata, "unexpected_tensor": NS(shape=(2, 2))})
    for mode in (False, True):
        instance = observer.ObservedScheduler(NS(scheduler_config=NS(async_scheduling=mode)))
        assert isinstance(instance, AsyncBase) == mode
        instance.stock_output = output()
        actual = instance.schedule(throttle_prefills=True)
        assert actual is instance.stock_output and len(instance.calls) == 1
        assert "VibeSimAlignmentIteration" not in capsys.readouterr().out
        result = {}
        assert instance.update_from_output(actual, result) is result
        assert len(instance.calls) == 2 and "VibeSimAlignmentIteration" in capsys.readouterr().out
    with pytest.raises(ValueError, match="log_stats"):
        observer.ObservedScheduler(NS(scheduler_config=NS(async_scheduling=False), log_stats=False))
    sys.modules.pop("alignment.neuron.vllm_observer", None)


def test_stock_randomized_ids_unwrap_without_changing_cuda_adapter():
    from alignment.profiler.engine_records import VLLM_NEURON_RECORDS, VLLM_RECORDS

    rid = "cmpl-independent_museum-01-0-a09bc2f6"
    assert VLLM_NEURON_RECORDS.unwrap_request_id(rid) == "independent_museum-01"
    assert VLLM_RECORDS.unwrap_request_id(rid) == "independent_museum-01-0-a09bc2f6"


def test_successful_http_responses_do_not_hide_observer_exceptions():
    vllm_runner.verify_server_log("Profiler stopped successfully")
    with pytest.raises(ValueError, match="exception"):
        vllm_runner.verify_server_log("WorkerProc hit an exception.\nIndexError")


def test_multiplexed_complete_worker_records_are_not_lost(tmp_path):
    path = tmp_path / "server.log"
    path.write_text(
        '(Worker_TP0 pid=42) VllmNeuronForward {"rank":0}'
        '(Worker_TP1 pid=43) VllmNeuronForward {"rank":1}\nother log\n'
    )
    assert vllm_runner._tagged(path, "VllmNeuronForward") == [{"rank": 0}, {"rank": 1}]
    path.write_text('VllmNeuronForward {"rank":')
    with pytest.raises(json.JSONDecodeError):
        vllm_runner._tagged(path, "VllmNeuronForward")


def test_museum_pool_exact_frontend_offsets(tmp_path):
    # Offline tokenizer-only check; no Torch, model construction, SDK or device.
    model = Path("/home/ec2-user/ServingStudio/tmp/models/llama3.1-8b")
    if not (model / "tokenizer.json").is_file():
        pytest.skip("local frozen tokenizer unavailable")
    materialize(model, tmp_path)
    cfg = config(tmp_path)
    workload = replace(
        cfg.workload,
        text_file=str(tmp_path / "museum-pool.txt"),
        frontend=IndependentFrontendConfig(str(tmp_path / "requests.csv")),
    )
    evidence = verify_corpus(workload, model)
    hashes = list(evidence["prompt_hashes"].values())
    assert hashes[:8] == hashes[8:] and len(set(hashes)) == 8
    (tmp_path / "museum-pool.txt").write_text("wrong corpus\n")
    with pytest.raises(ValueError, match="segments"):
        verify_corpus(workload, model)


@pytest.mark.parametrize("replay_fails", [False, True])
def test_server_cleanup_finishes_before_host_lease_release(tmp_path, monkeypatch, replay_fails):
    cfg = config(tmp_path)
    events = []
    device = NS(device_id=0, lnc=2, core_ids=(0, 1, 2, 3), busy=False)
    lease = NS(device=device, lock=NS(close=lambda: events.append("lease_released")))

    class Pool:
        def __init__(self, devices):
            assert devices == [0]

        def acquire_chunks(self, count):
            assert count == 1
            events.append("lease_acquired")
            yield lease

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert self.returncode == 0
            events.append("server_terminated")
            return 0

    proc = Process()

    def run(command, **kwargs):
        if "stop" in command:
            events.append("server_stopped")
            proc.returncode = 0
        return NS(returncode=0, stderr="")

    def replay(*args, measurement_ready, **kwargs):
        measurement_ready()
        if replay_fails:
            raise RuntimeError("client failed")
        return {"log_path": str(tmp_path / "replay.jsonl")}

    monkeypatch.setattr(vllm_runner, "LocalNeuronPool", Pool)
    monkeypatch.setattr(vllm_runner, "check_port", lambda *args: None)
    monkeypatch.setattr(vllm_runner, "source_provenance", lambda *args: {})
    monkeypatch.setattr(vllm_runner, "accepted_forward_evidence", lambda *args, **kwargs: {})
    monkeypatch.setattr(vllm_runner, "capture_binary_snapshot", lambda *args: {})
    monkeypatch.setattr(vllm_runner, "verify_checkpoint", lambda *args: {})
    monkeypatch.setattr(vllm_runner, "verify_corpus", lambda *args: {})
    monkeypatch.setattr(type(cfg.server.container()), "validate", lambda self: None)
    monkeypatch.setattr(vllm_runner.frontend, "prepare_replay", lambda *args: object())
    monkeypatch.setattr(vllm_runner.frontend, "run_replay", replay)
    monkeypatch.setattr(vllm_runner, "_http", lambda *args, **kwargs: {"server_load": 0})
    monkeypatch.setattr(vllm_runner.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(vllm_runner.subprocess, "run", run)
    monkeypatch.setattr(vllm_runner, "postprocess", lambda config: {"engine": config.engine})
    if replay_fails:
        with pytest.raises(RuntimeError, match="client failed"):
            vllm_runner.run_profile(cfg)
    else:
        assert vllm_runner.run_profile(cfg)["engine"] == "vllm_neuron"
    assert events == ["lease_acquired", "server_stopped", "server_terminated", "lease_released"]


@pytest.mark.parametrize("source", ["runtime", "exported"])
def test_native_drops_rejected_even_with_matching_execution_counts(source):
    from alignment.neuron.vllm_normalize import verify_no_event_drops

    warning = "System profile events were dropped due to full ring buffers"
    logs = {"server.log": warning if source == "runtime" else "normal shutdown"}
    trace = {"trace_event": [{"message": warning}] if source == "exported" else []}
    with pytest.raises(ValueError, match="event drops"):
        verify_no_event_drops(trace, logs)
    assert verify_no_event_drops({"trace_event": []}, {"server.log": "normal"})["passed"]


def test_accepted_graphs_do_not_grant_precision_to_a_new_graph(tmp_path):
    from alignment.neuron.vllm_records import token_hash
    from alignment.neuron.vllm_runner import accepted_forward_evidence, compare_accepted_graphs

    (tmp_path / "precision.json").write_text(
        json.dumps({"passed": True, "by_shape": {"b16": {"passed": True}}, "criterion": "vendor"})
    )
    records = [
        {"phase": phase, "token_bucket": bucket, "model_hash": str(bucket)}
        for phase, bucket in [("prefill", 512), ("decode", 1), ("decode", 16)]
    ]
    (tmp_path / "execution-records.json").write_text(json.dumps(records))
    (tmp_path / "accuracy-outputs.json").write_text(
        json.dumps([{"prompt_ids": [1, 2], "token_ids": [3]}])
    )
    (tmp_path / "profiled-outputs.json").write_text("[]")
    evidence = accepted_forward_evidence(tmp_path)
    assert token_hash([1, 2]) in evidence["subject_continuations_by_prompt_sha256"]
    assert evidence["http_generated_id_comparison"]["available"] is False
    assert compare_accepted_graphs({"16": ("decode", 16)}, evidence)["passed"]
    mismatch = compare_accepted_graphs({"new": ("decode", 16)}, evidence)
    assert not mismatch["passed"] and mismatch["mismatches"]["new"]["expected"] == "16"


# Context ladder: the C512 [1,16] contract above is one point of a general rule.
LADDER = [
    (512, (1, 16), 16),
    (2048, (1, 2, 4, 8, 16, 32, 64, 128), 16),
    (8192, (1, 2, 4, 8, 16, 32), 16),
]


def ladder_config(tmp_path, context, buckets, requests):
    cfg = config(tmp_path)
    return replace(
        cfg,
        server=replace(
            cfg.server, max_model_len=context, token_buckets=buckets, max_num_seqs=max(buckets)
        ),
        workload=replace(
            cfg.workload, max_model_len=context, max_concurrency=requests, max_items=requests
        ),
    )


@pytest.mark.parametrize("context,buckets,requests", LADDER)
def test_ladder_server_command_uses_configured_geometry(tmp_path, context, buckets, requests):
    cfg = ladder_config(tmp_path, context, buckets, requests)
    cfg.validate()
    device = NS(device_id=0, lnc=2, core_ids=(0, 1, 2, 3), busy=False)
    command = server_command(cfg, device, "ladder")
    assert command[command.index("--max-model-len") + 1] == str(context)
    assert command[command.index("--max-num-batched-tokens") + 1] == str(context)
    assert command[command.index("--max-num-seqs") + 1] == str(max(buckets))
    assert command[command.index("--num-gpu-blocks-override") + 1] == "6782"
    extra = json.loads(command[command.index("--additional-config") + 1])
    assert extra["neuron_config"] == {
        "num_batched_tokens_buckets": [context],
        "num_seqs_buckets": list(buckets),
    }
    if context == 512:
        assert command[command.index("--additional-config") + 1].startswith(
            '{"neuron_config":{"num_batched_tokens_buckets":[512],"num_seqs_buckets":[1,16]},'
        )


def test_ladder_config_accepts_inventory_without_bucket_one(tmp_path):
    # Stock num_seqs_buckets need not start at 1; C4096 accepts only 16 and 32.
    ladder_config(tmp_path, 4096, (16, 32), 16).validate()


@pytest.mark.parametrize(
    "context,buckets,requests,error",
    [(8192, (2, 4, 8, 16, 64), 16, "token_buckets")],  # above largest(C) without bucket 1
)
def test_ladder_config_rejects_partial_inventory_above_cap(
    tmp_path, context, buckets, requests, error
):
    with pytest.raises(ValueError, match=re.escape(error)):
        ladder_config(tmp_path, context, buckets, requests).validate()


@pytest.mark.parametrize(
    "context,buckets,requests,error",
    [
        (2048, (1, 2, 4, 8, 16, 32, 64, 128, 256), 16, "token_buckets"),  # above largest(C)
        (8192, (1, 2, 4, 8, 16, 32, 64), 16, "token_buckets"),
        (8192, (1, 3, 32), 16, "token_buckets"),  # powers of two only
        (8192, (1, 32, 16), 16, "token_buckets"),  # ascending
        (8192, (1, 2, 4, 8, 16, 32), 27, "1..26"),  # 217024 // 8192 resident cap
        (2048, (1, 2, 4, 8), 9, "1..8"),  # largest bucket cap
        (2048, (1, 2, 4, 8, 16, 32, 64, 128), 106, "1..105"),  # 217024 // 2048
        (2048, (1, 2, 4, 8), 0, "1..8"),
        (128, (1, 16), 16, "C in"),
        (3072, (1, 16), 16, "C in"),
    ],
)
def test_ladder_config_rejects_uncompiled_or_nonresident(
    tmp_path, context, buckets, requests, error
):
    with pytest.raises(ValueError, match=re.escape(error)):
        ladder_config(tmp_path, context, buckets, requests).validate()


def test_ladder_config_rejects_mismatched_seqs_or_workload(tmp_path):
    cfg = ladder_config(tmp_path, 2048, (1, 2, 4, 8, 16, 32, 64, 128), 16)
    cfg.validate()
    with pytest.raises(ValueError, match="max_num_seqs"):
        replace(cfg, server=replace(cfg.server, max_num_seqs=64)).validate()
    with pytest.raises(ValueError, match="context2048"):
        replace(cfg, workload=replace(cfg.workload, max_model_len=512)).validate()
    with pytest.raises(ValueError, match="max_items"):
        replace(cfg, workload=replace(cfg.workload, max_items=8)).validate()


def raw_snapshot(rows):
    return {
        "requests": [
            {"request_id": f"r{i}", "kv_len_before": kv, "q_tokens": q, "padded_q_tokens": pq}
            for i, (kv, q, pq) in enumerate(rows)
        ],
        "fingerprint": "f",
    }


@pytest.mark.parametrize(
    "context,buckets,count,expected",
    [
        (512, (1, 16), 1, 1),
        (512, (1, 16), 2, 16),
        (512, (1, 16), 16, 16),
        (2048, (1, 2, 4, 8, 16, 32, 64, 128), 5, 8),
        (2048, (1, 2, 4, 8, 16, 32, 64, 128), 33, 64),
        (2048, (1, 2, 4, 8, 16, 32, 64, 128), 128, 128),
        (8192, (1, 2, 4, 8, 16, 32), 17, 32),
        (8192, (1, 2, 4, 8, 16, 32), 1, 1),
    ],
)
def test_ladder_iteration_metric_selects_smallest_decode_bucket(context, buckets, count, expected):
    prompt = context - 8
    kwargs = dict(iteration=0, start_ns=1, stop_ns=2, context=context, decode_buckets=buckets)
    lengths = {f"r{i}": prompt for i in range(count + 1)}
    row = iteration_metric(
        raw_snapshot([(prompt + 6, 1, 1)] * count), prompt_lengths=lengths, **kwargs
    )
    assert row["compiled_shapes"] == {
        "context": context,
        "token_bucket": expected,
        "tp_size": 4,
        "kv_blocks": 6782,
        "block_size": 32,
    }
    prefill = iteration_metric(
        raw_snapshot([(0, prompt, context)]), prompt_lengths=lengths, **kwargs
    )
    assert prefill["compiled_shapes"]["token_bucket"] == context
    assert prefill["prefill_chunk_pairs"] == [[0, prompt]]
    with pytest.raises(ValueError, match="padded"):
        iteration_metric(raw_snapshot([(0, prompt, 512 if context != 512 else 1024)]),
                         prompt_lengths=lengths, **kwargs)
    with pytest.raises(ValueError, match="one-token decode"):
        iteration_metric(raw_snapshot([(context, 1, 1)]), prompt_lengths={"r0": 1}, **kwargs)
    over = {f"r{i}": prompt for i in range(max(buckets) + 1)}
    with pytest.raises(ValueError, match="oversized"):
        iteration_metric(
            raw_snapshot([(prompt + 1, 1, 1)] * (max(buckets) + 1)), prompt_lengths=over, **kwargs
        )


@pytest.mark.parametrize("context,buckets,requests", LADDER)
def test_ladder_normalize_uses_compiled_context_width_and_name(context, buckets, requests):
    trace, records, info = capture(context, buckets)
    parsed = normalize(trace, records, info)
    decode = records["iterations"][1]["compiled_shapes"]["token_bucket"]
    assert set(parsed["kernel_names"].values()) == {
        f"Stock vLLM Neuron Llama3.1-8B TP4 prefill B{context} C{context}",
        f"Stock vLLM Neuron Llama3.1-8B TP4 decode B{decode} C{context}",
    }
    for row in records["forwards"]:
        if row["block_table_shapes"][0][0] == decode:
            row["block_table_shapes"] = [[decode, 16 if context != 512 else 32]]
    with pytest.raises(ValueError, match=f"C{context}"):
        normalize(trace, records, info)


@pytest.mark.parametrize("context,buckets,requests", [*LADDER, (2048, (1, 2, 4), 3)])
def test_ladder_request_mapping_and_graphs(tmp_path, context, buckets, requests):
    from alignment.neuron.vllm_runner import accepted_forward_evidence

    prompt = context - 8
    rows, iterations = [], []
    for i in range(requests):
        rows.append({"outcome": {
            "request_id": f"independent_{i}", "status": "SUCCESS", "output_len_actual": 8,
            "first_token_id_ms": 1.0, "token_delivery_tpot_ms": 2.0,
            "server_usage": {"prompt_tokens": prompt, "completion_tokens": 8,
                             "cached_prompt_tokens": 0},
        }})
        for step in range(8):
            iterations.append({"iteration_index": len(iterations), "requests": [{
                "request_id": f"cmpl-independent_{i}-0", "prompt_tokens": prompt,
                "kv_len_before": 0 if step == 0 else prompt - 1 + step,
                "q_tokens": prompt if step == 0 else 1,
            }]})
    mapping = validate_request_mapping(iterations, rows, context=context, requests=requests)
    assert (mapping["request_count"], mapping["prompt_tokens"]) == (requests, prompt)
    with pytest.raises(ValueError, match=f"exactly {requests + 1}"):
        validate_request_mapping(iterations, rows, context=context, requests=requests + 1)
    with pytest.raises(ValueError, match=f"{prompt + 8}\\+8"):
        validate_request_mapping(iterations, rows, context=context + 8, requests=requests)

    (tmp_path / "precision.json").write_text(
        json.dumps({"passed": True, "by_shape": {"b": {"passed": True}}, "criterion": "vendor"})
    )
    graphs = [("prefill", context), *(("decode", b) for b in buckets)]
    (tmp_path / "execution-records.json").write_text(json.dumps([
        {"phase": phase, "token_bucket": bucket, "model_hash": f"{phase}{bucket}"}
        for phase, bucket in graphs
    ]))
    (tmp_path / "accuracy-outputs.json").write_text(
        json.dumps([{"prompt_ids": [1], "token_ids": [2]}])
    )
    (tmp_path / "profiled-outputs.json").write_text("[]")
    evidence = accepted_forward_evidence(tmp_path, context=context, token_buckets=buckets)
    assert set(evidence["graphs"]) == {f"{p}:{b}" for p, b in graphs}
    with pytest.raises(ValueError, match="configured stock graphs"):
        accepted_forward_evidence(tmp_path, context=context, token_buckets=buckets[:-1])


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return NS(ids=[0] * add_special_tokens + [ord(c) for c in text])


@pytest.mark.parametrize("context,requests", [(512, 16), (2048, 3), (8192, 26)])
def test_ladder_corpus_prompts_follow_profiler_batch(context, requests):
    from alignment.neuron.vllm_corpus import prompts

    ids = prompts(FakeTokenizer(), context, requests)
    assert len(ids) == requests and {len(p) for p in ids} == {context - 8}
    subjects = prompts(FakeTokenizer(), context, 8)
    assert len({tuple(p) for p in subjects}) == 8
    assert all(ids[i] == subjects[i % 8] for i in range(requests))
    with pytest.raises(ValueError, match="KV pool"):
        prompts(FakeTokenizer(), context, 217024 // context + 1)


def test_ladder_real_corpus_c8192(tmp_path):
    model = Path("/home/ec2-user/ServingStudio/tmp/models/llama3.1-8b")
    if not (model / "tokenizer.json").is_file():
        pytest.skip("local frozen tokenizer unavailable")
    materialize(model, tmp_path, 8192, 16)
    cfg = ladder_config(tmp_path, 8192, (1, 2, 4, 8, 16, 32), 16)
    workload = replace(
        cfg.workload,
        text_file=str(tmp_path / "museum-pool.txt"),
        frontend=IndependentFrontendConfig(str(tmp_path / "requests.csv")),
    )
    evidence = verify_corpus(workload, model)
    hashes = list(evidence["prompt_hashes"].values())
    assert hashes[:8] == hashes[8:] and len(set(hashes)) == 8
    assert evidence["consumed_prompt_tokens"] == 16 * 8184
    with pytest.raises(ValueError, match="16 simultaneous 4088\\+8"):
        verify_corpus(replace(workload, max_model_len=4096), model)
