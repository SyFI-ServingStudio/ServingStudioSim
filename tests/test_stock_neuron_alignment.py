"""Stock Neuron geometry, observational dispatch, and strict asynchronous native joins."""

import copy
import importlib
import json
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
    raw["server"]["compiled_path"] = "/old/nxdi"
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


def replace_ns(value, **fields):
    return NS(**{**vars(value), **fields})


def output(prefill=True):
    return NS(
        scheduled_new_reqs=[
            NS(req_id="cmpl-a-0", num_computed_tokens=0, prompt_token_ids=[1] * 504)
        ]
        if prefill
        else [],
        scheduled_cached_reqs=NS(
            req_ids=[] if prefill else ["cmpl-a-0", "cmpl-b-0"],
            num_computed_tokens=[] if prefill else [504, 507],
        ),
        num_scheduled_tokens={"cmpl-a-0": 504} if prefill else {"cmpl-a-0": 1, "cmpl-b-0": 1},
        num_scheduled_tokens_padded={"cmpl-a-0": 512}
        if prefill
        else {"cmpl-a-0": 1, "cmpl-b-0": 1},
    )


def metric(prefill=True, index=0):
    return iteration_metric(
        scheduler_snapshot(output(prefill)),
        prompt_lengths={"cmpl-a-0": 504, "cmpl-b-0": 504},
        iteration=index,
        start_ns=1000,
        stop_ns=2000,
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


def capture():
    metrics = [metric(), metric(False, 1)]
    forwards, events = [], []
    info = {"a" * 32: ("prefill", 512), "b" * 32: ("decode", 16)}
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
                    "block_table_shapes": [[1 if i == 0 else bucket, 16]],
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
    monkeypatch.setattr(vllm_runner, "accepted_forward_evidence", lambda *args: {})
    monkeypatch.setattr(vllm_runner, "capture_binary_snapshot", lambda *args: {})
    monkeypatch.setattr(vllm_runner, "verify_checkpoint", lambda *args: {})
    monkeypatch.setattr(vllm_runner, "verify_corpus", lambda *args: {})
    monkeypatch.setattr(type(cfg.server.container_env()), "validate", lambda self: None)
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
