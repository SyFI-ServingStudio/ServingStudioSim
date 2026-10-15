"""Cache-free command, bounded capture and invalid-evidence rejection on CPU."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from profiling import cli, ops, perf_api
from profiling.runners.neuron import vllm_ops, vllm_ops_engine, vllm_segment_trace

SHAPE = {
    "phase": "decode",
    "token_bucket": 16,
    "max_model_len": 512,
    "kv_blocks": 6782,
    "block_size": 32,
    "tp_size": 4,
    "dtype": "bf16",
}


def manifest():
    return {
        "schema_version": 1,
        "kernel": {"kind": "neuron_llama_forward", "backend": "vllm_neuron"},
        "shape": SHAPE,
        "hardware": {"name": "AWS Trainium2 LNC2", "gpu_count": 1},
        "sessions": [],
    }


@pytest.mark.parametrize("runtime", [{"median": 55.0}, 55.0, {"median": float("nan")}])
def test_public_ops_offline_validates_actual_measurement_contract(
    tmp_path, monkeypatch, capsys, runtime
):
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps(manifest()))
    output = tmp_path / "result"
    calls = []

    def measured(kind, shape, **kwargs):
        calls.append((kind, shape, kwargs))
        root = kwargs["output_dir"]
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "runtime_ms": runtime,
                    "per_op": [{"operation": "kv_gather"}],
                }
            )
        )
        (root / "operations.csv").write_text("operation,active_us\nkv_gather,900\n")
        (root / "neuron_ops.png").write_bytes(b"image")
        return {
            "shape": shape,
            "observed_gpu_name": "AWS Trainium2 LNC2",
            "gpu_count": 1,
            "artifacts": [
                str(root / name) for name in ("summary.json", "operations.csv", "neuron_ops.png")
            ],
            "plots": [str(root / "neuron_ops.png")],
            "persisted": False,
        }

    monkeypatch.setattr(perf_api, "profile_ops", measured)
    monkeypatch.setattr(
        perf_api,
        "get_neuron_llama_forward_times",
        lambda *a, **k: pytest.fail("timing DB path used"),
    )
    code = cli.main(["ops", "--from-capture", str(capture), "--output-dir", str(output), "--json"])
    if runtime != {"median": 55.0}:
        assert code == 2
        assert not (output / "kernel-measurement.meta.json").exists()
        return
    assert code == 0
    metadata = json.loads((output / "kernel-measurement.meta.json").read_text())
    assert metadata["measurement_id"].startswith("km_")
    assert metadata["duration_s"] is None
    assert metadata["plots"] == ["neuron_ops.png"]
    assert metadata["gpu"] == {"cache_key": None, "observed_name": "AWS Trainium2 LNC2", "count": 1}
    assert calls[0][0:2] == ("neuron_llama_forward", SHAPE)
    assert calls[0][2]["from_capture"] == capture
    assert json.loads(capsys.readouterr().out)["persisted"] is False


def test_offline_rejects_explicit_identity_conflict_before_capture(tmp_path, monkeypatch):
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps(manifest()))
    monkeypatch.setattr(perf_api, "profile_ops", lambda *a, **k: pytest.fail("capture invoked"))
    assert (
        cli.main(
            [
                "ops",
                "single_gemm",
                "--from-capture",
                str(capture),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )


def test_ops_refuses_overwriting_a_complete_artifact(tmp_path, monkeypatch):
    (tmp_path / "summary.json").write_text("prior evidence")
    monkeypatch.setattr(perf_api, "profile_ops", lambda *a, **k: pytest.fail("capture invoked"))
    assert (
        cli.main(
            [
                "ops",
                "neuron_llama_forward",
                "--backend",
                "vllm_neuron",
                "--spec",
                json.dumps(SHAPE),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert (tmp_path / "summary.json").read_text() == "prior evidence"


def test_ops_offline_uses_bounded_child_and_never_requests_device(tmp_path, monkeypatch):
    from profiling import gpu_policy

    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps(manifest()))
    output = tmp_path / "out"
    monkeypatch.setattr(
        gpu_policy, "require_gpu", lambda *a: pytest.fail("offline requested hardware")
    )

    def run(command, **kwargs):
        assert command[1:3] == ["-m", "profiling.ops"]
        assert command[-2:] == ["--memory-limit-gb", "20"]
        (output / "analysis-result.json").write_text(
            json.dumps({"artifacts": [], "observed_gpu_name": "AWS Trainium2 LNC2"})
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ops.subprocess, "run", run)
    result = ops.profile_ops(
        "neuron_llama_forward",
        SHAPE,
        backend="vllm_neuron",
        output_dir=output,
        from_capture=capture,
        memory_limit_gb=20,
    )
    assert result["persisted"] is False
    assert str(output / "capture-manifest.json") in result["artifacts"]


@pytest.mark.parametrize("buffer", ["dma_buffer_mb", "instruction_buffer_mb"])
def test_invalid_notification_queue_size_fails_before_device_or_analysis(
    tmp_path, monkeypatch, buffer
):
    from profiling import gpu_policy

    monkeypatch.setattr(gpu_policy, "require_gpu", lambda *a: pytest.fail("device requested"))
    monkeypatch.setattr(ops.subprocess, "run", lambda *a, **k: pytest.fail("analysis started"))
    with pytest.raises(ValueError, match="power of two"):
        ops.profile_ops(
            "neuron_llama_forward",
            SHAPE,
            backend="vllm_neuron",
            output_dir=tmp_path / "out",
            **{buffer: 384},
        )


@pytest.mark.parametrize(
    "phase,batch,delay", [("prefill", 16, 16), ("decode", 1, 4), ("decode", 16, 19)]
)
@pytest.mark.parametrize("forwards", [1, 2])
def test_capture_configuration_preserves_production_graph_inputs(
    tmp_path, phase, batch, delay, forwards
):
    original = {
        "model": "/model",
        "dtype": "bfloat16",
        "max_model_len": 512,
        "max_num_seqs": 16,
        "tensor_parallel_size": 4,
        "additional_config": {
            "neuron_config": {"num_seqs_buckets": [1, 16]},
            "neuron_profiler": {"activities": ["system_profile"]},
        },
        "profiler_config": {"profiler": "cuda"},
    }
    before = copy.deepcopy(original)
    if phase == "prefill" and forwards == 2:
        with pytest.raises(ValueError, match="one selected forward"):
            vllm_ops_engine.capture_config(
                original, tmp_path, {"ranks": [0], "forwards": forwards},
                {"prompts": [[1]] * batch}, phase,
            )
        assert original == before
        return
    configured = vllm_ops_engine.capture_config(
        original,
        tmp_path,
        {"ranks": [0], "forwards": forwards},
        {"prompts": [[1]] * batch},
        phase,
    )
    assert original == before
    assert configured["profiler_config"] == {
        "profiler": "cuda",
        "delay_iterations": 0 if phase == "prefill" else delay,
        "max_iterations": 0 if phase == "prefill" else forwards + 1,
    }
    assert (
        configured["additional_config"]["neuron_config"]
        == original["additional_config"]["neuron_config"]
    )
    configured.pop("profiler_config")
    original.pop("profiler_config")
    configured["additional_config"].pop("neuron_profiler")
    original["additional_config"].pop("neuron_profiler")
    if phase == "prefill":
        from profiling.runners.neuron.vllm_ops_control import WORKER_CLASS

        assert configured.pop("worker_cls") == WORKER_CLASS
    assert configured == original


@pytest.mark.parametrize("context", [128, 512, 1024, 2048, 4096, 8192])
def test_prefill_guard_window_uses_canonical_serial_prompt_sequence(tmp_path, context):
    from profiling.runners.neuron.vllm_forward_engine import cases_for_plan, engine_config

    shape = {**SHAPE, "phase": "prefill", "token_bucket": context, "max_model_len": context}
    plan = {"context": context, "specs": [shape], "model": "/model"}
    tokenizer = SimpleNamespace(encode=lambda text: list(range(context)))
    cases = cases_for_plan(plan, tokenizer)
    assert len(cases) == 1 and len(cases[0]["prompts"]) == 16
    original = engine_config(plan, cases, tmp_path, "profile")
    configured = vllm_ops_engine.capture_config(
        original, tmp_path, {"ranks": [0], "forwards": 1}, cases[0], "prefill"
    )
    assert configured["profiler_config"]["delay_iterations"] == 0
    assert configured["profiler_config"]["max_iterations"] == 0
    assert configured["max_num_seqs"] >= 16
    assert all(len(prompt) < configured["max_num_batched_tokens"]
               for prompt in cases[0]["prompts"])


def test_two_prefill_forwards_are_rejected_before_device(tmp_path, monkeypatch):
    from profiling import gpu_policy

    monkeypatch.setattr(gpu_policy, "require_gpu", lambda *a: pytest.fail("device requested"))
    with pytest.raises(ValueError, match="one selected forward"):
        ops.profile_ops(
            "neuron_llama_forward", {**SHAPE, "phase": "prefill", "token_bucket": 512},
            backend="vllm_neuron", output_dir=tmp_path / "out", forwards=2,
        )


@pytest.mark.parametrize("phase,expected", [("prefill", 1), ("decode", 2)])
def test_new_capture_defaults_selected_count_by_phase(tmp_path, monkeypatch, phase, expected):
    from profiling import exec as execution
    from profiling import gpu_policy

    shape = {**SHAPE, "phase": phase, "token_bucket": 512 if phase == "prefill" else 16}
    monkeypatch.setattr(gpu_policy, "require_gpu", lambda *a: None)

    def run(kind, spec, request):
        assert request["forwards"] == expected
        raise RuntimeError("capture reached")

    def chunks(*a, **k):
        yield SimpleNamespace(run_ops=run)

    monkeypatch.setattr(
        execution, "get_default_pool", lambda *a: SimpleNamespace(acquire_chunks=chunks)
    )
    with pytest.raises(RuntimeError, match="capture reached"):
        ops.profile_ops("neuron_llama_forward", shape, backend="vllm_neuron",
                        output_dir=tmp_path / "out")


def guard_fixture():
    graphs = {
        "a" * 32: {"phase": "prefill", "token_bucket": 512},
        "b" * 32: {"phase": "decode", "token_bucket": 16},
    }
    rows = [
        {"neff_name": f"graph_{'a' * 32}.neff", "execution_index": 7,
         "execution_start_ts": 100, "execution_end_ts": 200},
        {"neff_name": f"graph_{'b' * 32}.neff", "execution_index": 9,
         "execution_start_ts": 210, "execution_end_ts": 250},
    ]
    return graphs, rows


@pytest.mark.parametrize("failure", [None, "same_prefill", "missing", "extra", "unknown",
                                      "duplicate_index", "wrong_guard",
                                      "wrong_prefill", "wrong_prefill_graph",
                                      "wrong_prefill_bucket",
                                      "overlap", "reversed", "no_prefill", "barrier"])
def test_prefill_selection_requires_complete_validated_trailing_guard(failure):
    graphs, rows = guard_fixture()
    if failure == "same_prefill":
        rows[1]["neff_name"] = rows[0]["neff_name"]
    elif failure == "missing":
        rows.pop()
    elif failure == "extra":
        rows.append(dict(rows[1], execution_index=10))
    elif failure == "unknown":
        rows[1]["neff_name"] = f"graph_{'c' * 32}.neff"
    elif failure == "duplicate_index":
        rows[1]["execution_index"] = 7
    elif failure == "wrong_guard":
        graphs["b" * 32]["token_bucket"] = 1
    elif failure == "wrong_prefill":
        graphs["a" * 32]["token_bucket"] = 1024
    elif failure == "wrong_prefill_graph":
        graphs["c" * 32] = {"phase": "prefill", "token_bucket": 512}
        rows[1]["neff_name"] = f"graph_{'c' * 32}.neff"
    elif failure == "wrong_prefill_bucket":
        graphs["c" * 32] = {"phase": "prefill", "token_bucket": 1024}
        rows[1]["neff_name"] = f"graph_{'c' * 32}.neff"
    elif failure == "overlap":
        rows[1]["execution_start_ts"] = 150
    elif failure == "reversed":
        rows[1].update(execution_start_ts=50, execution_end_ts=90)
    elif failure == "no_prefill":
        rows[0]["neff_name"] = rows[1]["neff_name"]
    elif failure == "barrier":
        rows.append(dict(rows[1], neff_name="barrier_foo.neff", execution_index=11))
    shape = {**SHAPE, "phase": "prefill", "token_bucket": 512}
    request = {"forwards": 1, "guard_decode_bucket": 16}
    if failure is not None:
        with pytest.raises(ValueError, match="guard|unvalidated"):
            vllm_ops._select_executions(rows, graphs, shape, request)
    else:
        traced, selection = vllm_ops._select_executions(
            list(reversed(rows)), graphs, shape, request
        )
        assert traced == set(graphs)
        assert selection == {
            "policy": "prefill_with_trailing_decode_guard_v1",
            "selected_indices": [7],
            "guard_indices": [9],
        }


@pytest.mark.parametrize("changed_guard", [False, True])
def test_capture_keeps_guard_neff_with_its_actual_geometry(tmp_path, monkeypatch, changed_guard):
    from profiling.profilers import neuron_ops_labels
    from profiling.runners.neuron import parquet_columns
    from profiling.runners.neuron.vllm_identity import sha256

    graphs, rows = guard_fixture()
    session = tmp_path / "instance_pid_42_nc_0_session_0"
    session.mkdir()
    for graph_id, graph in graphs.items():
        path = tmp_path / "device-profiles/neffs" / graph_id / f"graph_{graph_id}.neff"
        path.parent.mkdir(parents=True)
        path.write_bytes(graph_id.encode())
        graph["binary"] = {"neff_sha256": sha256(path)}
        if changed_guard and graph["phase"] == "decode":
            path.write_bytes(b"changed guard")
    monkeypatch.setattr(vllm_segment_trace, "export_device_sessions", lambda *a, **k: [session])
    monkeypatch.setattr(parquet_columns, "read_rows", lambda *a: rows)
    monkeypatch.setattr(
        neuron_ops_labels, "export_static_map", lambda path: {"neff_sha256": sha256(path)}
    )

    def capture():
        return vllm_ops._capture_sessions(
            tmp_path, graphs, {**SHAPE, "phase": "prefill", "token_bucket": 512},
            {"ranks": [0], "forwards": 1, "guard_decode_bucket": 16},
        )

    if changed_guard:
        with pytest.raises(ValueError, match="sealed stock bytes"):
            capture()
        return
    result = capture()
    assert [(entry["phase"], entry["token_bucket"]) for entry in result[0]["neffs"]] == [
        ("prefill", 512), ("decode", 16),
    ]
    assert result[0]["execution_selection"]["guard_indices"] == [9]


@pytest.mark.parametrize("include_ops", [False, True])
def test_per_op_export_keeps_hierarchy_dma_and_events(tmp_path, monkeypatch, include_ops):
    profiles, output = tmp_path / "profiles", tmp_path / "out"
    rank = profiles / "instance_pid_42"
    rank.mkdir(parents=True)
    (rank / "trace.ntff").write_bytes(b"capture")
    (profiles / "neffs").mkdir()

    def run(command, **kwargs):
        for option in (
            "--ignore-event-trace",
            "--ignore-dma-trace",
            "--ignore-instruction-hierarchy",
        ):
            assert (option not in command) == include_ops
        parquet = Path(command[command.index("--output-file") + 1])
        session = parquet / "instance_pid_42_nc_0_session_0"
        session.mkdir(parents=True)
        (session / "ExecutionInfo.parquet").touch()
        status = Path(command[command.index("--processing-status-file") + 1])
        status.write_text(json.dumps({"status_code": 0, "profiles": [{"status": "success"}]}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(vllm_segment_trace.subprocess, "run", run)
    assert (
        len(vllm_segment_trace.export_device_sessions(profiles, output, include_ops=include_ops))
        == 1
    )


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_dma_notification_loss_cannot_be_accepted_by_per_op_capture(tmp_path, monkeypatch, phase):
    from profiling.runners.neuron import parquet_columns

    session = tmp_path / "instance_pid_42_nc_0_session_0"
    session.mkdir()
    (session / "Warning.parquet").touch()
    monkeypatch.setattr(vllm_segment_trace, "export_device_sessions", lambda *a, **k: [session])
    monkeypatch.setattr(
        parquet_columns,
        "read_rows",
        lambda *a, **k: [
            {
                "message": "Notification from block DMA of type DMA were dropped. "
                "This can cause corresponding data to be incorrect."
            }
        ],
    )
    with pytest.raises(ValueError, match="lost notifications"):
        vllm_ops._capture_sessions(tmp_path, {}, {**SHAPE, "phase": phase},
                                   {"ranks": [0], "forwards": 1, "guard_decode_bucket": 16})


@pytest.mark.parametrize("table", ["Warning", "ExecutionInfo"])
def test_capture_rejects_split_metadata_instead_of_ignoring_rows(tmp_path, monkeypatch, table):
    session = tmp_path / "instance_pid_42_nc_0_session_0"
    session.mkdir()
    (session / f"{table}_0.parquet").touch()
    monkeypatch.setattr(vllm_segment_trace, "export_device_sessions", lambda *a, **k: [session])
    with pytest.raises(ValueError, match="split metadata tables"):
        vllm_ops._capture_sessions(tmp_path, {}, SHAPE, {"ranks": [0], "forwards": 2})


def test_manifest_relocation_resolves_input_paths_without_inventing_hardware(tmp_path):
    original = manifest()
    original.pop("hardware")
    original["sessions"] = [
        {
            "session_dir": "trace/session",
            "neffs": [
                {
                    "path": "graph.neff",
                    "static_instructions": "labels.json",
                    "sha256": "abc",
                }
            ],
        }
    ]
    original["semantic_source_identity"] = {"source_hashes": {"model.py": "abc"}}
    relocated = ops.normalize_manifest_paths(original, tmp_path)
    assert relocated["sessions"][0]["session_dir"] == str(tmp_path / "trace/session")
    assert relocated["sessions"][0]["neffs"][0]["path"] == str(tmp_path / "graph.neff")
    assert relocated["semantic_source_identity"] == original["semantic_source_identity"]
    assert "hardware" not in relocated
    assert original["sessions"][0]["neffs"][0]["path"] == "graph.neff"


def numerical_manifest(root):
    from profiling.runners.neuron.vllm_identity import sha256

    value = manifest()
    neff = {"graph_id": "graph", "phase": "decode", "token_bucket": 16, "sha256": "a" * 64}
    value["sessions"] = [{"neffs": [neff]}]
    accuracy = root / "precision.json"
    accuracy.write_text(json.dumps({"passed": True}))
    binary = root / "binary.json"
    binary.write_text(json.dumps({
        "precision_passed": True,
        "binary_binding_available": True,
        "scope": "warm_cache_neff_bytes_stable_across_public_accuracy_calls",
        "files": {"precision.json": sha256(accuracy)},
        "graphs": {"graph": {"phase": "decode", "token_bucket": 16,
                             "binary": {"neff_sha256": neff["sha256"]}}},
    }))
    gate = root / "gate.json"
    gate.write_text(json.dumps({
        "passed": True, "binary_receipt_sha256": sha256(binary),
        "graphs": {"graph": {"neff_sha256": neff["sha256"]}},
    }))
    value["execution"] = {}
    for name, path in (("accuracy_receipt", accuracy), ("binary_receipt", binary),
                       ("identity_gate", gate)):
        value["execution"][name] = str(path)
        value["execution"][f"{name}_sha256"] = sha256(path)
    return value


@pytest.mark.parametrize("name", ["accuracy_receipt", "binary_receipt", "identity_gate"])
@pytest.mark.parametrize("failure", [None, "missing", "changed", "missing_hash"])
def test_offline_receipts_require_retained_exact_bytes(tmp_path, name, failure):
    capture = numerical_manifest(tmp_path)
    path = Path(capture["execution"][name])
    if failure == "missing":
        path.unlink()
    elif failure == "changed":
        path.write_text("{}")
    elif failure == "missing_hash":
        capture["execution"].pop(f"{name}_sha256")
    if failure:
        with pytest.raises(ValueError, match="receipt"):
            ops.validate_execution_receipts(capture)
    else:
        assert ops.validate_execution_receipts(capture)["status"] == "passed"


@pytest.mark.parametrize("failure", ["failed", "unbound", "other_neff"])
def test_matching_receipt_hashes_do_not_replace_numerical_or_neff_proof(tmp_path, failure):
    from profiling.runners.neuron.vllm_identity import sha256

    capture = numerical_manifest(tmp_path)
    if failure == "other_neff":
        capture["sessions"][0]["neffs"][0]["sha256"] = "b" * 64
    else:
        name = "accuracy_receipt" if failure == "failed" else "binary_receipt"
        path = Path(capture["execution"][name])
        receipt = json.loads(path.read_text())
        receipt["passed" if failure == "failed" else "binary_binding_available"] = False
        path.write_text(json.dumps(receipt))
        capture["execution"][f"{name}_sha256"] = sha256(path)
    with pytest.raises(ValueError, match="validation|bound"):
        ops.validate_execution_receipts(capture)


def test_old_research_manifests_explicitly_lack_numerical_validation():
    result = ops.validate_execution_receipts(manifest())
    assert result["status"] == "absent"
    assert result["validated"] is False


@pytest.mark.parametrize("stage", ["capture", "labels"])
@pytest.mark.parametrize("failure", [False, True])
def test_long_artifact_roots_use_short_owned_ipc_paths(tmp_path, monkeypatch, stage, failure):
    root = tmp_path / ("long" * 30) / "debug-labels"
    root.mkdir(parents=True)
    sockets = []

    def run(path, selected_stage, *, engine, env):
        assert path == root and selected_stage == stage
        socket_dir = Path(env["VLLM_RPC_BASE_PATH"])
        sockets.append(socket_dir)
        assert socket_dir.is_dir()
        assert len(str(socket_dir).encode()) + 1 + 36 <= 107
        assert env["TMPDIR"] == str(root / "tmp")
        if failure:
            raise RuntimeError("stage failed")

    monkeypatch.setattr(vllm_ops, "run_stage", run)
    if failure:
        with pytest.raises(RuntimeError, match="stage failed"):
            vllm_ops._run_ops_stage(root, stage, {"TMPDIR": str(root / "tmp")})
    else:
        vllm_ops._run_ops_stage(root, stage, {"TMPDIR": str(root / "tmp")})
    assert len(sockets) == 1 and not sockets[0].exists()
    assert root.is_dir()


@pytest.mark.parametrize("mismatch", [None, "neff", "static"])
def test_labels_receipt_is_bound_to_exact_stock_bytes(tmp_path, mismatch):
    from profiling.runners.neuron.vllm_identity import sha256

    static = tmp_path / "static.json"
    static.write_text(json.dumps({"neff_sha256": "abc"}))
    receipt = tmp_path / "labels.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "graphs": {
                    "graph": {
                        "neff_sha256": "other" if mismatch == "neff" else "abc",
                        "static_instructions": "static.json",
                        "static_instructions_sha256": "other"
                        if mismatch == "static"
                        else sha256(static),
                    }
                },
            }
        )
    )
    capture = {"sessions": [{"neffs": [{"graph_id": "graph", "sha256": "abc"}]}]}
    if mismatch:
        with pytest.raises(ValueError, match="different stock NEFF|instructions changed"):
            ops.apply_labels_map(capture, receipt)
    else:
        ops.apply_labels_map(capture, receipt)
        assert capture["sessions"][0]["neffs"][0]["static_instructions"] == str(static)


def test_debug_labels_execute_checked_tokens_without_starting_timing(tmp_path, monkeypatch):
    from profiling.runners.neuron import vllm_identity

    calls = []

    class LLM:
        def __init__(self, **config):
            assert "profiler_config" not in config
            assert "neuron_profiler" not in config["additional_config"]

        def generate(self, prompts, params, **kwargs):
            calls.append(prompts)
            return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(token_ids=[2])])]

        def start_profile(self):
            pytest.fail("debug label timings must not be captured")

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: kwargs)
    )
    monkeypatch.setattr(vllm_identity, "stock_compile_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(vllm_identity, "compile_cache_snapshot", lambda path: {"graph": {}})
    (tmp_path / "accuracy-outputs.json").write_text(
        json.dumps(
            [
                {"case_id": "case-slot0", "token_ids": [2]},
            ]
        )
    )
    (tmp_path / "ops-plan.json").write_text(
        json.dumps(
            {
                "validated_run": str(tmp_path),
                "case": {"id": "case", "shape_id": "b1-s1", "bucket": 1, "prompts": [[1]]},
                "profile_config": {
                    "profiler_config": {},
                    "additional_config": {"neuron_profiler": {}},
                },
            }
        )
    )
    vllm_ops_engine.main(tmp_path, "labels")
    assert len(calls) == 1
    assert json.loads((tmp_path / "label-compile-cache.json").read_text())["graphs"] == {
        "graph": {}
    }


def test_debug_labels_leave_trailing_guard_stock_map_unchanged(tmp_path, monkeypatch):
    from profiling.profilers import neuron_ops_labels
    from profiling.runners.neuron import vllm_forward_trace
    from profiling.runners.neuron.vllm_identity import sha256

    entries = []
    for phase, bucket, graph_id in (("prefill", 512, "a" * 32), ("decode", 16, "b" * 32)):
        path = tmp_path / f"static-{graph_id}.json"
        path.write_text(json.dumps({"neff_sha256": graph_id * 2}))
        entries.append({"phase": phase, "token_bucket": bucket, "graph_id": graph_id,
                        "sha256": graph_id * 2, "static_instructions": str(path),
                        "static_instructions_sha256": sha256(path)})
    guard_before = dict(entries[1])

    def compile_labels(root, stage, env):
        graph = "c" * 32
        path = root / "cache" / graph / f"graph_{graph}.neff"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"debug prefill")
        (root / "label-compile-cache.json").write_text(json.dumps({
            "compile_cache_dir": str(root / "cache"),
            "graphs": {graph: {"neff_sha256": sha256(path), "input_metadata": "prefill"}},
        }))

    calls = []

    def transfer(stock, debug, **kwargs):
        calls.append(stock["neff_sha256"])
        return {**stock, "debug_transfer": True}

    monkeypatch.setattr(vllm_ops, "_run_ops_stage", compile_labels)
    monkeypatch.setattr(vllm_forward_trace, "plan_geometry", lambda *a: ("prefill", 512))
    monkeypatch.setattr(neuron_ops_labels, "export_static_map", lambda path: {"debug": True})
    monkeypatch.setattr(neuron_ops_labels, "transfer_debug_labels", transfer)
    vllm_ops._debug_labels(
        tmp_path, {"shape": {**SHAPE, "phase": "prefill", "token_bucket": 512}},
        [{"neffs": entries}], {},
    )
    assert calls == ["a" * 64]
    assert entries[1] == guard_before
    assert json.loads(Path(entries[0]["static_instructions"]).read_text())["debug_transfer"]


def test_plot_reads_only_declared_per_op_summary(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    from profiling.profilers.neuron_ops_plot import render_ops

    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "per_op": [
                    {
                        "phase": "decode",
                        "token_bucket": 16,
                        "operation": "kv_gather",
                        "active_us": 200,
                        "overlap_us": 40,
                    },
                    {
                        "phase": "decode", "token_bucket": 16,
                        "operation": "compiler_transpose", "active_us": 100, "overlap_us": 20,
                        "attribution_category": "compiler_primitive",
                    },
                ]
            }
        )
    )
    captured = []
    original = Axes.set_yticks

    def capture_labels(axis, ticks, labels=None, **kwargs):
        if labels:
            captured.extend(labels)
        return original(axis, ticks, labels=labels, **kwargs)

    monkeypatch.setattr(Axes, "set_yticks", capture_labels)
    output = tmp_path / "ops.png"
    assert render_ops(summary, output) == output
    assert captured == [
        "decode B16 · kv_gather", "decode B16 · compiler_transpose [compiler primitive]"
    ]
    assert output.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("primitive", [False, True])
def test_coverage_plot_preserves_overlapping_label_fractions(tmp_path, monkeypatch, primitive):
    from matplotlib.axes import Axes

    from profiling.profilers.neuron_ops_plot import render_coverage

    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "quality": {
                    "per_engine_active_coverage": {
                        "TensorMatrix": {
                            "active_ns": 100,
                            "labeled_active_ns": 40,
                            "unresolved_active_ns": 80,
                            **({"model_labeled_active_ns": 20, "compiler_primitive_active_ns": 50}
                               if primitive else {}),
                        },
                        "Sync": {"active_ns": 0, "labeled_active_ns": 0, "unresolved_active_ns": 0},
                    }
                }
            }
        )
    )
    widths = []
    original = Axes.barh

    def capture_bars(axis, y, width, **kwargs):
        widths.append(width)
        return original(axis, y, width, **kwargs)

    monkeypatch.setattr(Axes, "barh", capture_bars)
    output = tmp_path / "coverage.png"
    assert render_coverage(summary, output) == output
    assert widths == ([[20.0], [50.0], [80.0]] if primitive else [[40.0], [80.0]])
    assert output.read_bytes().startswith(b"\x89PNG")


def test_coverage_plot_is_optional_for_reports_without_engine_coverage(tmp_path):
    from profiling.profilers.neuron_ops_plot import render_coverage

    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"per_op": []}))
    output = tmp_path / "coverage.png"
    assert render_coverage(summary, output) is None
    assert not output.exists()


def test_analysis_child_declares_both_rendered_plots(tmp_path, monkeypatch):
    import resource

    from profiling.profilers import neuron_ops, neuron_ops_plot

    monkeypatch.setattr(resource, "setrlimit", lambda *args: None)
    capture = manifest()
    capture["sampling"] = {"window": "prefill_with_decode_tail_v3",
                           "worker_delay_iterations": 16}
    (tmp_path / "capture.json").write_text(json.dumps(capture))
    monkeypatch.setattr(neuron_ops, "analyze_capture", lambda *args: {
        "artifacts": [], "plots": [], "summary": {"quality": {}},
    })

    def render(summary, output):
        output.write_bytes(b"image")
        return output

    monkeypatch.setattr(neuron_ops_plot, "render_ops", render)
    monkeypatch.setattr(neuron_ops_plot, "render_coverage", render)
    ops.main([str(tmp_path / "capture.json"), str(tmp_path), "--memory-limit-gb", "1"])
    result = json.loads((tmp_path / "analysis-result.json").read_text())
    assert result["summary"]["sampling"] == capture["sampling"]
    expected = [str(tmp_path / name) for name in ("neuron_ops.png", "neuron_ops_coverage.png")]
    assert result["plots"] == result["artifacts"] == expected
    assert result["summary"]["quality"]["numerical_provenance"]["status"] == "absent"
    assert json.loads((tmp_path / "quality.json").read_text())["numerical_provenance"] == (
        result["summary"]["quality"]["numerical_provenance"]
    )


@pytest.mark.parametrize("first_pass", ["cold", "failed", "unknown"])
def test_only_successful_cold_validation_gets_one_warm_recheck(tmp_path, monkeypatch, first_pass):
    stages = []
    identity = {"runtime": "pinned"}

    def stage(root, name):
        stages.append((root.name, name))
        if name == "accuracy":
            (root / "accuracy-binaries-before-load.json").write_text(
                json.dumps({"graph": {}} if first_pass == "unknown" else {})
            )
        else:
            (root / "precision.json").write_text(
                json.dumps(
                    {
                        "by_shape": {"b16-s504": {"passed": True}},
                    }
                )
            )

    seals = []

    def seal(root):
        seals.append(root)
        return {
            "binary_binding_available": len(seals) > 1,
            "precision_passed": first_pass != "failed",
            "graphs": {"graph": {}},
            "load": {"loader_coverage": {"ranks_by_graph": {"graph": []}}},
        }

    monkeypatch.setattr(vllm_ops, "run_stage", stage)
    monkeypatch.setattr(vllm_ops, "seal_accuracy_binaries", seal)
    monkeypatch.setattr(
        vllm_ops,
        "load_accuracy_binaries",
        lambda root: {
            "available": True,
            "load": {"source": {}},
            "graphs": {},
        },
    )
    monkeypatch.setattr(vllm_ops, "accuracy_source_identity", lambda: {})
    monkeypatch.setattr(vllm_ops, "stock_compile_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(vllm_ops, "verify_cache_binaries", lambda *args: {})
    if first_pass == "cold":
        validated, _, _ = vllm_ops._validation(tmp_path, SHAPE, tmp_path, tmp_path, identity, None)
        assert validated == tmp_path / "validation"
        assert (tmp_path / "compile-warmup/precision.json").exists()
        assert len(seals) == 2 and len(stages) == 4
    else:
        with pytest.raises(ValueError, match="initial numerical|other than cold"):
            vllm_ops._validation(tmp_path, SHAPE, tmp_path, tmp_path, identity, None)
        assert len(seals) == 1 and len(stages) == 2


def test_worker_ops_observes_reservation_before_lazy_capture(tmp_path, monkeypatch):
    from dataclasses import replace

    from profiling.db.registry import find_kernel_profiler_spec
    from profiling.exec import local_worker, neuron

    order = []

    def capture(shape, request):
        assert shape == SHAPE and request == {"forwards": 2}
        order.append("capture")
        return {"manifest_path": str(tmp_path / "capture.json")}

    profiler = replace(
        find_kernel_profiler_spec("neuron_llama_forward", "vllm_neuron"),
        ops_runner_ref=SimpleNamespace(load=lambda: capture),
    )
    monkeypatch.setattr(local_worker, "find_kernel_profiler_spec", lambda *args: profiler)
    monkeypatch.setattr(local_worker, "unsupported_device", lambda *args: None)
    monkeypatch.setattr(local_worker, "_neuron_versions", lambda: {"runtime": "checked"})

    def observe(request):
        order.append("reservation")
        return SimpleNamespace(gpu_name="AWS Trainium2 LNC2")

    monkeypatch.setattr(neuron, "observe_neuron_reservation", observe)
    request, output = tmp_path / "request.json", tmp_path / "output.json"
    request.write_text(
        json.dumps(
            {
                "kernel_kind": "neuron_llama_forward",
                "specs": [{**SHAPE, "backend": "vllm_neuron"}],
                "execution": {},
                "ops": {"forwards": 2},
                "energy": False,
            }
        )
    )
    local_worker._worker_main(request, output)
    assert order == ["reservation", "capture"]
    result = json.loads(output.read_text())["ops"]
    assert result["observed_gpu_name"] == "AWS Trainium2 LNC2"
    assert result["runtime_versions"] == {"runtime": "checked"}
