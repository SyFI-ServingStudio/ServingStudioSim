"""CPU tests for the four explicit alignment launcher phase contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from alignment import runner as alignment_runner
from alignment.load_generator import runner as load_runner
from alignment.profiler import engine_records, record_extraction, sglang_server, vllm_server
from alignment.profiler.config import ServerConfig
from alignment.timing_predict_input import BuildRequest, build_inputs
from launcher import alignment as alignment_launcher
from launcher import exec as launcher_exec
from launcher import timing_predict as timing_predict_launcher
from launcher.alignment_config import load_analyze_config, load_profile_config


def _write_config(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config) if path.suffix == ".json" else yaml.safe_dump(config))


def _phase_configs(tmp_path: Path, suffix: str = ".yaml") -> dict[str, Path]:
    trace = tmp_path / "trace" / "shared.csv"
    corpus = tmp_path / "trace" / "corpus.txt"
    trace.parent.mkdir()
    trace.write_text("id,input_len,output_len,arrival_time\n")
    corpus.write_text("hello\n")

    paths = {
        "simulation": tmp_path / f"simulation{suffix}",
        "profile": tmp_path / f"profile{suffix}",
        "timing": tmp_path / f"timing_predict{suffix}",
        "analyze_kernel": tmp_path / f"analyze_kernel{suffix}",
        "analyze_e2e": tmp_path / f"analyze_e2e{suffix}",
        "labeled": tmp_path / "kernel_sequences_labeled.json",
    }
    _write_config(
        paths["simulation"],
        {
            "deployment": "unified",
            "workload": {
                "trace_files": [str(trace)],
                "arrival_mode": "trace_timed",
                "session_dependency": "independent",
            },
            "io": {"log_dir": str(tmp_path / "simulation_run")},
            "pools": {
                "main": {
                    "groups": [
                        {
                            "gpu": "NVIDIA H200",
                            "replicas": 1,
                            "arch": {
                                "type": "llama3_dense",
                                "model_config": "model/config/llama3_8b.json",
                            },
                            "worker": {"type": "barebone"},
                        }
                    ]
                }
            },
        },
    )
    _write_config(
        paths["profile"],
        {
            "schema_version": 1,
            "name": "profile_test",
            "log_dir": "./profile_run",
            "gpu": "NVIDIA H200",
            "server": {"model_path": "model"},
            "workload": {
                "frontend": {"type": "independent", "path": "./trace/shared.csv"},
                "text_file": "./trace/corpus.txt",
                "tokenizer": "model/tokenizer.json",
                "max_concurrency": 64,
            },
        },
    )
    _write_config(
        paths["timing"],
        {
            "schema_version": 1,
            # Single-pass DAG: timing-predict reads the sim *preset* for
            # gpu/arch/backends, so it runs before any completed simulation.
            "simulation_preset": f"./simulation{suffix}",
            "profile_log_dir": "./profile_run",
            "log_dir": "./timing_predict_run",
            "input_builder": {
                "type": "vllm_text",
                "measured_phase": "forward",
                "group_assignment": "single",
            },
        },
    )
    _write_config(
        paths["analyze_kernel"],
        {
            "schema_version": 1,
            "profile_log_dir": "./profile_run",
            "timing_predict_log_dir": "./timing_predict_run",
            "log_dir": "./analysis_kernel_run",
            "iteration": {
                "enabled": True,
                "labeled_kernel_sequences_file": "./kernel_sequences_labeled.json",
            },
        },
    )
    _write_config(
        paths["analyze_e2e"],
        {
            "schema_version": 1,
            "simulation_log_dir": "./simulation_run",
            "workload_profile_log_dir": "./profile_run",
            "log_dir": "./analysis_e2e_run",
            "workload": {"enabled": True},
            "e2e": {"enabled": True, "throughput_bins": 20},
        },
    )
    _write_config(
        paths["labeled"],
        {
            "schema_version": 2,
            "encoding": "folded-v1",
            "source_parsed": "profile_run/parsed.json",
            "folding_policy": {
                "kind": "exact_contiguous_repeat",
                "match_fields": ["name", "suggested_category"],
                "row_identity": "sequence_id:expanded_ordinal",
            },
            "phases": {
                "forward": {
                    "unique_sequences": [
                        {
                            "sequence_id": "sequence_test",
                            "iterations": [1],
                            "expanded_kernel_count": 1,
                            "program": [
                                {
                                    "kernels": [
                                        {
                                            "name": "attention_kernel",
                                            "suggested_category": "attention",
                                            "label": {
                                                "status": "mapped",
                                                "operation": "attention",
                                                "type": "attention",
                                                "role": "attention main",
                                                "simulated_slots": ["unified.attn"],
                                            },
                                        }
                                    ]
                                }
                            ],
                        }
                    ]
                }
            },
        },
    )
    return paths


def _write_completed_inputs(tmp_path: Path) -> tuple[Path, Path, dict]:
    simulation = tmp_path / "simulation_run"
    profile = tmp_path / "profile_run"
    params = {
        "deployment": "unified",
        "workload": {
            "trace_files": [str(tmp_path / "trace" / "shared.csv")],
            "arrival_mode": "trace_timed",
            "session_dependency": "independent",
        },
        "pools": {
            "main": {
                "groups": [
                    {
                        "gpu": "NVIDIA H200",
                        "replicas": 1,
                        "arch": {"type": "llama3_dense", "fp8": False},
                    }
                ]
            }
        },
    }
    params_path = simulation / "raw" / "params.json"
    params_path.parent.mkdir(parents=True)
    params_path.write_text(json.dumps(params))

    parsed = profile / "parsed.json"
    metrics = profile / "metrics.jsonl"
    replay = profile / "load_generator" / "replay.jsonl"
    parsed.parent.mkdir(parents=True)
    replay.parent.mkdir()
    parsed.write_text(json.dumps(_parsed_iteration()))
    metrics.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "input_adapter": "vllm_text",
                "iteration_index": 34,
                "observed_start_monotonic_ns": 1_000_000,
                "observed_end_monotonic_ns": 2_000_000,
                "observed_elapsed_ms": 1.0,
                "prefill_tokens": 0,
                "decode_requests": 1,
                "decode_tokens_scheduled": 1,
                "prefill_chunk_pairs": [],
                "decode_kv_lens": [100],
            }
        )
    )
    replay.write_text("")
    result = {
        "profile_kind": "workload_metrics",
        "log_dir": str(profile),
        "parsed_nsys": str(parsed),
        "metrics_jsonl": str(metrics),
        "replay_result": str(replay),
        "server_tp_size": 1,
        "drive_summary": {
            "source_trace": str((tmp_path / "trace" / "shared.csv").resolve()),
            "log_path": str(replay),
            "replay_start_monotonic_ns": 900_000,
            "replay_end_monotonic_ns": 2_100_000,
        },
    }
    (profile / "profile_result.json").write_text(json.dumps(result))
    return simulation, profile, result


def _parsed_iteration() -> dict:
    return {
        "iteration_details": [
            {
                "iteration": 34,
                "metrics": {
                    "schema_version": 1,
                    "input_adapter": "vllm_text",
                    "prefill_tokens": 0,
                    "decode_requests": 1,
                    "decode_tokens_scheduled": 1,
                    "prefill_chunk_pairs": [],
                    "decode_kv_lens": [100],
                },
                "ranges": [{"phase": "forward", "kernel_count": 1}],
            }
        ]
    }


def _write_timing_artifacts(tmp_path: Path) -> Path:
    _, profile, profile_result = _write_completed_inputs(tmp_path)
    output = tmp_path / "timing_predict_run"
    timing_config = alignment_launcher.load_timing_predict_config(tmp_path / "timing_predict.yaml")
    result = build_inputs(
        BuildRequest(
            simulation_preset=timing_config.simulation_preset,
            profile_log_dir=profile,
            parsed_nsys=Path(profile_result["parsed_nsys"]),
            output_dir=output,
            gpu="NVIDIA H200",
            arch={"type": "llama3_dense", "fp8": False},
            backends={},
            input_spec=timing_config.input_builder,
        )
    )
    (output / "raw" / "cost_manifest").mkdir(parents=True)
    (output / "raw" / "cost_log").mkdir(parents=True)
    return result.input_manifest


def test_alignment_sim_runs_only_existing_simulation(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    calls = []
    monkeypatch.setattr(
        alignment_launcher,
        "_launch_simulation",
        lambda path, **kwargs: calls.append((path, kwargs)) or 0,
    )

    assert alignment_launcher.main(["sim", str(paths["simulation"])]) == 0
    assert calls[0][0] == paths["simulation"]
    assert calls[0][1]["overrides"] == []
    assert json.loads((tmp_path / "artifact.meta.json").read_text()) == {
        "schema_version": 1,
        "artifact_kind": "alignment_bundle",
    }


def test_alignment_sim_dry_run_does_not_publish_bundle_marker(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    monkeypatch.setattr(alignment_launcher, "_launch_simulation", lambda *args, **kwargs: 0)

    assert alignment_launcher.main(["sim", str(paths["simulation"]), "--dry-run"]) == 0
    assert not (tmp_path / "artifact.meta.json").exists()


def test_alignment_sim_auto_injects_kernel_align_multiplier(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    kernel_align = tmp_path / "kernel_align_run"
    report = kernel_align / "reports" / "alignment_iteration_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"meta": {"recommended_gpu_time_multiplier": 1.329}}))
    calls = []
    monkeypatch.setattr(
        alignment_launcher,
        "_launch_simulation",
        lambda path, **kwargs: calls.append((path, kwargs)) or 0,
    )

    assert (
        alignment_launcher.main(
            ["sim", str(paths["simulation"]), "--gpu-time-multiplier-from", str(kernel_align)]
        )
        == 0
    )
    assert calls[0][1]["overrides"] == ["pools.main.groups.0.worker.gpu_time_multiplier=1.329"]


def test_alignment_sim_rejects_below_unity_multiplier(tmp_path, monkeypatch, capsys):
    paths = _phase_configs(tmp_path)
    kernel_align = tmp_path / "kernel_align_run"
    report = kernel_align / "reports" / "alignment_iteration_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"meta": {"recommended_gpu_time_multiplier": 0.5}}))
    monkeypatch.setattr(alignment_launcher, "_launch_simulation", lambda *args, **kwargs: 0)

    assert (
        alignment_launcher.main(
            ["sim", str(paths["simulation"]), "--gpu-time-multiplier-from", str(kernel_align)]
        )
        == 2
    )
    assert "recommended_gpu_time_multiplier" in capsys.readouterr().err


def test_profile_config_is_profile_only_and_config_relative(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    profiled = []
    resumed = []
    monkeypatch.setattr(
        alignment_launcher,
        "run_profile",
        lambda config, *, resume=False: (
            profiled.append(config)
            or resumed.append(resume)
            or {"parsed_nsys": str(tmp_path / "profile_run" / "parsed.json")}
        ),
    )

    assert alignment_launcher.main(["profile", str(paths["profile"])]) == 0
    assert resumed == [False]
    config = profiled[0]
    assert not hasattr(config, "analysis")
    assert not hasattr(config, "output_dirs")
    assert config.log_dir == str((tmp_path / "profile_run").resolve())
    assert config.workload.frontend.path == str((tmp_path / "trace" / "shared.csv").resolve())


def test_profile_rejects_old_combined_schema(tmp_path, capsys):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["output_dirs"] = {
        "profile": "./profile_run",
        "timing_predict": "./timing_predict_run",
        "analysis": "./analysis_run",
    }
    raw["analysis"] = {"iteration": {"enabled": True}}
    paths["profile"].write_text(yaml.safe_dump(raw))

    assert alignment_launcher.main(["profile", str(paths["profile"]), "--dry-run"]) == 2
    assert "unexpected keyword" in capsys.readouterr().err


def test_all_phase_parsers_accept_json(tmp_path):
    paths = _phase_configs(tmp_path, suffix=".json")
    assert load_profile_config(paths["profile"]).name == "profile_test"
    assert (
        alignment_launcher.load_timing_predict_config(paths["timing"]).log_dir
        == (tmp_path / "timing_predict_run").resolve()
    )
    assert alignment_launcher.load_analyze_config(paths["analyze_e2e"]).e2e.throughput_bins == 20


def test_profile_fork_python_preserves_virtualenv_symlink(tmp_path):
    paths = _phase_configs(tmp_path)
    target = tmp_path / "base-python"
    target.write_text("")
    fork_python = tmp_path / "venv" / "bin" / "python"
    fork_python.parent.mkdir(parents=True)
    fork_python.symlink_to(target)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["fork_python"] = "./venv/bin/python"
    paths["profile"].write_text(yaml.safe_dump(raw))

    config = load_profile_config(paths["profile"])

    assert config.fork_python == str(fork_python)
    assert Path(config.fork_python).resolve() == target


def test_profile_config_counts_tp_times_dp_visible_devices(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["cuda_visible_devices"] = "2,5"
    raw["server"]["tp_size"] = 1
    raw["server"]["dp_size"] = 2
    paths["profile"].write_text(yaml.safe_dump(raw))

    config = load_profile_config(paths["profile"])

    assert config.server.tp_size == 1
    assert config.server.dp_size == 2


def test_profile_config_accepts_an_explicit_pure_tp_expert_degree(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["cuda_visible_devices"] = "0,1,2,3"
    raw["server"]["tp_size"] = 4
    raw["server"]["dp_size"] = 1
    paths["profile"].write_text(yaml.safe_dump(raw))

    assert load_profile_config(paths["profile"]).server.expert_parallel_size is None

    raw["server"]["expert_parallel_size"] = 1
    paths["profile"].write_text(yaml.safe_dump(raw))
    assert load_profile_config(paths["profile"]).server.expert_parallel_size == 1


def test_sglang_popularity_requires_and_validates_its_per_replica_expert_degree(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["engine"] = "sglang"
    raw["profile_kind"] = "expert_popularity"
    raw["cuda_visible_devices"] = "0,1,2,3"
    raw["server"]["tp_size"] = 2
    raw["server"]["dp_size"] = 2
    paths["profile"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="SGLang expert_popularity requires"):
        load_profile_config(paths["profile"])

    raw["server"]["expert_parallel_size"] = 4
    paths["profile"].write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="must divide tp_size=2"):
        load_profile_config(paths["profile"])

    raw["server"]["expert_parallel_size"] = 1
    paths["profile"].write_text(yaml.safe_dump(raw))
    assert load_profile_config(paths["profile"]).server.expert_parallel_size == 1


@pytest.mark.parametrize("expert_parallel_size", [0, True, 3])
def test_profile_config_rejects_an_invalid_expert_degree(tmp_path, expert_parallel_size):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["cuda_visible_devices"] = "0,1,2,3"
    raw["server"]["tp_size"] = 4
    raw["server"]["expert_parallel_size"] = expert_parallel_size
    paths["profile"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="expert_parallel_size"):
        load_profile_config(paths["profile"])


def test_profile_server_env_drops_inherited_vllm_api_key(tmp_path, monkeypatch):
    fork_python = tmp_path / "venv" / "bin" / "python"
    fork_python.parent.mkdir(parents=True)
    fork_python.write_text("")
    monkeypatch.setenv("VLLM_API_KEY", "unrelated-shell-key")
    monkeypatch.setenv("PATH", "/ambient/bin")

    server_env = vllm_server.build_server_env(str(fork_python), "2,5")

    assert "VLLM_API_KEY" not in server_env
    assert server_env["CUDA_VISIBLE_DEVICES"] == "2,5"
    assert server_env["PATH"] == f"{fork_python.parent}:/ambient/bin"


def test_sglang_nsys_preflight_rejects_a_venv_without_nvtx(tmp_path):
    fork_python = tmp_path / "python"
    fork_python.write_text("#!/bin/sh\necho missing-nvtx >&2\nexit 1\n")
    fork_python.chmod(0o755)

    with pytest.raises(RuntimeError, match="cannot import `nvtx`") as error:
        sglang_server.validate_nsys_capture_environment(
            str(fork_python), env={"PATH": "/usr/bin"}, cwd=tmp_path
        )

    assert "missing-nvtx" in str(error.value)
    assert "uv pip install" in str(error.value)


def test_sglang_nsys_preflight_uses_the_server_environment(tmp_path, monkeypatch):
    server_env = {"PATH": "/server/venv/bin", "LD_LIBRARY_PATH": "/server/libs"}
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return sglang_server.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(sglang_server.subprocess, "run", fake_run)
    sglang_server.validate_nsys_capture_environment(
        "/server/venv/bin/python",
        env=server_env,
        cwd=tmp_path,
    )

    assert seen["env"] is server_env
    assert seen["cwd"] == tmp_path
    assert seen["timeout"] == 30


def test_nsys_preflight_is_scoped_to_new_captures():
    calls = []

    class Driver:
        @staticmethod
        def validate_nsys_capture_environment(fork_python, *, env, cwd):
            calls.append((fork_python, env, cwd))

    for profile_kind, resume in (
        ("workload_metrics", False),
        ("expert_popularity", False),
        ("nsys", True),
    ):
        alignment_runner._preflight_capture_environment(
            Driver,
            None if resume else "fork-python",
            {},
            profile_kind=profile_kind,
            resume=resume,
        )
    assert calls == []

    alignment_runner._preflight_capture_environment(
        Driver,
        "fork-python",
        {"sentinel": "server-env"},
        profile_kind="nsys",
        resume=False,
    )
    assert calls == [
        ("fork-python", {"sentinel": "server-env"}, alignment_runner.REPO_ROOT)
    ]


def test_profile_server_argv_enables_prompt_token_details_once():
    server_config = vllm_server.ServerConfig(
        model_path="model",
        extra_args=["--enable-prompt-tokens-details", "--trust-remote-code"],
    )

    server_argv = vllm_server.build_server_argv("fork-python", server_config)

    assert server_argv.count("--enable-prompt-tokens-details") == 1
    assert "--trust-remote-code" in server_argv


def test_sglang_server_argv_refuses_the_vllm_token_cudagraph_ceiling():
    base = dict(model_path="model", tp_size=4, chunk_size=2048)
    argv = sglang_server.build_server_argv("fork-python", ServerConfig(**base))
    assert argv[argv.index("--chunked-prefill-size") + 1] == "2048"
    assert not any(flag.startswith("--cuda-graph-max-bs") for flag in argv)
    eager_argv = sglang_server.build_server_argv(
        "fork-python", ServerConfig(**base, enforce_eager=True)
    )
    assert "--disable-cuda-graph" in eager_argv
    assert not any(flag.startswith("--cuda-graph-max-bs") for flag in eager_argv)

    with pytest.raises(ValueError, match="counts requests, not tokens"):
        sglang_server.build_server_argv(
            "fork-python", ServerConfig(**base, max_cudagraph_capture_size=2048)
        )

    argv = sglang_server.build_server_argv(
        "fork-python",
        ServerConfig(**base, extra_args=["--cuda-graph-max-bs-decode", "160"]),
    )
    assert argv[argv.index("--cuda-graph-max-bs-decode") + 1] == "160"


def test_extract_expert_popularity_aggregates_logical_counts(tmp_path):
    server_log = tmp_path / "server.log"
    records = [
        {
            "schema_version": 2,
            "model": "moe/model",
            "eplb_step": 10,
            "expert_parallel_size": 2,
            "experts_per_token": 2,
            "logical_expert_counts": [[1, 3], [2, 2]],
        },
        {
            "schema_version": 2,
            "model": "moe/model",
            "eplb_step": 11,
            "expert_parallel_size": 2,
            "experts_per_token": 2,
            "logical_expert_counts": [[4, 0], [1, 3]],
        },
    ]
    server_log.write_text(
        "\n".join(f"INFO VibeSimAlignmentExpertLoad {json.dumps(record)}" for record in records)
    )
    raw_path = tmp_path / "expert_load.jsonl"
    summary_path = tmp_path / "expert_popularity.json"

    count = record_extraction.extract_expert_popularity(
        server_log,
        raw_path,
        summary_path,
        expert_parallel_size=2,
        max_tokens_per_step=2,
    )
    summary = json.loads(summary_path.read_text())
    schema_path = Path(__file__).parents[1] / "alignment/schema/expert_popularity_v3.schema.json"
    schema = json.loads(schema_path.read_text())

    assert count == 2
    assert set(summary) == set(schema["required"]) == set(schema["properties"])
    assert set(summary["aggregation"]) == set(schema["properties"]["aggregation"]["required"])
    assert set(summary["expert_partitioning"]) == set(
        schema["properties"]["expert_partitioning"]["required"]
    )
    assert summary["schema_version"] == 3
    assert summary["expert_parallel_size"] == 2
    assert summary["experts_per_rank"] == 1
    assert summary["experts_per_token"] == 2
    assert summary["count_semantics"] == "logical_routed_token_assignments"
    assert summary["aggregation"] == {
        "scope": "captured_eplb_steps_within_token_ceiling",
        "observed_eplb_step_min": 10,
        "observed_eplb_step_max": 11,
        "record_count": 2,
        "raw_record_count": 2,
        "discarded_oversized_record_count": 0,
        "discarded_oversized_eplb_steps": [],
        "max_tokens_per_step": 2,
    }
    assert summary["expert_partitioning"] == {
        "kind": "contiguous_logical_expert_ids",
        "layout": "rank_major",
    }
    assert summary["counts_by_layer"] == [[5, 3], [3, 5]]
    assert summary["counts_all_layers"] == [8, 8]
    assert summary["probabilities_all_layers"] == [0.5, 0.5]


def test_extract_expert_popularity_excludes_oversized_warmup_flush(tmp_path):
    server_log = tmp_path / "server.log"
    records = [
        {
            "schema_version": 2,
            "model": "moe/model",
            "eplb_step": 20,
            "expert_parallel_size": 2,
            "experts_per_token": 2,
            "logical_expert_counts": [[12, 0], [12, 0]],
        },
        {
            "schema_version": 2,
            "model": "moe/model",
            "eplb_step": 21,
            "expert_parallel_size": 2,
            "experts_per_token": 2,
            "logical_expert_counts": [[2, 2], [1, 3]],
        },
    ]
    server_log.write_text(
        "\n".join(f"INFO VibeSimAlignmentExpertLoad {json.dumps(record)}" for record in records)
    )
    raw_path = tmp_path / "expert_load.jsonl"
    summary_path = tmp_path / "expert_popularity.json"

    count = record_extraction.extract_expert_popularity(
        server_log,
        raw_path,
        summary_path,
        expert_parallel_size=2,
        max_tokens_per_step=1,
    )
    summary = json.loads(summary_path.read_text())

    assert count == 1
    assert len(raw_path.read_text().splitlines()) == 2
    assert summary["aggregation"] == {
        "scope": "captured_eplb_steps_within_token_ceiling",
        "observed_eplb_step_min": 21,
        "observed_eplb_step_max": 21,
        "record_count": 1,
        "raw_record_count": 2,
        "discarded_oversized_record_count": 1,
        "discarded_oversized_eplb_steps": [20],
        "max_tokens_per_step": 1,
    }
    assert summary["counts_by_layer"] == [[2, 2], [1, 3]]


def test_extract_expert_popularity_rejects_partition_mismatch(tmp_path):
    server_log = tmp_path / "server.log"
    record = {
        "schema_version": 1,
        "model": "moe/model",
        "eplb_step": 10,
        "logical_expert_counts": [[1, 2, 3]],
    }
    server_log.write_text(f"INFO VibeSimAlignmentExpertLoad {json.dumps(record)}\n")

    with pytest.raises(ValueError, match="must be divisible"):
        record_extraction.extract_expert_popularity(
            server_log,
            tmp_path / "raw.jsonl",
            tmp_path / "summary.json",
            max_tokens_per_step=1,
            expert_parallel_size=2,
            experts_per_token=2,
        )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("nvidia/GLM-5.2-NVFP4", "nvidia/GLM-5.2-NVFP4"),
        ("./checkpoints/GLM-5.2-NVFP4", "./checkpoints/GLM-5.2-NVFP4"),
        (
            "/cache/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/deadbeef",
            "nvidia/GLM-5.2-NVFP4",
        ),
        (
            "/raid/checkpoints/GLM-5.2-NVFP4/snapshots/deadbeef",
            "GLM-5.2-NVFP4",
        ),
        ("/raid/checkpoints/GLM-5.2-NVFP4", "GLM-5.2-NVFP4"),
        ("/", "/"),
    ],
)
def test_normalize_model_id(model, expected):
    assert record_extraction.normalize_model_id(model) == expected


def test_expert_popularity_normalizes_only_the_summary_model_identity(tmp_path):
    model_path = "/cache/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/deadbeef"
    record = {
        "schema_version": 2,
        "model": model_path,
        "eplb_step": 1,
        "expert_parallel_size": 1,
        "experts_per_token": 1,
        "logical_expert_counts": [[1]],
    }
    server_log = tmp_path / "server.log"
    raw_path = tmp_path / "expert_load.jsonl"
    summary_path = tmp_path / "expert_popularity.json"
    server_log.write_text(f"INFO VibeSimAlignmentExpertLoad {json.dumps(record)}\n")

    record_extraction.extract_expert_popularity(
        server_log,
        raw_path,
        summary_path,
        expert_parallel_size=1,
        max_tokens_per_step=1,
    )

    assert json.loads(raw_path.read_text())["model"] == model_path
    assert json.loads(summary_path.read_text())["model"] == "nvidia/GLM-5.2-NVFP4"


def test_expert_popularity_ceiling_uses_the_count_reduction_group(tmp_path):
    record = {
        "schema_version": 2,
        "model": "moe/model",
        "eplb_step": 1,
        "expert_parallel_size": 1,
        "experts_per_token": 2,
        # One token routed top-2 on four TP replicas. The synchronized record
        # therefore carries eight assignments although experts are not sharded.
        "logical_expert_counts": [[4, 4]],
    }
    server_log = tmp_path / "server.log"
    server_log.write_text(f"INFO VibeSimAlignmentExpertLoad {json.dumps(record)}\n")

    with pytest.raises(ValueError, match="within the configured token ceiling"):
        record_extraction.extract_expert_popularity(
            server_log,
            tmp_path / "raw-with-ep-ceiling.jsonl",
            tmp_path / "summary-with-ep-ceiling.json",
            expert_parallel_size=1,
            max_tokens_per_step=1,
        )

    count = record_extraction.extract_expert_popularity(
        server_log,
        tmp_path / "raw-with-reduction-ceiling.jsonl",
        tmp_path / "summary-with-reduction-ceiling.json",
        expert_parallel_size=1,
        reduction_group_size=4,
        max_tokens_per_step=1,
    )

    assert count == 1


def test_sglang_tensor_parallel_expert_copies_are_not_multiplied(tmp_path):
    record = {
        "schema_version": 2,
        "model": "moe/model",
        "eplb_step": 1,
        "expert_parallel_size": 1,
        "experts_per_token": 2,
        "logical_expert_counts": [[4, 4]],
    }
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "\n".join(
            f"[2026-08-11 16:18:16 TP{tp_rank}] INFO "
            f"VibeSimAlignmentExpertLoad {json.dumps(record)}"
            for tp_rank in range(4)
        )
    )
    raw_path = tmp_path / "expert_load.jsonl"
    summary_path = tmp_path / "expert_popularity.json"

    count = record_extraction.extract_expert_popularity(
        server_log,
        raw_path,
        summary_path,
        records=engine_records.SGLANG_RECORDS,
        expert_parallel_size=1,
        reduction_group_size=4,
        max_tokens_per_step=1,
    )

    assert count == 1
    assert len(raw_path.read_text().splitlines()) == 1
    assert json.loads(summary_path.read_text())["counts_by_layer"] == [[4, 4]]


def test_sglang_expert_records_keep_one_owner_per_dp_group(tmp_path):
    records = [
        {
            "schema_version": 2,
            "model": "moe/model",
            "eplb_step": 1,
            "expert_parallel_size": 1,
            "experts_per_token": 2,
            "logical_expert_counts": [counts],
        }
        for counts in ([2, 2], [3, 1])
    ]
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "\n".join(
            f"[2026-08-11 16:18:16 DP{dp_rank} TP{tp_rank}] INFO "
            f"VibeSimAlignmentExpertLoad {json.dumps(records[dp_rank])}"
            for dp_rank in range(2)
            for tp_rank in range(2)
        )
    )
    raw_path = tmp_path / "expert_load.jsonl"
    summary_path = tmp_path / "expert_popularity.json"

    count = record_extraction.extract_expert_popularity(
        server_log,
        raw_path,
        summary_path,
        records=engine_records.SGLANG_RECORDS,
        expert_parallel_size=1,
        reduction_group_size=2,
        max_tokens_per_step=1,
        dp_size=2,
    )

    assert count == 2
    assert len(raw_path.read_text().splitlines()) == 2
    assert json.loads(summary_path.read_text())["counts_by_layer"] == [[5, 3]]


def test_sglang_duplicate_expert_record_from_one_dp_owner_is_rejected(tmp_path):
    record = {
        "schema_version": 2,
        "model": "moe/model",
        "eplb_step": 1,
        "expert_parallel_size": 1,
        "experts_per_token": 2,
        "logical_expert_counts": [[2, 2]],
    }
    line = f"[2026-08-11 16:18:16 DP0 TP0] INFO VibeSimAlignmentExpertLoad {json.dumps(record)}"
    server_log = tmp_path / "server.log"
    server_log.write_text(f"{line}\n{line}\n")

    with pytest.raises(ValueError, match="duplicate alignment expert-load record"):
        record_extraction.extract_expert_popularity(
            server_log,
            tmp_path / "expert_load.jsonl",
            tmp_path / "expert_popularity.json",
            records=engine_records.SGLANG_RECORDS,
            expert_parallel_size=1,
            reduction_group_size=2,
            max_tokens_per_step=1,
            dp_size=2,
        )


def test_engines_state_their_expert_count_reduction_population():
    args = {"tensor_parallel_size": 4, "expert_parallel_size": 1}

    assert engine_records.VLLM_RECORDS.expert_count_reduction_group_size(**args) == 1
    assert engine_records.SGLANG_RECORDS.expert_count_reduction_group_size(**args) == 4


def test_runner_passes_engine_specific_expert_popularity_group_sizes(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["profile_kind"] = "expert_popularity"
    raw["cuda_visible_devices"] = "0,1,2,3,4,5,6,7"
    raw["server"]["tp_size"] = 4
    raw["server"]["dp_size"] = 2
    paths["profile"].write_text(yaml.safe_dump(raw))

    vllm_config = load_profile_config(paths["profile"])
    assert alignment_runner._expert_popularity_group_sizes(
        vllm_config, engine_records.VLLM_RECORDS
    ) == (8, 8)

    raw["engine"] = "sglang"
    raw["server"]["expert_parallel_size"] = 1
    paths["profile"].write_text(yaml.safe_dump(raw))
    sglang_config = load_profile_config(paths["profile"])
    assert alignment_runner._expert_popularity_group_sizes(
        sglang_config, engine_records.SGLANG_RECORDS
    ) == (1, 4)


@pytest.mark.parametrize("reduction_group_size", [0, -1, True, 1.5])
def test_expert_popularity_rejects_invalid_reduction_group_size(tmp_path, reduction_group_size):
    with pytest.raises(ValueError, match="reduction_group_size must be a positive integer"):
        record_extraction.extract_expert_popularity(
            tmp_path / "server.log",
            tmp_path / "raw.jsonl",
            tmp_path / "summary.json",
            expert_parallel_size=1,
            experts_per_token=1,
            reduction_group_size=reduction_group_size,
            max_tokens_per_step=1,
        )


def test_analyze_rejects_removed_kernel_mapping_field(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["analyze_kernel"].read_text())
    raw["iteration"] = {"enabled": True, "kernel_mapping_file": "./mapping.yaml"}
    paths["analyze_kernel"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="kernel_mapping_file"):
        load_analyze_config(paths["analyze_kernel"])


def test_analyze_rejects_mixing_kernel_align_and_e2e_phases(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["analyze_kernel"].read_text())
    raw["e2e"] = {"enabled": True, "throughput_bins": 20}
    paths["analyze_kernel"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="one phase"):
        load_analyze_config(paths["analyze_kernel"])


def test_timing_predict_uses_one_phase_config_and_no_labeled_inventory(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    _write_completed_inputs(tmp_path)
    launched = []
    monkeypatch.setattr(
        alignment_launcher,
        "_launch_timing_predict",
        lambda path, **kwargs: launched.append((path, kwargs)) or 0,
    )

    assert alignment_launcher.main(["timing-predict", str(paths["timing"])]) == 0
    output = tmp_path / "timing_predict_run"
    assert launched == [(output / "timing_predict_config.json", {"build_type": "release"})]
    input_manifest = json.loads((output / "timing_predict_input_manifest.json").read_text())
    assert "replay_result" not in input_manifest
    assert not (tmp_path / "analysis_run").exists()
    assert not (output / "kernel_sequences_labeled.json").exists()


def test_timing_predict_launcher_keeps_post_run_analysis_enabled(tmp_path, monkeypatch):
    config_path = tmp_path / "timing_predict_config.json"
    config_path.write_text("{}")
    launched = []
    monkeypatch.setattr(
        timing_predict_launcher,
        "main",
        lambda argv: launched.append(argv) or 0,
    )

    assert alignment_launcher._launch_timing_predict(config_path, build_type="release") == 0
    assert launched == [[str(config_path), "--build-type", "release"]]


def test_timing_predict_preserves_simulation_backend_policy(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    _write_completed_inputs(tmp_path)
    # The preset carries backends in the flat "pool/role" form; timing-predict
    # re-nests them into {pool: {role: [...]}} for the offline predictor so the
    # predicted CostTree matches the simulation's backend identity.
    preset = yaml.safe_load(paths["simulation"].read_text())
    preset["backends"] = {"main/unified.pre_attn.qkv_proj": ["torch_linear"]}
    paths["simulation"].write_text(yaml.safe_dump(preset))
    monkeypatch.setattr(alignment_launcher, "_launch_timing_predict", lambda *args, **kwargs: 0)

    assert alignment_launcher.main(["timing-predict", str(paths["timing"])]) == 0
    predict_config = json.loads(
        (tmp_path / "timing_predict_run" / "timing_predict_config.json").read_text()
    )
    assert predict_config["backends"] == {"main": {"unified.pre_attn.qkv_proj": ["torch_linear"]}}


def test_timing_predict_warns_on_trace_mismatch(tmp_path, monkeypatch, capsys):
    paths = _phase_configs(tmp_path)
    _write_completed_inputs(tmp_path)
    result_path = tmp_path / "profile_run" / "profile_result.json"
    result = json.loads(result_path.read_text())
    result["drive_summary"]["source_trace"] = str(tmp_path / "trace" / "different.csv")
    result_path.write_text(json.dumps(result))
    monkeypatch.setattr(alignment_launcher, "_launch_timing_predict", lambda *args, **kwargs: 0)

    assert alignment_launcher.main(["timing-predict", str(paths["timing"])]) == 0
    assert "[warn] alignment trace mismatch" in capsys.readouterr().err


def test_analyze_kernel_align_writes_kernel_only_manifest(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    _write_timing_artifacts(tmp_path)
    calls = []
    monkeypatch.setattr(
        alignment_launcher,
        "_launch_alignment_analysis",
        lambda log_dir, **kwargs: calls.append((log_dir, kwargs)) or True,
    )

    assert alignment_launcher.main(["analyze", str(paths["analyze_kernel"])]) == 0
    analysis = tmp_path / "analysis_kernel_run"
    manifest = json.loads((analysis / "alignment_manifest.json").read_text())
    assert manifest["schema_version"] == 9
    assert manifest["labeled_kernel_sequences"] == str(analysis / "kernel_sequences_labeled.json")
    assert manifest["predict_log_dir"] == str(tmp_path / "timing_predict_run")
    # kernel-align names no simulation, replay, or subject-enabled bookkeeping.
    assert "simulation_log_dir" not in manifest
    assert "replay_result" not in manifest
    assert "iteration" not in manifest and "workload" not in manifest and "e2e" not in manifest
    # The capture predates the host sidecar, so the manifest names none — and
    # says so explicitly rather than omitting the key.
    assert manifest["host_timeline"] is None
    # `alignment-timeline` reads exactly what `alignment-iteration` reads and is
    # never useful without it, so kernel-align runs the pair.
    assert calls == [
        (
            analysis.resolve(),
            {
                "build_type": "release",
                "subjects": ["alignment-iteration", "alignment-timeline"],
            },
        )
    ]


def test_analyze_e2e_align_writes_simulation_manifest_without_labels(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    _write_timing_artifacts(tmp_path)
    calls = []
    monkeypatch.setattr(
        alignment_launcher,
        "_launch_alignment_analysis",
        lambda log_dir, **kwargs: calls.append((log_dir, kwargs)) or True,
    )

    assert alignment_launcher.main(["analyze", str(paths["analyze_e2e"])]) == 0
    analysis = tmp_path / "analysis_e2e_run"
    manifest = json.loads((analysis / "alignment_manifest.json").read_text())
    assert manifest["schema_version"] == 9
    assert manifest["simulation_log_dir"] == str(tmp_path / "simulation_run")
    assert manifest["workload_profile_log_dir"] == str(tmp_path / "profile_run")
    assert manifest["metrics_jsonl"] == str(tmp_path / "profile_run" / "metrics.jsonl")
    assert manifest["replay_start_monotonic_ns"] == 900_000
    assert manifest["replay_end_monotonic_ns"] == 2_100_000
    assert "profile_log_dir" not in manifest
    assert "parsed_nsys" not in manifest
    assert manifest["throughput_bins"] == 20
    # e2e-align names no timing-predict labels or per-subject flags.
    assert "labeled_kernel_sequences" not in manifest
    assert "iteration" not in manifest and "workload" not in manifest and "e2e" not in manifest
    assert calls == [
        (
            analysis.resolve(),
            {"build_type": "release", "subjects": ["alignment-workload", "alignment-e2e"]},
        )
    ]


def test_analyze_e2e_uses_distinct_full_run_workload_profile(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    _write_timing_artifacts(tmp_path)
    workload_profile = tmp_path / "workload_profile_run"
    workload_profile.mkdir()
    metrics = workload_profile / "metrics.jsonl"
    replay = workload_profile / "replay.jsonl"
    metrics.write_text("{}\n")
    replay.write_text("")
    base_profile = json.loads((tmp_path / "profile_run" / "profile_result.json").read_text())
    workload_result = {
        **base_profile,
        "profile_kind": "workload_metrics",
        "log_dir": str(workload_profile),
        "metrics_jsonl": str(metrics),
        "replay_result": str(replay),
    }
    copied_trace = tmp_path / "trace" / "copied-shared.csv"
    copied_trace.write_bytes((tmp_path / "trace" / "shared.csv").read_bytes())
    workload_result["drive_summary"] = {
        **base_profile["drive_summary"],
        "source_trace": str(copied_trace),
    }
    (workload_profile / "profile_result.json").write_text(json.dumps(workload_result))
    analyze_raw = yaml.safe_load(paths["analyze_e2e"].read_text())
    analyze_raw["workload_profile_log_dir"] = "./workload_profile_run"
    paths["analyze_e2e"].write_text(yaml.safe_dump(analyze_raw))
    monkeypatch.setattr(
        alignment_launcher, "_launch_alignment_analysis", lambda *args, **kwargs: True
    )

    assert alignment_launcher.main(["analyze", str(paths["analyze_e2e"])]) == 0

    manifest = json.loads((tmp_path / "analysis_e2e_run" / "alignment_manifest.json").read_text())
    assert manifest["workload_profile_log_dir"] == str(workload_profile)
    assert manifest["metrics_jsonl"] == str(metrics)
    assert manifest["replay_result"] == str(replay)


def test_analyze_e2e_rejects_nsys_profile_field(tmp_path, capsys):
    paths = _phase_configs(tmp_path)
    _write_timing_artifacts(tmp_path)
    analyze_raw = yaml.safe_load(paths["analyze_e2e"].read_text())
    analyze_raw["profile_log_dir"] = "./profile_run"
    paths["analyze_e2e"].write_text(yaml.safe_dump(analyze_raw))

    assert alignment_launcher.main(["analyze", str(paths["analyze_e2e"])]) == 2
    assert "profile_log_dir does not belong to workload/e2e alignment" in capsys.readouterr().err


def test_analyze_rejects_all_subjects_disabled(tmp_path, capsys):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["analyze_kernel"].read_text())
    raw["iteration"] = {"enabled": False}
    paths["analyze_kernel"].write_text(yaml.safe_dump(raw))

    assert alignment_launcher.main(["analyze", str(paths["analyze_kernel"])]) == 2
    assert "at least one analysis subject" in capsys.readouterr().err


def test_timing_predict_snapshot_accepts_inputs_already_in_output_root(tmp_path, capsys):
    config_path = tmp_path / "timing_predict_config.json"
    cases_path = tmp_path / "timing_predict_cases.json"
    cases_path.write_text("[]")
    config = {"log_dir": str(tmp_path), "cases_file": str(cases_path)}
    config_path.write_text(json.dumps(config))

    timing_predict_launcher._snapshot_inputs(config_path, config, tmp_path)

    assert capsys.readouterr().err == ""
    assert json.loads(config_path.read_text()) == config


def test_timing_predict_snapshots_private_labeler_params(tmp_path):
    config_path = tmp_path / "timing_predict_config.json"
    cases_path = tmp_path / "timing_predict_cases.json"
    cases_path.write_text("[]")
    config = {
        "arch": {
            "iter": {
                "type": "glm52_dsa_moe",
                "model_config": "model/config/glm52.json",
                "fp8": False,
                "ep_size": 8,
            }
        },
        "gpu": "NVIDIA H200",
        "log_dir": str(tmp_path),
        "cases_file": str(cases_path),
    }
    config_path.write_text(json.dumps(config))

    timing_predict_launcher._snapshot_inputs(config_path, config, tmp_path)

    assert json.loads((tmp_path / "raw" / "params.json").read_text()) == {
        "pools": {
            "predict": {
                "groups": [
                    {
                        "arch": config["arch"]["iter"],
                        "gpu": "NVIDIA H200",
                    }
                ]
            }
        }
    }


def test_timing_predict_publishes_stable_analyzer_resource_metadata(tmp_path):
    config_path = tmp_path / "timing_predict_config.json"
    cases_path = tmp_path / "timing_predict_cases.json"
    cases_path.write_text("[]")
    config = {
        "arch": {"iter": {"type": "llama3_dense"}},
        "gpu": "NVIDIA H200",
        "log_dir": str(tmp_path),
        "cases_file": str(cases_path),
    }
    config_path.write_text(json.dumps(config))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "prediction_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gpu_name": "NVIDIA H200",
                "gpu_count": 4,
            }
        )
    )

    prediction_id = timing_predict_launcher._prediction_id(tmp_path)
    timing_predict_launcher._write_prediction_metadata(
        config_path,
        config,
        tmp_path,
        prediction_id,
    )

    assert prediction_id.startswith("p_")
    assert timing_predict_launcher._prediction_id(tmp_path) == prediction_id
    assert json.loads((tmp_path / "prediction.meta.json").read_text()) == {
        "schema_version": 1,
        "prediction_id": prediction_id,
        "selector": "iter",
        "arch_type": "llama3_dense",
        "gpu": "NVIDIA H200",
        "gpu_count": 4,
        "config_file": config_path.name,
        "cases_file": "prediction.cases.json",
        "case_count": 0,
    }
    assert json.loads((tmp_path / "prediction.cases.json").read_text()) == []
    assert timing_predict_launcher._predict_descriptor(config_path, config) == {
        "selector": "iter",
        "configName": config_path.name,
        "caseCount": 0,
    }


def test_timing_predict_does_not_reuse_noncanonical_prediction_id(tmp_path):
    (tmp_path / "prediction.meta.json").write_text(
        json.dumps({"prediction_id": "p_"}),
        encoding="utf-8",
    )

    prediction_id = timing_predict_launcher._prediction_id(tmp_path)

    assert prediction_id.startswith("p_")
    assert prediction_id != "p_"


def test_alignment_analysis_calls_only_selected_subjects(tmp_path, monkeypatch):
    analyzer = tmp_path / "analyze"
    analyzer.touch()
    log_dir = tmp_path / "analysis"
    log_dir.mkdir()
    invocations = []

    def fake_capture(argv):
        invocations.append(argv)
        return 0, "alignment accepted\n"

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda build_type: analyzer)
    monkeypatch.setattr(launcher_exec, "_run_capture_sync", fake_capture)

    launcher_exec.run_alignment_analysis(log_dir, "release", ["alignment-e2e"])
    assert invocations[0] == [str(analyzer), "alignment", str(log_dir), "alignment-e2e"]
    assert invocations[1][-1:] == ["alignment-e2e"]


def test_req_frontend_invocation_keeps_independent_as_a_typed_frontend(tmp_path, monkeypatch):
    from alignment.load_generator.config import IndependentFrontendConfig, LoadGeneratorConfig

    config = LoadGeneratorConfig(
        frontend=IndependentFrontendConfig(path="trace/smoke.csv"),
        text_file="corpus.txt",
        tokenizer="tokenizer.json",
        max_concurrency=64,
    )
    prepared = load_runner.PreparedReplay(
        trace_path=tmp_path / "trace.csv",
        text_file=tmp_path / "corpus.txt",
        tokenizer="tokenizer.json",
        log_path=tmp_path / "replay.jsonl",
        summary_path=tmp_path / "summary.json",
    )
    commands = []
    monkeypatch.setattr(
        load_runner.subprocess, "run", lambda argv, **kwargs: commands.append((argv, kwargs))
    )

    load_runner.run_replay(config, prepared, base_url="http://localhost:8000", model="m")
    argv = commands[0][0]
    assert argv[argv.index("--input-file-format") + 1] == "text-generation-independent"
    assert argv[argv.index("--backend") + 1] == "openai"
    assert argv[argv.index("--base-url") + 1] == "http://localhost:8000/v1"
    assert argv[argv.index("--max-concurrency") + 1] == "64"


def test_req_frontend_invocation_selects_vllm_tokens_backend(tmp_path, monkeypatch):
    from alignment.load_generator.config import (
        IndependentFrontendConfig,
        LoadGeneratorConfig,
        VllmTokensBackendConfig,
    )

    config = LoadGeneratorConfig(
        frontend=IndependentFrontendConfig(path="trace/smoke.csv"),
        text_file="corpus.txt",
        tokenizer="tokenizer.json",
        backend=VllmTokensBackendConfig(),
    )
    prepared = load_runner.PreparedReplay(
        trace_path=tmp_path / "trace.csv",
        text_file=tmp_path / "corpus.txt",
        tokenizer="tokenizer.json",
        log_path=tmp_path / "replay.jsonl",
        summary_path=tmp_path / "summary.json",
    )
    commands = []
    monkeypatch.setattr(
        load_runner.subprocess, "run", lambda argv, **kwargs: commands.append((argv, kwargs))
    )

    result = load_runner.run_replay(config, prepared, base_url="http://localhost:8000", model="m")

    argv = commands[0][0]
    assert argv[argv.index("--backend") + 1] == "vllm-tokens"
    assert argv[argv.index("--base-url") + 1] == "http://localhost:8000"
    assert result["backend_type"] == "vllm_tokens"


def test_req_frontend_invocation_passes_no_context_policy(tmp_path, monkeypatch):
    """A session run selects a materialized trace, not a rule for materializing one.

    The replay binary has no context-policy flag: the split was resolved by
    `tracegen` and is recorded in the manifest beside the trace.
    """
    from alignment.load_generator.config import (
        LoadGeneratorConfig,
        SessionFrontendConfig,
        VllmTokensBackendConfig,
    )

    config = LoadGeneratorConfig(
        frontend=SessionFrontendConfig(path="trace/execution.csv"),
        text_file="corpus.txt",
        tokenizer="tokenizer.json",
        backend=VllmTokensBackendConfig(),
    )
    prepared = load_runner.PreparedReplay(
        trace_path=tmp_path / "trace.csv",
        text_file=tmp_path / "corpus.txt",
        tokenizer="tokenizer.json",
        log_path=tmp_path / "replay.jsonl",
        summary_path=tmp_path / "summary.json",
    )
    commands = []
    monkeypatch.setattr(
        load_runner.subprocess, "run", lambda argv, **kwargs: commands.append((argv, kwargs))
    )

    result = load_runner.run_replay(config, prepared, base_url="http://localhost:8000", model="m")

    argv = commands[0][0]
    assert argv[argv.index("--input-file-format") + 1] == "text-generation-session-execution-v2"
    assert "--session-context-policy" not in argv
    assert "session_context_policy" not in result


@pytest.mark.parametrize(
    "config_field",
    ["skip_when_reaching_limit", "fail_on_context_overflow"],
)
def test_req_frontend_context_limit_skip_uses_canonical_cli_flag(
    tmp_path, monkeypatch, config_field
):
    from alignment.load_generator.config import (
        LoadGeneratorConfig,
        SessionFrontendConfig,
    )

    config = LoadGeneratorConfig(
        frontend=SessionFrontendConfig(path="trace/session.csv"),
        text_file="corpus.txt",
        tokenizer="tokenizer.json",
        max_model_len=128,
        **{config_field: True},
    )
    prepared = load_runner.PreparedReplay(
        trace_path=tmp_path / "trace.csv",
        text_file=tmp_path / "corpus.txt",
        tokenizer="tokenizer.json",
        log_path=tmp_path / "replay.jsonl",
        summary_path=tmp_path / "summary.json",
    )
    commands = []
    monkeypatch.setattr(
        load_runner.subprocess, "run", lambda argv, **kwargs: commands.append((argv, kwargs))
    )

    load_runner.run_replay(config, prepared, base_url="http://localhost:8000", model="m")

    argv = commands[0][0]
    assert "--skip-when-reaching-limit" in argv
    assert "--fail-on-context-overflow" not in argv


def test_req_frontend_context_limit_skip_requires_model_limit():
    from alignment.load_generator.config import (
        LoadGeneratorConfig,
        SessionFrontendConfig,
    )

    with pytest.raises(ValueError, match="requires max_model_len"):
        LoadGeneratorConfig(
            frontend=SessionFrontendConfig(path="trace/session.csv"),
            text_file="corpus.txt",
            tokenizer="tokenizer.json",
            skip_when_reaching_limit=True,
        )


def test_profile_config_accepts_a_session_frontend_with_openai_backend():
    from alignment.load_generator.config import (
        LoadGeneratorConfig,
        OpenAIBackendConfig,
        SessionFrontendConfig,
    )

    config = LoadGeneratorConfig.from_mapping(
        {
            "frontend": {"type": "session", "path": "trace/execution.csv"},
            "text_file": "corpus.txt",
            "tokenizer": "tokenizer.json",
        }
    )

    assert isinstance(config.frontend, SessionFrontendConfig)
    assert config.frontend.path == "trace/execution.csv"
    assert isinstance(config.backend, OpenAIBackendConfig)


def test_profile_config_rejects_a_stale_context_policy_key():
    """An older config must fail loudly rather than have the key ignored."""
    from alignment.load_generator.config import LoadGeneratorConfig

    with pytest.raises(ValueError, match="context_policy"):
        LoadGeneratorConfig.from_mapping(
            {
                "frontend": {
                    "type": "session",
                    "path": "trace/execution.csv",
                    "context_policy": "monotonic",
                },
                "text_file": "corpus.txt",
                "tokenizer": "tokenizer.json",
            }
        )


def test_profile_config_loads_vllm_tokens_backend(tmp_path):
    from alignment.load_generator.config import (
        LoadGeneratorConfig,
        VllmTokensBackendConfig,
    )

    config = LoadGeneratorConfig.from_mapping(
        {
            "frontend": {"type": "independent", "path": "trace/smoke.csv"},
            "backend": {"type": "vllm_tokens"},
            "text_file": "corpus.txt",
            "tokenizer": "tokenizer.json",
        }
    )

    assert isinstance(config.backend, VllmTokensBackendConfig)


def test_profile_config_rejects_unknown_load_generator_backend():
    from alignment.load_generator.config import LoadGeneratorConfig

    with pytest.raises(ValueError, match="unsupported backend.type"):
        LoadGeneratorConfig.from_mapping(
            {
                "frontend": {"type": "independent", "path": "trace/smoke.csv"},
                "backend": {"type": "unknown"},
                "text_file": "corpus.txt",
                "tokenizer": "tokenizer.json",
            }
        )


def test_vllm_tokens_backend_adds_server_flag_once(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["workload"]["backend"] = {"type": "vllm_tokens"}
    paths["profile"].write_text(yaml.safe_dump(raw))
    config = load_profile_config(paths["profile"])
    server_argv = ["python", "-m", "vllm.entrypoints.openai.api_server"]

    alignment_runner._append_backend_server_args(server_argv, config)
    alignment_runner._append_backend_server_args(server_argv, config)

    assert server_argv.count("--tokens-only") == 1


def test_launcher_main_dispatches_alignment_subcommand(monkeypatch):
    from launcher import __main__ as launcher_main

    received = []
    monkeypatch.setattr(alignment_launcher, "main", lambda argv: received.append(argv) or 0)
    assert launcher_main.main(["alignment", "profile", "profile.yaml"]) == 0
    assert received == [["profile", "profile.yaml"]]


def test_profile_resume_flag_reaches_the_runner(tmp_path, monkeypatch):
    """A post-capture failure must be recoverable without re-running the GPU."""
    paths = _phase_configs(tmp_path)
    resumed = []
    monkeypatch.setattr(
        alignment_launcher,
        "run_profile",
        lambda config, *, resume=False: (
            resumed.append(resume) or {"parsed_nsys": str(tmp_path / "profile_run" / "parsed.json")}
        ),
    )

    assert alignment_launcher.main(["profile", str(paths["profile"]), "--resume"]) == 0
    assert resumed == [True]
