"""CPU tests for the four explicit alignment launcher phase contracts."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from alignment import runner as alignment_runner
from alignment.load_generator import runner as load_runner
from alignment.profiler import (
    engine_records,
    record_extraction,
    runtime_artifacts,
    sglang_server,
    vllm_server,
)
from alignment.profiler.config import (
    ProfileConfig,
    PythonPackageArtifact,
    PythonRuntimeConfig,
    ServerConfig,
)
from alignment.timing_predict_input import BuildRequest, build_inputs
from launcher import alignment as alignment_launcher
from launcher import alignment_config as alignment_config_module
from launcher import exec as launcher_exec
from launcher import timing_predict as timing_predict_launcher
from launcher.alignment_config import load_analyze_config, load_profile_config


def _write_config(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config) if path.suffix == ".json" else yaml.safe_dump(config))


def _python_runtime_document() -> dict:
    return {
        "packages": [
            {
                "name": "flashinfer-jit-cache",
                "version": "0.6.15.post1",
                "local_version": "cu130",
                "index_url": "https://flashinfer.ai/whl/cu130",
                "required_files": [
                    "flashinfer_jit_cache/jit_cache/fused_moe_trtllm_sm100/"
                    "fused_moe_trtllm_sm100.so"
                ],
            }
        ],
        "environment": {"FLASHINFER_DISABLE_JIT": "1"},
        "lock_timeout_seconds": 60.0,
    }


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


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_explicit_speculative_input_config_round_trip(tmp_path, depth):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["timing"].read_text())
    raw["input_builder"] = {"type": "speculative_engine_text", "draft_tokens": depth}
    _write_config(paths["timing"], raw)
    spec = alignment_launcher.load_timing_predict_config(paths["timing"]).input_builder
    assert spec.to_mapping() == {
        "type": "speculative_engine_text", "draft_tokens": depth,
        "measured_phase": "forward", "group_assignment": "single",
    }


def test_speculative_input_config_has_no_implicit_depth(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["timing"].read_text())
    raw["input_builder"] = {"type": "speculative_engine_text"}
    _write_config(paths["timing"], raw)
    with pytest.raises(ValueError, match="draft_tokens"):
        alignment_launcher.load_timing_predict_config(paths["timing"])


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
    assert "overrides" not in calls[0][1]
    assert json.loads((tmp_path / "artifact.meta.json").read_text()) == {
        "schema_version": 1,
        "artifact_kind": "alignment_bundle",
    }


def test_alignment_sim_dry_run_does_not_publish_bundle_marker(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    monkeypatch.setattr(alignment_launcher, "_launch_simulation", lambda *args, **kwargs: 0)

    assert alignment_launcher.main(["sim", str(paths["simulation"]), "--dry-run"]) == 0
    assert not (tmp_path / "artifact.meta.json").exists()


def test_alignment_sim_preserves_explicit_worker_multiplier(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    preset = yaml.safe_load(paths["simulation"].read_text())
    preset["pools"]["main"]["groups"][0]["worker"]["gpu_time_multiplier"] = 1.17
    _write_config(paths["simulation"], preset)
    before = paths["simulation"].read_bytes()
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

    assert alignment_launcher.main(["sim", str(paths["simulation"])]) == 0
    assert calls[0][0] == paths["simulation"]
    assert "overrides" not in calls[0][1]
    assert paths["simulation"].read_bytes() == before


def test_alignment_sim_rejects_removed_calibration_flag(tmp_path):
    paths = _phase_configs(tmp_path)
    with pytest.raises(SystemExit) as error:
        alignment_launcher.main(
            ["sim", str(paths["simulation"]), "--gpu-time-multiplier-from", str(tmp_path)]
        )
    assert error.value.code == 2


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


def test_non_popularity_profile_does_not_require_expert_topology(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["cuda_visible_devices"] = "0,1,2,3"
    raw["server"]["tp_size"] = 4
    raw["server"]["dp_size"] = 1
    paths["profile"].write_text(yaml.safe_dump(raw))

    assert load_profile_config(paths["profile"]).server.expert_parallel_size is None
    assert load_profile_config(paths["profile"]).server.expert_count_reduction_group_size is None

    raw["server"]["expert_parallel_size"] = 1
    paths["profile"].write_text(yaml.safe_dump(raw))
    assert load_profile_config(paths["profile"]).server.expert_parallel_size == 1


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_a_popularity_pass_requires_explicit_engine_neutral_topology(tmp_path, engine):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["engine"] = engine
    if engine == "sglang":
        raw["python_runtime"] = _python_runtime_document()
    raw["profile_kind"] = "expert_popularity"
    raw["cuda_visible_devices"] = "0,1,2,3"
    raw["server"]["tp_size"] = 2
    raw["server"]["dp_size"] = 2
    paths["profile"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="expert_popularity requires explicit"):
        load_profile_config(paths["profile"])

    raw["server"]["expert_parallel_size"] = 1
    paths["profile"].write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="expert_popularity requires explicit"):
        load_profile_config(paths["profile"])

    # The same config is a valid corpus capture on vLLM: the topology is what a
    # marginal is reduced over, and a corpus records logical ids instead. On
    # SGLang there is no per-token route to return at all.
    raw["profile_kind"] = "token_corpus"
    paths["profile"].write_text(yaml.safe_dump(raw))
    if engine == "vllm":
        assert load_profile_config(paths["profile"]).server.expert_parallel_size == 1
    else:
        with pytest.raises(ValueError, match="token_corpus requires engine vllm"):
            load_profile_config(paths["profile"])

    raw["profile_kind"] = "expert_popularity"
    raw["server"]["expert_count_reduction_group_size"] = 2
    paths["profile"].write_text(yaml.safe_dump(raw))
    server = load_profile_config(paths["profile"]).server
    assert server.expert_parallel_size == 1
    assert server.expert_count_reduction_group_size == 2


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


@pytest.mark.parametrize("reduction_group_size", [0, True, 3])
def test_profile_config_rejects_an_invalid_expert_count_reduction_group(
    tmp_path, reduction_group_size
):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["cuda_visible_devices"] = "0,1,2,3"
    raw["server"]["tp_size"] = 4
    raw["server"]["expert_count_reduction_group_size"] = reduction_group_size
    paths["profile"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="expert_count_reduction_group_size"):
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


def test_sglang_profile_requires_an_explicit_prebuilt_runtime(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["engine"] = "sglang"
    paths["profile"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="explicit python_runtime"):
        load_profile_config(paths["profile"])
    resumed = load_profile_config(paths["profile"], require_python_runtime=False)
    assert resumed.engine == "sglang"
    assert resumed.python_runtime is None

    raw["python_runtime"] = _python_runtime_document()
    paths["profile"].write_text(yaml.safe_dump(raw))
    config = load_profile_config(paths["profile"])
    assert config.python_runtime is not None
    assert config.python_runtime.environment == {"FLASHINFER_DISABLE_JIT": "1"}


@pytest.mark.parametrize(
    "python_runtime",
    [None, PythonRuntimeConfig(packages=[], environment={})],
)
def test_direct_sglang_profile_fails_before_replay_without_strict_jit_policy(
    tmp_path, monkeypatch, python_runtime
):
    monkeypatch.setattr(
        load_runner,
        "prepare_replay",
        lambda *_args, **_kwargs: pytest.fail("runtime policy must fail before replay setup"),
    )
    config = ProfileConfig(
        name="direct-sglang",
        log_dir=str(tmp_path / "profile"),
        gpu="NVIDIA B200",
        engine="sglang",
        server=ServerConfig(model_path="model"),
        python_runtime=python_runtime,
    )

    with pytest.raises(ValueError, match="FLASHINFER_DISABLE_JIT=1"):
        alignment_runner.run_profile(config)


def test_python_runtime_cannot_override_launcher_owned_environment(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["python_runtime"] = _python_runtime_document()
    raw["python_runtime"]["environment"]["CUDA_VISIBLE_DEVICES"] = "7"
    paths["profile"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="launcher-owned variables"):
        load_profile_config(paths["profile"])


def test_python_runtime_required_files_must_be_exact_wheel_paths(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["python_runtime"] = _python_runtime_document()
    raw["python_runtime"]["packages"][0]["required_files"] = ["module.so"]
    paths["profile"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="exact wheel-relative path"):
        load_profile_config(paths["profile"])


@pytest.mark.parametrize("timeout", [True, float("nan"), 0.0, "60"])
def test_python_runtime_timeouts_are_finite_positive_numbers(timeout):
    config = PythonRuntimeConfig(
        packages=[],
        lock_timeout_seconds=timeout,
    )

    with pytest.raises(ValueError, match="finite number > 0"):
        alignment_config_module._validate_python_runtime(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("environment", [], "environment must map"),
        (
            "required_files",
            "artifact/module.so",
            "required_files must be a list",
        ),
    ],
)
def test_python_runtime_collection_fields_reject_wrong_container_types(field, value, message):
    package = PythonPackageArtifact(
        name="artifact",
        version="1.0",
        index_url="https://packages.invalid/simple",
    )
    if field == "required_files":
        package = PythonPackageArtifact(
            name=package.name,
            version=package.version,
            index_url=package.index_url,
            required_files=value,
        )
        config = PythonRuntimeConfig(packages=[package])
    else:
        config = PythonRuntimeConfig(packages=[package], environment=value)

    with pytest.raises(ValueError, match=message):
        alignment_config_module._validate_python_runtime(config)


def test_runtime_artifacts_do_not_install_or_refresh_a_satisfied_venv(tmp_path, monkeypatch):
    fork_python = tmp_path / "venv" / "bin" / "python"
    fork_python.parent.mkdir(parents=True)
    fork_python.write_text("")
    package = PythonPackageArtifact(
        name="flashinfer-jit-cache",
        version="0.6.15.post1",
        local_version="cu130",
        index_url="https://flashinfer.ai/whl/cu130",
        required_files=["artifact/module.so"],
    )
    state = {
        package.name: {
            "installed_version": "0.6.15.post1+cu130",
            "required_files": {
                "artifact/module.so": [{"path": "/wheel/artifact/module.so", "size_bytes": 123}]
            },
        }
    }
    monkeypatch.setattr(runtime_artifacts, "_inspect", lambda *_: state)
    monkeypatch.setattr(
        runtime_artifacts,
        "_install_missing",
        lambda *_: pytest.fail("a satisfied runtime must not invoke the installer"),
    )

    provenance = runtime_artifacts.prepare_python_runtime(
        str(fork_python),
        PythonRuntimeConfig(packages=[package], lock_timeout_seconds=1.0),
    )

    assert provenance is not None
    assert provenance["packages"][0]["installed_version"] == "0.6.15.post1+cu130"
    assert (tmp_path / "venv" / ".vibesim-python-runtime.lock").is_file()


def test_runtime_artifact_probe_does_not_inherit_another_python_environment(
    monkeypatch,
):
    package = PythonPackageArtifact(
        name="artifact",
        version="1.0",
        index_url="https://packages.invalid/simple",
    )
    monkeypatch.setenv("PYTHONPATH", "/wrong/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/wrong/python")
    monkeypatch.setenv("VIRTUAL_ENV", "/wrong/venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/wrong/uv-venv")
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return runtime_artifacts.subprocess.CompletedProcess(
            argv, 0, json.dumps({"artifact": None}), ""
        )

    monkeypatch.setattr(runtime_artifacts.subprocess, "run", fake_run)
    assert runtime_artifacts._inspect("/target/venv/bin/python", [package]) == {"artifact": None}

    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        assert key not in seen["env"]
    assert seen["env"]["PYTHONNOUSERSITE"] == "1"


def test_runtime_artifact_probe_requires_the_exact_wheel_relative_path(tmp_path):
    site = tmp_path / "site"
    dist_info = site / "artifact-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: artifact\nVersion: 1.0\n"
    )
    (dist_info / "RECORD").write_text("wrong/module.so,,\nexpected/module.so,,\n")
    wrong = site / "wrong" / "module.so"
    wrong.parent.mkdir()
    wrong.write_bytes(b"wrong artifact with the same basename")

    fork_python = tmp_path / "venv" / "bin" / "python"
    fork_python.parent.mkdir(parents=True)
    fork_python.write_text(
        "#!/bin/sh\n"
        f"PYTHONPATH='{site}' exec '{sys.executable}' \"$@\"\n"
    )
    fork_python.chmod(0o755)
    package = PythonPackageArtifact(
        name="artifact",
        version="1.0",
        index_url="https://packages.invalid/simple",
        required_files=["expected/module.so"],
    )

    state = runtime_artifacts._inspect(str(fork_python), [package])

    assert state["artifact"] == {
        "distribution_count": 1,
        "installed_version": "1.0",
        "required_files": {"expected/module.so": []},
    }


def test_runtime_artifact_installer_accepts_wheels_only(monkeypatch):
    package = PythonPackageArtifact(
        name="artifact",
        version="1.0",
        index_url="https://packages.invalid/simple",
    )
    seen = {}
    monkeypatch.setattr(runtime_artifacts.shutil, "which", lambda _name: "/usr/bin/uv")

    def fake_run(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return runtime_artifacts.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(runtime_artifacts.subprocess, "run", fake_run)
    runtime_artifacts._install_missing("/target/venv/bin/python", [package], 123.0)

    only_binary = seen["argv"].index("--only-binary")
    assert seen["argv"][only_binary + 1] == ":all:"
    assert seen["timeout"] == 123.0


def test_runtime_artifact_installer_has_a_bounded_timeout(monkeypatch):
    package = PythonPackageArtifact(
        name="artifact",
        version="1.0",
        index_url="https://packages.invalid/simple",
    )
    monkeypatch.setattr(runtime_artifacts.shutil, "which", lambda _name: "/usr/bin/uv")

    def timeout(argv, **kwargs):
        raise runtime_artifacts.subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(runtime_artifacts.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out after 7.0s"):
        runtime_artifacts._install_missing("/target/venv/bin/python", [package], 7.0)


def test_runtime_artifacts_install_only_missing_packages_under_the_lock(tmp_path, monkeypatch):
    fork_python = tmp_path / "venv" / "bin" / "python"
    fork_python.parent.mkdir(parents=True)
    fork_python.write_text("")
    package = PythonPackageArtifact(
        name="artifact",
        version="1.2.3",
        index_url="https://packages.invalid/simple",
        required_files=["artifact/artifact.so"],
    )
    states = iter(
        [
            {package.name: None},
            {
                package.name: {
                    "installed_version": "1.2.3",
                    "required_files": {
                        "artifact/artifact.so": [
                            {"path": "/wheel/artifact/artifact.so", "size_bytes": 7}
                        ]
                    },
                }
            },
        ]
    )
    installs = []
    monkeypatch.setattr(runtime_artifacts, "_inspect", lambda *_: next(states))
    monkeypatch.setattr(
        runtime_artifacts,
        "_install_missing",
        lambda python, packages, timeout: installs.append((python, packages, timeout)),
    )

    runtime_artifacts.prepare_python_runtime(
        str(fork_python), PythonRuntimeConfig(packages=[package])
    )

    assert installs == [(str(fork_python), [package], 1800.0)]


def test_runtime_artifact_lock_prevents_two_profiles_from_installing_together(
    tmp_path, monkeypatch
):
    fork_python = tmp_path / "venv" / "bin" / "python"
    fork_python.parent.mkdir(parents=True)
    fork_python.write_text("")
    package = PythonPackageArtifact(
        name="artifact",
        version="1.2.3",
        index_url="https://packages.invalid/simple",
    )
    installed = False
    install_started = threading.Event()
    release_install = threading.Event()
    installs = []

    def inspect(*_args):
        return {
            package.name: (
                {"installed_version": "1.2.3", "required_files": {}} if installed else None
            )
        }

    def install(*_args):
        nonlocal installed
        installs.append(threading.get_ident())
        install_started.set()
        assert release_install.wait(timeout=2)
        installed = True

    monkeypatch.setattr(runtime_artifacts, "_inspect", inspect)
    monkeypatch.setattr(runtime_artifacts, "_install_missing", install)
    config = PythonRuntimeConfig(packages=[package], lock_timeout_seconds=2.0)
    errors = []

    def prepare():
        try:
            runtime_artifacts.prepare_python_runtime(str(fork_python), config)
        except BaseException as exc:  # surfaced in the parent after both joins
            errors.append(exc)

    first = threading.Thread(target=prepare)
    second = threading.Thread(target=prepare)
    first.start()
    assert install_started.wait(timeout=2)
    second.start()
    time.sleep(0.15)
    assert second.is_alive(), "the second profile must wait for the venv preparation lock"
    release_install.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert len(installs) == 1


def test_runtime_artifacts_refuse_to_replace_an_existing_version(tmp_path, monkeypatch):
    fork_python = tmp_path / "venv" / "bin" / "python"
    fork_python.parent.mkdir(parents=True)
    fork_python.write_text("")
    package = PythonPackageArtifact(
        name="artifact",
        version="2.0",
        index_url="https://packages.invalid/simple",
    )
    monkeypatch.setattr(
        runtime_artifacts,
        "_inspect",
        lambda *_: {package.name: {"installed_version": "1.0", "required_files": {}}},
    )
    monkeypatch.setattr(
        runtime_artifacts,
        "_install_missing",
        lambda *_: pytest.fail("a version conflict must not refresh the venv"),
    )

    with pytest.raises(RuntimeError, match="Refusing to mutate"):
        runtime_artifacts.prepare_python_runtime(
            str(fork_python), PythonRuntimeConfig(packages=[package])
        )


def test_runtime_artifacts_refuse_duplicate_installed_distributions(tmp_path, monkeypatch):
    fork_python = tmp_path / "venv" / "bin" / "python"
    fork_python.parent.mkdir(parents=True)
    fork_python.write_text("")
    package = PythonPackageArtifact(
        name="artifact",
        version="2.0",
        index_url="https://packages.invalid/simple",
    )
    monkeypatch.setattr(
        runtime_artifacts,
        "_inspect",
        lambda *_: {
            package.name: {
                "distribution_count": 2,
                "installed_versions": ["1.0", "2.0"],
                "required_files": {},
            }
        },
    )
    monkeypatch.setattr(
        runtime_artifacts,
        "_install_missing",
        lambda *_: pytest.fail("an ambiguous environment must not invoke the installer"),
    )

    with pytest.raises(RuntimeError, match="2 installed distributions"):
        runtime_artifacts.prepare_python_runtime(
            str(fork_python), PythonRuntimeConfig(packages=[package])
        )


def test_nsys_preflight_is_scoped_to_new_captures():
    calls = []

    class Driver:
        @staticmethod
        def validate_nsys_capture_environment(fork_python, *, env, cwd):
            calls.append((fork_python, env, cwd))

    for profile_kind, resume in (
        ("workload_metrics", False),
        ("expert_popularity", False),
        ("token_corpus", False),
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
        reduction_group_size=2,
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
        reduction_group_size=2,
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
            reduction_group_size=2,
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
        reduction_group_size=1,
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
            reduction_group_size=1,
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


def _routing_profile(tmp_path: Path, profile_kind: str, **server) -> ProfileConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["profile_kind"] = profile_kind
    raw["cuda_visible_devices"] = "0,1,2,3,4,5,6,7"
    raw["server"]["tp_size"] = 4
    raw["server"]["dp_size"] = 2
    raw["server"].update(server)
    paths["profile"].write_text(yaml.safe_dump(raw))
    return load_profile_config(paths["profile"])


def test_the_yaml_expert_topology_reaches_the_expert_load_extractor(tmp_path, monkeypatch):
    seen: dict = {}

    def record(*_args, **kwargs):
        seen.update(kwargs)
        return 7

    monkeypatch.setattr(alignment_runner.record_extraction, "extract_expert_popularity", record)
    cfg = _routing_profile(
        tmp_path,
        "expert_popularity",
        extra_args=["--enable-expert-parallel"],
        expert_parallel_size=2,
        expert_count_reduction_group_size=4,
    )

    artifacts = alignment_runner._extract_expert_load(
        cfg,
        tmp_path / "server.log",
        tmp_path,
        tmp_path,
        engine_records.VLLM_RECORDS,
        speculative=False,
        window={},
    )

    assert (seen["expert_parallel_size"], seen["reduction_group_size"]) == (2, 4)
    assert artifacts["expert_record_count"] == 7


def test_a_pass_that_asks_for_the_expert_load_stream_also_requires_it(tmp_path):
    """Asking and requiring are one question, and the answer is the deployment.

    A corpus capture on a deployment with no expert parallelism cannot produce
    the stream at all, and still records the routes themselves; one that turns
    EPLB on and then finds nothing has a defect, not a missing by-product.
    """
    for kind, server in (
        ("expert_popularity", {}),
        ("token_corpus", {"extra_args": ["--enable-expert-parallel"]}),
    ):
        with pytest.raises(ValueError, match="expert_parallel_size"):
            alignment_runner._extract_expert_load(
                _routing_profile(tmp_path / kind, kind, **server),
                tmp_path / "server.log",
                tmp_path,
                tmp_path,
                engine_records.VLLM_RECORDS,
                speculative=False,
                window={},
            )

    assert (
        alignment_runner._extract_expert_load(
            _routing_profile(tmp_path / "no-ep", "token_corpus"),
            tmp_path / "server.log",
            tmp_path,
            tmp_path,
            engine_records.VLLM_RECORDS,
            speculative=False,
            window={},
        )
        == {}
    )


def test_a_replay_cannot_pack_a_previous_captures_requests(tmp_path):
    """The packer concatenates every file it finds, so the directory is emptied."""
    cfg = _routing_profile(tmp_path / "corpus", "token_corpus")
    routes = Path(cfg.log_dir) / alignment_runner.ROUTED_EXPERTS_DIR
    routes.mkdir(parents=True)
    stale = routes / "from-a-previous-run-0000.npy"
    stale.write_bytes(b"stale")

    prepared = alignment_runner._prepared_routes_dir(cfg, Path(cfg.log_dir))

    assert prepared == routes
    assert not stale.exists()
    assert list(routes.iterdir()) == []


@pytest.mark.parametrize(
    "authored",
    [
        ["--eplb-config", '{"window_size": 1000}'],
        ['--eplb-config={"window_size": 1000}'],
    ],
    ids=["separate", "joined"],
)
def test_an_authored_eplb_config_keeps_its_settings_and_gains_the_log(tmp_path, authored):
    """The pass adds what it needs to a tuned object rather than skipping it.

    argparse keeps the last occurrence, so a second flag would drop the tuning.
    """
    argv = ["--enable-expert-parallel", *authored]
    alignment_runner._append_backend_server_args(
        argv,
        _routing_profile(
            tmp_path / "tuned",
            "token_corpus",
            tp_size=4,
            extra_args=["--enable-expert-parallel"],
            expert_parallel_size=4,
            expert_count_reduction_group_size=4,
        ),
    )

    [config] = [arg for arg in argv if arg.startswith("--eplb-config")]
    merged = json.loads(config.split("=", 1)[1])
    assert merged == {"window_size": 1000, "log_balancedness": True, "rearrange": False}


def test_a_routing_pass_asks_for_the_expert_load_stream_only_where_it_exists(tmp_path):
    """vLLM's balancedness log is its only source, and it refuses EPLB without EP."""
    argv: list[str] = ["--enable-expert-parallel"]
    alignment_runner._append_backend_server_args(
        argv,
        _routing_profile(
            tmp_path / "ep",
            "token_corpus",
            extra_args=["--enable-expert-parallel"],
            expert_parallel_size=2,
            expert_count_reduction_group_size=4,
        ),
    )
    assert json.loads(argv[argv.index("--eplb-config") + 1]) == {
        "log_balancedness": True,
        "rearrange": False,
    }
    assert "--enable-return-routed-experts" in argv

    bare: list[str] = []
    alignment_runner._append_backend_server_args(
        bare, _routing_profile(tmp_path / "tp", "token_corpus")
    )
    assert "--enable-eplb" not in bare
    assert "--enable-return-routed-experts" in bare


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


def test_timing_predict_dry_run_builds_cases_in_scratch_and_writes_nothing(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    _write_completed_inputs(tmp_path)
    seen = []

    def launch(path, **kwargs):
        # The predictor sees real cases while the scratch directory still exists.
        seen.append((json.loads(path.read_text()), kwargs))
        assert json.loads(Path(seen[0][0]["cases_file"]).read_text())
        return 0

    monkeypatch.setattr(alignment_launcher, "_launch_timing_predict", launch)
    before = sorted(tmp_path.rglob("*"))

    assert alignment_launcher.main(["timing-predict", str(paths["timing"]), "--dry-run"]) == 0
    [(predict_config, kwargs)] = seen
    assert kwargs == {"build_type": "release", "dry_run": True}
    assert not Path(predict_config["log_dir"]).exists()
    assert sorted(tmp_path.rglob("*")) == before


def test_timing_predict_dry_run_is_forwarded_to_the_predictor(tmp_path, monkeypatch):
    config_path = tmp_path / "timing_predict_config.json"
    config_path.write_text("{}")
    launched = []
    monkeypatch.setattr(timing_predict_launcher, "main", lambda argv: launched.append(argv) or 0)

    assert (
        alignment_launcher._launch_timing_predict(config_path, build_type="release", dry_run=True)
        == 0
    )
    assert launched == [[str(config_path), "--build-type", "release", "--dry-run"]]


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
                "type": "glm52_vllm_dsa_moe",
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
        if argv[1] == "alignment":
            (log_dir / "reports").mkdir(exist_ok=True)
            (log_dir / "reports" / "analyzer_timing.json").write_text(
                json.dumps({"subjects": [{"name": "alignment-e2e", "status": "ok"}]})
            )
        return 0, "alignment accepted\n"

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda build_type: analyzer)
    monkeypatch.setattr(launcher_exec, "_run_capture_sync", fake_capture)

    assert launcher_exec.run_alignment_analysis(log_dir, "release", ["alignment-e2e"])
    assert invocations[0] == [str(analyzer), "alignment", str(log_dir), "alignment-e2e"]
    assert invocations[1][-1:] == ["alignment-e2e"]


@pytest.mark.parametrize("outcome", ["failed", "missing", "stale"])
def test_alignment_analysis_does_not_publish_failed_compute(tmp_path, monkeypatch, outcome):
    analyzer = tmp_path / "analyze"
    analyzer.touch()
    log_dir = tmp_path / "analysis"
    reports = log_dir / "reports"
    reports.mkdir(parents=True)
    timing = reports / "analyzer_timing.json"
    timing.write_text(json.dumps({"subjects": [{"name": "alignment-iteration", "status": "ok"}]}))
    invocations = []

    def fake_capture(argv):
        invocations.append(argv)
        if outcome != "stale":
            rows = [] if outcome == "missing" else [
                {"name": "alignment-iteration", "status": "failed"}
            ]
            timing.write_text(json.dumps({"subjects": rows}))
        return 0, "compute process exited zero\n"

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda build_type: analyzer)
    monkeypatch.setattr(launcher_exec, "_run_capture_sync", fake_capture)
    assert not launcher_exec.run_alignment_analysis(log_dir, "release", ["alignment-iteration"])
    assert len(invocations) == 1


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


def test_a_popularity_pass_refuses_the_eplb_opt_out(tmp_path):
    """The marginal is the stream's only product; the replay would run for nothing."""
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["profile_kind"] = "expert_popularity"
    raw["server"]["extra_args"] = ["--no-enable-eplb"]
    paths["profile"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="use token_corpus"):
        load_profile_config(paths["profile"])


def test_a_corpus_pass_refuses_a_protocol_that_returns_no_routes(tmp_path):
    """Every response would fail to fold; a launch would only spend the job."""
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["profile_kind"] = "token_corpus"
    raw["workload"]["backend"] = {"type": "vllm_tokens"}
    paths["profile"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="'vllm_tokens' does not"):
        load_profile_config(paths["profile"])


def test_a_corpus_pass_refuses_the_v1_model_runner(tmp_path):
    """The server refuses MTP route capture under V1 only after the job is scheduled."""
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["profile"].read_text())
    raw["profile_kind"] = "token_corpus"
    raw["python_runtime"] = {
        "packages": [
            {"name": "nvtx", "version": "0.2.16", "index_url": "https://pypi.org/simple"}
        ],
        "environment": {"VLLM_USE_V2_MODEL_RUNNER": "0"},
    }
    paths["profile"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="model runner V2"):
        load_profile_config(paths["profile"])


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


@pytest.mark.parametrize(
    ("success", "failed", "files", "accepted"),
    [(2, 0, 2, True), (2, 0, 1, False), (1, 1, 1, False)],
    ids=["whole", "a_success_without_routes", "a_failed_request"],
)
def test_a_corpus_is_published_only_when_every_request_carried_routes(
    tmp_path, success, failed, files, accepted
):
    """The load generator exits cleanly when a response lacks routes."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"replay": {"common": {"success_steps": success, "failed_steps": failed}}})
    )
    routes = tmp_path / "routed_experts"
    routes.mkdir()
    for index in range(files):
        (routes / f"request-{index}.npy").write_bytes(b"")

    if accepted:
        alignment_runner._check_routes_cover_replay(summary, routes)
    else:
        with pytest.raises(ValueError, match="carried routes"):
            alignment_runner._check_routes_cover_replay(summary, routes)


def test_a_hub_model_id_reads_its_config_from_the_hub_cache(tmp_path, monkeypatch):
    """`model_path` is what vLLM's --model accepted, a repo id included."""
    import huggingface_hub

    cached = tmp_path / "snapshot" / "config.json"
    cached.parent.mkdir()
    cached.write_text(json.dumps({"n_routed_experts": 256}))
    fetched = []

    def fake_download(repo_id, filename, revision):
        fetched.append((repo_id, filename, revision))
        return str(cached)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.chdir(tmp_path)
    server = SimpleNamespace(model_path="nvidia/GLM-5.2-NVFP4", extra_args=[])

    assert alignment_runner._checkpoint_config(server) == cached
    # The revision the server loaded, in either spelling.
    server.extra_args = ["--revision", "abc123"]
    alignment_runner._checkpoint_config(server)
    server.extra_args = ["--revision=def456"]
    alignment_runner._checkpoint_config(server)
    assert fetched == [
        ("nvidia/GLM-5.2-NVFP4", "config.json", None),
        ("nvidia/GLM-5.2-NVFP4", "config.json", "abc123"),
        ("nvidia/GLM-5.2-NVFP4", "config.json", "def456"),
    ]

    local = tmp_path / "checkpoint"
    local.mkdir()
    (local / "config.json").write_text("{}")
    assert (
        alignment_runner._checkpoint_config(SimpleNamespace(model_path=str(local), extra_args=[]))
        == local / "config.json"
    )
    # A separate config path wins over the weights, as it does in the server.
    overridden = tmp_path / "config-only"
    overridden.mkdir()
    (overridden / "config.json").write_text("{}")
    assert (
        alignment_runner._checkpoint_config(
            SimpleNamespace(model_path=str(local), extra_args=["--hf-config-path", str(overridden)])
        )
        == overridden / "config.json"
    )
    assert len(fetched) == 3


def test_a_model_without_eplb_opts_out_of_the_expert_load_stream(tmp_path):
    """Only loading the model shows whether it balances; the operator says so."""
    argv = ["--enable-expert-parallel", "--no-enable-eplb"]
    cfg = _routing_profile(
        tmp_path / "no-eplb",
        "token_corpus",
        # No topology declared: a pass without the stream has nothing to reduce.
        extra_args=["--enable-expert-parallel", "--no-enable-eplb"],
    )
    alignment_runner._append_backend_server_args(argv, cfg)

    assert not cfg.captures_expert_load
    assert "--enable-eplb" not in argv
    assert "--enable-return-routed-experts" in argv
