"""CPU tests for the four explicit alignment launcher phase contracts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from alignment.load_generator import runner as load_runner
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
        "analyze": tmp_path / f"analyze{suffix}",
        "labeled": tmp_path / "kernel_sequences_labeled.json",
    }
    _write_config(
        paths["simulation"],
        {
            "deployment": "unified",
            "workload": {"trace_files": [str(trace)]},
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
                "frontend": {"type": "vibesim", "path": "./trace/shared.csv"},
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
            "simulation_log_dir": "./simulation_run",
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
        paths["analyze"],
        {
            "schema_version": 1,
            "simulation_log_dir": "./simulation_run",
            "profile_log_dir": "./profile_run",
            "timing_predict_log_dir": "./timing_predict_run",
            "log_dir": "./analysis_run",
            "iteration": {
                "enabled": True,
                "labeled_kernel_sequences_file": "./kernel_sequences_labeled.json",
            },
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
        "workload": {"trace_files": [str(tmp_path / "trace" / "shared.csv")]},
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
    replay = profile / "load_generator" / "replay.jsonl"
    parsed.parent.mkdir(parents=True)
    replay.parent.mkdir()
    parsed.write_text(json.dumps(_parsed_iteration()))
    replay.write_text("")
    result = {
        "log_dir": str(profile),
        "parsed_nsys": str(parsed),
        "replay_result": str(replay),
        "server_tp_size": 1,
        "drive_summary": {
            "source_trace": str((tmp_path / "trace" / "shared.csv").resolve()),
            "log_path": str(replay),
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
    simulation, profile, profile_result = _write_completed_inputs(tmp_path)
    output = tmp_path / "timing_predict_run"
    result = build_inputs(
        BuildRequest(
            simulation_log_dir=simulation,
            profile_log_dir=profile,
            parsed_nsys=Path(profile_result["parsed_nsys"]),
            output_dir=output,
            gpu="NVIDIA H200",
            arch={"type": "llama3_dense", "fp8": False},
            backends={},
            input_spec=alignment_launcher.load_timing_predict_config(
                tmp_path / "timing_predict.yaml"
            ).input_builder,
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


def test_profile_config_is_profile_only_and_config_relative(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    profiled = []
    monkeypatch.setattr(
        alignment_launcher,
        "run_profile",
        lambda config: (
            profiled.append(config)
            or {"parsed_nsys": str(tmp_path / "profile_run" / "parsed.json")}
        ),
    )

    assert alignment_launcher.main(["profile", str(paths["profile"])]) == 0
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
    assert alignment_launcher.load_analyze_config(paths["analyze"]).e2e.throughput_bins == 20


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


def test_analyze_rejects_removed_kernel_mapping_field(tmp_path):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["analyze"].read_text())
    raw["iteration"] = {"enabled": True, "kernel_mapping_file": "./mapping.yaml"}
    paths["analyze"].write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="kernel_mapping_file"):
        load_analyze_config(paths["analyze"])


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


def test_timing_predict_preserves_simulation_backend_policy(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    simulation, _, _ = _write_completed_inputs(tmp_path)
    params_path = simulation / "raw" / "params.json"
    params = json.loads(params_path.read_text())
    params["backends"] = {
        "main": {"unified.pre_attn.qkv_proj": ["torch_linear"]}
    }
    params_path.write_text(json.dumps(params))
    monkeypatch.setattr(alignment_launcher, "_launch_timing_predict", lambda *args, **kwargs: 0)

    assert alignment_launcher.main(["timing-predict", str(paths["timing"])]) == 0
    predict_config = json.loads(
        (tmp_path / "timing_predict_run" / "timing_predict_config.json").read_text()
    )
    assert predict_config["backends"] == params["backends"]


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


def test_analyze_creates_manifest_and_runs_selected_subjects(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    _write_timing_artifacts(tmp_path)
    calls = []
    monkeypatch.setattr(
        alignment_launcher,
        "_launch_alignment_analysis",
        lambda log_dir, **kwargs: calls.append((log_dir, kwargs)) or True,
    )

    assert alignment_launcher.main(["analyze", str(paths["analyze"])]) == 0
    analysis = tmp_path / "analysis_run"
    manifest = json.loads((analysis / "alignment_manifest.json").read_text())
    assert manifest["schema_version"] == 3
    assert manifest["labeled_kernel_sequences"] == str(analysis / "kernel_sequences_labeled.json")
    assert manifest["iteration"] == {"enabled": True}
    assert manifest["workload"] == {"enabled": True}
    assert calls == [
        (
            analysis.resolve(),
            {
                "build_type": "release",
                "subjects": ["alignment-iteration", "alignment-workload", "alignment-e2e"],
            },
        )
    ]


def test_analyze_e2e_only_does_not_require_labeled_inventory(tmp_path, monkeypatch):
    paths = _phase_configs(tmp_path)
    _write_timing_artifacts(tmp_path)
    raw = yaml.safe_load(paths["analyze"].read_text())
    raw["iteration"] = {"enabled": False}
    raw["workload"] = {"enabled": False}
    paths["analyze"].write_text(yaml.safe_dump(raw))
    calls = []
    monkeypatch.setattr(
        alignment_launcher,
        "_launch_alignment_analysis",
        lambda log_dir, **kwargs: calls.append(kwargs["subjects"]) or True,
    )

    assert alignment_launcher.main(["analyze", str(paths["analyze"])]) == 0
    assert calls == [["alignment-e2e"]]


def test_analyze_rejects_all_subjects_disabled(tmp_path, capsys):
    paths = _phase_configs(tmp_path)
    raw = yaml.safe_load(paths["analyze"].read_text())
    raw["iteration"] = {"enabled": False}
    raw["workload"] = {"enabled": False}
    raw["e2e"] = {"enabled": False}
    paths["analyze"].write_text(yaml.safe_dump(raw))

    assert alignment_launcher.main(["analyze", str(paths["analyze"])]) == 2
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


def test_alignment_analysis_calls_only_selected_subjects(tmp_path, monkeypatch):
    analyzer = tmp_path / "analyze"
    analyzer.touch()
    log_dir = tmp_path / "analysis"
    log_dir.mkdir()
    invocations = []

    async def fake_capture(argv):
        invocations.append(argv)
        return 0, "alignment accepted\n"

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda build_type: analyzer)
    monkeypatch.setattr(launcher_exec, "_run_capture", fake_capture)

    asyncio.run(launcher_exec.run_alignment_analysis(log_dir, "release", ["alignment-e2e"]))
    assert invocations[0] == [str(analyzer), "alignment", str(log_dir), "alignment-e2e"]
    assert invocations[1][-1:] == ["alignment-e2e"]


def test_tracelab_invocation_keeps_vibesim_as_a_typed_frontend(tmp_path, monkeypatch):
    from alignment.load_generator.config import LoadGeneratorConfig, VibeSimFrontendConfig

    config = LoadGeneratorConfig(
        frontend=VibeSimFrontendConfig(path="trace/smoke.csv"),
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
    assert argv[argv.index("--trace-format") + 1] == "vibesim"
    assert argv[argv.index("--max-concurrency") + 1] == "64"


def test_launcher_main_dispatches_alignment_subcommand(monkeypatch):
    from launcher import __main__ as launcher_main

    received = []
    monkeypatch.setattr(alignment_launcher, "main", lambda argv: received.append(argv) or 0)
    assert launcher_main.main(["alignment", "profile", "profile.yaml"]) == 0
    assert received == [["profile", "profile.yaml"]]
