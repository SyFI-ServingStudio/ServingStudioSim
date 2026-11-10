"""Alignment Phase 1 cases, tracked traces, and inputs read from the capture.

A Phase 1 case replays `trace/diverse_100.csv` at its recorded arrival times and
reads `attn_gpu_memory_gb` and `speculative_acceptance` from its own workload
pass. The fixture texts below are copies of real lines from a DeepSeek-V4.1
DSpark TP4/EP4 diverse_100 workload pass (vLLM fork, 2026-11-07).
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest
import yaml

from launcher.alignment_campaign import calibration, cli, execute
from launcher.alignment_campaign import check as check_module
from launcher.alignment_campaign import compare as compare_module
from launcher.alignment_campaign.metrics import measure_case, metric_names
from launcher.alignment_campaign.pack import PackError, load_pack
from launcher.alignment_campaign.render import (
    SIMULATION_PHASE,
    TIMING_PREDICT_PHASE,
    case_documents,
    render_case,
    simulation_document,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = REPO_ROOT / "presets" / "alignment"
BASE_PACK = PACK_ROOT / "glm53_nvfp4_b200_dflash2"  # vLLM, speculative, has a workload pass
DIVERSE = REPO_ROOT / "trace" / "diverse_100.csv"

SPEC_METRICS = {
    "schema_version": 1,
    "num_drafts": 35205,
    "draft_tokens": 176025,
    "accepted_tokens": 100092,
    "accepted_tokens_per_position": {"0": 28012, "1": 22465, "2": 18875, "3": 16347, "4": 14393},
}
EXPECTED_CHAIN = [0.7957, 0.802, 0.8402, 0.8661, 0.8805]
SERVER_LOG = "".join(
    line + "\n"
    for line in (
        "(Worker_TP0_EP0 pid=1) INFO 11-07 06:14:46 [gpu_worker.py:813] "
        "Available KV cache memory: 84.78 GiB",
        "(EngineCore pid=2) INFO 11-07 06:16:34 [kv_cache_utils.py:1168] "
        "GPU KV cache size: 40,746,855 tokens",
        "(Worker_TP3_EP3 pid=4) INFO 11-07 06:16:34 [gpu_worker.py:497] "
        "Extended KV cache to 566724 blocks (80.27 GiB).",
        "(Worker_TP0_EP0 pid=1) INFO 11-07 06:16:34 [gpu_worker.py:497] "
        "Extended KV cache to 566723 blocks (80.26 GiB).",
    )
)
# 80.26 GiB = 80.26 * 2**30 / 1e9 GB
EXPECTED_KV_GB = 86.178519


# ── fixtures ─────────────────────────────────────────────────────────────────


def _phase1_pack(tmp_path: Path, *, engine: str = "vllm", acceptance: bool = True):
    """The base pack plus one Phase 1 diverse_100 case on a depth-5 variant."""
    root = tmp_path / "pack"
    shutil.copytree(BASE_PACK, root)
    document = yaml.safe_load((root / "campaign.yaml").read_text())
    base_name, base = next(iter(document["variants"].items()))
    variant = json.loads(json.dumps(base))
    variant["engine"] = engine
    variant["arch"]["draft_tokens"] = 5
    variant["worker"]["draft_tokens"] = 5
    if engine != "vllm":
        variant["server"]["extra_args"] = [
            arg for arg in variant["server"]["extra_args"] if "max-model-len" not in arg
        ]
    document["variants"]["p1"] = variant
    first = document["cases"][0]
    case = {
        "index": 90,
        "name": "diverse_100",
        "variant": "p1",
        "phase": 1,
        "purpose": "alignment Phase 1: the per-deployment diverse_100 check",
        "coverage": ["diverse"],
        "max_model_len": 131072,
        "profile_arrival_mode": "trace-timed",
        "simulation_arrival_mode": "trace_timed",
        "gpu_memory_utilization": 0.9,
        "capture_seconds": 30.0,
        "device_role": first["device_role"],
        "workload_trace": {"path": "trace/diverse_100.csv"},
        "attn_gpu_memory_gb": {"status": "from_capture"},
    }
    if acceptance:
        case["speculative_acceptance"] = {"status": "from_capture"}
    document["cases"].append(case)
    (root / "campaign.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    pack = load_pack(root)
    check_module.write_invariants(pack)
    return load_pack(root)


def _write_workload_pass(case_dir: Path, *, log: str = SERVER_LOG, engine: str = "vllm") -> None:
    """A workload pass as `alignment profile` leaves it, recorded from elsewhere."""
    directory = case_dir / "profile_workload"
    (directory / engine).mkdir(parents=True, exist_ok=True)
    recorded = "/elsewhere/campaign/90_diverse_100/profile_workload"
    (directory / "spec_decode_metrics.json").write_text(json.dumps(SPEC_METRICS))
    (directory / engine / "p1_server.log").write_text(log)
    (directory / "profile_result.json").write_text(json.dumps({
        "engine": engine,
        "profile_kind": "workload_metrics",
        "log_dir": recorded,
        "server_log": f"{recorded}/{engine}/p1_server.log",
        "spec_decode_metrics_json": f"{recorded}/spec_decode_metrics.json",
    }))


def _render(pack, case, out_root):
    return render_case(pack, case, check_module.host_for(pack, None), out_root, REPO_ROOT)


def _phase1_case(pack):
    (case,) = pack.phase1_cases
    return case


# ── alignment Phase 1 / 2 selection ──────────────────────────────────────────


def test_phase_defaults_to_two_and_phase1_selects_only_phase_one(tmp_path):
    pack = _phase1_pack(tmp_path)
    case = _phase1_case(pack)
    assert case.slug == "90_diverse_100" and case.phase == 1
    assert all(item.phase == 2 for item in pack.cases if item is not case)
    assert cli._selected_cases(pack, None, phase1=True) == [case]
    # alignment Phase 2 is the whole matrix, Phase 1 included.
    assert cli._selected_cases(pack, None) == list(pack.cases)
    with pytest.raises(PackError, match="not an alignment Phase 1 case"):
        cli._selected_cases(pack, [pack.cases[0].slug], phase1=True)


def test_phase1_on_a_pack_without_one_is_a_clear_error(tmp_path, capsys):
    pack = load_pack(BASE_PACK)
    with pytest.raises(PackError, match="declares no alignment Phase 1 case"):
        cli._selected_cases(pack, None, phase1=True)
    code = cli.main([
        "run", "--pack", str(BASE_PACK), "--out-root", str(tmp_path), "--phase", "simulation",
        "--phase1", "--dry-run",
    ])
    assert code == 2
    assert "declares no alignment Phase 1 case" in capsys.readouterr().err
    assert cli.main(["extract", "--runs", str(tmp_path), "--out", str(tmp_path / "m.json"),
                     "--phase1"]) == 2


@pytest.mark.parametrize("phase", [0, 3, "1", True])
def test_phase_rejects_anything_but_one_or_two(tmp_path, phase):
    pack = _phase1_pack(tmp_path)
    document = yaml.safe_load((pack.root / "campaign.yaml").read_text())
    document["cases"][-1]["phase"] = phase
    (pack.root / "campaign.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(PackError, match="phase must be 1"):
        load_pack(pack.root)


def test_extract_phase1_keeps_only_phase_one_cases(tmp_path):
    from launcher.alignment_campaign import extract as extract_module

    pack = _phase1_pack(tmp_path)
    extraction = extract_module.extract([tmp_path / "runs"], pack, only={"90_diverse_100"})
    assert list(extraction.cases) == ["p1/90_diverse_100"]


# ── the tracked trace ────────────────────────────────────────────────────────


def test_tracked_trace_replays_the_file_at_its_recorded_times(tmp_path):
    pack = _phase1_pack(tmp_path)
    case = _phase1_case(pack)
    assert case.workload_trace.request_count == 100
    rendered = _render(pack, case, tmp_path / "runs")
    case_dir = rendered.directory
    assert (case_dir / "trace.csv").read_bytes() == DIVERSE.read_bytes()
    profile = yaml.safe_load((case_dir / "profile_workload.yaml").read_text())
    assert profile["workload"]["arrival_mode"] == "trace-timed"
    assert "rate" not in profile["workload"]  # recorded arrivals, scale 1

    recorded = json.loads(check_module.invariants_path(pack).read_text())["traces"]
    digest = hashlib.sha256(DIVERSE.read_bytes()).hexdigest()
    assert recorded["trace/diverse_100.csv"]["sha256"] == digest
    findings = check_module.check_pack(pack, metric_names=metric_names(None))
    assert not [item for item in findings if item.level == "error"], findings


def test_tracked_trace_sha_mismatch_is_a_check_error(tmp_path):
    pack = _phase1_pack(tmp_path)
    path = check_module.invariants_path(pack)
    document = json.loads(path.read_text())
    document["traces"]["trace/diverse_100.csv"]["sha256"] = "0" * 64
    path.write_text(json.dumps(document))
    findings = check_module.check_pack(pack, metric_names=metric_names(None))
    assert any(
        item.level == "error" and "trace/diverse_100.csv invariants differ" in item.message
        for item in findings
    )


def test_trace_timed_without_a_rate_is_only_for_a_tracked_trace(tmp_path):
    pack = _phase1_pack(tmp_path)
    document = yaml.safe_load((pack.root / "campaign.yaml").read_text())
    document["cases"][-1]["workload_trace"] = {"file": "traces/x.csv", "shapes": [[16, 16]]}
    (pack.root / "campaign.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(PackError, match="trace-timed arrival needs a rate"):
        load_pack(pack.root)


@pytest.mark.parametrize("trace", [
    {"path": "/abs/diverse_100.csv"},
    {"path": "trace/no_such.csv"},
    {"path": "trace/diverse_100.csv", "shapes": [[1, 1]]},
])
def test_tracked_trace_rejects_a_bad_path(tmp_path, trace):
    pack = _phase1_pack(tmp_path)
    document = yaml.safe_load((pack.root / "campaign.yaml").read_text())
    document["cases"][-1]["workload_trace"] = trace
    (pack.root / "campaign.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(PackError, match="workload_trace"):
        load_pack(pack.root)


# ── the two extractors ───────────────────────────────────────────────────────


def test_acceptance_chain_is_conditional_per_position():
    chain = calibration.acceptance_chain(SPEC_METRICS)
    assert [round(value, 4) for value in chain] == EXPECTED_CHAIN
    assert chain[0] == round(28012 / 35205, 6)
    assert chain[1] == round(22465 / 28012, 6)


@pytest.mark.parametrize(
    "metrics", [{}, {"num_drafts": 0, "accepted_tokens_per_position": {"0": 1}}]
)
def test_acceptance_chain_rejects_missing_counters(metrics):
    with pytest.raises(calibration.CalibrationError):
        calibration.acceptance_chain(metrics)


def test_vllm_kv_pool_is_the_smallest_ranks_extended_pool():
    pool = calibration.vllm_kv_pool(SERVER_LOG)
    assert (pool.gib, pool.blocks, pool.gb) == (80.26, 566723, EXPECTED_KV_GB)
    assert pool.line == "Extended KV cache to 566723 blocks (80.26 GiB)"


def test_vllm_kv_pool_falls_back_to_available_memory():
    log = (
        "(Worker_TP1 pid=1) INFO [gpu_worker.py:813] Available KV cache memory: 73.85 GiB\n"
        "(Worker_TP0 pid=1) INFO [gpu_worker.py:813] Available KV cache memory: 73.76 GiB\n"
    )
    pool = calibration.vllm_kv_pool(log)
    assert pool.gib == 73.76 and pool.blocks is None
    assert pool.gb == round(73.76 * 1.073741824, 6)
    with pytest.raises(calibration.CalibrationError, match="Extended KV cache"):
        calibration.vllm_kv_pool("INFO nothing about the pool\n")


# ── blocking and calibration.json ────────────────────────────────────────────


def test_simulation_waits_for_the_workload_pass(tmp_path):
    pack = _phase1_pack(tmp_path)
    case = _phase1_case(pack)
    out_root = tmp_path / "runs"
    rendered = _render(pack, case, out_root)
    assert rendered.pending and "profile_workload has not run" in rendered.pending[0]
    assert not (rendered.directory / "simulation.yaml").exists()
    assert not (rendered.directory / "calibration.json").exists()
    assert (rendered.directory / "timing_predict.yaml").is_file()
    for phase in (SIMULATION_PHASE, TIMING_PREDICT_PHASE):
        (plan,) = execute.plan_phase(pack, out_root, phase, cases=[case])
        assert plan.state == "blocked"
        assert "from_capture" in plan.reasons[0]
        assert "profile_workload has not run" in plan.reasons[0]
    host = check_module.host_for(pack, None)
    documents = case_documents(pack, case, host, rendered.directory, REPO_ROOT)
    assert "simulation.yaml" not in documents
    with pytest.raises(PackError, match="from_capture"):
        simulation_document(pack, case, pack.variant_of(case), rendered.directory, REPO_ROOT)


def test_run_reads_the_capture_and_writes_calibration_json(tmp_path, monkeypatch):
    pack = _phase1_pack(tmp_path)
    case = _phase1_case(pack)
    out_root = tmp_path / "runs"
    case_dir = _render(pack, case, out_root).directory
    _write_workload_pass(case_dir)

    # Planning is pure: ready, but nothing written yet.
    (plan,) = execute.plan_phase(pack, out_root, SIMULATION_PHASE, cases=[case])
    assert plan.state == "ready"
    assert not (case_dir / "simulation.yaml").exists()

    async def no_launch(*args, **kwargs):
        return []

    monkeypatch.setattr(execute, "_run_all", no_launch)
    report = execute.run_phase(pack, out_root, SIMULATION_PHASE, cases=[case])
    assert [plan.state for plan in report.plans] == ["ready"]

    document = json.loads((case_dir / "calibration.json").read_text())
    assert document["case"] == "90_diverse_100"
    assert document["workload_pass"] == "profile_workload"
    kv = document["fields"]["attn_gpu_memory_gb"]
    assert kv["value"] == EXPECTED_KV_GB
    assert kv["source"] == "profile_workload/vllm/p1_server.log"
    assert kv["log_line"] == "Extended KV cache to 566723 blocks (80.26 GiB)"
    accept = document["fields"]["speculative_acceptance"]
    assert [round(value, 4) for value in accept["value"]] == EXPECTED_CHAIN
    assert accept["source"] == "profile_workload/spec_decode_metrics.json"

    simulation = yaml.safe_load((case_dir / "simulation.yaml").read_text())
    worker = simulation["pools"]["main"]["groups"][0]["worker"]
    assert worker["attn_gpu_memory_gb"] == EXPECTED_KV_GB
    assert simulation["workload"]["arrival_mode"] == "trace_timed"
    assert simulation["workload"]["request_rate"] == 1.0
    assert simulation["workload"]["trace_files"][0].endswith("trace_speculative.csv")
    rows = list(csv.DictReader(io.StringIO((case_dir / "trace_speculative.csv").read_text())))
    replay = list(csv.DictReader(DIVERSE.open()))
    assert [{k: v for k, v in row.items() if k != "accept_rate"} for row in rows] == replay
    assert all(json.loads(row["accept_rate"]) == accept["value"] for row in rows)

    # Re-rendering now resolves in place, and the measured value is what it uses.
    rendered = _render(pack, case, out_root)
    assert not rendered.pending
    assert yaml.safe_load((case_dir / "simulation.yaml").read_text()) == simulation


def test_an_unreadable_capture_blocks_with_its_reason(tmp_path):
    pack = _phase1_pack(tmp_path)
    case = _phase1_case(pack)
    out_root = tmp_path / "runs"
    case_dir = _render(pack, case, out_root).directory
    _write_workload_pass(case_dir, log="INFO server up\n")
    (plan,) = execute.plan_phase(pack, out_root, SIMULATION_PHASE, cases=[case])
    assert plan.state == "blocked"
    assert "Extended KV cache" in plan.reasons[0]


def test_sglang_kv_pool_is_a_clear_error(tmp_path):
    pack = _phase1_pack(tmp_path, engine="sglang", acceptance=False)
    case = _phase1_case(pack)
    findings = check_module._check_capture_sources(pack)
    assert any("no sglang extractor" in item.message for item in findings)
    _write_workload_pass(tmp_path / "case", engine="sglang")
    with pytest.raises(calibration.CalibrationError, match="max_total_num_tokens"):
        calibration.resolve(pack, case, tmp_path / "case")


@pytest.mark.parametrize("raw", [
    {"status": "from_capture", "value": 80.0},
    {"status": "from_capture", "derived_from": "x"},
])
def test_from_capture_carries_no_value(tmp_path, raw):
    pack = _phase1_pack(tmp_path)
    document = yaml.safe_load((pack.root / "campaign.yaml").read_text())
    document["cases"][-1]["attn_gpu_memory_gb"] = raw
    (pack.root / "campaign.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(PackError, match="from_capture"):
        load_pack(pack.root)


def test_from_capture_is_only_for_fields_with_an_extractor(tmp_path):
    pack = _phase1_pack(tmp_path)
    document = yaml.safe_load((pack.root / "campaign.yaml").read_text())
    document["gpu_time_multiplier"] = {"status": "from_capture"}
    (pack.root / "campaign.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(PackError, match="only supported for"):
        load_pack(pack.root)


# ── the values reach the report and the golden's provenance ─────────────────


def test_calibration_reaches_extract_compare_and_record(tmp_path):
    pack = _phase1_pack(tmp_path)
    case = _phase1_case(pack)
    case_dir = tmp_path / "runs" / case.slug
    case_dir.mkdir(parents=True)
    _write_workload_pass(case_dir)
    calibration.write(pack, case, case_dir, calibration.resolve(pack, case, case_dir))

    measured = measure_case(case_dir)
    captured = measured.provenance["capture_calibration"]
    assert captured["attn_gpu_memory_gb"]["value"] == EXPECTED_KV_GB

    key = f"p1/{case.slug}"
    body = {"available": True, "metrics": {}, "provenance": measured.provenance}
    extraction = {"cases": {key: body}}
    comparison = compare_module.compare(extraction, pack)
    assert comparison.capture_calibration[key] == captured
    text = compare_module.render_text(comparison)
    assert f"{key}  attn_gpu_memory_gb = {EXPECTED_KV_GB}" in text
    assert compare_module.as_json(comparison)["capture_calibration"][key] == captured
    assert "## Inputs read from the capture" in compare_module.render_markdown(comparison, ())

    assert compare_module._missing_capture_calibration(extraction, pack) == []
    bare = {"cases": {key: {"available": True, "metrics": {}, "provenance": {}}}}
    assert compare_module._missing_capture_calibration(bare, pack) == [
        f"{key}.attn_gpu_memory_gb", f"{key}.speculative_acceptance"
    ]

    sidecar = compare_module._write_provenance(tmp_path / "golden.json", extraction, pack, [])
    assert json.loads(sidecar.read_text())["capture_calibration"][key] == captured


def test_record_refuses_a_from_capture_case_without_calibration(tmp_path, monkeypatch):
    pack = _phase1_pack(tmp_path)
    pack = dataclasses.replace(pack, acceptance={})
    key = "p1/90_diverse_100"
    extraction = {"cases": {key: {"available": True, "metrics": {}, "provenance": {}}}}
    with pytest.raises(compare_module.RecordRefused, match="from_capture"):
        compare_module.record(extraction, pack, accept_provisional=True)
