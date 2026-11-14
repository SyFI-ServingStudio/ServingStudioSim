"""A profile result names its artifacts so the profile directory can be copied."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from alignment import artifact_paths
from launcher.alignment import _load_profile_result, _profile_artifact
from launcher.alignment_campaign.execute import _recorded_artifacts_exist


def _profile(root: Path) -> dict:
    (root / "vllm").mkdir(parents=True)
    for name in ("parsed.json", "parsed.kernel_rows.parquet", "vllm/m_metrics.jsonl"):
        (root / name).write_text("{}")
    return {
        "profile_kind": "nsys",
        "log_dir": str(root),
        "parsed_nsys": str(root / "parsed.json"),
        "parsed_kernel_rows": str(root / "parsed.kernel_rows.parquet"),
        "metrics_jsonl": str(root / "vllm" / "m_metrics.jsonl"),
        "gpu": "H100",
        "drive_summary": {"source_trace": "/data/trace.csv"},
    }


def test_record_stores_paths_inside_the_profile_relative(tmp_path):
    root = tmp_path / "run" / "profile_nsys"
    result = _profile(root)
    result["replay_result"] = "/elsewhere/replay.jsonl"
    recorded = artifact_paths.record(result, root)
    assert recorded["parsed_nsys"] == "parsed.json"
    assert recorded["metrics_jsonl"] == "vllm/m_metrics.jsonl"
    # Outside the directory, and fields that are not artifacts, stay as they are.
    assert recorded["replay_result"] == "/elsewhere/replay.jsonl"
    assert recorded["log_dir"] == str(root)
    assert recorded["drive_summary"] == result["drive_summary"]


def test_record_sees_through_a_symlinked_root(tmp_path):
    real = tmp_path / "workdir" / "logs" / "p"
    _profile(real)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "logs").symlink_to(tmp_path / "workdir" / "logs")
    linked = tmp_path / "src" / "logs" / "p"
    # As the runner records it on a remote: replay resolved, log_dir not.
    value = str((linked / "parsed.json").resolve())
    assert artifact_paths.relative_to(value, linked) == "parsed.json"


def test_a_copied_profile_reads_where_it_lands(tmp_path):
    remote = tmp_path / "remote" / "logs" / "case" / "profile_nsys"
    result = _profile(remote)
    (remote / "profile_result.json").write_text(
        json.dumps(artifact_paths.record(result, remote))
    )
    here = tmp_path / "here" / "logs" / "case" / "profile_nsys"
    shutil.copytree(remote, here)
    shutil.rmtree(tmp_path / "remote")

    loaded = _load_profile_result(here)
    assert _profile_artifact(loaded, "parsed_nsys") == (here / "parsed.json").resolve()
    assert Path(loaded["metrics_jsonl"]) == here / "vllm" / "m_metrics.jsonl"
    assert _recorded_artifacts_exist(here, "nsys")


def test_an_old_absolute_result_is_rebased_onto_its_new_directory(tmp_path):
    remote = tmp_path / "remote" / "profile_nsys"
    result = _profile(remote)
    (remote / "profile_result.json").write_text(json.dumps(result))
    here = tmp_path / "here" / "profile_nsys"
    shutil.copytree(remote, here)
    shutil.rmtree(tmp_path / "remote")

    loaded = _load_profile_result(here)
    assert Path(loaded["parsed_nsys"]) == here / "parsed.json"
    assert _recorded_artifacts_exist(here, "nsys")


def test_a_copy_beside_its_original_reads_its_own_files(tmp_path):
    original = tmp_path / "a" / "profile_nsys"
    result = _profile(original)
    (original / "profile_result.json").write_text(json.dumps(result))
    copy = tmp_path / "b" / "profile_nsys"
    shutil.copytree(original, copy)
    assert Path(_load_profile_result(copy)["parsed_nsys"]) == copy / "parsed.json"


def test_the_case_trace_of_a_copied_back_capture_reads_beside_it(tmp_path):
    # A remote capture records its case trace (outside the phase dir) by the
    # remote's path; the copied-back case dir has the same trace beside it.
    remote = "/remote/workdir/logs/run/case_a/profile_nsys"
    here = tmp_path / "logs" / "run" / "case_a" / "profile_nsys"
    here.mkdir(parents=True)
    (here.parent / "trace.csv").write_text("t,in,out\n")
    trace = "/remote/workdir/logs/run/case_a/trace.csv"
    assert artifact_paths.resolve_beside(trace, here, remote) == here.parent / "trace.csv"
    # A path that exists here, or that has no counterpart here, is left alone.
    assert artifact_paths.resolve_beside(str(here.parent / "trace.csv"), here, remote) == (
        here.parent / "trace.csv"
    )
    gone = "/remote/workdir/logs/run/case_b/trace.csv"
    assert artifact_paths.resolve_beside(gone, here, remote) == Path(gone)
