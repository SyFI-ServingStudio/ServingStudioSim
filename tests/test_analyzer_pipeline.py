from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from launcher import analyzer_pipeline
from launcher import exec as launcher_exec
from launcher.analyzer_pipeline import (
    PIPELINE_STATE_RELATIVE_PATH,
    AnalyzerPipelinePublisher,
    atomic_write_json,
)


def _read_state(log_dir: Path) -> dict[str, object]:
    return json.loads((log_dir / PIPELINE_STATE_RELATIVE_PATH).read_text())


def _producer() -> dict[str, object]:
    return {
        "name": "vibesim-analyzer",
        "version": "0.1.0",
        "revision": "a" * 40,
        "binary_sha256": f"sha256:{'b' * 64}",
    }


def test_pipeline_is_not_complete_until_trace_finishes(tmp_path: Path) -> None:
    publisher = AnalyzerPipelinePublisher.begin(tmp_path, _producer())
    initial_revision = publisher.state["artifact_revision"]

    publisher.complete_stage("compute")
    publisher.start_stage("render")
    publisher.complete_stage("render")
    publisher.start_stage("trace")
    publisher.complete_stage("trace")

    before_finish = _read_state(tmp_path)
    assert before_finish["status"] == "pending"
    assert before_finish["artifact_revision"] == initial_revision

    publisher.finish()
    complete = _read_state(tmp_path)
    assert complete["status"] == "complete"
    assert complete["artifact_revision"] == initial_revision
    assert complete["producer"] == _producer()
    publisher.close()


def test_new_generation_waits_for_lease_and_old_publisher_cannot_overwrite(
    tmp_path: Path,
) -> None:
    first = AnalyzerPipelinePublisher.begin(tmp_path, _producer())
    first_generation = first.generation_id

    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(AnalyzerPipelinePublisher.begin, tmp_path, _producer())
        assert not waiting.done()
        first.close()
        second = waiting.result(timeout=2)

    assert second.generation_id != first_generation
    second_state = _read_state(tmp_path)
    assert second_state["generation_id"] == second.generation_id
    with pytest.raises(RuntimeError, match="no longer owns"):
        first.complete_stage("compute")
    assert _read_state(tmp_path)["generation_id"] == second.generation_id
    second.close()


def test_atomic_json_replaces_from_same_directory(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "reports" / "state.json"
    atomic_write_json(destination, {"generation": "old"})

    original_replace = os.replace
    replace_paths: list[tuple[Path, Path]] = []

    def observing_replace(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        target_path = Path(target)
        replace_paths.append((source_path, target_path))
        assert source_path.parent == target_path.parent
        original_replace(source_path, target_path)

    monkeypatch.setattr(os, "replace", observing_replace)
    atomic_write_json(destination, {"generation": "new"})

    assert json.loads(destination.read_text()) == {"generation": "new"}
    assert replace_paths == [(replace_paths[0][0], destination)]
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_atomic_json_failure_preserves_previous_file(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "reports" / "state.json"
    atomic_write_json(destination, {"generation": "old"})

    def fail_dump(*_args, **_kwargs) -> None:
        raise OSError("injected serialization write failure")

    monkeypatch.setattr(analyzer_pipeline.json, "dump", fail_dump)
    with pytest.raises(OSError, match="injected"):
        atomic_write_json(destination, {"generation": "new"})

    assert json.loads(destination.read_text()) == {"generation": "old"}
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_run_analysis_publishes_complete_generation(tmp_path: Path, monkeypatch) -> None:
    analyzer = tmp_path / "analyze"
    analyzer.touch()
    calls: list[list[str]] = []

    async def fake_capture(argv: list[str]) -> tuple[int, str]:
        calls.append(argv)
        return 0, "ok\n"

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda _build: analyzer)
    monkeypatch.setattr(launcher_exec, "discover_producer_identity", lambda *_args: _producer())
    monkeypatch.setattr(launcher_exec, "_run_capture", fake_capture)

    asyncio.run(launcher_exec.run_analysis(tmp_path, subjects=["throughput"]))

    state = _read_state(tmp_path)
    assert state["status"] == "complete"
    assert [call[1] if call[0] == str(analyzer) else call[2] for call in calls] == [
        "run",
        "render",
        "trace",
    ]
    assert state["stages"]["trace"]["status"] == "complete"  # type: ignore[index]


def test_run_analysis_keeps_render_failure_until_trace_stops(tmp_path: Path, monkeypatch) -> None:
    analyzer = tmp_path / "analyze"
    analyzer.touch()
    called_stages: list[str] = []

    async def fake_capture(argv: list[str]) -> tuple[int, str]:
        stage = argv[1] if argv[0] == str(analyzer) else argv[2]
        called_stages.append(stage)
        return (1 if stage == "render" else 0), ""

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda _build: analyzer)
    monkeypatch.setattr(launcher_exec, "discover_producer_identity", lambda *_args: _producer())
    monkeypatch.setattr(launcher_exec, "_run_capture", fake_capture)

    asyncio.run(launcher_exec.run_analysis(tmp_path))

    state = _read_state(tmp_path)
    assert called_stages == ["run", "render", "trace"]
    assert state["status"] == "complete"
    assert state.get("failure_code") is None
    assert state["stages"]["render"]["code"] == "render_failed"  # type: ignore[index]
    assert state["stages"]["trace"]["status"] == "complete"  # type: ignore[index]


def test_run_analysis_distinguishes_trace_failure(tmp_path: Path, monkeypatch) -> None:
    analyzer = tmp_path / "analyze"
    analyzer.touch()

    async def fake_capture(argv: list[str]) -> tuple[int, str]:
        stage = argv[1] if argv[0] == str(analyzer) else argv[2]
        return (1 if stage == "trace" else 0), ""

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda _build: analyzer)
    monkeypatch.setattr(launcher_exec, "discover_producer_identity", lambda *_args: _producer())
    monkeypatch.setattr(launcher_exec, "_run_capture", fake_capture)

    asyncio.run(launcher_exec.run_analysis(tmp_path))

    state = _read_state(tmp_path)
    assert state["status"] == "complete"
    assert state.get("failure_code") is None
    assert state["stages"]["trace"]["code"] == "trace_failed"  # type: ignore[index]


def test_run_analysis_compute_failure_never_starts_later_stages(
    tmp_path: Path, monkeypatch
) -> None:
    analyzer = tmp_path / "analyze"
    analyzer.touch()
    calls: list[list[str]] = []

    async def fake_capture(argv: list[str]) -> tuple[int, str]:
        calls.append(argv)
        return 1, ""

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda _build: analyzer)
    monkeypatch.setattr(launcher_exec, "discover_producer_identity", lambda *_args: _producer())
    monkeypatch.setattr(launcher_exec, "_run_capture", fake_capture)

    asyncio.run(launcher_exec.run_analysis(tmp_path))

    state = _read_state(tmp_path)
    assert len(calls) == 1
    assert state["status"] == "failed"
    assert state["failure_code"] == "compute_failed"
    assert state["stages"]["render"]["status"] == "not_started"  # type: ignore[index]
    assert state["stages"]["trace"]["status"] == "not_started"  # type: ignore[index]


def test_run_analysis_missing_binary_is_a_terminal_compute_failure(
    tmp_path: Path, monkeypatch
) -> None:
    analyzer = tmp_path / "missing-analyze"
    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda _build: analyzer)
    monkeypatch.setattr(launcher_exec, "discover_producer_identity", lambda *_args: _producer())

    asyncio.run(launcher_exec.run_analysis(tmp_path))

    state = _read_state(tmp_path)
    assert state["status"] == "failed"
    assert state["failure_code"] == "analyzer_unavailable"
    assert state["stages"]["compute"]["status"] == "failed"  # type: ignore[index]


def test_run_analysis_subprocess_exception_becomes_terminal_state(
    tmp_path: Path, monkeypatch
) -> None:
    analyzer = tmp_path / "analyze"
    analyzer.touch()

    async def fail_capture(_argv: list[str]) -> tuple[int, str]:
        raise OSError("cannot spawn")

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda _build: analyzer)
    monkeypatch.setattr(launcher_exec, "discover_producer_identity", lambda *_args: _producer())
    monkeypatch.setattr(launcher_exec, "_run_capture", fail_capture)

    asyncio.run(launcher_exec.run_analysis(tmp_path))

    state = _read_state(tmp_path)
    assert state["status"] == "failed"
    assert state["failure_code"] == "compute_failed"
