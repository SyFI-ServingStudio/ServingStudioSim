from __future__ import annotations

import asyncio
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest

from launcher import analyzer_pipeline
from launcher import exec as launcher_exec
from launcher.analyzer_pipeline import (
    PIPELINE_STATE_RELATIVE_PATH,
    AnalyzerBinaryContract,
    AnalyzerPipelinePublisher,
    atomic_write_json,
    discover_analyzer_contract,
    snapshot_analyzer_binary,
)

PIPELINE_LIFECYCLE_FIXTURE = (
    Path(__file__).parent / "fixtures" / "analyzer_pipeline_lifecycle_v1.json"
)
LifecycleShape = tuple[str, str, str, str]


def _read_state(log_dir: Path) -> dict[str, object]:
    return json.loads((log_dir / PIPELINE_STATE_RELATIVE_PATH).read_text())


def _lifecycle_shape(state: object) -> LifecycleShape:
    assert isinstance(state, dict)
    stages = state["stages"]
    assert isinstance(stages, dict)
    compute = stages["compute"]
    render = stages["render"]
    trace = stages["trace"]
    assert isinstance(compute, dict)
    assert isinstance(render, dict)
    assert isinstance(trace, dict)
    return (
        str(state["status"]),
        str(compute["status"]),
        str(render["status"]),
        str(trace["status"]),
    )


@pytest.fixture(autouse=True)
def assert_published_pipeline_states_are_rust_valid(monkeypatch):
    """Check every durable publisher write against the Rust lifecycle table."""

    lifecycle_contract = json.loads(PIPELINE_LIFECYCLE_FIXTURE.read_text())
    valid_shapes = {
        (
            row["pipeline"],
            row["compute"],
            row["render"],
            row["trace"],
        )
        for row in lifecycle_contract["valid_states"]
    }
    published_shapes: list[LifecycleShape] = []
    original_atomic_write_json = analyzer_pipeline.atomic_write_json

    def recording_atomic_write_json(path: Path, value: object) -> None:
        original_atomic_write_json(path, value)
        if path.name == PIPELINE_STATE_RELATIVE_PATH.name:
            # Record only after the durable replace succeeds. Later transitions
            # mutate the publisher's in-memory dictionary in place.
            published_shapes.append(_lifecycle_shape(value))

    monkeypatch.setattr(
        analyzer_pipeline,
        "atomic_write_json",
        recording_atomic_write_json,
    )
    yield

    unexpected_shapes = [shape for shape in published_shapes if shape not in valid_shapes]
    assert not unexpected_shapes, (
        "publisher emitted lifecycle snapshots rejected by the Rust v1 validator: "
        f"{unexpected_shapes}"
    )


def _producer() -> dict[str, object]:
    return {
        "name": "vibesim-analyzer",
        "version": "0.1.0",
        "revision": "a" * 40,
        "binary_sha256": f"sha256:{'b' * 64}",
    }


def _contract(analyzer: Path | None = None, *, revision: str = "a" * 40) -> AnalyzerBinaryContract:
    fingerprint = (0, 0, 0, 0)
    if analyzer is not None:
        stat = analyzer.stat()
        fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    return AnalyzerBinaryContract(
        version="0.1.0",
        revision=revision,
        binary_sha256=f"sha256:{'b' * 64}",
        subject_scopes=(
            ("slo-general", "run"),
            ("throughput", "run"),
            ("alignment-e2e", "alignment"),
        ),
        binary_fingerprint=fingerprint,
    )


def test_pipeline_is_not_complete_until_trace_finishes(tmp_path: Path) -> None:
    publisher = AnalyzerPipelinePublisher.begin(tmp_path, _producer())
    initial_revision = publisher.state["artifact_revision"]

    publisher.complete_compute_and_start_render()
    publisher.complete_render_and_start_trace()

    before_finish = _read_state(tmp_path)
    assert before_finish["status"] == "pending"
    assert before_finish["artifact_revision"] == initial_revision
    assert before_finish["stages"]["trace"]["status"] == "pending"  # type: ignore[index]

    publisher.complete_trace_and_finish(artifact="traces/test.pftrace.gz")
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
        first.complete_compute_and_start_render()
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


def test_binary_contract_comes_from_machine_identity_and_binary_digest(tmp_path: Path) -> None:
    analyzer = tmp_path / "analyze"
    identity = {
        "schema_version": 1,
        "name": "vibesim-analyzer",
        "version": "9.8.7",
        "revision": "c" * 40,
        "subjects": [
            {"name": "throughput", "scope": "run"},
            {"name": "alignment-e2e", "scope": "alignment"},
        ],
    }
    analyzer.write_text(f"#!/usr/bin/env python3\nimport json\nprint(json.dumps({identity!r}))\n")
    analyzer.chmod(0o755)

    contract = discover_analyzer_contract(analyzer, tmp_path)

    assert contract.version == "9.8.7"
    assert contract.revision == "c" * 40
    assert contract.binary_sha256 == f"sha256:{sha256(analyzer.read_bytes()).hexdigest()}"
    assert contract.subject_scopes == (
        ("throughput", "run"),
        ("alignment-e2e", "alignment"),
    )


def test_content_addressed_snapshot_survives_cargo_output_replacement(tmp_path: Path) -> None:
    analyzer = tmp_path / "analyze"
    identity = {
        "schema_version": 1,
        "name": "vibesim-analyzer",
        "version": "1.2.3",
        "revision": "d" * 40,
        "subjects": [{"name": "throughput", "scope": "run"}],
    }
    analyzer.write_text(f"#!/usr/bin/env python3\nimport json\nprint(json.dumps({identity!r}))\n")
    analyzer.chmod(0o755)
    original_bytes = analyzer.read_bytes()
    source_inode = analyzer.stat().st_ino

    snapshot, contract = snapshot_analyzer_binary(analyzer, tmp_path)
    cached_snapshot, cached_contract = snapshot_analyzer_binary(analyzer, tmp_path)
    replacement = tmp_path / "replacement-analyze"
    replacement.write_bytes(b"replacement cargo output")
    os.replace(replacement, analyzer)

    assert snapshot != analyzer
    assert snapshot.stat().st_ino == source_inode
    assert cached_snapshot == snapshot
    assert cached_contract == contract
    assert snapshot.read_bytes() == original_bytes
    assert snapshot.name == f"analyze-{sha256(original_bytes).hexdigest()}"
    assert contract.matches_executable(snapshot)
    assert (
        subprocess.run(
            [str(snapshot), "identity"], capture_output=True, text=True, check=False
        ).returncode
        == 0
    )


def test_identical_cargo_rebuild_reuses_existing_snapshot_inode(tmp_path: Path) -> None:
    analyzer = tmp_path / "analyze"
    identity = {
        "schema_version": 1,
        "name": "vibesim-analyzer",
        "version": "1.2.3",
        "revision": "e" * 40,
        "subjects": [{"name": "throughput", "scope": "run"}],
    }
    analyzer.write_text(f"#!/usr/bin/env python3\nimport json\nprint(json.dumps({identity!r}))\n")
    analyzer.chmod(0o755)
    snapshot, first_contract = snapshot_analyzer_binary(analyzer, tmp_path)
    snapshot_inode = snapshot.stat().st_ino

    replacement = tmp_path / "replacement-analyze"
    replacement.write_bytes(analyzer.read_bytes())
    replacement.chmod(0o755)
    os.replace(replacement, analyzer)
    rebuilt_snapshot, rebuilt_contract = snapshot_analyzer_binary(analyzer, tmp_path)

    assert rebuilt_snapshot == snapshot
    assert rebuilt_snapshot.stat().st_ino == snapshot_inode
    assert rebuilt_contract == first_contract


def test_binary_contract_canonicalizes_only_known_run_subjects() -> None:
    contract = _contract()
    assert contract.canonical_run_subjects(["throughput", "slo-general", "throughput"]) == [
        "slo-general",
        "throughput",
    ]
    with pytest.raises(ValueError, match="unknown run analyzer subject"):
        contract.canonical_run_subjects(["through-put"])
    with pytest.raises(ValueError, match="has alignment scope"):
        contract.canonical_run_subjects(["alignment-e2e"])
    assert contract.canonical_subjects(["alignment-e2e"], scope="alignment") == ["alignment-e2e"]


def test_cargo_target_env_controls_all_launcher_artifact_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path))
    assert launcher_exec.binary_path("release") == tmp_path / "release" / "simulator"
    assert launcher_exec.analyzer_binary_path("release") == tmp_path / "release" / "analyze"
    assert launcher_exec.schema_json_path("release") == (
        tmp_path / "release" / "deployment_schema.json"
    )


def test_cargo_metadata_target_directory_honors_cargo_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    launcher_exec._cargo_target_directory.cache_clear()

    def fake_metadata(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["cargo", "metadata"],
            returncode=0,
            stdout=json.dumps({"target_directory": str(tmp_path)}),
            stderr="",
        )

    monkeypatch.setattr(launcher_exec.subprocess, "run", fake_metadata)
    try:
        assert launcher_exec.cargo_target_directory() == tmp_path
    finally:
        launcher_exec._cargo_target_directory.cache_clear()


def test_run_analysis_publishes_complete_generation(tmp_path: Path, monkeypatch) -> None:
    analyzer = tmp_path / "analyze"
    analyzer.touch()
    calls: list[list[str]] = []

    async def fake_capture(argv: list[str]) -> tuple[int, str]:
        calls.append(argv)
        return 0, "ok\n"

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda _build: analyzer)
    monkeypatch.setattr(
        launcher_exec,
        "snapshot_analyzer_binary",
        lambda *_args: (analyzer, _contract(analyzer)),
    )
    monkeypatch.setattr(launcher_exec, "_run_capture", fake_capture)

    asyncio.run(launcher_exec.run_analysis(tmp_path, subjects=["throughput"]))

    state = _read_state(tmp_path)
    assert state["status"] == "complete"
    assert state["producer"] == _producer()
    assert state["requested_subjects"] == ["throughput"]
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
    monkeypatch.setattr(
        launcher_exec,
        "snapshot_analyzer_binary",
        lambda *_args: (analyzer, _contract(analyzer)),
    )
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
    monkeypatch.setattr(
        launcher_exec,
        "snapshot_analyzer_binary",
        lambda *_args: (analyzer, _contract(analyzer)),
    )
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
    monkeypatch.setattr(
        launcher_exec,
        "snapshot_analyzer_binary",
        lambda *_args: (analyzer, _contract(analyzer)),
    )
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
    monkeypatch.setattr(
        launcher_exec,
        "snapshot_analyzer_binary",
        lambda *_args: (analyzer, _contract(analyzer)),
    )
    monkeypatch.setattr(launcher_exec, "_run_capture", fail_capture)

    asyncio.run(launcher_exec.run_analysis(tmp_path))

    state = _read_state(tmp_path)
    assert state["status"] == "failed"
    assert state["failure_code"] == "compute_failed"


def test_run_analysis_identity_failure_is_terminal_before_compute(
    tmp_path: Path, monkeypatch
) -> None:
    analyzer = tmp_path / "analyze"
    analyzer.touch()
    compute_called = False

    def fail_identity(*_args):
        raise RuntimeError("identity unavailable")

    async def fake_capture(_argv: list[str]) -> tuple[int, str]:
        nonlocal compute_called
        compute_called = True
        return 0, ""

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda _build: analyzer)
    monkeypatch.setattr(launcher_exec, "snapshot_analyzer_binary", fail_identity)
    monkeypatch.setattr(launcher_exec, "_run_capture", fake_capture)

    asyncio.run(launcher_exec.run_analysis(tmp_path))

    state = _read_state(tmp_path)
    assert not compute_called
    assert state["status"] == "failed"
    assert state["failure_code"] == "producer_identity_unavailable"
    assert state["producer"]["revision"] == "unavailable"  # type: ignore[index]


@pytest.mark.parametrize("subject", ["through-put", "alignment-e2e"])
def test_run_analysis_invalid_run_subject_is_terminal_before_compute(
    tmp_path: Path, monkeypatch, subject: str
) -> None:
    analyzer = tmp_path / "analyze"
    analyzer.touch()
    compute_called = False

    async def fake_capture(_argv: list[str]) -> tuple[int, str]:
        nonlocal compute_called
        compute_called = True
        return 0, ""

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda _build: analyzer)
    monkeypatch.setattr(
        launcher_exec,
        "snapshot_analyzer_binary",
        lambda *_args: (analyzer, _contract(analyzer)),
    )
    monkeypatch.setattr(launcher_exec, "_run_capture", fake_capture)

    asyncio.run(launcher_exec.run_analysis(tmp_path, subjects=[subject]))

    state = _read_state(tmp_path)
    assert not compute_called
    assert state["status"] == "failed"
    assert state["failure_code"] == "subject_selection_invalid"
    assert state["requested_subjects"] is None


def test_run_analysis_binary_change_after_compute_is_terminal(tmp_path: Path, monkeypatch) -> None:
    analyzer = tmp_path / "analyze"
    analyzer.touch()
    original_contract = _contract(analyzer)

    async def fake_capture(argv: list[str]) -> tuple[int, str]:
        if argv[1] == "run":
            analyzer.write_bytes(b"rebuilt analyzer")
        return 0, ""

    monkeypatch.setattr(launcher_exec, "analyzer_binary_path", lambda _build: analyzer)
    monkeypatch.setattr(
        launcher_exec,
        "snapshot_analyzer_binary",
        lambda *_args: (analyzer, original_contract),
    )
    monkeypatch.setattr(launcher_exec, "_run_capture", fake_capture)

    asyncio.run(launcher_exec.run_analysis(tmp_path))

    state = _read_state(tmp_path)
    assert state["status"] == "failed"
    assert state["failure_code"] == "producer_identity_changed"
    assert state["stages"]["render"]["status"] == "not_started"  # type: ignore[index]
