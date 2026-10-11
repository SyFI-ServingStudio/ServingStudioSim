"""`VIBESIM_BIN_DIR`, the source stamp, and how a shipped tree records provenance."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from launcher import binaries, metadata
from launcher.binaries import BIN_DIR_ENV, PrebuiltBinaryError
from launcher.source_stamp import STAMP_NAME, read_source_stamp


def _stamp(root: Path, **fields) -> dict:
    stamp = {
        "schema_version": 1,
        "version": "v1",
        "build_hash": "b1",
        "head_oneline": "deadbee Some commit",
        "status": " M launcher/cli.py",
        "diff": "diff --git a/x b/x",
        "profiler_git_hash": "deadbeef",
        **fields,
    }
    (root / STAMP_NAME).write_text(json.dumps(stamp))
    return stamp


def _bin_dir(tmp_path: Path, build_hash: str | None = "b1") -> Path:
    directory = tmp_path / "bin"
    directory.mkdir()
    simulator = directory / "simulator"
    simulator.write_text("#!/bin/sh\n")
    simulator.chmod(0o755)
    (directory / "deployment_schema.json").write_text("{}")
    if build_hash is not None:
        (directory / ".build-hash").write_text(build_hash + "\n")
    return directory


def test_without_the_variable_binaries_come_from_cargo_target(monkeypatch):
    monkeypatch.delenv(BIN_DIR_ENV, raising=False)
    assert binaries.build_output_dir("dev") == binaries.REPO_ROOT / "target" / "dev"


def test_the_variable_replaces_the_release_output_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(BIN_DIR_ENV, str(tmp_path))
    assert binaries.build_output_dir("release") == tmp_path
    from launcher.exec import analyzer_binary_path, binary_path, schema_json_path

    assert binary_path("release") == tmp_path / "simulator"
    assert analyzer_binary_path("release") == tmp_path / "analyze"
    assert schema_json_path("release") == tmp_path / "deployment_schema.json"
    with pytest.raises(PrebuiltBinaryError, match="--build-type dev"):
        binaries.build_output_dir("dev")


def test_a_prebuilt_set_must_be_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(binaries, "REPO_ROOT", tmp_path)
    directory = _bin_dir(tmp_path)
    binaries.check_prebuilt(directory)  # no stamp here: completeness only
    (directory / "deployment_schema.json").unlink()
    with pytest.raises(PrebuiltBinaryError, match="deployment_schema.json"):
        binaries.check_prebuilt(directory)
    (directory / "simulator").chmod(0o644)
    with pytest.raises(PrebuiltBinaryError, match="no executable simulator"):
        binaries.check_prebuilt(directory)


@pytest.mark.parametrize("built_from", ["b0", None])
def test_a_shipped_tree_refuses_binaries_from_other_sources(tmp_path, monkeypatch, built_from):
    monkeypatch.setattr(binaries, "REPO_ROOT", tmp_path)
    _stamp(tmp_path, build_hash="b1")
    binaries.check_prebuilt(_bin_dir(tmp_path, "b1"))
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(PrebuiltBinaryError, match="build inputs are b1"):
        binaries.check_prebuilt(_bin_dir(other, built_from))


def test_a_stamp_with_an_unknown_schema_is_an_error(tmp_path):
    assert read_source_stamp(tmp_path) is None
    (tmp_path / STAMP_NAME).write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError, match="unsupported source stamp"):
        read_source_stamp(tmp_path)


def test_a_shipped_tree_writes_its_git_snapshot_from_the_stamp(tmp_path, monkeypatch):
    root = tmp_path / "src"
    root.mkdir()
    _stamp(root)
    monkeypatch.setattr(metadata, "REPO_ROOT", root)

    metadata.save_git_snapshot(tmp_path / "run", datetime(2026, 10, 11, 12, 0))

    snapshot = tmp_path / "run" / "git_snapshot"
    commit = (snapshot / "commit.txt").read_text()
    assert "deadbee Some commit" in commit and "shipped source version v1" in commit
    assert (snapshot / "status.txt").read_text() == " M launcher/cli.py\n"
    assert (snapshot / "diff.patch").read_text() == "diff --git a/x b/x\n"
