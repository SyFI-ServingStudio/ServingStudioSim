from __future__ import annotations

import json
from pathlib import Path

from launcher.artifact_kind import ARTIFACT_METADATA_FILENAME
from launcher.migrate_artifact_kinds import main


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_migration_preflights_then_marks_registry_and_descriptor_roots(tmp_path):
    registry_path = tmp_path / "agent-workspaces/registry.json"
    _write_json(
        registry_path,
        {
            "schema_version": 1,
            "workspaces": [
                {
                    "workspace_id": "w_registry",
                    "logs_root": "../registry-logs",
                }
            ],
        },
    )
    _write_json(
        registry_path.parent / "archived/workspace.json",
        {
            "schema_version": 1,
            "workspace_id": "w_archived",
            "logs_path": "repo/logs",
        },
    )
    run_root = tmp_path / "registry-logs/run"
    _write_json(run_root / "raw/params.json", {"deployment": "unified"})
    prediction_root = registry_path.parent / "archived/repo/logs/prediction"
    _write_json(
        prediction_root / "prediction.meta.json",
        {"schema_version": 1, "prediction_id": "p_test"},
    )

    assert main(["--registry", str(registry_path), "--check"]) == 1
    assert not (run_root / ARTIFACT_METADATA_FILENAME).exists()
    assert main(["--registry", str(registry_path), "--apply"]) == 0
    assert main(["--registry", str(registry_path), "--check"]) == 0
    assert json.loads((run_root / ARTIFACT_METADATA_FILENAME).read_text()) == {
        "schema_version": 1,
        "artifact_kind": "simulation_run",
    }
    assert json.loads((prediction_root / ARTIFACT_METADATA_FILENAME).read_text()) == {
        "schema_version": 1,
        "artifact_kind": "timing_prediction",
    }


def test_migration_writes_nothing_when_any_candidate_is_ambiguous(tmp_path):
    registry_path = tmp_path / "agent-workspaces/registry.json"
    _write_json(
        registry_path,
        {
            "schema_version": 1,
            "workspaces": [{"workspace_id": "w_test", "logs_root": "../logs"}],
        },
    )
    valid_root = tmp_path / "logs/valid"
    _write_json(valid_root / "raw/params.json", {"deployment": "unified"})
    ambiguous_root = tmp_path / "logs/ambiguous"
    _write_json(ambiguous_root / "raw/params.json", {"deployment": "unified"})
    _write_json(
        ambiguous_root / "prediction.meta.json",
        {"schema_version": 1, "prediction_id": "p_ambiguous"},
    )

    assert main(["--registry", str(registry_path), "--apply"]) == 2
    assert not (valid_root / ARTIFACT_METADATA_FILENAME).exists()
    assert not (ambiguous_root / ARTIFACT_METADATA_FILENAME).exists()
