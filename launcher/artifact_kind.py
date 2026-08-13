"""Explicit on-disk type markers for Analyzer first-class artifacts."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

ARTIFACT_METADATA_FILENAME = "artifact.meta.json"


class ArtifactKind(StrEnum):
    SIMULATION_RUN = "simulation_run"
    SIMULATION_SWEEP = "simulation_sweep"
    TIMING_PREDICTION = "timing_prediction"
    ALIGNMENT_BUNDLE = "alignment_bundle"
    KERNEL_PROFILE = "kernel_profile"
    KERNEL_MEASUREMENT = "kernel_measurement"


def read_artifact_kind(artifact_root: Path) -> ArtifactKind | None:
    metadata_path = artifact_root / ARTIFACT_METADATA_FILENAME
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if set(metadata) != {"schema_version", "artifact_kind"}:
        raise ValueError(f"invalid artifact marker shape: {metadata_path}")
    if metadata.get("schema_version") != 1:
        raise ValueError(f"unsupported artifact marker schema: {metadata_path}")
    try:
        return ArtifactKind(metadata.get("artifact_kind"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown artifact kind in {metadata_path}") from error


def write_artifact_kind(artifact_root: Path, artifact_kind: ArtifactKind) -> Path:
    """Atomically publish one immutable artifact kind, allowing idempotent reruns."""
    artifact_root.mkdir(parents=True, exist_ok=True)
    metadata_path = artifact_root / ARTIFACT_METADATA_FILENAME
    existing_kind = read_artifact_kind(artifact_root)
    if existing_kind is not None:
        if existing_kind != artifact_kind:
            raise ValueError(
                f"artifact root {artifact_root} is {existing_kind.value}, "
                f"not {artifact_kind.value}"
            )
        return metadata_path
    temporary_path = artifact_root / f".{ARTIFACT_METADATA_FILENAME}.tmp"
    temporary_path.write_text(
        json.dumps(
            {"schema_version": 1, "artifact_kind": artifact_kind.value},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(metadata_path)
    return metadata_path
