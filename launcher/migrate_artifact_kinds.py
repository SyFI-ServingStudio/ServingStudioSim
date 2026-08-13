"""One-shot migration from legacy artifact shapes to explicit type markers.

Runtime discovery intentionally contains no legacy inference. This command owns
the only shape-based classification path and performs a complete preflight
before writing any marker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_kind import ARTIFACT_METADATA_FILENAME, ArtifactKind, write_artifact_kind

IGNORED_DIRECTORY_NAMES = {"old-logs", "__pycache__", "node_modules", "target"}


@dataclass(frozen=True)
class WorkspaceRoot:
    workspace_id: str
    path: Path


@dataclass(frozen=True)
class Candidate:
    workspace_id: str
    path: Path
    artifact_kind: ArtifactKind


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _load_workspace_roots(registry_path: Path) -> list[WorkspaceRoot]:
    roots: dict[Path, WorkspaceRoot] = {}
    registry = _read_json(registry_path)
    if not isinstance(registry, dict) or not isinstance(registry.get("workspaces"), list):
        raise ValueError(f"invalid workspace registry: {registry_path}")
    for workspace in registry["workspaces"]:
        if not isinstance(workspace, dict):
            continue
        workspace_id = workspace.get("workspace_id")
        logs_root = workspace.get("logs_root")
        if not isinstance(workspace_id, str) or not isinstance(logs_root, str):
            continue
        path = (registry_path.parent / logs_root).resolve()
        roots.setdefault(path, WorkspaceRoot(workspace_id, path))

    for descriptor_path in sorted(registry_path.parent.glob("*/workspace.json")):
        descriptor = _read_json(descriptor_path)
        if not isinstance(descriptor, dict):
            continue
        workspace_id = descriptor.get("workspace_id")
        logs_path = descriptor.get("logs_path")
        if not isinstance(workspace_id, str) or not isinstance(logs_path, str):
            continue
        path = (descriptor_path.parent / logs_path).resolve()
        roots.setdefault(path, WorkspaceRoot(workspace_id, path))
    return sorted(roots.values(), key=lambda root: (root.workspace_id, str(root.path)))


def _is_prediction(directory: Path) -> bool:
    metadata = _read_json(directory / "prediction.meta.json")
    return isinstance(metadata, dict) and metadata.get("schema_version") == 1 and isinstance(
        metadata.get("prediction_id"), str
    )


def _is_simulation_run(directory: Path) -> bool:
    params = _read_json(directory / "raw/params.json")
    return isinstance(params, dict) and params.get("deployment") in {"unified", "pd", "afd"}


def _is_sweep(directory: Path) -> bool:
    manifest = _read_json(directory / "sweep_manifest.json")
    return (
        isinstance(manifest, dict)
        and manifest.get("schema_version") == 1
        and isinstance(manifest.get("axes"), list)
        and isinstance(manifest.get("runs"), list)
    )


def _is_alignment(directory: Path) -> bool:
    return any(
        isinstance(_read_json(directory / analysis_dir / "alignment_manifest.json"), dict)
        for analysis_dir in ("analysis_kernel", "analysis_e2e")
    )


def _is_kernel_profile(directory: Path) -> bool:
    metadata = _read_json(directory / "kernel-profile.meta.json")
    if isinstance(metadata, dict) and metadata.get("schema_version") == 1:
        return True
    return (directory / "job.meta.json").is_file() and (directory / "curve.json").is_file()


def _is_kernel_measurement(directory: Path) -> bool:
    metadata = _read_json(directory / "kernel-measurement.meta.json")
    if isinstance(metadata, dict) and metadata.get("schema_version") == 1:
        return True
    summary = _read_json(directory / "summary.json")
    return (
        isinstance(summary, dict)
        and summary.get("schema_version") == 1
        and isinstance(summary.get("runtime_ms"), dict)
        and isinstance(summary["runtime_ms"].get("median"), (int, float))
    )


CLASSIFIERS = (
    (ArtifactKind.TIMING_PREDICTION, _is_prediction),
    (ArtifactKind.SIMULATION_RUN, _is_simulation_run),
    (ArtifactKind.SIMULATION_SWEEP, _is_sweep),
    (ArtifactKind.ALIGNMENT_BUNDLE, _is_alignment),
    (ArtifactKind.KERNEL_PROFILE, _is_kernel_profile),
    (ArtifactKind.KERNEL_MEASUREMENT, _is_kernel_measurement),
)


def _walk_directories(root: Path):
    if not root.is_dir():
        return
    for directory_name, child_names, _file_names in os.walk(root, followlinks=False):
        child_names[:] = sorted(
            child_name
            for child_name in child_names
            if not child_name.startswith(".") and child_name not in IGNORED_DIRECTORY_NAMES
        )
        yield Path(directory_name)


def _existing_marker(directory: Path) -> tuple[ArtifactKind | None, str | None]:
    marker_path = directory / ARTIFACT_METADATA_FILENAME
    if not marker_path.is_file():
        return None, None
    metadata = _read_json(marker_path)
    if not isinstance(metadata, dict) or set(metadata) != {"schema_version", "artifact_kind"}:
        return None, f"invalid marker shape: {marker_path}"
    if metadata.get("schema_version") != 1:
        return None, f"unsupported marker schema: {marker_path}"
    try:
        return ArtifactKind(metadata["artifact_kind"]), None
    except (KeyError, ValueError):
        return None, f"unknown artifact kind: {marker_path}"


def _preflight(roots: list[WorkspaceRoot]) -> tuple[list[Candidate], list[str]]:
    candidates: list[Candidate] = []
    errors: list[str] = []
    for root in roots:
        for directory in _walk_directories(root.path):
            inferred = [kind for kind, classifier in CLASSIFIERS if classifier(directory)]
            marker_kind, marker_error = _existing_marker(directory)
            if marker_error is not None:
                errors.append(marker_error)
                continue
            if len(inferred) > 1:
                errors.append(
                    f"ambiguous artifact {directory}: {[kind.value for kind in inferred]}"
                )
                continue
            if marker_kind is not None and inferred and marker_kind != inferred[0]:
                errors.append(
                    f"marker mismatch {directory}: {marker_kind.value} != {inferred[0].value}"
                )
                continue
            artifact_kind = marker_kind or (inferred[0] if inferred else None)
            if artifact_kind is not None:
                candidates.append(Candidate(root.workspace_id, directory, artifact_kind))
    return candidates, errors


def _print_report(candidates: list[Candidate], errors: list[str]) -> None:
    counts = Counter(
        (candidate.workspace_id, candidate.artifact_kind.value) for candidate in candidates
    )
    missing = sum(
        not (candidate.path / ARTIFACT_METADATA_FILENAME).is_file() for candidate in candidates
    )
    for (workspace_id, artifact_kind), count in sorted(counts.items()):
        print(f"{workspace_id}\t{artifact_kind}\t{count}")
    print(f"total={len(candidates)} missing_markers={missing} errors={len(errors)}")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m launcher migrate-artifact-kinds",
        description="Bulk-mark every legacy Analyzer artifact under all workspace roots.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "agent-workspaces/registry.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Report; fail if any marker is missing.")
    mode.add_argument(
        "--apply", action="store_true", help="Preflight all roots, then write markers."
    )
    args = parser.parse_args(argv)

    roots = _load_workspace_roots(args.registry.resolve())
    candidates, errors = _preflight(roots)
    _print_report(candidates, errors)
    if errors:
        return 2
    missing = [
        candidate
        for candidate in candidates
        if not (candidate.path / ARTIFACT_METADATA_FILENAME).is_file()
    ]
    if args.check:
        return 1 if missing else 0
    for candidate in missing:
        write_artifact_kind(candidate.path, candidate.artifact_kind)
    print(f"wrote_markers={len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
