"""Optional managed-run registration for Launcher invocations.

The normal development CLI has no managed context and only writes a stable
``experiment.meta.json`` sidecar.  UI/headless Agent runtimes inject one
short-lived capability path through ``VIBESIM_MANAGED_RUN_CONTEXT``.  In that mode
registration is mandatory and happens before any official run artifact is
created.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANAGED_CONTEXT_ENV = "VIBESIM_MANAGED_RUN_CONTEXT"
EXPERIMENT_METADATA_FILENAME = "experiment.meta.json"
MANAGED_AGGREGATE_SUBJECTS = ("slo-general", "throughput", "utilization")


def managed_analysis_subjects(
    managed_run: ManagedRun | None,
    requested_subjects: list[str] | None,
) -> list[str] | None:
    """Complete a narrowed managed run with the Analyzer's baseline panels.

    ``None`` and ``[]`` already mean all applicable subjects, while direct CLI
    runs must preserve the user's exact selection.
    """
    if managed_run is None or not requested_subjects:
        return requested_subjects
    return list(dict.fromkeys([*requested_subjects, *MANAGED_AGGREGATE_SUBJECTS]))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"managed run context could not be read: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("managed run context must contain a JSON object")
    return payload


@dataclass(slots=True)
class ManagedRun:
    backend_url: str
    capability_token: str
    job_id: str | None = None
    experiment_id: str | None = None
    approved_root: str | None = None
    _reported_statuses: set[str] = field(default_factory=set)
    _status_lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_environment(cls) -> ManagedRun | None:
        configured_path = os.environ.get(MANAGED_CONTEXT_ENV, "").strip()
        if not configured_path:
            return None
        payload = _read_json(Path(configured_path))
        if payload.get("schema_version") != 1:
            raise RuntimeError(
                "managed run context has unsupported schema_version "
                f"{payload.get('schema_version')!r}"
            )
        backend_url = payload.get("backend_url")
        capability_token = payload.get("capability_token")
        if not isinstance(backend_url, str) or not backend_url.startswith(
            ("http://", "https://")
        ):
            raise RuntimeError("managed run context has invalid backend_url")
        if not isinstance(capability_token, str) or not capability_token:
            raise RuntimeError("managed run context is missing capability_token")
        return cls(
            backend_url=backend_url.rstrip("/"),
            capability_token=capability_token,
        )

    def register(
        self,
        experiment_root: Path,
        *,
        run_count: int,
        axes: list[str],
    ) -> None:
        response = self._request(
            "/api/internal/managed-runs/register",
            {
                "experimentRoot": str(experiment_root),
                "runCount": run_count,
                "axes": axes,
            },
        )
        self.job_id = _required_string(response, "jobId")
        self.experiment_id = _required_string(response, "experimentId")
        self.approved_root = _required_string(response, "approvedRoot")
        if Path(self.approved_root).resolve() != experiment_root.resolve():
            raise RuntimeError(
                "managed backend approved a different experiment root: "
                f"{self.approved_root!r} != {str(experiment_root)!r}"
            )

    def report(self, status: str) -> None:
        if self.job_id is None:
            raise RuntimeError("managed run must register before reporting status")
        with self._status_lock:
            if status in self._reported_statuses:
                return
            self._request(
                f"/api/internal/managed-runs/{self.job_id}/status",
                {"status": status},
            )
            self._reported_statuses.add(status)

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.backend_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.capability_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                body = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            detail = ""
            if isinstance(error, urllib.error.HTTPError):
                try:
                    detail = error.read().decode("utf-8", "replace")[:500]
                except OSError:
                    detail = ""
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"managed run backend request failed ({path}){suffix}"
            ) from error
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"managed run backend returned invalid JSON ({path})"
            ) from error
        if not isinstance(decoded, dict):
            raise RuntimeError(
                f"managed run backend returned a non-object response ({path})"
            )
        return decoded


def prepare_experiment(
    experiment_root: Path,
    *,
    run_count: int,
    axes: list[str],
) -> ManagedRun | None:
    """Register a managed run or write direct-development provenance."""
    managed_run = ManagedRun.from_environment()
    if managed_run is not None:
        managed_run.register(
            experiment_root,
            run_count=run_count,
            axes=axes,
        )
        return managed_run
    _write_development_metadata(experiment_root)
    return None


def _write_development_metadata(experiment_root: Path) -> None:
    experiment_root.mkdir(parents=True, exist_ok=True)
    metadata_path = experiment_root / EXPERIMENT_METADATA_FILENAME
    if metadata_path.is_file():
        payload = _read_json(metadata_path)
        if payload.get("schema_version") != 1 or not isinstance(
            payload.get("experiment_id"), str
        ):
            raise RuntimeError(f"incompatible experiment metadata: {metadata_path}")
        return
    payload = {
        "schema_version": 1,
        "experiment_id": f"e_{uuid.uuid4().hex}",
        "origin": {"kind": "development"},
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
    temporary.replace(metadata_path)


def _required_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"managed run backend response is missing {field_name!r}"
        )
    return value
