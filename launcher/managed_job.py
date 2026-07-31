"""Capability-gated lifecycle callbacks for non-simulation managed jobs.

The conversation backend injects an opaque context file into agent processes.
Commands keep choosing their own artifact root, while this client proves the
workspace/conversation/turn identity to the backend before creating artifacts.
Direct developer invocations have no context and therefore remain local-only.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANAGED_JOB_CONTEXT_ENV = "VIBESIM_MANAGED_JOB_CONTEXT"
LEGACY_MANAGED_RUN_CONTEXT_ENV = "VIBESIM_MANAGED_RUN_CONTEXT"


def _read_context() -> dict[str, Any] | None:
    configured_path = os.environ.get(MANAGED_JOB_CONTEXT_ENV, "").strip()
    if not configured_path:
        configured_path = os.environ.get(LEGACY_MANAGED_RUN_CONTEXT_ENV, "").strip()
    if not configured_path:
        return None
    path = Path(configured_path)
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"managed job context could not be read: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("managed job context must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise RuntimeError(
            f"managed job context has unsupported schema_version {payload.get('schema_version')!r}"
        )
    return payload


@dataclass(slots=True)
class ManagedJob:
    """One typed job registered against the current conversation capability."""

    backend_url: str
    capability_token: str
    job_kind: str
    job_id: str | None = None
    approved_root: str | None = None
    resource_id: str | None = None
    _reported_statuses: set[str] = field(default_factory=set)
    _status_lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_environment(cls, job_kind: str) -> ManagedJob | None:
        payload = _read_context()
        if payload is None:
            return None
        backend_url = payload.get("backend_url")
        capability_token = payload.get("capability_token")
        if not isinstance(backend_url, str) or not backend_url.startswith(("http://", "https://")):
            raise RuntimeError("managed job context has invalid backend_url")
        if not isinstance(capability_token, str) or not capability_token:
            raise RuntimeError("managed job context is missing capability_token")
        if not job_kind:
            raise ValueError("managed job kind must not be empty")
        return cls(
            backend_url=backend_url.rstrip("/"),
            capability_token=capability_token,
            job_kind=job_kind,
        )

    def register(
        self,
        artifact_root: Path,
        *,
        descriptor: dict[str, Any],
    ) -> None:
        response = self._request(
            "/api/internal/managed-jobs/register",
            {
                "jobKind": self.job_kind,
                "artifactRoot": str(artifact_root),
                "descriptor": descriptor,
            },
        )
        self.job_id = _required_string(response, "jobId")
        self.approved_root = _required_string(response, "approvedRoot")
        resource_id = response.get("resourceId")
        if resource_id is not None and not isinstance(resource_id, str):
            raise RuntimeError("managed job backend returned invalid resourceId")
        self.resource_id = resource_id
        if Path(self.approved_root).resolve() != artifact_root.resolve():
            raise RuntimeError(
                "managed backend approved a different artifact root: "
                f"{self.approved_root!r} != {str(artifact_root)!r}"
            )

    def report(self, status: str, *, summary: dict[str, Any] | None = None) -> None:
        if self.job_id is None:
            raise RuntimeError("managed job must register before reporting status")
        with self._status_lock:
            if status in self._reported_statuses and summary is None:
                return
            payload: dict[str, Any] = {"status": status}
            if summary is not None:
                payload["summary"] = summary
            self._request(
                f"/api/internal/managed-jobs/{self.job_id}/status",
                payload,
            )
            self._reported_statuses.add(status)

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        http_request = urllib.request.Request(
            f"{self.backend_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.capability_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=15) as response:
                body = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            detail = ""
            if isinstance(error, urllib.error.HTTPError):
                try:
                    detail = error.read().decode("utf-8", "replace")[:500]
                except OSError:
                    detail = ""
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"managed job backend request failed ({path}){suffix}") from error
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"managed job backend returned invalid JSON ({path})") from error
        if not isinstance(decoded, dict):
            raise RuntimeError(f"managed job backend returned a non-object response ({path})")
        return decoded


def prepare_managed_job(
    job_kind: str,
    artifact_root: Path,
    *,
    descriptor: dict[str, Any],
) -> ManagedJob | None:
    """Register before the caller creates any official job artifacts."""
    managed_job = ManagedJob.from_environment(job_kind)
    if managed_job is not None:
        managed_job.register(artifact_root, descriptor=descriptor)
    return managed_job


def _required_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"managed job backend response is missing {field_name!r}")
    return value
