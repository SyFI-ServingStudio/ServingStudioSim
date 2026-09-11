"""Capability-gated lifecycle callbacks for non-simulation managed jobs.

The conversation backend injects an opaque context file into agent processes.
Commands keep choosing their own artifact root, while this client proves the
workspace/conversation/turn identity to the backend before creating artifacts.
Direct developer invocations have no context and therefore remain local-only.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .managed_client import (
    LEGACY_MANAGED_RUN_CONTEXT_ENV as LEGACY_MANAGED_RUN_CONTEXT_ENV,
)
from .managed_client import (
    MANAGED_JOB_CONTEXT_ENV as MANAGED_JOB_CONTEXT_ENV,
)
from .managed_client import (
    callback_prefix,
    post,
    read_context,
)


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
    managed_jobs_api: str | None = field(default=None, kw_only=True)

    @classmethod
    def from_environment(cls, job_kind: str) -> ManagedJob | None:
        payload = read_context()
        if payload is None:
            return None
        if not job_kind:
            raise ValueError("managed job kind must not be empty")
        return cls(
            backend_url=payload["backend_url"].rstrip("/"),
            capability_token=payload["capability_token"],
            job_kind=job_kind,
            managed_jobs_api=payload.get("managed_jobs_api"),
        )

    def register(
        self,
        artifact_root: Path,
        *,
        descriptor: dict[str, Any],
        analyzer_resource_id: str | None = None,
    ) -> None:
        request_payload: dict[str, Any] = {
            "jobKind": self.job_kind,
            "artifactRoot": str(artifact_root),
            "descriptor": descriptor,
        }
        if analyzer_resource_id is not None:
            request_payload["analyzerResourceId"] = analyzer_resource_id
        response = self._request(
            self._prefix() + "/register",
            request_payload,
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
                f"{self._prefix()}/{self.job_id}/status",
                payload,
            )
            self._reported_statuses.add(status)

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return post(self.backend_url, self.capability_token, path, payload)

    def _prefix(self) -> str:
        return callback_prefix(self.managed_jobs_api, simulation=False)


def prepare_managed_job(
    job_kind: str,
    artifact_root: Path,
    *,
    descriptor: dict[str, Any],
    analyzer_resource_id: str | None = None,
) -> ManagedJob | None:
    """Register before the caller creates any official job artifacts."""
    managed_job = ManagedJob.from_environment(job_kind)
    if managed_job is not None:
        managed_job.register(
            artifact_root,
            descriptor=descriptor,
            analyzer_resource_id=analyzer_resource_id,
        )
    return managed_job


def _required_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"managed job backend response is missing {field_name!r}")
    return value
