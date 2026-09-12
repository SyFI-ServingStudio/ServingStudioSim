"""Shared context protocol selection and one-shot managed callback transport."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MANAGED_JOB_CONTEXT_ENV = "VIBESIM_MANAGED_JOB_CONTEXT"
LEGACY_MANAGED_RUN_CONTEXT_ENV = "VIBESIM_MANAGED_RUN_CONTEXT"
UNIFIED_JOBS_API = "agent-v1"


def read_context() -> dict[str, Any] | None:
    configured_path = os.environ.get(MANAGED_JOB_CONTEXT_ENV, "").strip()
    if not configured_path:
        configured_path = os.environ.get(LEGACY_MANAGED_RUN_CONTEXT_ENV, "").strip()
    if not configured_path:
        return None
    path = Path(configured_path)
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"managed job context could not be read: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("managed job context must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise RuntimeError("managed job context has unsupported schema_version")
    backend_url = payload.get("backend_url")
    if not isinstance(backend_url, str) or not backend_url.startswith(("http://", "https://")):
        raise RuntimeError("managed job context has invalid backend_url")
    token = payload.get("capability_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("managed job context is missing capability_token")
    if "managed_jobs_api" in payload and payload["managed_jobs_api"] != UNIFIED_JOBS_API:
        raise RuntimeError("managed job context has unsupported managed_jobs_api")
    return payload


def callback_prefix(managed_jobs_api: str | None, *, simulation: bool) -> str:
    if managed_jobs_api == UNIFIED_JOBS_API:
        return "/api/agent/v1/internal/jobs"
    if managed_jobs_api is not None:
        raise RuntimeError("unsupported managed_jobs_api")
    return "/api/internal/managed-runs" if simulation else "/api/internal/managed-jobs"


def post(backend_url: str, capability_token: str, path: str, payload: dict[str, Any]) -> dict:
    request = urllib.request.Request(
        f"{backend_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {capability_token}",
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
                pass
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"managed job backend request failed ({path}){suffix}") from error
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeError) as error:
        raise RuntimeError(f"managed job backend returned invalid JSON ({path})") from error
    if not isinstance(decoded, dict):
        raise RuntimeError(f"managed job backend returned a non-object response ({path})")
    return decoded
