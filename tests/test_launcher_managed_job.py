from __future__ import annotations

import json
from pathlib import Path
from urllib import request

from launcher.managed_job import (
    LEGACY_MANAGED_RUN_CONTEXT_ENV,
    MANAGED_JOB_CONTEXT_ENV,
    ManagedJob,
    prepare_managed_job,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_direct_job_has_no_callback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(MANAGED_JOB_CONTEXT_ENV, raising=False)
    monkeypatch.delenv(LEGACY_MANAGED_RUN_CONTEXT_ENV, raising=False)

    assert (
        prepare_managed_job(
            "timing_predict",
            tmp_path / "predict",
            descriptor={"selector": "iter"},
        )
        is None
    )


def test_typed_job_registration_and_status(monkeypatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "logs" / "profile"
    context_path = tmp_path / "managed-job.json"
    context_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_url": "http://backend.test",
                "capability_token": "secret",
            }
        )
    )
    monkeypatch.setenv(MANAGED_JOB_CONTEXT_ENV, str(context_path))
    captured: list[request.Request] = []

    def fake_urlopen(http_request: request.Request, timeout: int):
        captured.append(http_request)
        assert timeout == 15
        if http_request.full_url.endswith("/register"):
            assert not artifact_root.exists()
            return FakeResponse(
                {
                    "jobId": "j_profile",
                    "resourceId": "kp_profile",
                    "approvedRoot": str(artifact_root),
                }
            )
        return FakeResponse({"ok": True})

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    managed_job = prepare_managed_job(
        "kernel_profile",
        artifact_root,
        descriptor={"table": "single_gemm", "pointCount": 2},
    )
    assert isinstance(managed_job, ManagedJob)
    managed_job.report("running")
    managed_job.report("running")
    managed_job.report("ready", summary={"axes": ["m"]})

    registration = json.loads(captured[0].data or b"{}")
    assert registration == {
        "jobKind": "kernel_profile",
        "artifactRoot": str(artifact_root),
        "descriptor": {"table": "single_gemm", "pointCount": 2},
    }
    status_payloads = [json.loads(item.data or b"{}") for item in captured[1:]]
    assert status_payloads == [
        {"status": "running"},
        {"status": "ready", "summary": {"axes": ["m"]}},
    ]


def test_legacy_managed_run_context_remains_accepted(monkeypatch, tmp_path: Path) -> None:
    context_path = tmp_path / "managed-run.json"
    context_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_url": "http://backend.test",
                "capability_token": "legacy",
            }
        )
    )
    monkeypatch.delenv(MANAGED_JOB_CONTEXT_ENV, raising=False)
    monkeypatch.setenv(LEGACY_MANAGED_RUN_CONTEXT_ENV, str(context_path))

    managed_job = ManagedJob.from_environment("timing_predict")

    assert managed_job is not None
    assert managed_job.capability_token == "legacy"
