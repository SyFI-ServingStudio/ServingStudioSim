from __future__ import annotations

import json
from pathlib import Path
from urllib import request

import pytest

from launcher.managed_run import (
    EXPERIMENT_METADATA_FILENAME,
    MANAGED_CONTEXT_ENV,
    ManagedRun,
    prepare_experiment,
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


def test_direct_development_metadata_is_stable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(MANAGED_CONTEXT_ENV, raising=False)
    experiment_root = tmp_path / "logs" / "experiment"

    assert prepare_experiment(experiment_root, run_count=2, axes=["rate"]) is None
    first = json.loads(
        (experiment_root / EXPERIMENT_METADATA_FILENAME).read_text("utf-8")
    )
    assert prepare_experiment(experiment_root, run_count=2, axes=["rate"]) is None
    second = json.loads(
        (experiment_root / EXPERIMENT_METADATA_FILENAME).read_text("utf-8")
    )

    assert first == second
    assert first["experiment_id"].startswith("e_")
    assert first["origin"] == {"kind": "development"}


def test_managed_registration_precedes_experiment_directory_creation(
    monkeypatch, tmp_path: Path
) -> None:
    experiment_root = tmp_path / "logs" / "managed"
    context_path = tmp_path / "managed-run.json"
    context_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_url": "http://backend.test",
                "capability_token": "secret",
            }
        )
    )
    monkeypatch.setenv(MANAGED_CONTEXT_ENV, str(context_path))
    captured: list[request.Request] = []

    def fake_urlopen(http_request: request.Request, timeout: int):
        captured.append(http_request)
        assert timeout == 15
        assert not experiment_root.exists()
        return FakeResponse(
            {
                "jobId": "j_test",
                "experimentId": "e_test",
                "approvedRoot": str(experiment_root),
            }
        )

    monkeypatch.setattr(request, "urlopen", fake_urlopen)

    managed_run = prepare_experiment(
        experiment_root,
        run_count=4,
        axes=["request_rate", "tensor_parallel"],
    )

    assert isinstance(managed_run, ManagedRun)
    assert managed_run.job_id == "j_test"
    assert not experiment_root.exists()
    posted = json.loads(captured[0].data or b"{}")
    assert posted == {
        "experimentRoot": str(experiment_root),
        "runCount": 4,
        "axes": ["request_rate", "tensor_parallel"],
    }
    assert captured[0].headers["Authorization"] == "Bearer secret"


def test_managed_context_failure_is_fatal(monkeypatch, tmp_path: Path) -> None:
    context_path = tmp_path / "broken.json"
    context_path.write_text("{}")
    monkeypatch.setenv(MANAGED_CONTEXT_ENV, str(context_path))

    with pytest.raises(RuntimeError, match="schema_version"):
        prepare_experiment(tmp_path / "logs" / "experiment", run_count=1, axes=[])


def test_status_updates_are_idempotent(monkeypatch) -> None:
    managed_run = ManagedRun(
        backend_url="http://backend.test",
        capability_token="secret",
        job_id="j_test",
    )
    statuses: list[str] = []

    def fake_request(_self: ManagedRun, path: str, payload: dict):
        statuses.append(payload["status"])
        return {}

    monkeypatch.setattr(ManagedRun, "_request", fake_request)
    managed_run.report("running")
    managed_run.report("running")
    managed_run.report("ready")

    assert statuses == ["running", "ready"]
