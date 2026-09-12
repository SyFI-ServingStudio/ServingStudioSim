"""Managed context protocol selection at the outbound HTTP boundary."""

import io
import json
from urllib import error, request

import pytest

from launcher.managed_job import ManagedJob, prepare_managed_job
from launcher.managed_run import ManagedRun, prepare_experiment

JOB_ENV = "VIBESIM_MANAGED_JOB_CONTEXT"
RUN_ENV = "VIBESIM_MANAGED_RUN_CONTEXT"
KINDS = ("simulation", "timing_predict", "kernel_profile", "kernel_measure")
RESOURCE_IDS = {
    "timing_predict": "p_test",
    "kernel_profile": "kp_test",
    "kernel_measure": "km_test",
}


class Response:
    def __init__(self, root):
        self.root = root

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(
            {
                "jobId": "j_test",
                "experimentId": "e_test",
                "resourceId": "jr_test",
                "approvedRoot": str(self.root),
            }
        ).encode()


def context(path, **extra):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_url": "http://backend.test/",
                "capability_token": "opaque-test-token",
                **extra,
            }
        )
    )
    return path


def prepare(kind, root):
    if kind == "simulation":
        return prepare_experiment(root, run_count=3, axes=["tp"])
    return prepare_managed_job(
        kind,
        root,
        descriptor={"selector": "iter"},
        analyzer_resource_id=RESOURCE_IDS[kind],
    )


@pytest.fixture(autouse=True)
def clear_context(monkeypatch):
    monkeypatch.delenv(JOB_ENV, raising=False)
    monkeypatch.delenv(RUN_ENV, raising=False)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("protocol", [None, "agent-v1"])
def test_four_kinds_select_exact_register_and_status_protocol(
    monkeypatch, tmp_path, kind, protocol
):
    root = tmp_path / "artifacts"
    extra = {} if protocol is None else {"managed_jobs_api": protocol}
    monkeypatch.setenv(RUN_ENV, str(context(tmp_path / "context.json", **extra)))
    captured = []

    def urlopen(http_request, timeout):
        assert timeout == 15
        assert not root.exists()
        assert http_request.method == "POST"
        assert http_request.headers["Authorization"] == "Bearer opaque-test-token"
        assert http_request.headers["Content-type"] == "application/json"
        captured.append(http_request)
        return Response(root)

    monkeypatch.setattr(request, "urlopen", urlopen)
    managed = prepare(kind, root)
    managed.report("running")
    managed.report("running")
    managed.report("ready")
    family = "managed-runs" if kind == "simulation" else "managed-jobs"
    prefix = "/api/agent/v1/internal/jobs" if protocol else "/api/internal/" + family
    assert [item.full_url for item in captured] == [
        "http://backend.test" + prefix + suffix
        for suffix in ("/register", "/j_test/status", "/j_test/status")
    ]
    expected = (
        {"runCount": 3, "axes": ["tp"]}
        if kind == "simulation"
        else {
            "jobKind": kind,
            "artifactRoot": str(root),
            "descriptor": {"selector": "iter"},
            "analyzerResourceId": RESOURCE_IDS[kind],
        }
    )
    if kind == "simulation":
        expected.update(
            {"jobKind": "simulation", "artifactRoot": str(root)}
            if protocol
            else {"experimentRoot": str(root)}
        )
    assert json.loads(captured[0].data) == expected
    assert [json.loads(item.data) for item in captured[1:]] == [
        {"status": "running"},
        {"status": "ready"},
    ]


@pytest.mark.parametrize("kind", ["simulation", "timing_predict"])
def test_job_context_precedes_run_and_invalid_preferred_context_never_falls_back(
    monkeypatch, tmp_path, kind
):
    root = tmp_path / "artifacts"
    preferred = context(
        tmp_path / "job.json", managed_jobs_api="agent-v1", capability_token="preferred"
    )
    legacy = context(tmp_path / "run.json", capability_token="legacy")
    monkeypatch.setenv(JOB_ENV, str(preferred))
    monkeypatch.setenv(RUN_ENV, str(legacy))
    captured = []

    def urlopen(http_request, timeout):
        captured.append(http_request)
        return Response(root)

    monkeypatch.setattr(request, "urlopen", urlopen)
    prepare(kind, root)
    assert captured[0].headers["Authorization"] == "Bearer preferred"
    assert captured[0].full_url.endswith("/api/agent/v1/internal/jobs/register")
    preferred.write_text("{}")
    with pytest.raises(RuntimeError, match="schema_version"):
        prepare(kind, root)
    assert len(captured) == 1
    assert not root.exists()


@pytest.mark.parametrize("kind", ["simulation", "timing_predict"])
@pytest.mark.parametrize("marker", [None, "unknown", 1, {}])
def test_explicit_unsupported_marker_fails_before_http_or_artifacts(
    monkeypatch, tmp_path, kind, marker
):
    monkeypatch.setenv(JOB_ENV, str(context(tmp_path / "context.json", managed_jobs_api=marker)))
    captured = []
    monkeypatch.setattr(request, "urlopen", lambda *args, **kwargs: captured.append(args))
    root = tmp_path / "artifacts"
    with pytest.raises(RuntimeError, match="managed_jobs_api"):
        prepare(kind, root)
    assert captured == []
    assert not root.exists()


def test_without_context_development_metadata_stays_local(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(request, "urlopen", lambda *args, **kwargs: captured.append(args))
    assert ManagedRun.from_environment() is None
    assert ManagedJob.from_environment("timing_predict") is None
    root = tmp_path / "simulation"
    assert prepare("simulation", root) is None
    metadata = (root / "experiment.meta.json").read_bytes()
    assert prepare("simulation", root) is None
    assert (root / "experiment.meta.json").read_bytes() == metadata
    assert json.loads(metadata)["origin"] == {"kind": "development"}
    for kind in KINDS[1:]:
        target = tmp_path / kind
        assert prepare(kind, target) is None
        assert not target.exists()
    assert captured == []


@pytest.mark.parametrize("kind", ["simulation", "timing_predict"])
@pytest.mark.parametrize("protocol", [None, "agent-v1"])
@pytest.mark.parametrize("failure", [404, 500, "timeout"])
def test_failed_http_never_falls_back_and_explicit_retry_is_not_deduplicated(
    monkeypatch, tmp_path, kind, protocol, failure
):
    root = tmp_path / "artifacts"
    extra = {} if protocol is None else {"managed_jobs_api": protocol}
    monkeypatch.setenv(JOB_ENV, str(context(tmp_path / "context.json", **extra)))
    captured = []
    fail = True

    def urlopen(http_request, timeout):
        assert timeout == 15
        captured.append(http_request.full_url)
        if fail:
            if failure == "timeout":
                raise TimeoutError("timed out")
            raise error.HTTPError(
                http_request.full_url, failure, "not found", {}, io.BytesIO(b"not found")
            )
        return Response(root)

    monkeypatch.setattr(request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="request failed"):
        prepare(kind, root)
    assert len(captured) == 1
    assert not root.exists()
    fail = False
    managed = prepare(kind, root)
    assert captured[1] == captured[0]
    fail = True
    with pytest.raises(RuntimeError, match="request failed"):
        managed.report("running")
    assert len(captured) == 3
    fail = False
    managed.report("running")
    assert captured[3] == captured[2]
    managed.report("running")
    assert len(captured) == 4


def test_legacy_positional_constructors_and_summary_repeat_contract(monkeypatch):
    run = ManagedRun("http://backend.test", "token", "j_run", "e_run", "/tmp/run")
    job = ManagedJob(
        "http://backend.test", "token", "timing_predict", "j_job", "/tmp/job", "jr_job"
    )
    assert run.job_id == "j_run" and run.experiment_id == "e_run"
    assert job.job_id == "j_job" and job.resource_id == "jr_job"
    assert run.managed_jobs_api is None and job.managed_jobs_api is None
    captured = []

    def post(self, path, payload):
        captured.append((path, payload))
        return {}

    monkeypatch.setattr(ManagedJob, "_request", post)
    job.report("ready")
    job.report("ready")
    job.report("ready", summary={"count": 2})
    assert captured == [
        ("/api/internal/managed-jobs/j_job/status", {"status": "ready"}),
        ("/api/internal/managed-jobs/j_job/status", {"status": "ready", "summary": {"count": 2}}),
    ]
