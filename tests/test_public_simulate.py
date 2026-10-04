"""``/simulations`` and ``/simulate``: sim presets, request checks, the queue
and the rate limits.

The arch presets are fixtures (a dense one, and an MoE speculative one with a
capture row and a max_model_len); the sim presets are written for each test;
a run is a small script standing in for the launcher's child process.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from public_api import preset as public_preset
from public_api import simulate
from public_api.app import PREFIX, create_app
from public_api.deployments import DeploymentIndex, Member, Preset
from public_api.kernels import KernelLibrary
from public_api.limits import RateLimiter
from public_api.sim_preset import SimIndex, SimPresetError, load
from public_api.sources import Sources

HF = "hf://datasets/UW-SyFI/servingstudio-workload@" + "a" * 40
CAPTURE_DIR = f"{HF}/glm/vllm/capa/capture/20260101"
DENSE_TRACE = "glm/vllm/capa/capture/20260101"
MAX_MODEL_LEN = 1000
DRAFT = 3
# id, input_len, output_len: the third request does not fit MAX_MODEL_LEN.
TRACE_ROWS = [(0, 100, 50), (1, 200, 60), (2, 900, 200), (3, 50, 10)]

DENSE_SIM = """\
deployment: unified
pools:
  main:
    groups:
      - replicas: ${replicas}
        arch:
          preset: dense
          tp_size: ${tp_size}
        worker: {type: barebone}
sweep:
  tp_size: [1, 2]
  replicas: [1, 2]
"""
SPEC_SIM = f"""\
deployment: unified
pools:
  main:
    groups:
      - replicas: 1
        arch: {{preset: spec}}
        worker: {{type: speculative, draft_tokens: {DRAFT}}}
"""
PD_SIM = """\
deployment: pd
pools:
  prefill:
    groups:
      - replicas: 1
        arch: {preset: dense, tp_size: 1}
        worker: {type: pd_prefill}
  decode:
    groups:
      - replicas: ${decode_replicas}
        arch: {preset: dense, tp_size: 2}
        worker: {type: pd_decode}
sweep:
  decode_replicas: [1, 2]
"""


def _arch_index() -> DeploymentIndex:
    index = DeploymentIndex("abc123", {}, {})
    index.presets["Llama/dense"] = Preset(
        id="Llama/dense",
        checkpoint="meta-llama/Meta-Llama-3-8B",
        arch="dense",
        gpu="NVIDIA H200",
        axes=[{"name": "tp_size", "values": [1, 2]}],
        members=[
            Member(
                preset="Llama/dense",
                params={"tp_size": tp},
                gpu="NVIDIA H200",
                arch={"type": "dense", "tp_size": tp},
                block={},
                gpus_per_replica=tp,
            )
            for tp in (1, 2)
        ],
    )
    rows = {
        "capa": {
            "routing": "popularity",
            "expert_popularity_file": f"{CAPTURE_DIR}/popularity.json",
        },
        "uniform": {"routing": "uniform"},
    }
    index.presets["GLM/spec"] = Preset(
        id="GLM/spec",
        checkpoint="zai-org/GLM",
        arch="moe",
        gpu="NVIDIA B200",
        axes=[{"name": "workload", "values": list(rows), "rows": rows}],
        members=[
            Member(
                preset="GLM/spec",
                params={"workload": label},
                gpu="NVIDIA B200",
                arch={"type": "moe", "max_model_len": MAX_MODEL_LEN, **row},
                block={},
                gpus_per_replica=4,
            )
            for label, row in rows.items()
        ],
    )
    return index


@pytest.fixture
def arch_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "public"
    for preset_id in ("Llama/dense", "GLM/spec"):
        (root / preset_id).parent.mkdir(parents=True, exist_ok=True)
        (root / f"{preset_id}.yaml").write_text("{}\n")
    monkeypatch.setattr(public_preset, "PRESET_ROOT", root)
    return root


def _write(root: Path, preset_id: str, text: str) -> Path:
    path = root / f"{preset_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def sims(tmp_path: Path, arch_presets: Path) -> SimIndex:
    """Three sim presets; the dense one's tp 2 x 2 replicas member lacks a row."""
    root = tmp_path / "public_sim"
    paths = [
        _write(root, "Llama/dense_barebone", DENSE_SIM),
        _write(root, "Llama/dense_pd", PD_SIM),
        _write(root, "GLM/spec_speculative", SPEC_SIM),
    ]
    sims = SimIndex.build(_arch_index(), paths)

    def dry_run(member, capture):
        pool = member.pools["main"] if "main" in member.pools else None
        if pool and pool["replicas"] == 2 and pool["arch_params"] == {"tp_size": 2}:
            return {"layer.qkv": 1}
        return {}

    sims.check(dry_run, jobs=2)
    return sims


@pytest.fixture
def trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "trace.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "input_len", "output_len", "arrival_time"])
        for n, (rid, input_len, output_len) in enumerate(TRACE_ROWS):
            writer.writerow([rid, input_len, output_len, n * 0.5])
    monkeypatch.setattr(simulate, "resolve_reference", lambda reference: str(path))
    # The launcher's expansion against the simulator's schema is its own tests'.
    monkeypatch.setattr(simulate, "concrete", lambda tree, registry: tree)
    return path


RUN = """\
import json, pathlib, sys, time
run = pathlib.Path(sys.argv[1]); mode = sys.argv[2]
if mode == "sleep":
    time.sleep(60)
if mode == "fail":
    print("thread 'main' panicked at src/x.rs:1:1:\\nthe run broke", file=sys.stderr)
    sys.exit(101)
(run / "summary.json").write_text(json.dumps({
    "cause": "finished", "requests_total": 2, "requests_finished": 2, "sim_ms": 1500.0,
    "num_gpus": 1, "total_tok_s": 100.0, "prefill_tok_s": 80.0, "decode_tok_s": 20.0,
    "total_tok_s_per_gpu": 100.0, "completed_req_s": 1.3}))
(run / "reports").mkdir()
(run / "reports" / "slo_general_report.json").write_text(json.dumps({"metrics": {
    "ttft": {"p50": 10.0, "p99": 20.0}, "tpot": {"p50": 1.0}, "e2e": {"p50": 100.0}}}))
"""


class Runner:
    """The queue's command: what the next run does is ``mode``."""

    def __init__(self) -> None:
        self.mode = "ok"

    def __call__(self, run_dir: Path, build_type: str) -> list[str]:
        return [sys.executable, "-c", RUN, str(run_dir), self.mode]


@pytest.fixture
def runner() -> Runner:
    return Runner()


def _queue(tmp_path: Path, runner: Runner, **limits) -> simulate.Simulations:
    return simulate.Simulations(
        tmp_path / "sims", command=runner, run_id=lambda sim_id: f"run-{sim_id}", **limits
    )


def _client(sims: SimIndex, queue: simulate.Simulations, **limits) -> TestClient:
    service = simulate.SimulationService(sims, None, queue)
    kernels = KernelLibrary(Sources(db_path=Path("/nonexistent")), sims.index)
    return TestClient(create_app(kernels, None, None, service, **limits))


@pytest.fixture
def client(sims, trace, tmp_path, runner) -> TestClient:
    return _client(sims, _queue(tmp_path, runner))


def _wait(client: TestClient, sim_id: str, until=("done", "failed", "timed_out")) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        answer = client.get(f"{PREFIX}/simulations/{sim_id}").json()
        if answer["status"] in until:
            return answer
        time.sleep(0.05)
    raise AssertionError(f"{sim_id} still {answer['status']}")


def _post(client: TestClient, preset: str, params: dict, **workload):
    return client.post(
        f"{PREFIX}/simulate", json={"preset": preset, "params": params, "workload": workload}
    )


DENSE = ("Llama/dense_barebone", {"tp_size": 1, "replicas": 1})
SPEC = ("GLM/spec_speculative", {})


# -- presets -------------------------------------------------------------------


def test_presets_list_members_captures_and_what_cannot_run(client: TestClient) -> None:
    answer = client.get(f"{PREFIX}/simulations/presets").json()

    assert answer["sim_commit"] == "abc123"
    assert answer["limits"]["max_requests"] == simulate.MAX_REQUESTS
    assert "accept_rate" in answer["workload"]["properties"]
    presets = {preset["id"]: preset for preset in answer["presets"]}
    assert set(presets) == {"Llama/dense_barebone", "Llama/dense_pd", "GLM/spec_speculative"}

    dense = presets["Llama/dense_barebone"]
    # A dense arch routes nothing: it replays any published capture's trace.
    assert [c["name"] for c in dense["captures"]] == [DENSE_TRACE]
    assert dense["pools"]["main"] == {
        "arch_preset": "Llama/dense",
        "arch": "dense",
        "gpu": "NVIDIA H200",
        "worker": "barebone",
    }
    members = {json.dumps(m["params"], sort_keys=True): m for m in dense["members"]}
    assert len(members) == 4
    lacking = members[json.dumps({"replicas": 2, "tp_size": 2})]
    assert lacking["gpus"] == 4
    assert lacking["unavailable"] == {DENSE_TRACE: "lacks profile.db rows: layer.qkv 1"}
    assert members[json.dumps({"replicas": 1, "tp_size": 1})]["unavailable"] == {}

    # An MoE arch offers its capture rows, never the uniform one.
    spec = presets["GLM/spec_speculative"]
    assert spec["captures"] == [
        {"name": "capa", "trace": f"{CAPTURE_DIR}/trace.csv", "routing": "popularity"}
    ]
    pd = presets["Llama/dense_pd"]
    assert [m["gpus"] for m in pd["members"]] == [3, 5]


def test_without_a_simulation_service_the_routes_answer_503(sims) -> None:
    kernels = KernelLibrary(Sources(db_path=Path("/nonexistent")), sims.index)
    client = TestClient(create_app(kernels))
    assert client.get(f"{PREFIX}/simulations/presets").status_code == 503
    assert _post(client, *DENSE).status_code == 503
    assert client.get(f"{PREFIX}/simulations/abc").status_code == 503
    assert client.get(f"{PREFIX}/analyzer/runs/abc/descriptor").status_code == 503
    # No catalog of runs: a reader opens the simulation they made.
    assert client.get(f"{PREFIX}/analyzer/runs").status_code == 404


# -- request checks ------------------------------------------------------------


@pytest.mark.parametrize(
    ("preset", "params", "workload", "status", "message"),
    [
        ("Llama/nope", {}, {}, 404, None),
        ("Llama/dense_barebone", {"tp_size": 4, "replicas": 1}, {}, 400, "has no member"),
        ("Llama/dense_barebone", {"tp_size": 1}, {}, 400, "missing ['replicas']"),
        (*DENSE, {"capture": "nope"}, 400, "has no capture 'nope'"),
        (*DENSE, {"num_requests": 5}, 400, "has 4 requests, not 5"),
        (*DENSE, {"session_dependency": "chained"}, 400, "needs a session trace"),
        (*DENSE, {"run_to_end": False}, 400, "needs duration_ms"),
        (*DENSE, {"accept_rate": 0.5}, 400, "only to a speculative worker"),
        (*SPEC, {"num_requests": 2}, 400, "needs workload.accept_rate"),
        (*SPEC, {"num_requests": 2, "accept_rate": [0.5, 0.4]}, 400, "has 2 positions"),
        (*SPEC, {"num_requests": 2, "accept_rate": 1.5}, 400, "between 0 and 1"),
        # Request 2 needs 900 + 200 + 3 draft tokens > 1000.
        (*SPEC, {"accept_rate": 0.5}, 400, "1 of 4 requests exceed pool main's max_model_len"),
        ("Llama/dense_barebone", {"tp_size": 2, "replicas": 2}, {}, 409, "lacks profile.db"),
        (*DENSE, {"num_requests": 0}, 422, None),
        (*DENSE, {"arrival_mode": "poisson"}, 422, None),
        (*DENSE, {"request_rate": 0}, 422, None),
    ],
)
def test_a_request_the_member_cannot_run_is_refused_before_it_queues(
    client: TestClient, tmp_path: Path, preset, params, workload, status, message
) -> None:
    answer = _post(client, preset, params, **workload)

    assert answer.status_code == status, answer.json()
    detail = answer.json()["detail"]
    if message:
        assert message in (detail if isinstance(detail, str) else detail["message"])
    # Nothing is left behind.
    assert not any((tmp_path / "sims").iterdir())


def test_at_most_max_requests_per_simulation(client, monkeypatch) -> None:
    monkeypatch.setattr(simulate, "MAX_REQUESTS", 3)
    answer = _post(client, *DENSE)
    assert answer.status_code == 400
    assert "at most 3 requests" in answer.json()["detail"]
    assert _post(client, *DENSE, num_requests=3).status_code == 202


# -- runs ----------------------------------------------------------------------


def test_a_simulation_runs_reports_its_summary_and_is_deleted(client, tmp_path) -> None:
    started = _post(client, *DENSE, num_requests=2, request_rate=2.0, max_concurrency=8)
    assert started.status_code == 202
    sim_id = started.json()["simulation_id"]

    answer = _wait(client, sim_id)
    assert answer["status"] == "done", answer
    assert answer["preset"] == DENSE[0] and answer["params"] == DENSE[1]
    assert answer["workload"]["capture"] == DENSE_TRACE
    assert answer["workload"]["num_requests"] == 2
    assert answer["gpus"] == 1
    assert answer["run_id"] == f"run-{sim_id}"
    summary = answer["summary"]
    assert summary["throughput"]["total_tok_s"] == 100.0
    assert summary["requests"] == {"total": 2, "finished": 2}
    assert summary["ttft_ms"]["p99"] == 20.0 and summary["e2e_ms"]["p50"] == 100.0

    run = tmp_path / "sims" / sim_id
    config = json.loads((run / simulate.RUN_CONFIG).read_text())
    workload = config["workload"]
    assert workload["trace_files"] == [str(run / "workload.csv")]
    assert workload["request_rate"] == 2.0 and workload["max_concurrency"] == 8
    assert workload["input_file_tags"] == []
    assert config["io"]["log_dir"] == str(run)
    assert config["pools"]["main"]["groups"] == [
        {
            "gpu": "NVIDIA H200",
            "replicas": 1,
            "arch": {"type": "dense", "tp_size": 1},
            "worker": {"type": "barebone"},
        }
    ]
    with (run / "workload.csv").open() as stream:
        assert [row["id"] for row in csv.DictReader(stream)] == ["0", "1"]

    assert client.delete(f"{PREFIX}/simulations/{sim_id}").json() == {
        "simulation_id": sim_id,
        "status": "deleted",
    }
    assert not run.exists()
    assert client.get(f"{PREFIX}/simulations/{sim_id}").status_code == 404
    assert client.delete(f"{PREFIX}/simulations/{sim_id}").status_code == 404


def test_a_speculative_run_gets_its_acceptance_and_the_capture_routing(client, tmp_path) -> None:
    started = _post(client, *SPEC, num_requests=2, accept_rate=[0.9, 0.7, 0.5])
    sim_id = started.json()["simulation_id"]
    assert _wait(client, sim_id)["status"] == "done"

    run = tmp_path / "sims" / sim_id
    config = json.loads((run / simulate.RUN_CONFIG).read_text())
    assert config["workload"]["input_file_tags"] == ["speculative"]
    arch = config["pools"]["main"]["groups"][0]["arch"]
    assert arch["routing"] == "popularity"
    assert arch["expert_popularity_file"] == f"{CAPTURE_DIR}/popularity.json"
    with (run / "workload.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["accept_rate"] for row in rows] == ["[0.9, 0.7, 0.5]"] * 2


def test_a_failed_run_reports_its_cause(client, runner) -> None:
    runner.mode = "fail"
    answer = _wait(client, _post(client, *DENSE).json()["simulation_id"])
    assert answer["status"] == "failed"
    assert "the run broke" in answer["error"]
    assert "summary" not in answer


def test_runs_queue_past_max_running_and_a_deleted_run_is_stopped(
    sims, trace, tmp_path, runner
) -> None:
    runner.mode = "sleep"
    client = _client(sims, _queue(tmp_path, runner, max_running=2, max_queued=1))
    ids = [_post(client, *DENSE).json()["simulation_id"] for _ in range(2)]
    for sim_id in ids:
        _wait(client, sim_id, until=("running",))
    third = _post(client, *DENSE).json()["simulation_id"]
    assert client.get(f"{PREFIX}/simulations/{third}").json()["queue_position"] == 0
    full = _post(client, *DENSE)
    assert full.status_code == 429
    assert "already waiting" in full.json()["detail"]

    # Deleting a running one stops it, and the queued one starts.
    assert client.delete(f"{PREFIX}/simulations/{ids[0]}").status_code == 200
    _wait(client, third, until=("running",))
    for sim_id in (ids[1], third):
        client.delete(f"{PREFIX}/simulations/{sim_id}")


def test_a_run_past_the_wall_clock_limit_is_stopped(sims, trace, tmp_path, runner) -> None:
    runner.mode = "sleep"
    client = _client(sims, _queue(tmp_path, runner, timeout_s=0.5))
    started = time.monotonic()
    answer = _wait(client, _post(client, *DENSE).json()["simulation_id"])
    assert answer["status"] == "timed_out"
    assert "wall-clock limit" in answer["error"]
    assert time.monotonic() - started < 15


def test_a_restart_fails_unfinished_runs_and_drops_expired_ones(tmp_path, runner) -> None:
    root = tmp_path / "sims"
    old = time.time() - 2 * simulate.KEEP_S
    for sim_id, status, created in (("live", "running", time.time()), ("old", "done", old)):
        sim = simulate.Simulation(
            id=sim_id, directory=root / sim_id, request={"preset": "x"}, status=status
        )
        sim.created_at = created
        sim.directory.mkdir(parents=True)
        sim.save()
    (root / "stray").mkdir()

    queue = _queue(tmp_path, runner)

    live = queue.get("live")
    assert live["status"] == "failed" and "restarted" in live["error"]
    assert live["preset"] == "x"
    with pytest.raises(simulate.UnknownSimulation):
        queue.get("old")
    assert not (root / "old").exists()
    # A directory without a record goes once it is a day old.
    assert (root / "stray").exists()


# -- rate limits ---------------------------------------------------------------


def test_simulate_and_predict_are_rate_limited_per_client(sims, trace, tmp_path, runner) -> None:
    client = _client(
        sims,
        _queue(tmp_path, runner),
        predict_limit=RateLimiter(2, 60.0),
        simulate_limit=RateLimiter(1, 60.0),
    )
    assert _post(client, *DENSE, num_requests=1).status_code == 202
    refused = _post(client, *DENSE, num_requests=1)
    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) >= 1

    body = {"preset": "Llama/dense", "params": {"tp_size": 1}, "cases": []}
    # This service keeps no runs directory, but the limit is counted first.
    assert [client.post(f"{PREFIX}/predict", json=body).status_code for _ in range(3)] == [
        503,
        503,
        429,
    ]


def test_a_rate_limit_is_a_sliding_window_per_client() -> None:
    now = [0.0]
    limiter = RateLimiter(2, 60.0, clock=lambda: now[0])
    assert limiter.admit("a") is None
    now[0] = 30.0
    assert limiter.admit("a") is None
    assert limiter.admit("a") == pytest.approx(30.0)
    assert limiter.admit("b") is None
    now[0] = 60.0  # the first request leaves the window
    assert limiter.admit("a") is None
    assert limiter.admit("a") == pytest.approx(30.0)


# -- sim preset shape ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (DENSE_SIM + "workload: {request_rate: 1}\n", "a sim preset has no workload"),
        (DENSE_SIM.replace("preset: dense", "preset: nope"), "names no arch preset"),
        (DENSE_SIM.replace("preset: dense", "preset: ${arch}"), "literal `preset`"),
        (DENSE_SIM.replace("tp_size: ${tp_size}", "workload: capa"), "fixes `workload`"),
        (DENSE_SIM.replace("worker:", "gpu: NVIDIA H200\n        worker:"), "its GPU is"),
        (
            DENSE_SIM.replace(
                "        worker: {type: barebone}\n",
                "        worker: {type: barebone}\n"
                "      - replicas: 1\n"
                "        arch: {preset: dense, tp_size: 1}\n"
                "        worker: {type: barebone}\n",
            ),
            "exactly one group",
        ),
    ],
)
def test_a_sim_preset_must_reference_its_arch_preset(tmp_path, arch_presets, text, message) -> None:
    path = _write(tmp_path / "public_sim", "Llama/bad", text)
    with pytest.raises(SimPresetError) as error:
        load(path)
    assert message in str(error.value)
