"""``/simulations``, ``/simulate`` and ``/workloads``: sim presets, request
checks, generated and uploaded workloads, the queue and the rate limits.

The arch presets are fixtures (a dense one and an MoE speculative one with a
capture row); the sim presets are written for each test; a run is a small
script standing in for the launcher's child process. The simulator's trace
loading (``simulator workload-plan``) is stood in for by reading the trace's
rows, except in the tests marked ``needs_binary``, which run it; tracegen runs
where it is built. Whether a trace fits a pool is the simulator's check on the
run config, which these fixture archs cannot build: it is the real members'
test (``test_public_sim_presets``), and here a stand-in that refuses.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alignment.load_generator.runner import TRACEGEN
from public_api import preset as public_preset
from public_api import sim_preset, simulate, workloads
from public_api.app import PREFIX, create_app
from public_api.deployments import DeploymentIndex, Member, Preset
from public_api.kernels import KernelLibrary
from public_api.limits import RateLimiter
from public_api.sim_preset import SimIndex, SimPresetError, load
from public_api.sources import Sources

HF = "hf://datasets/UW-SyFI/servingstudio-workload@" + "a" * 40
CAPTURE_DIR = f"{HF}/glm/vllm/capa/capture/20260101"
# A dense member names a capture by its workload label.
DENSE_TRACE = "capa"
DRAFT = 3
# The speculative pool's bound: request 2 (900 + 200 tokens) fits it only
# without the draft window.
MAX_MODEL_LEN = 1100
# id, input_len, output_len.
TRACE_ROWS = [(0, 100, 50), (1, 200, 60), (2, 900, 200), (3, 50, 10)]
needs_tracegen = pytest.mark.skipif(not TRACEGEN.is_file(), reason="tracegen not built")

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
        arch: {{preset: spec, max_model_len: {MAX_MODEL_LEN}}}
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


def _arch_index(model_config: Path) -> DeploymentIndex:
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
                arch={"type": "dense", "tp_size": tp, "model_config": str(model_config)},
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
        axes=[
            {"name": "max_model_len", "values": [MAX_MODEL_LEN]},
            {"name": "workload", "values": list(rows), "rows": rows},
        ],
        members=[
            Member(
                preset="GLM/spec",
                params={"max_model_len": MAX_MODEL_LEN, "workload": label},
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
    # Every fixture capture is its own request list.
    monkeypatch.setattr(sim_preset, "_content", lambda reference: reference)
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
    config = tmp_path / "llama.json"
    config.write_text(json.dumps({"max_position_embeddings": 131072}))
    sims = SimIndex.build(_arch_index(config), paths)

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


def _rows_plan(block: dict, build_type: str = "release") -> list[dict]:
    """Stands in for ``simulator workload-plan``: the trace's rows, as its plan
    names them."""
    (path,) = block["trace_files"]
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            "request_id": row.get("request_id", row.get("id")),
            "predecessor_request_id": None,
            "session_arrival_time_ms": float(row.get("arrival_time", 0)) * 1000,
            "prefix_len": int(row.get("prefix_len", 0)),
            "input_len": int(row["input_len"]),
            "output_len": int(row["output_len"]),
        }
        for row in rows
    ]


class RunPlans:
    """Stands in for ``simulator workload-plan --config``: the run configs it
    was given, and the refusal it answers with when ``refusal`` is set."""

    def __init__(self) -> None:
        self.configs: list[dict] = []
        self.refusal: str | None = None

    def __call__(self, config: dict, build_type: str = "release") -> list[dict]:
        self.configs.append(config)
        if self.refusal:
            raise workloads.BadWorkload(self.refusal)
        return _rows_plan(config["workload"], build_type)


@pytest.fixture
def run_plans(monkeypatch: pytest.MonkeyPatch) -> RunPlans:
    plans = RunPlans()
    monkeypatch.setattr(simulate, "plan_run", plans)
    return plans


# `simulator trace-formats`, as far as these tests read it.
FORMATS = {
    "formats": [
        {
            "name": "text-generation-independent",
            "columns": ["id", "arrival_time", "input_len", "output_len"],
            "tags": ["session", "slo", "priority", "speculative"],
        },
        {
            "name": "text-generation-session-execution-v2",
            "columns": ["request_id", "session_id", "round_idx", "arrival_time_ms"]
            + ["prefix_len", "input_len", "output_len", "tool_wait_after_ms"],
            "tags": ["slo", "priority", "speculative"],
        },
    ],
    "tags": [
        {"name": "session", "columns": ["session_id", "prefix_kv", "tool_wait_after_ms"]},
        {"name": "slo", "columns": ["ttft_slo_ms", "tpot_slo_ms", "e2e_slo_ms"]},
        {"name": "priority", "columns": ["priority"]},
        {"name": "speculative", "columns": ["accept_rate"]},
    ],
}


@pytest.fixture
def rows_plan(monkeypatch: pytest.MonkeyPatch, run_plans: RunPlans) -> None:
    monkeypatch.setattr(workloads, "plan", _rows_plan)
    monkeypatch.setattr(simulate, "plan", _rows_plan)
    monkeypatch.setattr(workloads, "trace_formats", lambda build_type="release": FORMATS)


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


def _client(
    sims: SimIndex, queue: simulate.Simulations, runs_dir: Path | None = None, **limits
) -> TestClient:
    uploads = workloads.Workloads(queue.runs_dir.parent / "uploads")
    service = simulate.SimulationService(sims, None, queue, uploads)
    kernels = KernelLibrary(Sources(db_path=Path("/nonexistent")), sims.index)
    return TestClient(create_app(kernels, runs_dir, None, service, **limits))


@pytest.fixture
def client(sims, trace, rows_plan, tmp_path, runner) -> TestClient:
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
SMALL = {
    "type": "synthetic",
    "sessions": 3,
    "rounds": "1",
    "input_len": "100",
    "output_len": "20",
    "seed": 1,
}


def _upload(client: TestClient, text: str, **query):
    return client.post(
        f"{PREFIX}/workloads",
        content=text.encode(),
        params=query,
        headers={"content-type": "text/csv"},
    )


def _sessions(rows) -> str:
    lines = [",".join(FORMATS["formats"][1]["columns"])]
    lines += [
        f"session_{n}_round_000000,{n},0,{n * 500},0,{input_len},{output_len},0"
        for n, (_, input_len, output_len) in enumerate(rows)
    ]
    return "\n".join(lines) + "\n"


def _independent(rows) -> str:
    lines = ["id,arrival_time,input_len,output_len"]
    lines += [
        f"{rid},{n * 0.5},{input_len},{output_len}"
        for n, (rid, input_len, output_len) in enumerate(rows)
    ]
    return "\n".join(lines) + "\n"


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
    assert lacking["unavailable"] == {DENSE_TRACE: {"missing": {"layer.qkv": 1}}}
    assert members[json.dumps({"replicas": 1, "tp_size": 1})]["unavailable"] == {}

    # An MoE arch offers its capture rows, never the uniform one.
    spec = presets["GLM/spec_speculative"]
    # Each capture says what its trace holds, as the simulator reads it.
    facts = {
        "requests": 4,
        "sessions": False,
        "prompt_tokens": 312.5,
        "output_tokens": 80.0,
        "rate": 2.0,
        "input_file_tags": [],
    }
    assert spec["captures"] == [
        {
            "name": "capa",
            "trace": f"{CAPTURE_DIR}/trace.csv",
            "routing": "popularity",
            "facts": facts,
        }
    ]
    pd = presets["Llama/dense_pd"]
    assert [m["gpus"] for m in pd["members"]] == [3, 5]


def test_a_capture_too_long_for_a_member_is_that_captures_misfit(sims) -> None:
    """A capture whose requests the member cannot hold is offered as a
    misfit with the simulator's counts; the member's other captures and the
    other members still run."""
    spec = sims.preset("GLM/spec_speculative").members[0]
    names = [capture.name for capture in spec.captures]
    too_long = {"reason": "3 of 4 requests exceed", "requests": 3, "total": 4, "max_model_len": 8}

    def misfit(member, capture):
        return too_long if member is spec and capture.name == names[0] else None

    sims.check(lambda member, capture: {}, misfit, jobs=2)
    assert spec.summary(sims.index)["unavailable"] == {names[0]: {"misfit": too_long}}
    assert spec.runnable(spec.capture(names[0])) == "3 of 4 requests exceed"
    # Its routing still serves requests that are not its own.
    assert spec.runnable(spec.capture(names[0]), replayed=False) is None
    assert all(spec.runnable(spec.capture(name)) is None for name in names[1:])
    others = [m for p in sims.presets.values() for m in p.members if m is not spec]
    assert not any(m.misfits for m in others)


def test_a_dense_member_lists_each_request_list_once(monkeypatch) -> None:
    """Captures of one workload on several models often record the same
    requests: a dense member lists each list once, by its workload label (with
    a model when a label has two lists), the one most captures record first."""
    contents = {
        "m1/vllm/grid/capture/1": "grid",
        "m1/vllm/diverse/capture/1": "diverse",
        "m2/sglang/diverse/capture/2": "diverse",
        "m1/vllm/ctx/capture/1": "ctx-a",
        "m2/vllm/ctx/capture/1": "ctx-b",
    }
    rows = {
        f"r{n}": {"routing": "popularity", "expert_popularity_file": f"{HF}/{d}/popularity.json"}
        for n, d in enumerate(contents)
    }
    index = DeploymentIndex("abc123", {}, {})
    index.presets["M/moe"] = Preset(
        id="M/moe",
        checkpoint="m",
        arch="moe",
        gpu="NVIDIA B200",
        axes=[{"name": "workload", "values": list(rows), "rows": rows}],
        members=[],
    )

    def directory(trace: str) -> str:
        return trace.removeprefix(f"{HF}/").removesuffix("/trace.csv")

    monkeypatch.setattr(sim_preset, "_content", lambda reference: contents[directory(reference)])
    captures = sim_preset._all_traces(index)
    assert [(c.name, directory(c.trace)) for c in captures] == [
        ("diverse", "m1/vllm/diverse/capture/1"),
        ("ctx/m1", "m1/vllm/ctx/capture/1"),
        ("ctx/m2", "m2/vllm/ctx/capture/1"),
        ("grid", "m1/vllm/grid/capture/1"),
    ]


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
        (*DENSE, {"run_to_end": False}, 400, "needs duration_ms"),
        (*DENSE, {"source": "generated"}, 400, "a generated workload needs workload.generator"),
        (*DENSE, {"source": "upload"}, 400, "an upload workload needs workload.upload"),
        (*DENSE, {"upload": "abc"}, 400, "a capture workload takes no generator or upload"),
        (*DENSE, {"source": "upload", "upload": "abc"}, 400, "no upload 'abc'"),
        (*DENSE, {"accept_rate": 0.5}, 400, "only to a speculative worker"),
        (*SPEC, {}, 400, "needs workload.accept_rate"),
        ("Llama/dense_barebone", {"tp_size": 2, "replicas": 2}, {}, 409, "lacks profile.db"),
        (*DENSE, {"num_requests": 2}, 422, None),
        (*DENSE, {"load": {}}, 422, None),
        (*DENSE, {"load": {"rate": 1, "concurrency": 8}}, 422, None),
        (*DENSE, {"load": {"rate": 0}}, 422, None),
        (*DENSE, {"load": {"concurrency": 0}}, 422, None),
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


def test_the_simulators_check_of_the_run_refuses_before_it_queues(
    client, run_plans, tmp_path
) -> None:
    run_plans.refusal = "1 of 4 requests exceed pool main's max_model_len 1000 (...)"
    answer = _post(client, *SPEC, accept_rate=[0.9, 0.8, 0.7])

    assert answer.status_code == 400
    assert answer.json()["detail"] == run_plans.refusal
    assert not any((tmp_path / "sims").iterdir())
    # It checked the whole run: the trace the run replays, on the member's pools.
    (config,) = run_plans.configs
    assert config["pools"]["main"]["groups"][0]["worker"]["draft_tokens"] == DRAFT
    assert config["workload"]["input_file_tags"] == ["speculative"]
    (trace,) = config["workload"]["trace_files"]
    assert Path(trace).name == "workload.csv"


def test_at_most_max_requests_per_simulation(client, monkeypatch) -> None:
    monkeypatch.setattr(simulate, "MAX_REQUESTS", 3)
    answer = _post(client, *DENSE)
    assert answer.status_code == 400
    assert "has 4 requests; at most 3 per simulation" in answer.json()["detail"]


# -- runs ----------------------------------------------------------------------


def test_a_simulation_runs_reports_its_summary_and_is_deleted(client, tmp_path) -> None:
    # The capture's four requests arrive at 2 per second; 4 per second replays
    # them twice as fast.
    started = _post(client, *DENSE, load={"rate": 4.0})
    assert started.status_code == 202
    sim_id = started.json()["simulation_id"]

    assert started.json()["routing"] == {
        "capture": None,
        "label": f"requests from capture {DENSE_TRACE}; the model routes no experts",
    }

    answer = _wait(client, sim_id)
    assert answer["status"] == "done", answer
    assert answer["preset"] == DENSE[0] and answer["params"] == DENSE[1]
    assert answer["workload"]["capture"] == DENSE_TRACE
    assert answer["routing"] == started.json()["routing"]
    assert answer["workload"]["trace"]["requests"] == 4
    assert answer["workload"]["load"] == {"rate": 4.0, "concurrency": None}
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
    assert workload["arrival_mode"] == "trace_timed" and workload["request_rate"] == 2.0
    assert "max_concurrency" not in workload
    assert workload["input_file_tags"] == []
    assert config["io"]["log_dir"] == str(run)
    assert config["pools"]["main"]["groups"] == [
        {
            "gpu": "NVIDIA H200",
            "replicas": 1,
            "arch": {"type": "dense", "tp_size": 1, "model_config": str(tmp_path / "llama.json")},
            "worker": {"type": "barebone"},
        }
    ]
    with (run / "workload.csv").open() as stream:
        assert [row["id"] for row in csv.DictReader(stream)] == ["0", "1", "2", "3"]

    assert client.delete(f"{PREFIX}/simulations/{sim_id}").json() == {
        "simulation_id": sim_id,
        "status": "deleted",
    }
    assert not run.exists()
    assert client.get(f"{PREFIX}/simulations/{sim_id}").status_code == 404
    assert client.delete(f"{PREFIX}/simulations/{sim_id}").status_code == 404


def test_a_speculative_run_gets_its_acceptance_and_the_capture_routing(client, tmp_path) -> None:
    started = _post(client, *SPEC, load={"concurrency": 8}, accept_rate=[0.9, 0.7, 0.5])
    sim_id = started.json()["simulation_id"]
    assert started.json()["routing"] == {
        "capture": "capa",
        "label": "requests and routing from capture capa",
    }
    assert _wait(client, sim_id)["status"] == "done"

    run = tmp_path / "sims" / sim_id
    config = json.loads((run / simulate.RUN_CONFIG).read_text())
    assert config["workload"]["input_file_tags"] == ["speculative"]
    assert config["workload"]["arrival_mode"] == "saturated"
    assert config["workload"]["max_concurrency"] == 8
    arch = config["pools"]["main"]["groups"][0]["arch"]
    assert arch["routing"] == "popularity"
    assert arch["expert_popularity_file"] == f"{CAPTURE_DIR}/popularity.json"
    with (run / "workload.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["accept_rate"] for row in rows] == ["[0.9, 0.7, 0.5]"] * 4


def _with_acceptance(trace: Path) -> None:
    """Give the capture's trace the acceptance a speculative capture records."""
    with trace.open() as stream:
        rows = list(csv.DictReader(stream))
    with trace.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, [*rows[0], "accept_rate"])
        writer.writeheader()
        for n, row in enumerate(rows):
            writer.writerow(row | {"accept_rate": f"[0.{n + 5},0.5,0.5]"})


def test_a_speculative_capture_runs_on_the_acceptance_it_recorded(client, trace, tmp_path):
    _with_acceptance(trace)
    presets = client.get(f"{PREFIX}/simulations/presets").json()["presets"]
    (spec,) = [preset for preset in presets if preset["id"] == SPEC[0]]
    assert spec["captures"][0]["facts"]["input_file_tags"] == ["speculative"]

    # A speculative worker replays each request's own recorded chain.
    started = _post(client, *SPEC)
    assert started.status_code == 202, started.json()
    run = tmp_path / "sims" / started.json()["simulation_id"]
    config = json.loads((run / simulate.RUN_CONFIG).read_text())
    assert config["workload"]["input_file_tags"] == ["speculative"]
    with (run / "workload.csv").open() as stream:
        assert [row["accept_rate"] for row in csv.DictReader(stream)] == [
            f"[0.{n + 5},0.5,0.5]" for n in range(len(TRACE_ROWS))
        ]
    # Its own column is the acceptance: a second one is refused.
    both = _post(client, *SPEC, accept_rate=0.5)
    assert both.status_code == 400
    assert "carries its own accept_rate column" in both.json()["detail"]

    # A worker that drafts nothing runs the same requests without it.
    dense = _post(client, *DENSE)
    assert dense.status_code == 202, dense.json()
    run = tmp_path / "sims" / dense.json()["simulation_id"]
    config = json.loads((run / simulate.RUN_CONFIG).read_text())
    assert config["workload"]["input_file_tags"] == []
    with (run / "workload.csv").open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["id", "input_len", "output_len", "arrival_time"]
        assert len(list(reader)) == len(TRACE_ROWS)


def test_the_requests_of_a_trace_digest_without_a_tags_columns(trace, rows_plan, tmp_path):
    plain = tmp_path / "plain.csv"
    plain.write_bytes(trace.read_bytes())
    _with_acceptance(trace)
    assert workloads.requests_digest(trace) == workloads.requests_digest(plain)
    lengths = tmp_path / "lengths.csv"
    lengths.write_text(plain.read_text().replace("\n0,100,", "\n0,101,"))
    assert workloads.requests_digest(lengths) != workloads.requests_digest(plain)


def test_a_request_that_misses_the_draft_window_by_a_few_tokens_is_shortened(client, tmp_path):
    started = _post(client, *SPEC, accept_rate=0.5)
    assert started.status_code == 202, started.json()
    sim_id = started.json()["simulation_id"]
    with (tmp_path / "sims" / sim_id / "workload.csv").open() as stream:
        rows = [(row["input_len"], row["output_len"]) for row in csv.DictReader(stream)]
    # 900 + 200 + 3 drafted tokens is 3 past 1100: its output loses those 3.
    assert rows == [("100", "50"), ("200", "60"), ("900", "197"), ("50", "10")]
    assert _wait(client, sim_id)["workload"]["shortened"] == 1

    # A worker that drafts nothing serves it whole.
    dense = _post(client, *DENSE).json()["simulation_id"]
    with (tmp_path / "sims" / dense / "workload.csv").open() as stream:
        assert [row["output_len"] for row in csv.DictReader(stream)][2] == "200"


def test_a_failed_run_reports_its_cause(client, runner) -> None:
    runner.mode = "fail"
    answer = _wait(client, _post(client, *DENSE).json()["simulation_id"])
    assert answer["status"] == "failed"
    assert "the run broke" in answer["error"]
    assert "summary" not in answer


def test_runs_queue_past_max_running_and_a_deleted_run_is_stopped(
    sims, trace, rows_plan, tmp_path, runner
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


def test_a_run_past_the_wall_clock_limit_is_stopped(
    sims, trace, rows_plan, tmp_path, runner
) -> None:
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


def test_simulate_and_predict_are_rate_limited_per_client(
    sims, trace, rows_plan, tmp_path, runner
) -> None:
    client = _client(
        sims,
        _queue(tmp_path, runner),
        tmp_path / "predictions",
        predict_limit=RateLimiter(2, 60.0),
        simulate_limit=RateLimiter(1, 60.0),
        upload_limit=RateLimiter(1, 60.0),
    )
    # A refused request is no work done: it does not count.
    assert _post(client, *DENSE, capture="nope").status_code == 400
    assert _post(client, *DENSE).status_code == 202
    refused = _post(client, *DENSE)
    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) >= 1

    body = {"preset": "Llama/dense", "params": {"tp_size": 9}, "cases": []}
    assert [client.post(f"{PREFIX}/predict", json=body).status_code for _ in range(3)] == [400] * 3

    too_many = [(n, 1, 1) for n in range(simulate.MAX_REQUESTS + 1)]
    assert _upload(client, _independent(too_many)).status_code == 400
    assert _upload(client, _independent(TRACE_ROWS)).status_code == 201
    assert _upload(client, _independent(TRACE_ROWS)).status_code == 429


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
    limiter.refund("a")  # the route rejected the latest one
    assert limiter.admit("a") is None


# -- generated and uploaded workloads ----------------------------------------


@needs_tracegen
def test_workloads_describe_the_generators_arguments(client) -> None:
    generated = client.get(f"{PREFIX}/workloads").json()["sources"]["generated"]
    assert generated["input_file_format"] == "text-generation-session-execution-v2"
    (synthetic,) = generated["generators"]
    arguments = {argument["name"]: argument for argument in synthetic["arguments"]}
    # The service names the output itself.
    assert "out" not in arguments
    assert arguments["input_len"]["flag"] == "--input-len"
    assert arguments["arrival_pattern"]["choices"] == ["poisson", "constant"]


@needs_tracegen
@pytest.mark.parametrize("accept_rate", [0.7, [0.9, 0.7, 0.5]])
def test_a_generated_workload_routes_as_the_members_capture(client, tmp_path, accept_rate) -> None:
    started = _post(client, *SPEC, source="generated", generator=SMALL, accept_rate=accept_rate)
    assert started.status_code == 202, started.json()
    assert started.json()["routing"] == {
        "capture": "capa",
        "label": "requests from your workload, routing from capture capa",
    }
    sim_id = started.json()["simulation_id"]
    answer = _wait(client, sim_id)
    assert answer["workload"]["capture"] == "capa" and answer["workload"]["trace"]["requests"] == 3

    run = tmp_path / "sims" / sim_id
    config = json.loads((run / simulate.RUN_CONFIG).read_text())
    workload = config["workload"]
    assert workload["input_file_format"] == "text-generation-session-execution-v2"
    assert workload["input_file_tags"] == ["speculative"]
    arch = config["pools"]["main"]["groups"][0]["arch"]
    assert arch["expert_popularity_file"] == f"{CAPTURE_DIR}/popularity.json"
    with (run / "workload.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["input_len"] for row in rows] == ["100"] * 3
    assert {row["accept_rate"] for row in rows} == {
        json.dumps(accept_rate) if isinstance(accept_rate, list) else "0.7"
    }
    # tracegen's record of how it drew the trace stays with the run.
    assert json.loads((run / "generated.manifest.json").read_text())["parameters"]["seed"] == 1


@needs_tracegen
@pytest.mark.parametrize(
    ("generator", "message"),
    [
        ({"type": "coding-session"}, "must be one of ['synthetic']"),
        (SMALL | {"out": "/tmp/x.csv"}, "unknown ['out']"),
        (SMALL | {"bogus": 1}, "unknown ['bogus']"),
        (SMALL | {"input_len": "bogus"}, "is not a distribution"),
        (SMALL | {"sessions": 0}, "--sessions must be greater than 0"),
        (SMALL | {"seed": [1]}, "takes one number or string"),
    ],
)
def test_a_generator_the_service_cannot_run_is_refused(
    client, tmp_path, generator, message
) -> None:
    answer = _post(client, *DENSE, source="generated", generator=generator)
    assert answer.status_code == 400
    assert message in answer.json()["detail"]
    assert not any((tmp_path / "sims").iterdir())


def test_an_uploaded_workload_is_kept_and_simulated(client, tmp_path) -> None:
    uploaded = _upload(client, _independent([("a", 10, 5), ("b", 20, 5)]))
    assert uploaded.status_code == 201, uploaded.json()
    record = uploaded.json()
    # Read as the format its header fits; what it holds, as the simulator reads it.
    assert record["input_file_format"] == "text-generation-independent"
    assert record["input_file_tags"] == []
    assert record["requests"] == 2 and record["rate"] == 2.0
    assert record["prompt_tokens"] == 15.0 and record["output_tokens"] == 5.0

    started = _post(client, *DENSE, source="upload", upload=record["workload_id"])
    assert started.status_code == 202, started.json()
    assert started.json()["routing"]["label"] == (
        "requests from your workload; the model routes no experts"
    )
    sim_id = started.json()["simulation_id"]
    assert _wait(client, sim_id)["workload"]["capture"] is None
    with (tmp_path / "sims" / sim_id / "workload.csv").open() as stream:
        assert [row["id"] for row in csv.DictReader(stream)] == ["a", "b"]

    # An MoE member routes it as its capture.
    spec = _post(client, *SPEC, source="upload", upload=record["workload_id"], accept_rate=0.5)
    assert spec.json()["routing"]["label"] == (
        "requests from your workload, routing from capture capa"
    )


def test_an_upload_carries_its_own_acceptance_or_takes_the_requests(client, tmp_path) -> None:
    rows = 'id,arrival_time,input_len,output_len,accept_rate\na,0.0,10,5,"[0.9,0.8,0.7]"\n'
    record = _upload(client, rows).json()
    # Its accept_rate column is the speculative tag's.
    assert record["input_file_tags"] == ["speculative"]

    # The upload's column is the acceptance; the simulator checks its width.
    started = _post(client, *SPEC, source="upload", upload=record["workload_id"])
    assert started.status_code == 202, started.json()
    run = tmp_path / "sims" / started.json()["simulation_id"]
    config = json.loads((run / simulate.RUN_CONFIG).read_text())
    assert config["workload"]["input_file_tags"] == ["speculative"]
    with (run / "workload.csv").open() as stream:
        assert [row["accept_rate"] for row in csv.DictReader(stream)] == ["[0.9,0.8,0.7]"]

    # Two acceptances for one request is a mistake, not a precedence rule.
    both = _post(client, *SPEC, source="upload", upload=record["workload_id"], accept_rate=0.5)
    assert both.status_code == 400
    assert "carries its own accept_rate column" in both.json()["detail"]


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("id,arrival_time,input_len", "fits no trace format"),
        ("id,arrival_time,input_len,output_len,priority,bogus", "fits no trace format"),
    ],
)
def test_an_upload_whose_header_fits_no_format_is_refused(client, tmp_path, header, message):
    answer = _upload(client, header + "\n")
    assert answer.status_code == 400
    assert message in answer.json()["detail"]
    assert "text-generation-independent: id, arrival_time" in answer.json()["detail"]
    assert not any((tmp_path / "uploads").iterdir())


def test_tags_without_a_format_are_refused(client) -> None:
    answer = _upload(client, _independent(TRACE_ROWS), tags="slo")
    assert answer.status_code == 400
    assert "tags need a format" in answer.json()["detail"]


def test_an_upload_past_the_limits_is_refused(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(simulate, "MAX_REQUESTS", 3)
    answer = _upload(client, _independent(TRACE_ROWS))
    assert answer.status_code == 400
    assert "the upload has 4 requests; a simulation runs at most 3" in answer.json()["detail"]
    monkeypatch.setattr(workloads, "MAX_UPLOAD_BYTES", 16)
    assert _upload(client, _independent(TRACE_ROWS)).status_code == 413
    assert not any((tmp_path / "uploads").iterdir())


@pytest.fixture
def binary_client(sims, trace, tmp_path, runner, monkeypatch) -> TestClient:
    """Like ``client``, but the simulator itself loads every trace. The fixture
    archs do not build, so a run's trace is loaded without its pools."""
    monkeypatch.setattr(
        simulate,
        "plan_run",
        lambda config, build_type="release": workloads.plan(config["workload"], build_type),
    )
    return _client(sims, _queue(tmp_path, runner))


@pytest.mark.needs_binary
def test_the_simulator_refuses_what_a_run_would_refuse(binary_client) -> None:
    # An upload is read as the format it declares, before it is kept.
    answer = _upload(
        binary_client, _independent(TRACE_ROWS), format="text-generation-session-execution-v2"
    )
    assert answer.status_code == 400
    assert answer.json()["detail"].startswith("trace.csv: header does not match")
    slo = _upload(
        binary_client, _independent(TRACE_ROWS), format="text-generation-independent", tags="slo"
    )
    assert slo.status_code == 400
    # With no format, the one its header fits, from the simulator's own list.
    assert _upload(binary_client, _independent(TRACE_ROWS)).json()["requests"] == 4
    sessions = _upload(binary_client, _sessions(TRACE_ROWS)).json()
    assert sessions["input_file_format"] == "text-generation-session-execution-v2"


@pytest.mark.needs_binary
@needs_tracegen
def test_a_generated_session_trace_runs_through_the_simulators_loader(binary_client) -> None:
    generator = SMALL | {"rounds": "2"}
    answer = _post(
        binary_client,
        *SPEC,
        source="generated",
        generator=generator,
        accept_rate=0.6,
        duration_ms=1000.0,
    )
    assert answer.status_code == 202, answer.json()
    formats = binary_client.get(f"{PREFIX}/workloads").json()["sources"]["upload"]["formats"]
    assert [fmt["name"] for fmt in formats] == [
        "text-generation-independent",
        "text-generation-session-execution-v2",
    ]
    # An upload may carry its own acceptance.
    assert "speculative" in formats[0]["tags"]


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
