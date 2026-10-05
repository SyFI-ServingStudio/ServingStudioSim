"""``POST /simulate``: one sim preset member run on a reader's workload.

A request names a member of a sim preset (:mod:`public_api.sim_preset`) and a
workload. The workload's requests come from one of three sources: a
capture the member offers (its arch preset's ``workload`` row, whose
``trace.csv`` the run replays), a trace req-frontend's ``tracegen`` draws, or a
CSV the reader uploaded (:mod:`public_api.workloads`). Its ``load`` (a request
rate or a concurrency, :class:`Load`) applies to all three. An MoE member's routing is always
one of its captures, the ``capture`` the request names or its first: a
generated or uploaded workload is labelled "requests from your workload,
routing from capture X", and nothing defaults to a synthetic routing.

The service checks the request before queueing it, so a run never starts on
input that would fail it: the member builds and is measured, a speculative
worker gets an ``accept_rate`` (the request's, or an uploaded trace's own
column), and the simulator loads the trace as the run will and refuses what a
pool cannot serve (``simulator workload-plan --config``: a request past a pool's
``max_model_len``, an acceptance vector of another width than the worker
drafts). Then it writes the run's directory under the service's
simulations directory: the trace it replays (the source's rows, with the
acceptance column a speculative worker reads)
and the concrete run config. A queued run is the launcher's standard single run
(``launcher.sweep.run_single``, analyzed, without plots) in a child process
(:mod:`public_api.simulate_run`), at most ``max_running`` at once. The Analyzer
reads the finished directory; the answer's summary is the run's own
``summary.json`` and the Analyzer's ``slo-general`` report.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from launcher.__main__ import InvalidPreset, expand_preset
from launcher.corpus import resolve_reference
from launcher.exec import ERROR_JSON, _build_subprocess_env, binary_error, binary_path
from launcher.process.leases import LauncherLeases
from launcher.schema.argv import build_cli_command
from launcher.schema.loader import Registry
from public_api import workloads
from public_api.deployments import DeploymentIndex
from public_api.predict import _cause, missing_by_role
from public_api.sim_preset import Capture, SimMember
from public_api.workloads import (
    BadWorkload,
    TraceSource,
    Workloads,
    check_count,
    describe,
    plan,
    plan_run,
    read_block,
    trace_facts,
    trace_source,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

MAX_RUNNING = 2
MAX_QUEUED = 32
TIMEOUT_S = 10 * 60
KEEP_S = 24 * 3600
# Seconds a killed run gets to exit after SIGTERM before SIGKILL.
_KILL_GRACE_S = 10.0

_LEASES = LauncherLeases(REPO_ROOT)
_RECORD = "simulation.json"
_TRACE = "workload.csv"
RUN_CONFIG = "simulation.run.json"
RUN_PRESET = "simulation.preset.json"
# A generated workload's trace, as tracegen writes it (with its manifest and
# plan beside it); the run replays `workload.csv`, cut from it.
_GENERATED = "generated.csv"


class NotRunnable(RuntimeError):
    """The member does not build, or lacks profile.db rows, for this capture."""


class QueueFull(RuntimeError):
    """Too many simulations are waiting."""


class UnknownSimulation(LookupError):
    """No simulation has this id (or it was removed)."""


class Load(BaseModel):
    """How hard a simulation drives its requests: ``rate``, the requests (a
    session trace's sessions) per second they arrive at, the trace's own spacing
    scaled to that mean; or ``concurrency``, every request ready at once and at
    most this many in flight. A workload with no load replays the trace's own
    arrival times."""

    model_config = ConfigDict(extra="forbid")

    rate: float | None = Field(default=None, gt=0)
    concurrency: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _one(self) -> Load:
        if (self.rate is None) == (self.concurrency is None):
            raise ValueError("give exactly one of rate and concurrency")
        return self


class Workload(BaseModel):
    """What a simulation replays. ``source`` says where its requests come from:
    a capture the member offers, a trace ``generator`` draws, or an ``upload``
    (``GET /workloads`` describes both). Every request of the source runs."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["capture", "generated", "upload"] = "capture"
    # A capture the member offers (`/simulations/presets`); its first by default.
    # Its trace is the requests of a `capture` workload; an MoE member routes
    # every workload as this capture does.
    capture: str | None = None
    # `generated`: `{"type": "synthetic", <tracegen argument>: value, ...}`.
    generator: dict[str, Any] | None = None
    # `upload`: the `workload_id` POST /workloads answered.
    upload: str | None = None
    load: Load | None = None
    # Simulated time to run (ms); with run_to_end, the least it runs.
    duration_ms: float | None = Field(default=None, gt=0)
    run_to_end: bool = True
    # A speculative worker's per-request acceptance: one probability for every
    # draft position, or one per position (draft_tokens of them).
    accept_rate: float | list[float] | None = None


# -- run configs ---------------------------------------------------------------


def _drafts(member: SimMember) -> bool:
    """Whether one of the member's pools runs a speculative worker."""
    return any(pool["worker"]["type"] == "speculative" for pool in member.pools.values())


def _draft_tokens(bounds: list[dict] | None) -> int | None:
    """How many tokens the drafting pool drafts, as the simulator's ``bounds``
    (:attr:`SimMember.bounds`) say; None when no pool drafts or before a build."""
    return next((b["draft_tokens"] for b in bounds or [] if b["draft_tokens"]), None)


def run_tree(
    index: DeploymentIndex,
    member: SimMember,
    capture: Capture,
    workload: dict,
    log_dir: Path,
) -> dict:
    """The run config of ``member`` on ``capture`` (its pools' arch blocks those
    arch members' own, captures still ``hf://``), with ``workload``."""
    pools = {}
    for role, pool in member.pools.items():
        arch = member.arch_member(index, role, capture)
        pools[role] = {
            **({"placement": pool["placement"]} if "placement" in pool else {}),
            "groups": [
                {
                    "gpu": arch.gpu,
                    "replicas": pool["replicas"],
                    "arch": arch.arch,
                    "worker": pool["worker"],
                }
            ],
        }
    return {
        "deployment": member.deployment,
        "workload": workload,
        "io": {"log_dir": str(log_dir), "quiet": True},
        "pools": pools,
    }


def concrete(tree: dict, registry: Registry) -> dict:
    """``tree`` as the launcher hands it to a run: validated against the
    simulator's schema, every default filled, captures fetched to local files
    (the launcher's :func:`~launcher.__main__.expand_preset` of one run)."""
    try:
        (candidate,) = expand_preset(tree, registry, "the simulation")
    except InvalidPreset as error:
        raise BadWorkload(str(error)) from None
    return candidate


def check_capture(
    index: DeploymentIndex,
    member: SimMember,
    capture: Capture,
    registry: Registry,
    build_type: str = "release",
    *,
    build: bool = True,
) -> tuple[dict[str, int] | None, dict | None, list[dict] | None]:
    """What a run of ``member`` on ``capture`` meets, from the run's own checks
    of one run config, its trace written as a run writes it
    (:func:`write_trace`):

    - with ``build``, the profile.db rows its kernels lack, ``{kernel role:
      count}`` (``simulator dry-run``: the deployment's ``build_flow``, nothing
      simulated); raises with the simulator's message when it does not build;
    - how the capture's requests do not fit its pools (:func:`plan_run`):
      ``{"reason", "requests", "total", "max_model_len"}``, the counts when the
      simulator gives them; None when they fit;
    - the pools' request bounds (:attr:`SimMember.bounds`): the build's, else
      the member's own.

    A drafting member stands in an acceptance the capture lacks: the reader
    gives one, and its value changes neither answer."""
    source = trace_source(Path(resolve_reference(capture.trace)), capture.name, build_type)
    stand_in = None
    if _drafts(member) and "speculative" not in source.input_file_tags:
        stand_in = 1.0
    workload = Workload(source="capture", capture=capture.name, accept_rate=stand_in)
    with tempfile.TemporaryDirectory(prefix="public-sim-check-") as directory:
        scratch = Path(directory)
        path = scratch / _TRACE
        # The build gives the pools' bounds, which decide the trace's
        # shortening: written once for the build, again with them.
        bounds = None if build else member.bounds
        tags, _ = write_trace(member, source, workload, path, bounds)
        trace = TraceSource(path, source.input_file_format, tuple(tags), source.name)
        config = concrete(run_tree(index, member, capture, read_block(trace), scratch), registry)
        missing = None
        if build:
            missing, bounds = _dry_run(config, scratch, build_type)
            write_trace(member, source, workload, path, bounds)
        try:
            plan_run(config, build_type)
            misfit = None
        except BadWorkload as refusal:
            misfit = {"reason": str(refusal), **(refusal.too_long or {})}
    return missing, misfit, bounds


def _dry_run(config: dict, scratch: Path, build_type: str) -> tuple[dict[str, int], list[dict]]:
    """``simulator dry-run`` of ``config``: the profile.db rows it lacks, and
    its pools' request bounds."""
    report, error = scratch / "dry_run_report.json", scratch / ERROR_JSON
    argv = build_cli_command(
        config, binary_path(build_type), scratch / "run_config.yaml", "dry-run", error_json=error
    )
    env = {**_build_subprocess_env(), "RUST_LOG": "warn"}
    with _LEASES.profile_database(write=False):
        result = subprocess.run(
            [*argv, "--report-json", str(report)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(_cause(binary_error(error), result.stderr, scratch))
    document = json.loads(report.read_text())
    return missing_by_role(document), document["pools"]


# -- workloads -----------------------------------------------------------------


def _accept_rate(
    member: SimMember, workload: Workload, source: TraceSource, draft: int | None
) -> str | None:
    """The ``accept_rate`` cell to add to every row, or None to add none. A
    member that drafts nothing takes no acceptance; a speculative one takes the
    request's ``accept_rate`` or the trace's own column (a source tagged
    ``speculative``), never both, since one would silently override the other.
    The simulator checks the values against the worker (``plan_run``); ``draft``
    is its draft width, when known, for the message."""
    if not _drafts(member):
        if workload.accept_rate is not None:
            raise BadWorkload("accept_rate applies only to a speculative worker")
        return None
    own = "speculative" in source.input_file_tags
    rate = workload.accept_rate
    if rate is None:
        if own:
            return None
        width = f"{draft}, " if draft else ""
        raise BadWorkload(
            "a speculative worker needs workload.accept_rate: one probability, or "
            f"{width}one per draft position; or upload a trace tagged speculative with "
            "its own accept_rate column"
        )
    if own:
        raise BadWorkload(
            f"{source.name} carries its own accept_rate column; leave workload.accept_rate unset"
        )
    return json.dumps(rate) if isinstance(rate, list) else repr(float(rate))


def _fit_draft_window(bounds: list[dict] | None, rows: list[dict]) -> int:
    """Shorten, in place, each request a speculative worker could not finish
    only because its last verify reads ``draft_tokens`` past the request: the
    simulator verifies a fixed width, where vLLM drafts fewer tokens near
    ``max_model_len`` and serves it. ``bounds`` are the simulator's (the pools'
    ``max_model_len`` and ``draft_tokens``, :attr:`SimMember.bounds`). Such a
    request loses at most ``draft_tokens`` tokens, from its output while it
    keeps one, then its input. A request longer than ``max_model_len`` itself, or
    one with too little input to lose, is left for the simulator to refuse.
    Returns how many requests were shortened."""
    drafting = [b for b in bounds or [] if b["draft_tokens"]]
    if not drafting:
        return 0
    pool = min(drafting, key=lambda b: b["max_model_len"] - b["draft_tokens"])
    bound, draft = pool["max_model_len"], pool["draft_tokens"]
    shortened = 0
    for row in rows:
        output, given = int(row["output_len"]), int(row["input_len"])
        over = _prefix(row) + given + output + draft - bound
        cut = min(over, output - 1)
        if not 0 < over <= draft or over - cut >= given:
            continue
        row["output_len"] = str(output - cut)
        row["input_len"] = str(given - (over - cut))
        shortened += 1
    return shortened


def _prefix(row: dict) -> int:
    """The prefix the simulator counts toward a request's length, as its trace
    reader takes it (``sim/frontend/schema.rs``): a chained round's
    ``prefix_len``, or an independent row's ``prefix_kv`` when it names a
    session."""
    if row.get("prefix_len"):
        return int(row["prefix_len"])
    if row.get("session_id") and row.get("prefix_kv"):
        return int(row["prefix_kv"])
    return 0


def write_trace(
    member: SimMember,
    source: TraceSource,
    workload: Workload,
    out: Path,
    bounds: list[dict] | None,
) -> tuple[list[str], int]:
    """Write the trace this run replays to ``out``: the source's rows, every
    column kept, with an ``accept_rate`` column for a speculative worker. A
    worker that drafts nothing runs a speculative trace (a speculative
    capture's, which records its acceptance) without that column and tag: the
    acceptance belongs to the capture's proposer, not to the requests.
    A request that misses a speculative worker's draft window by a few tokens
    is shortened to fit the pools' ``bounds`` (:func:`_fit_draft_window`). Returns the trace's tags
    and how many requests were shortened; raises :class:`BadWorkload` on a
    trace the member cannot serve."""
    with open(source.path, newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    check_count(source.name, len(rows))
    accept = _accept_rate(member, workload, source, _draft_tokens(bounds))
    tags = list(source.input_file_tags)
    if not _drafts(member) and "speculative" in tags:
        tags.remove("speculative")
        fields.remove("accept_rate")
        rows = [{k: v for k, v in row.items() if k != "accept_rate"} for row in rows]
    shortened = _fit_draft_window(bounds, rows)
    if accept:
        fields.append("accept_rate")
        tags.append("speculative")
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row | ({"accept_rate": accept} if accept else {}))
    return tags, shortened


def workload_block(workload: Workload, trace: TraceSource, facts: dict) -> dict:
    """The run config's ``workload`` for this request. ``workload.load`` in the
    simulator's terms: a rate is the trace-timed replay's speed-up, the asked
    rate over the trace's own (``facts``, :func:`trace_facts`); a concurrency
    is a saturated replay capped at it."""
    load = workload.load
    block: dict[str, Any] = read_block(trace) | {"run_to_end": workload.run_to_end}
    del block["duration_ms"]
    if load is not None and load.concurrency is not None:
        block["arrival_mode"] = "saturated"
        block["max_concurrency"] = load.concurrency
    elif load is not None:
        if facts["rate"] is None:
            raise BadWorkload(
                f"{trace.name} has no rate of its own to scale: its requests all arrive "
                "at once; give load.concurrency"
            )
        block["request_rate"] = load.rate / facts["rate"]
    if workload.duration_ms is not None:
        block["duration_ms"] = workload.duration_ms
    elif not workload.run_to_end:
        raise BadWorkload("run_to_end false needs duration_ms")
    return block


# -- results -------------------------------------------------------------------


def summary(log_dir: Path) -> dict | None:
    """A finished run's headline numbers: the run's ``summary.json`` (throughput)
    and the Analyzer's ``slo-general`` report (TTFT, TPOT, E2E percentiles, ms)."""
    path = log_dir / "summary.json"
    if not path.is_file():
        return None
    run = json.loads(path.read_text())
    out: dict[str, Any] = {
        "cause": run.get("cause"),
        "requests": {"total": run.get("requests_total"), "finished": run.get("requests_finished")},
        "sim_ms": run.get("sim_ms"),
        "num_gpus": run.get("num_gpus"),
        "throughput": {
            key: run.get(key)
            for key in (
                "total_tok_s",
                "prefill_tok_s",
                "decode_tok_s",
                "total_tok_s_per_gpu",
                "completed_req_s",
            )
        },
    }
    slo = log_dir / "reports" / "slo_general_report.json"
    if slo.is_file():
        metrics = json.loads(slo.read_text()).get("metrics", {})
        for name in ("ttft", "tpot", "e2e"):
            out[f"{name}_ms"] = metrics.get(name)
    return out


# -- the queue -----------------------------------------------------------------


def _descendants(pid: int) -> list[int]:
    """Every live descendant of ``pid``, by parent link: the launcher starts
    each stage in its own session, so a process group does not reach them."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as stream:
                parent = int(stream.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(parent, []).append(int(entry))
    out, pending = [], [pid]
    while pending:
        for child in children.get(pending.pop(), []):
            out.append(child)
            pending.append(child)
    return out


def kill_tree(process: subprocess.Popen) -> None:
    """Stop a run's process and everything it started: SIGTERM, then SIGKILL."""
    pids = [process.pid, *_descendants(process.pid)]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            continue
        pids = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
        if not pids:
            return


def _now() -> float:
    return time.time()


def _iso(stamp: float | None) -> str | None:
    if stamp is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp))


@dataclass
class Simulation:
    id: str
    directory: Path
    request: dict
    status: str = "queued"  # queued, running, done, failed, timed_out, cancelled
    created_at: float = field(default_factory=_now)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    run_id: str | None = None
    process: subprocess.Popen | None = None

    def record(self) -> dict:
        return {
            "simulation_id": self.id,
            "status": self.status,
            **self.request,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "error": self.error,
            "run_id": self.run_id,
            "_stamps": [self.created_at, self.started_at, self.finished_at],
        }

    def save(self) -> None:
        path = self.directory / _RECORD
        tmp = path.with_suffix(".tmp")
        # The request again as it was asked, so a restart reads it back whole.
        tmp.write_text(json.dumps(self.record() | {"_request": self.request}, indent=1))
        os.replace(tmp, path)


def default_command(run_dir: Path, build_type: str) -> list[str]:
    return [sys.executable, "-m", "public_api.simulate_run", str(run_dir), build_type]


class Simulations:
    """The simulation queue: at most ``max_running`` runs at once, each killed
    and marked ``timed_out`` after ``timeout_s``; every run's directory and
    record removed ``keep_s`` after it was created."""

    def __init__(
        self,
        runs_dir: Path,
        build_type: str = "release",
        *,
        max_running: int = MAX_RUNNING,
        max_queued: int = MAX_QUEUED,
        timeout_s: float = TIMEOUT_S,
        keep_s: float = KEEP_S,
        command: Callable[[Path, str], list[str]] = default_command,
        run_id: Callable[[str], str | None] | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.build_type = build_type
        self.max_running = max_running
        self.max_queued = max_queued
        self.timeout_s = timeout_s
        self.keep_s = keep_s
        self.command = command
        self.find_run_id = run_id
        self._lock = threading.Condition()
        self._sims: dict[str, Simulation] = {}
        self._queue: list[str] = []
        self._load()
        self._workers = [
            threading.Thread(target=self._work, name=f"simulation-{n}", daemon=True)
            for n in range(max_running)
        ]
        for worker in self._workers:
            worker.start()

    def limits(self) -> dict:
        return {
            "max_running": self.max_running,
            "max_queued": self.max_queued,
            "max_requests": workloads.MAX_REQUESTS,
            "timeout_s": self.timeout_s,
            "keep_s": self.keep_s,
        }

    def _load(self) -> None:
        """Pick up the runs a previous service left: a finished one stays
        readable until it expires; one it was still running or queueing failed."""
        for record in self.runs_dir.glob(f"*/{_RECORD}"):
            try:
                data = json.loads(record.read_text())
                created, started, finished = data.pop("_stamps")
                request = data.pop("_request")
            except (OSError, ValueError, KeyError):
                continue
            sim = Simulation(
                id=data["simulation_id"],
                directory=record.parent,
                request=request,
                status=data["status"],
                created_at=created,
                started_at=started,
                finished_at=finished,
                error=data.get("error"),
                run_id=data.get("run_id"),
            )
            if sim.status in ("queued", "running"):
                sim.status, sim.error = "failed", "the service restarted before it finished"
                sim.finished_at = _now()
                sim.save()
            self._sims[sim.id] = sim
        self.prune()

    def prune(self) -> None:
        """Remove the runs older than ``keep_s``, and any directory without a record."""
        cutoff = _now() - self.keep_s
        with self._lock:
            expired = [
                sim
                for sim in self._sims.values()
                if sim.created_at < cutoff and sim.status not in ("queued", "running")
            ]
            for sim in expired:
                del self._sims[sim.id]
            known = {sim.directory for sim in self._sims.values()}
        for sim in expired:
            shutil.rmtree(sim.directory, ignore_errors=True)
        for directory in self.runs_dir.iterdir():
            if directory.is_dir() and directory not in known and directory.stat().st_mtime < cutoff:
                shutil.rmtree(directory, ignore_errors=True)

    def new_directory(self) -> tuple[str, Path]:
        """A fresh id and its run directory, for :meth:`submit` to queue."""
        simulation_id = uuid.uuid4().hex
        directory = self.runs_dir / simulation_id
        directory.mkdir()
        return simulation_id, directory

    def submit(self, simulation_id: str, directory: Path, request: dict) -> Simulation:
        """Queue a run whose directory holds its :data:`RUN_CONFIG` and :data:`RUN_PRESET`."""
        self.prune()
        with self._lock:
            if len(self._queue) >= self.max_queued:
                raise QueueFull(f"{len(self._queue)} simulations are already waiting")
            sim = Simulation(id=simulation_id, directory=directory, request=request)
            sim.save()
            self._sims[sim.id] = sim
            self._queue.append(sim.id)
            self._lock.notify()
        return sim

    def get(self, simulation_id: str) -> dict:
        with self._lock:
            sim = self._sims.get(simulation_id)
            if sim is None:
                raise UnknownSimulation(simulation_id)
            status = sim.status
            position = self._queue.index(sim.id) if status == "queued" else None
        if status == "done" and sim.run_id is None and self.find_run_id is not None:
            sim.run_id = self.find_run_id(sim.id)
            if sim.run_id:
                sim.save()
        out = {k: v for k, v in sim.record().items() if not k.startswith("_")}
        if position is not None:
            out["queue_position"] = position
        if status == "done":
            out["summary"] = summary(sim.directory)
        return out

    def delete(self, simulation_id: str) -> None:
        """Cancel a queued or running run and remove it."""
        with self._lock:
            sim = self._sims.pop(simulation_id, None)
            if sim is None:
                raise UnknownSimulation(simulation_id)
            if sim.id in self._queue:
                self._queue.remove(sim.id)
            process = sim.process
            if sim.status in ("queued", "running"):
                sim.status = "cancelled"
        if process is not None and process.poll() is None:
            kill_tree(process)
        shutil.rmtree(sim.directory, ignore_errors=True)

    def _work(self) -> None:
        while True:
            with self._lock:
                while not self._queue:
                    self._lock.wait()
                sim = self._sims[self._queue.pop(0)]
                sim.status, sim.started_at = "running", _now()
                try:
                    with (sim.directory / "launcher.log").open("w") as log:
                        sim.process = subprocess.Popen(
                            self.command(sim.directory, self.build_type),
                            cwd=REPO_ROOT,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                except OSError as error:
                    sim.status, sim.error, sim.finished_at = "failed", str(error), _now()
                    sim.save()
                    continue
                sim.save()
            self._finish(sim)

    def _finish(self, sim: Simulation) -> None:
        try:
            code = sim.process.wait(timeout=self.timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            kill_tree(sim.process)
            code, timed_out = sim.process.returncode, True
        with self._lock:
            sim.process = None
            if sim.status != "running":  # deleted meanwhile
                return
            sim.finished_at = _now()
            if timed_out:
                sim.status = "timed_out"
                sim.error = f"stopped after the {self.timeout_s:.0f} s wall-clock limit"
            elif code == 0 and (sim.directory / "summary.json").is_file():
                sim.status = "done"
            else:
                sim.status = "failed"
                sim.error = self._failure(sim.directory, code)
            if sim.directory.is_dir():
                sim.save()

    @staticmethod
    def _failure(directory: Path, code: int | None) -> str:
        logs = (directory / "stdout.log", directory / "launcher.log")
        texts = (log.read_text(errors="replace").strip() for log in logs if log.is_file())
        error = binary_error(directory / "raw" / ERROR_JSON)
        cause = _cause(error, next((text for text in texts if text), ""), directory)
        return cause[-2000:] if cause else f"the run exited with {code}"


# -- the service ---------------------------------------------------------------


def analyzer_run_id(analyzer: str, simulation_id: str) -> str | None:
    """The Analyzer's id of a finished simulation's run: the run whose path
    under the simulations directory is its id."""
    import httpx

    try:
        catalog = httpx.get(f"{analyzer}/api/analyzer/v1/runs", timeout=30).json()
    except (httpx.HTTPError, ValueError):
        return None
    for run in catalog.get("runs", []):
        if run.get("display_name") == simulation_id:
            return run["run_id"]
    return None


def routing(member: SimMember, capture: Capture, source: str) -> dict:
    """Where a simulation's requests and its expert routing come from: the
    capture it routes as (none for a dense member), and the answer's label."""
    requests = f"capture {capture.name}" if source == "capture" else "your workload"
    if member.dense:
        return {"capture": None, "label": f"requests from {requests}; the model routes no experts"}
    if source == "capture":
        label = f"requests and routing from capture {capture.name}"
    else:
        label = f"requests from your workload, routing from capture {capture.name}"
    return {"capture": capture.name, "label": label}


@dataclass
class SimulationService:
    """What ``/simulations``, ``/simulate`` and ``/workloads`` answer from: the
    sim presets (built and checked), the simulator's schema, the queue and the
    generated and uploaded workloads."""

    sims: Any  # SimIndex
    registry: Registry
    queue: Simulations
    workloads: Workloads
    # Each capture trace's `trace_facts`, by its reference: read once.
    _facts: dict[str, dict] = field(default_factory=dict)

    def capture_facts(self, reference: str) -> dict:
        """What a capture's trace holds (:func:`trace_facts`), as the simulator
        reads it."""
        if reference not in self._facts:
            source = self._capture_source(reference, reference)
            facts = trace_facts(plan(read_block(source), self.queue.build_type))
            self._facts[reference] = facts | {"input_file_tags": list(source.input_file_tags)}
        return self._facts[reference]

    def _capture_source(self, reference: str, name: str) -> TraceSource:
        return trace_source(Path(resolve_reference(reference)), name, self.queue.build_type)

    def presets(self) -> dict:
        catalog = self.sims.catalog()
        for preset in catalog:
            for capture in preset["captures"]:
                capture["facts"] = self.capture_facts(capture["trace"])
        return {
            "sim_commit": self.sims.index.sim_commit,
            "limits": self.queue.limits(),
            "workload": Workload.model_json_schema(),
            "presets": catalog,
        }

    def describe_workloads(self) -> dict:
        return describe(self.workloads, self.queue.limits(), Workload.model_json_schema())

    def _source(self, workload: Workload, capture: Capture, directory: Path) -> TraceSource:
        """The file this request's requests come from."""
        given = {"generator": workload.generator, "upload": workload.upload}
        stray = [name for name, value in given.items() if value is not None]
        wanted = {"capture": [], "generated": ["generator"], "upload": ["upload"]}[workload.source]
        if stray != wanted:
            needs = f"needs workload.{wanted[0]}" if wanted else "takes no generator or upload"
            article = "an" if workload.source == "upload" else "a"
            raise BadWorkload(f"{article} {workload.source} workload {needs}")
        if workload.source == "generated":
            return self.workloads.generate(workload.generator, directory / _GENERATED)
        if workload.source == "upload":
            return self.workloads.uploaded(workload.upload)
        return self._capture_source(capture.trace, f"capture {capture.name}")

    def start(self, preset: str, params: dict, workload: Workload) -> dict:
        """Check the request, write its run directory and queue it."""
        index = self.sims.index
        member = self.sims.member(preset, params)
        try:
            capture = member.capture(workload.capture)
        except KeyError:
            names = [capture.name for capture in member.captures]
            raise BadWorkload(f"{preset} has no capture {workload.capture!r}; it has {names}")
        replayed = workload.source == "capture"
        blocker = member.blocker(capture, replayed=replayed)
        if blocker is not None and "misfit" in blocker:
            # The capture's own requests do not fit: refused as any trace
            # that does not fit is, with the limit.
            too_long = {k: v for k, v in blocker["misfit"].items() if k != "reason"}
            raise BadWorkload(blocker["misfit"]["reason"], too_long or None)
        reason = member.runnable(capture, replayed=replayed)
        if reason is not None:
            raise NotRunnable(f"{preset} {member.params} on {capture.name}: {reason}")
        simulation_id, directory = self.queue.new_directory()
        try:
            source = self._source(workload, capture, directory)
            path = directory / _TRACE
            tags, shortened = write_trace(member, source, workload, path, member.bounds)
            trace = TraceSource(path, source.input_file_format, tuple(tags), source.name)
            facts = trace_facts(plan(read_block(trace), self.queue.build_type))
            block = workload_block(workload, trace, facts)
            tree = run_tree(index, member, capture, block, directory)
            config = concrete(tree, self.registry)
            plan_run(config, self.queue.build_type)
            (directory / RUN_CONFIG).write_text(json.dumps(config, indent=1))
            (directory / RUN_PRESET).write_text(json.dumps(tree, indent=1))
            routed = routing(member, capture, workload.source)
            # The capture the run reads: its trace, its routing, or both.
            used = capture.name if workload.source == "capture" else routed["capture"]
            request = {
                "preset": preset,
                "params": member.params,
                "workload": workload.model_dump()
                | {"capture": used, "trace": facts, "shortened": shortened},
                "routing": routed,
                "gpus": member.summary(index)["gpus"],
            }
            sim = self.queue.submit(simulation_id, directory, request)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return {"simulation_id": sim.id, "status": sim.status, "routing": routed}
