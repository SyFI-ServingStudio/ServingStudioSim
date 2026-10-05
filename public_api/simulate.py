"""``POST /simulate``: one sim preset member run on a reader's workload.

A request names a member of a sim preset (:mod:`public_api.sim_preset`) and a
workload. Today the workload is a capture the member offers (its arch preset's
``workload`` row), whose ``trace.csv`` the run replays and whose routing file
its MoE arch reads, plus the knobs below; generated and uploaded workloads will
be other ``source`` values.

The service checks the request before queueing it, so a run never starts on
input that would fail it: the member builds and is measured, every request fits
the arch's ``max_model_len``, a speculative worker gets an ``accept_rate``.
Then it writes the run's directory under the service's simulations directory:
the trace it replays (the capture's first ``num_requests`` rows, with the
acceptance column a speculative worker reads) and the concrete run config. A
queued run is the launcher's standard single run (``launcher.sweep.run_single``,
analyzed, without plots) in a child process (:mod:`public_api.simulate_run`),
at most ``max_running`` at once. The Analyzer reads the finished directory; the
answer's summary is the run's own ``summary.json`` and the Analyzer's
``slo-general`` report.
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

from pydantic import BaseModel, Field

from launcher.corpus import resolve_hf_references, resolve_reference
from launcher.exec import ERROR_JSON, _build_subprocess_env, binary_error, binary_path
from launcher.process.leases import LauncherLeases
from launcher.schema import (
    _format_log_dir,
    expand_sweep_params,
    normalize_params,
    validate_expanded,
    validate_params,
)
from launcher.schema.argv import write_config
from launcher.schema.loader import Registry
from public_api.deployments import DeploymentIndex
from public_api.predict import _cause, missing_by_role
from public_api.sim_preset import Capture, SimMember

REPO_ROOT = Path(__file__).resolve().parents[1]

MAX_REQUESTS = 2000
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


class BadWorkload(ValueError):
    """A workload this member cannot run; the message says why."""


class NotRunnable(RuntimeError):
    """The member does not build, or lacks profile.db rows, for this capture."""


class QueueFull(RuntimeError):
    """Too many simulations are waiting."""


class UnknownSimulation(LookupError):
    """No simulation has this id (or it was removed)."""


class Workload(BaseModel):
    """What a simulation replays. ``source`` says where its requests come from;
    only ``capture`` exists today."""

    source: Literal["capture"] = "capture"
    # A capture the member offers (`/simulations/presets`); its first by default.
    capture: str | None = None
    # The capture's first `num_requests` rows; all of them by default.
    num_requests: int | None = Field(default=None, ge=1)
    # `trace_timed` replays each request at arrival_time / request_rate;
    # `saturated` releases every request at once.
    arrival_mode: Literal["trace_timed", "saturated"] = "trace_timed"
    request_rate: float = Field(default=1.0, gt=0)
    max_concurrency: int | None = Field(default=None, ge=1)
    # Simulated time to run (ms); with run_to_end, the least it runs.
    duration_ms: float | None = Field(default=None, gt=0)
    run_to_end: bool = True
    session_dependency: Literal["independent", "chained"] = "independent"
    # A speculative worker's per-request acceptance: one probability for every
    # draft position, or one per position (draft_tokens of them).
    accept_rate: float | list[float] | None = None


# -- run configs ---------------------------------------------------------------


def _speculative_draft(member: SimMember) -> int | None:
    """The draft width of the member's speculative worker; None without one."""
    for pool in member.pools.values():
        if pool["worker"]["type"] == "speculative":
            return int(pool["worker"].get("draft_tokens", 5))
    return None


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
    simulator's schema, every default filled, captures fetched to local files.
    The launcher's own expansion of one preset, without the sweep."""
    errors = validate_params(tree, registry)
    if errors:
        raise BadWorkload("; ".join(errors))
    (candidate,) = expand_sweep_params(tree, registry)
    errors = validate_expanded(candidate, registry)
    if errors:
        raise BadWorkload("; ".join(errors))
    return resolve_hf_references(_format_log_dir(normalize_params(candidate, registry)))


def missing_rows(
    index: DeploymentIndex,
    member: SimMember,
    capture: Capture,
    registry: Registry,
    build_type: str = "release",
) -> dict[str, int]:
    """Build ``member`` on ``capture`` the way a run does (``simulator
    dry-run``: the deployment's ``build_flow``, nothing simulated) and return
    the profile.db rows its kernels lack, ``{kernel role: count}``. Raises with
    the simulator's message when it does not build."""
    workload = {
        "trace_files": [capture.trace],
        "input_file_format": "text-generation-independent",
        "input_file_tags": ["speculative"] if _speculative_draft(member) else [],
        "arrival_mode": "trace_timed",
        "session_dependency": "independent",
        "run_to_end": True,
        "request_rate": 1.0,
    }
    with tempfile.TemporaryDirectory(prefix="public-sim-check-") as directory:
        scratch = Path(directory)
        config = concrete(run_tree(index, member, capture, workload, scratch), registry)
        path = write_config(config, scratch / "run_config.yaml")
        report, error = scratch / "dry_run_report.json", scratch / ERROR_JSON
        argv = [str(binary_path(build_type)), "dry-run", str(path)]
        argv += ["--report-json", str(report), "--error-json", str(error)]
        env = {**_build_subprocess_env(), "RUST_LOG": "warn"}
        with _LEASES.profile_database(write=False):
            result = subprocess.run(
                argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False
            )
        if result.returncode:
            raise RuntimeError(_cause(binary_error(error), result.stderr, scratch))
        return missing_by_role(json.loads(report.read_text()))


# -- workloads -----------------------------------------------------------------


def _max_model_lens(index: DeploymentIndex, member: SimMember, capture: Capture) -> dict:
    """Each pool's arch ``max_model_len``, for the archs that have one."""
    out = {}
    for role in member.pools:
        limit = member.arch_member(index, role, capture).arch.get("max_model_len")
        if limit is not None:
            out[role] = int(limit)
    return out


def _accept_rate(member: SimMember, workload: Workload) -> str | None:
    """The trace's ``accept_rate`` cell, or None for a member that drafts nothing."""
    draft = _speculative_draft(member)
    if draft is None:
        if workload.accept_rate is not None:
            raise BadWorkload("accept_rate applies only to a speculative worker")
        return None
    rate = workload.accept_rate
    if rate is None:
        raise BadWorkload(
            f"a speculative worker drafting {draft} tokens needs workload.accept_rate: "
            f"one probability, or {draft}, one per draft position"
        )
    rates = rate if isinstance(rate, list) else [rate]
    if isinstance(rate, list) and len(rate) != draft:
        raise BadWorkload(f"accept_rate has {len(rate)} positions; this worker drafts {draft}")
    if any(not 0.0 <= r <= 1.0 for r in rates):
        raise BadWorkload("accept_rate probabilities must be between 0 and 1")
    return json.dumps(rate) if isinstance(rate, list) else repr(float(rate))


def write_trace(
    index: DeploymentIndex,
    member: SimMember,
    capture: Capture,
    workload: Workload,
    out: Path,
) -> int:
    """Write the trace this run replays to ``out``: the capture's first
    ``num_requests`` rows, with an ``accept_rate`` column for a speculative
    worker. Returns the row count; raises :class:`BadWorkload` on a trace the
    member cannot serve."""
    if workload.session_dependency == "chained":
        raise BadWorkload("session_dependency chained needs a session trace; a capture has none")
    with open(resolve_reference(capture.trace), newline="") as stream:
        rows = list(csv.DictReader(stream))
    count = len(rows) if workload.num_requests is None else workload.num_requests
    if count > len(rows):
        raise BadWorkload(f"capture {capture.name} has {len(rows)} requests, not {count}")
    if count > MAX_REQUESTS:
        raise BadWorkload(f"at most {MAX_REQUESTS} requests per simulation")
    rows = rows[:count]
    accept = _accept_rate(member, workload)
    # A speculative request's last verify reads up to draft_tokens past its output.
    extra = _speculative_draft(member) or 0
    for role, limit in _max_model_lens(index, member, capture).items():
        over = [
            row["id"]
            for row in rows
            if int(row["input_len"]) + int(row["output_len"]) + extra > limit
        ]
        if over:
            fit = "input_len + output_len" + (f" + {extra} draft tokens" if extra else "")
            raise BadWorkload(
                f"{len(over)} of {count} requests exceed pool {role}'s max_model_len {limit} "
                f"({fit}), first ids {over[:5]}; pick a member with a longer max_model_len "
                "or fewer requests"
            )
    fields = ["id", "input_len", "output_len", "arrival_time"]
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, [*fields, *(["accept_rate"] if accept else [])])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: row[name] for name in fields} | ({"accept_rate": accept} if accept else {})
            )
    return count


def workload_block(member: SimMember, workload: Workload, trace: Path) -> dict:
    """The run config's ``workload`` for this request."""
    block: dict[str, Any] = {
        "trace_files": [str(trace)],
        "input_file_format": "text-generation-independent",
        "input_file_tags": ["speculative"] if _speculative_draft(member) else [],
        "arrival_mode": workload.arrival_mode,
        "request_rate": workload.request_rate,
        "run_to_end": workload.run_to_end,
        "session_dependency": workload.session_dependency,
    }
    if workload.max_concurrency is not None:
        block["max_concurrency"] = workload.max_concurrency
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
        tmp.write_text(json.dumps(self.record(), indent=1))
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
            "max_requests": MAX_REQUESTS,
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
            except (OSError, ValueError, KeyError):
                continue
            request = {
                key: data[key] for key in ("preset", "params", "workload", "gpus") if key in data
            }
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


@dataclass
class SimulationService:
    """What ``/simulations`` and ``/simulate`` answer from: the sim presets
    (built and checked), the simulator's schema and the queue."""

    sims: Any  # SimIndex
    registry: Registry
    queue: Simulations

    def presets(self) -> dict:
        return {
            "sim_commit": self.sims.index.sim_commit,
            "limits": self.queue.limits(),
            "workload": Workload.model_json_schema(),
            "presets": self.sims.catalog(),
        }

    def start(self, preset: str, params: dict, workload: Workload) -> dict:
        """Check the request, write its run directory and queue it."""
        index = self.sims.index
        member = self.sims.member(preset, params)
        try:
            capture = member.capture(workload.capture)
        except KeyError:
            names = [capture.name for capture in member.captures]
            raise BadWorkload(f"{preset} has no capture {workload.capture!r}; it has {names}")
        reason = member.runnable(capture)
        if reason is not None:
            raise NotRunnable(f"{preset} {member.params} on {capture.name}: {reason}")
        simulation_id, directory = self.queue.new_directory()
        try:
            trace = directory / _TRACE
            count = write_trace(index, member, capture, workload, trace)
            tree = run_tree(
                index, member, capture, workload_block(member, workload, trace), directory
            )
            config = concrete(tree, self.registry)
            (directory / RUN_CONFIG).write_text(json.dumps(config, indent=1))
            (directory / RUN_PRESET).write_text(json.dumps(tree, indent=1))
            request = {
                "preset": preset,
                "params": member.params,
                "workload": workload.model_dump()
                | {"capture": capture.name, "num_requests": count},
                "gpus": member.summary(index)["gpus"],
            }
            sim = self.queue.submit(simulation_id, directory, request)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return {"simulation_id": sim.id, "status": sim.status}
