"""`alignment-campaign run --phase P` — one phase across every ready case.

## The batch dimension is a phase, never a pipeline

`alignment/README.md` states the invariant: "No phase implicitly launches the
next one." That forbids **chaining phases**, not running one phase across many
cases — and `skills/operate-run-alignment` positively requires the latter
("complete shared prerequisites once, then fan the cases out and collect their
exit codes separately; do not serialize the campaign for attribution's sake").

So there is a `--phase` and there is no `--all`. Every arrow between phases stays
a human checkpoint; what gets automated is typing the same command fifteen times.
Simulation reads its worker settings directly from the rendered preset.
A case with `from_capture` inputs gets that preset written by `run` itself, from
its workload pass, right before `timing_predict` or `simulation` launches
(`render.materialize`); until that pass has run, both phases are blocked with
the reason.

## Readiness before spend

A phase's inputs are checked before it is launched, and a case that is missing
one is reported with what it lacks rather than being allowed to fail inside an
expensive subprocess. Completion is `.complete` (INV-5, `process/markers.py`)
**and** the artifacts the phase is supposed to have produced — a marker left by
an interrupted phase must not suppress the re-run. Per-phase markers are also
what implements "never re-run an expensive GPU profile because labeling or
analysis failed afterwards": a later failure cannot clear an earlier marker.

Scheduling is delegated, per the same skill: `ResourceScheduler` for in-process
budgets (GPU-exclusive phases take the simulation slot, analysis phases the
analysis slot), `LauncherLeases` for cross-process exclusion, `RunJournal` for
per-case stage state and exit codes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..process import ProcessResult, ProcessSpec, ProcessSupervisor
from ..process.journal import RunJournal, StageState
from ..process.leases import LauncherLeases
from ..process.markers import clear_marker, has_marker, mark_complete
from . import calibration
from .pack import REPO_ROOT, Case, Pack, PackError, Variant
from .render import (
    ANALYSIS_E2E_PHASE,
    ANALYSIS_KERNEL_PHASE,
    SIMULATION_PHASE,
    TIMING_PREDICT_PHASE,
    config_stem,
    materialize,
    phase_names,
)

STALE_SIMULATION = (
    "re-run: simulation.yaml or calibration.json changed since this simulation ran"
)

#: Phases that read `simulation.yaml`, so they wait for a case's from_capture
#: inputs and get that preset written right before they launch.
CAPTURE_DEPENDENT_PHASES = frozenset({TIMING_PREDICT_PHASE, SIMULATION_PHASE})

LABELED_SEQUENCES_NAME = "kernel_sequences_labeled.json"
_PROCESS_SUPERVISOR = ProcessSupervisor()

#: Which concurrency budget a phase draws from. GPU-exclusive work takes the
#: simulation slot even when it is a profile rather than a simulation: the
#: distinction the budget encodes is "owns a device" versus "reads artifacts".
GPU_PHASES = frozenset({SIMULATION_PHASE})

#: Artifacts that prove a phase ran, keyed by phase kind. Checked alongside the
#: marker so an interrupted phase is re-run rather than skipped.
_PROFILE_ARTIFACTS = {
    "nsys": ("profile_result.json", "parsed.json", "kernel_sequences.json"),
    "workload_metrics": ("profile_result.json",),
    "expert_popularity": ("profile_result.json",),
    "token_corpus": ("profile_result.json", "token_corpus/manifest.json"),
}
_PIPELINE_ARTIFACTS = {
    TIMING_PREDICT_PHASE: ("timing_predict_input_manifest.json", "prediction.cases.json"),
    ANALYSIS_KERNEL_PHASE: ("alignment_manifest.json", "reports"),
    SIMULATION_PHASE: ("summary.json",),
    ANALYSIS_E2E_PHASE: ("alignment_manifest.json", "reports"),
}


@dataclass(frozen=True)
class PhasePlan:
    """One case's disposition for the requested phase."""

    case_slug: str
    phase: str
    directory: Path
    state: str  # "ready" | "complete" | "blocked" | "missing"
    reasons: tuple[str, ...] = ()
    command: tuple[str, ...] = ()

    @property
    def selected(self) -> bool:
        return self.state == "ready"

    def describe(self) -> str:
        if self.state == "ready":
            note = f"  ({'; '.join(self.reasons)})" if self.reasons else ""
            return f"  [run]      {self.case_slug}  {shlex.join(self.command)}{note}"
        if self.state == "complete":
            return f"  [skip]     {self.case_slug}  already complete (--refresh to re-run)"
        detail = "; ".join(self.reasons) or "unknown"
        label = "missing" if self.state == "missing" else "blocked"
        return f"  [{label}]{'':<{max(0, 4 - len(label))}} {self.case_slug}  {detail}"


@dataclass
class PhaseResult:
    plan: PhasePlan
    returncode: int | None = None
    skipped: bool = False


@dataclass
class RunReport:
    phase: str
    plans: tuple[PhasePlan, ...]
    results: list[PhaseResult] = field(default_factory=list)

    @property
    def failures(self) -> list[PhaseResult]:
        return [item for item in self.results if item.returncode not in (0, None)]

    @property
    def blocked(self) -> list[PhasePlan]:
        return [item for item in self.plans if item.state in ("blocked", "missing")]


# ── completion + readiness ───────────────────────────────────────────────────


def _artifacts_for(variant: Variant, phase: str) -> tuple[str, ...]:
    profile_pass = variant.pass_named(phase)
    if profile_pass is not None:
        return _PROFILE_ARTIFACTS[profile_pass.kind]
    return _PIPELINE_ARTIFACTS.get(phase, ())


#: Artifacts a profile names in its `profile_result.json` only when it wrote
#: them. A schema-6 NSYS parse keeps its kernel rows beside `parsed.json`,
#: which is unreadable without them; an older capture has no such sibling and
#: names none, so it is still complete.
_RECORDED_PROFILE_ARTIFACTS = {"nsys": ("parsed_kernel_rows",)}


def _recorded_artifacts_exist(directory: Path, kind: str) -> bool:
    keys = _RECORDED_PROFILE_ARTIFACTS.get(kind, ())
    if not keys:
        return True
    try:
        result = json.loads((directory / "profile_result.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(result, dict):
        return False
    # By name inside this phase directory: the recorded path is absolute, and a
    # moved or copied campaign must not look incomplete and re-profile.
    return all(
        isinstance(result[key], str) and (directory / Path(result[key]).name).is_file()
        for key in keys
        if result.get(key) is not None
    )


#: Written into the simulation directory when a campaign-launched simulation
#: succeeds: the sha256 of the preset and of `calibration.json` it ran with,
#: and a copy of that calibration. Re-rendering after a new workload pass
#: rewrites both inputs, and a simulation that read the old ones must neither
#: count as complete nor lend its report the new calibration.
SIMULATION_INPUTS_NAME = "campaign_inputs.json"


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def simulation_inputs(case_dir: Path, variant: Variant) -> dict:
    """The inputs a simulation of this case would run with right now."""
    case_dir = Path(case_dir)
    preset = case_dir / f"{config_stem(variant, SIMULATION_PHASE)}.yaml"
    calibration_path = case_dir / calibration.CALIBRATION_NAME
    return {
        "schema_version": 1,
        "simulation_yaml_sha256": _sha256(preset),
        "calibration_sha256": _sha256(calibration_path),
    }


def recorded_simulation_inputs(case_dir: Path) -> dict | None:
    """What the campaign recorded when this case's simulation succeeded."""
    path = Path(case_dir) / SIMULATION_PHASE / SIMULATION_INPUTS_NAME
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text())
    except ValueError:
        return None
    return document if isinstance(document, dict) else None


def simulation_inputs_current(case_dir: Path, variant: Variant) -> bool:
    """Whether the simulation on disk ran with today's preset and calibration.

    A case with a `calibration.json` has a preset only the campaign writes, so
    its simulation needs the record. Any other case may have been simulated
    before the record existed; the marker alone still speaks for it then.
    """
    recorded = recorded_simulation_inputs(case_dir)
    current = simulation_inputs(case_dir, variant)
    if recorded is None:
        return current["calibration_sha256"] is None
    return all(recorded.get(key) == current[key] for key in
               ("simulation_yaml_sha256", "calibration_sha256"))


def _record_simulation_inputs(case_dir: Path, inputs: dict) -> None:
    document = dict(inputs)
    calibration_path = Path(case_dir) / calibration.CALIBRATION_NAME
    if inputs["calibration_sha256"] is not None and calibration_path.is_file():
        snapshot = json.loads(calibration_path.read_text())
        # Only if it is still the file that was hashed before launch.
        if _sha256(calibration_path) == inputs["calibration_sha256"]:
            document["calibration"] = snapshot
    (Path(case_dir) / SIMULATION_PHASE / SIMULATION_INPUTS_NAME).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )


def phase_complete(case_dir: Path, variant: Variant, phase: str) -> bool:
    """Exit status *and* artifacts. `simulation` already carries a marker written
    by the ordinary launcher, so the campaign reads it rather than writing it;
    it is complete only while its preset and calibration are the ones it ran
    with (`simulation_inputs_current`)."""
    directory = case_dir / phase
    if not directory.is_dir():
        return False
    artifacts = _artifacts_for(variant, phase)
    if artifacts and not all((directory / name).exists() for name in artifacts):
        return False
    profile_pass = variant.pass_named(phase)
    if profile_pass is not None and not _recorded_artifacts_exist(directory, profile_pass.kind):
        return False
    if phase == SIMULATION_PHASE:
        return has_marker(directory) and simulation_inputs_current(case_dir, variant)
    return has_marker(directory) or bool(artifacts)


def phase_requirements(
    case_dir: Path,
    variant: Variant,
    phase: str,
    *,
    pack: Pack | None = None,
    case: Case | None = None,
) -> list[str]:
    """What this phase still needs. Empty means ready.

    Deliberately expressed as *inputs of P*, never as "run the previous phase":
    the campaign must not be able to schedule a predecessor as a side effect of
    being asked for P. With `pack` and `case`, a from_capture input is one of
    those: it needs the workload pass's files, and `simulation.yaml` is then
    written at launch rather than required beforehand.
    """
    missing: list[str] = []
    stem = config_stem(variant, phase)
    captured = (
        pack is not None
        and case is not None
        and bool(case.capture_fields)
        and phase in CAPTURE_DEPENDENT_PHASES
    )
    if captured:
        missing += calibration.requirements(pack, case, case_dir)
    if not (case_dir / f"{stem}.yaml").is_file() and not (captured and phase == SIMULATION_PHASE):
        missing.append(f"{stem}.yaml not rendered")

    if variant.pass_named(phase) is not None:
        profile_pass = variant.pass_named(phase)
        trace = "trace_nsys.csv" if profile_pass.trace == "kernel" else "trace.csv"
        if not (case_dir / trace).is_file() and not (case_dir / "trace.csv").is_file():
            missing.append(f"{trace} not rendered")
        return missing

    kernel_pass = next((item for item in variant.profile_passes if item.kind == "nsys"), None)
    workload_pass = next(
        (item for item in variant.profile_passes if item.kind == "workload_metrics"), None
    )
    if phase == TIMING_PREDICT_PHASE:
        if kernel_pass is None or not phase_complete(case_dir, variant, kernel_pass.name):
            missing.append(f"{kernel_pass.name if kernel_pass else 'nsys profile'} not complete")
    elif phase == ANALYSIS_KERNEL_PHASE:
        if kernel_pass is None or not phase_complete(case_dir, variant, kernel_pass.name):
            missing.append(f"{kernel_pass.name if kernel_pass else 'nsys profile'} not complete")
        if not phase_complete(case_dir, variant, TIMING_PREDICT_PHASE):
            missing.append(f"{TIMING_PREDICT_PHASE} not complete")
        if not (case_dir / LABELED_SEQUENCES_NAME).is_file():
            missing.append(
                f"{LABELED_SEQUENCES_NAME} absent — run `alignment-campaign label` first"
            )
    elif phase == ANALYSIS_E2E_PHASE:
        if not phase_complete(case_dir, variant, SIMULATION_PHASE):
            missing.append(f"{SIMULATION_PHASE} not complete")
        if workload_pass is None or not phase_complete(case_dir, variant, workload_pass.name):
            missing.append(
                f"{workload_pass.name if workload_pass else 'workload profile'} not complete"
            )
    return missing


def phase_command(
    case_dir: Path,
    variant: Variant,
    phase: str,
    *,
    resume: bool,
    refresh: bool,
    runtime: str = "image",
    allow_autotune: bool = False,
) -> tuple[str, ...]:
    """The `python -m launcher alignment ...` invocation for one case's phase.

    `runtime` reaches only the profile passes: a capture can run in the
    alignment image, and everything after it reads artifacts on the host.
    A resume re-derives artifacts from an existing capture, on the host.
    """
    config = case_dir / f"{config_stem(variant, phase)}.yaml"
    base = (sys.executable, "-m", "launcher", "alignment")
    if variant.pass_named(phase) is not None:
        if resume:
            return base + ("profile", str(config), "--resume")
        return (
            base
            + ("profile", str(config))
            + ("--runtime", runtime)
            + (("--allow-autotune",) if allow_autotune else ())
        )
    if phase == TIMING_PREDICT_PHASE:
        return base + ("timing-predict", str(config))
    if phase in (ANALYSIS_KERNEL_PHASE, ANALYSIS_E2E_PHASE):
        return base + ("analyze", str(config))
    if phase == SIMULATION_PHASE:
        return base + ("sim", str(config)) + (("--refresh",) if refresh else ())
    raise ValueError(f"unknown phase {phase!r}")


def plan_phase(
    pack: Pack,
    out_root: Path,
    phase: str,
    *,
    cases: list[Case] | None = None,
    refresh: bool = False,
    resume: bool = False,
    runtime: str = "image",
    allow_autotune: bool = False,
) -> tuple[PhasePlan, ...]:
    """Decide each case's disposition. Pure: no process is started, nothing is
    written, so the CPU tier tests this directly against a faked artifact tree."""
    out_root = Path(out_root).resolve()
    plans: list[PhasePlan] = []
    for case in cases if cases is not None else pack.cases:
        variant = pack.variant_of(case)
        if phase not in phase_names(variant):
            plans.append(
                PhasePlan(
                    case.slug,
                    phase,
                    out_root / case.slug,
                    "missing",
                    (
                        f"variant {variant.name} has no phase {phase!r}; "
                        f"it offers {list(phase_names(variant))}",
                    ),
                )
            )
            continue
        case_dir = out_root / case.slug
        if not case_dir.is_dir():
            plans.append(
                PhasePlan(
                    case.slug, phase, case_dir, "missing", ("no run directory — render first",)
                )
            )
            continue
        if (
            not refresh
            and phase_complete(case_dir, variant, phase)
            and not _capture_moved_on(pack, case, case_dir, phase)
        ):
            plans.append(PhasePlan(case.slug, phase, case_dir, "complete"))
            continue
        # A finished simulation whose preset or calibration has changed since:
        # the launcher would skip it on its own marker, so it is re-run.
        stale = (
            phase == SIMULATION_PHASE
            and has_marker(case_dir / phase)
            and (
                not simulation_inputs_current(case_dir, variant)
                or _capture_moved_on(pack, case, case_dir, phase)
            )
        )
        blockers = phase_requirements(case_dir, variant, phase, pack=pack, case=case)
        if blockers:
            plans.append(PhasePlan(case.slug, phase, case_dir, "blocked", tuple(blockers)))
            continue
        plans.append(
            PhasePlan(
                case.slug,
                phase,
                case_dir,
                "ready",
                (STALE_SIMULATION,) if stale else (),
                phase_command(
                    case_dir,
                    variant,
                    phase,
                    resume=resume,
                    refresh=refresh or stale,
                    runtime=runtime,
                    allow_autotune=allow_autotune,
                ),
            )
        )
    return tuple(plans)


def _capture_moved_on(pack: Pack, case: Case, case_dir: Path, phase: str) -> bool:
    """A re-run workload pass now resolves to inputs other than the
    `calibration.json` the simulation used, though nothing re-rendered yet."""
    if phase != SIMULATION_PHASE or not case.capture_fields:
        return False
    try:
        current = calibration.resolve(pack, case, case_dir)
        written = calibration.load(case_dir)
    except (calibration.CalibrationError, OSError, ValueError, KeyError):
        return False
    return written is None or written.get("fields") != json.loads(json.dumps(current))


def describe_plan(phase: str, plans: tuple[PhasePlan, ...]) -> str:
    ready = sum(1 for item in plans if item.selected)
    complete = sum(1 for item in plans if item.state == "complete")
    stuck = len(plans) - ready - complete
    lines = [
        f"[plan] phase {phase}: {ready} to run, {complete} already complete, {stuck} not ready"
    ]
    lines += [item.describe() for item in plans]
    if any(item.state == "blocked" for item in plans):
        lines.append(
            "[note] a blocked case names the inputs it lacks; run that phase yourself — "
            "this command never schedules a phase you did not ask for"
        )
    return "\n".join(lines)


# ── execution ────────────────────────────────────────────────────────────────


async def _run_one(
    plan: PhasePlan,
    variant: Variant,
    scheduler,
    leases: LauncherLeases,
    refresh: bool,
) -> PhaseResult:
    directory = plan.directory / plan.phase
    uses_simulation_slot = plan.phase in GPU_PHASES or variant.pass_named(plan.phase) is not None
    spec = ProcessSpec(
        argv=plan.command,
        cwd=REPO_ROOT,
        name=f"alignment_campaign_{plan.phase}",
    )
    journal = RunJournal(plan.directory)
    journal.begin_attempts([plan.phase])
    journal.update(
        plan.phase,
        StageState.WAITING_RESOURCE,
        spec=spec,
        resources=[
            "simulation-slot" if uses_simulation_slot else "analysis-slot",
            "run-directory-lease",
        ],
    )
    result: ProcessResult | None = None
    try:
        slot = scheduler.simulation_slot() if uses_simulation_slot else scheduler.analysis_slot()
        async with slot:
            async with leases.run_directory(plan.directory):
                if (refresh or STALE_SIMULATION in plan.reasons) and directory.is_dir():
                    clear_marker(directory)
                inputs = None
                if plan.phase == SIMULATION_PHASE:
                    (directory / SIMULATION_INPUTS_NAME).unlink(missing_ok=True)
                    inputs = simulation_inputs(plan.directory, variant)
                journal.update(plan.phase, StageState.RUNNING, spec=spec)
                result = await _PROCESS_SUPERVISOR.run(spec)
                if result.succeeded and inputs is not None and directory.is_dir():
                    _record_simulation_inputs(plan.directory, inputs)
                journal.update(plan.phase, StageState.EXITED, spec=spec, result=result)

        # Resource release is part of the attempt. Publish no terminal state or
        # marker until both the slot and cross-process lease exited cleanly.
        if not result.succeeded:
            journal.update(
                plan.phase,
                StageState.FAILED,
                spec=spec,
                result=result,
                error="process failed or left process-group descendants",
            )
            effective_returncode = result.exit_code if result.exit_code != 0 else 1
            return PhaseResult(plan=plan, returncode=effective_returncode)

        # INV-5: the marker is written only after a successful process. The
        # simulation phase already carries one written by the ordinary
        # launcher, so re-writing it here would claim ownership twice.
        if plan.phase != SIMULATION_PHASE and directory.is_dir():
            mark_complete(directory)
        journal.update(plan.phase, StageState.SUCCEEDED, spec=spec, result=result)
        return PhaseResult(plan=plan, returncode=0)
    except asyncio.CancelledError:
        journal.update(
            plan.phase,
            StageState.CANCELLED,
            spec=spec,
            result=result,
        )
        raise
    except OSError as error:
        # One case failing to acquire resources, spawn, or publish its marker
        # must not cancel the other cases in this phase's fanout.
        journal.update(
            plan.phase,
            StageState.FAILED,
            spec=spec,
            result=result,
            error=f"{type(error).__name__}: {error}",
        )
        return PhaseResult(plan=plan, returncode=1)


async def _run_all(
    pack: Pack,
    plans: tuple[PhasePlan, ...],
    *,
    parallelism: int,
    refresh: bool,
) -> list[PhaseResult]:
    from ..workflow import ResourceScheduler

    scheduler = ResourceScheduler(
        simulation_parallelism=1 if parallelism <= 0 else parallelism,
        analysis_parallelism=None,
    )
    leases = LauncherLeases(REPO_ROOT)
    by_slug = {case.slug: case for case in pack.cases}
    selected = [item for item in plans if item.selected]
    tasks = [
        _run_one(item, pack.variant_of(by_slug[item.case_slug]), scheduler, leases, refresh)
        for item in selected
    ]
    if not tasks:
        return []
    return list(await asyncio.gather(*tasks))


def run_phase(
    pack: Pack,
    out_root: Path,
    phase: str,
    *,
    cases: list[Case] | None = None,
    refresh: bool = False,
    resume: bool = False,
    parallelism: int = 1,
    runtime: str = "image",
    allow_autotune: bool = False,
) -> RunReport:
    """Fan one phase across every ready case and collect exit codes separately.

    A GPU-owning phase defaults to `parallelism=1`: on a single-replica host the
    cases contend for the same devices, and the point of the budget is to let the
    resource manager schedule that rather than to promise concurrency.
    """
    plans = plan_phase(
        pack,
        out_root,
        phase,
        cases=cases,
        refresh=refresh,
        resume=resume,
        runtime=runtime,
        allow_autotune=allow_autotune,
    )
    plans = _materialize_captured(pack, plans)
    report = RunReport(phase=phase, plans=plans)
    report.results = asyncio.run(_run_all(pack, plans, parallelism=parallelism, refresh=refresh))
    return report


def _materialize_captured(pack: Pack, plans: tuple[PhasePlan, ...]) -> tuple[PhasePlan, ...]:
    """Write each ready case's from_capture preset before its phase launches.

    A case whose capture cannot be read after all is blocked with the reason,
    exactly like an input missing at planning time.
    """
    by_slug = {case.slug: case for case in pack.cases}
    updated: list[PhasePlan] = []
    for plan in plans:
        case = by_slug.get(plan.case_slug)
        if (
            plan.selected
            and case is not None
            and case.capture_fields
            and plan.phase in CAPTURE_DEPENDENT_PHASES
        ):
            try:
                materialize(pack, case, plan.directory, REPO_ROOT)
            except (PackError, OSError, ValueError, KeyError) as error:
                plan = PhasePlan(
                    plan.case_slug, plan.phase, plan.directory, "blocked", (str(error),)
                )
        updated.append(plan)
    return tuple(updated)
