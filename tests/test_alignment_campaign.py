"""CPU gate for the alignment campaign engine and every tracked pack.

Parameterized over `presets/alignment/*/`, so a pack added tomorrow is covered
the moment it lands — that parameterization is what "reusable" actually cashes
out to here.

Two tiers:

  - unmarked (cpu): everything that runs without build artifacts. The pack self
    check, the readiness/plan logic, the metric formulas, and a cross-model
    regression on a schema-1 report set.
  - ``needs_binary``: the parts that need `deployment_schema.json`, which lives
    under the gitignored `target/<profile>/` and requires a release build. That
    is where the arch tag and its parameter set are validated against the
    schema the Rust side exported — the check that makes "a new model needs no
    engine change" true rather than hoped.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import shutil
from pathlib import Path

import pytest

from alignment.labeling import load_rules, subsumptions
from launcher.alignment_campaign import check as check_module
from launcher.alignment_campaign import compare as compare_module
from launcher.alignment_campaign import execute, label
from launcher.alignment_campaign import extract as extract_module
from launcher.alignment_campaign.metrics import (
    REPORT_LOCATIONS,
    SUPPORTED_SCHEMAS,
    measure_case,
    metric_names,
    metric_specs,
)
from launcher.alignment_campaign.pack import PackError, load_host, load_pack
from launcher.alignment_campaign.render import (
    ANALYSIS_KERNEL_PHASE,
    CONTEXT_LIMIT_FLAG,
    PHASE_CONFIG_STEMS,
    SIMULATION_PHASE,
    TIMING_PREDICT_PHASE,
    _extra_args,
    case_documents,
    phase_names,
    render_case,
)
from launcher.golden import GOLDEN_ROOT
from launcher.process import ProcessResult, ProcessSpec
from launcher.process.journal import RunJournal, StageState
from launcher.process.markers import has_marker, mark_complete

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = REPO_ROOT / "presets" / "alignment"
HOST_ROOT = PACK_ROOT / "hosts"


def _pack_dirs() -> list[Path]:
    if not PACK_ROOT.is_dir():
        return []
    return sorted(
        child for child in PACK_ROOT.iterdir()
        if child.is_dir() and (child / "campaign.yaml").is_file()
    )


PACK_DIRS = _pack_dirs()
PACK_IDS = [path.name for path in PACK_DIRS]


@pytest.fixture(scope="module", params=PACK_DIRS, ids=PACK_IDS)
def pack(request):
    return load_pack(request.param)


# ── the pack itself ──────────────────────────────────────────────────────────

def test_at_least_one_pack_is_tracked():
    """The gate is only meaningful if it has something to gate."""
    assert PACK_DIRS, f"no alignment pack under {PACK_ROOT}"


def test_pack_check_reports_no_errors(pack):
    """`alignment-campaign check` end to end: renders every case, hands it to the
    real phase loaders, reproduces every trace, and validates the policy."""
    findings = check_module.check_pack(
        pack, metric_names=metric_names(_schemas(pack))
    )
    errors = [item for item in findings if item.level == "error"]
    assert not errors, "\n".join(str(item) for item in errors)


def test_pack_declares_provisional_inputs_explicitly(pack):
    """A provisional value is allowed — GLM case 09's rate knee is one in the
    accepted run — but it must be named, so `--record` can refuse by default."""
    for name in pack.provisional_fields:
        slug, _, field = name.partition(".")
        case = pack.case_named(slug)
        assert case is not None
        assert case.calibrated_fields[field].derived_from.strip(), (
            f"{name} is provisional but does not say how it will be measured"
        )


def test_pack_rules_cannot_depend_on_the_order_they_are_listed_in(pack):
    """`apply_rules` stops at the first matching rule, so a rule set with two
    rules claiming one position and disagreeing about it labels by list order.

    This pack used to be seven files applied in a fixed sequence, and the labels
    were a property of how the filenames sorted. `subsumptions` proves such a
    pair from the rule text alone; the empirical half runs at the fixpoint in
    `label`. The union is checked, not each file: splitting a pair across two
    files is exactly how the dependence hid.
    """
    for variant in pack.variants.values():
        rules = [
            rule
            for path in label.rule_files(pack, variant)
            for rule in load_rules(path)
        ]
        assert rules, f"{variant.label_rules} declares no rules"
        assert not subsumptions(rules), "\n".join(subsumptions(rules))


def test_pack_rule_manifest_does_not_declare_an_order(pack):
    """`ordered_rule_files` is refused rather than quietly re-read as a set:
    a manifest that still names it was written when order carried meaning."""
    for variant in pack.variants.values():
        manifest = json.loads((pack.root / variant.label_rules).read_text())
        assert "ordered_rule_files" not in manifest
        assert manifest["rule_files"]


def test_trace_rows_are_not_committed_only_their_hashes(pack):
    """A trace is `shapes` x `repeats`; committing the rows stores it twice."""
    assert not list((pack.root / "traces").glob("*.csv"))
    recorded = json.loads(check_module.invariants_path(pack).read_text())["traces"]
    referenced = {
        spec.file
        for case in pack.cases
        for spec in (case.workload_trace, case.kernel_trace)
        if spec is not None
    }
    assert set(recorded) == referenced
    assert all(body["sha256"] for body in recorded.values())


def test_update_invariants_adds_a_trace_but_will_not_rewrite_one(pack, tmp_path):
    """Append-only is what keeps the sha256 an anchor rather than a cache.

    Without the CSVs, the recorded hash is the only tie between the matrix and
    the workload that was measured. If `--update-invariants` could refresh it,
    changing a shape and re-running the command would turn a failing check green
    in a one-line diff.
    """
    shutil.copytree(pack.root, tmp_path / pack.name)
    copy = load_pack(tmp_path / pack.name)
    path = check_module.invariants_path(copy)

    # Re-recording an unchanged pack is a no-op, not a rewrite.
    before = path.read_text()
    check_module.write_invariants(copy)
    assert path.read_text() == before

    # A case whose shapes moved is refused, and the record is left alone.
    case = copy.cases[0]
    spec = dataclasses.replace(case.workload_trace, repeats=case.workload_trace.repeats + 1)
    moved = dataclasses.replace(copy, cases=(dataclasses.replace(case, workload_trace=spec),))
    with pytest.raises(PackError, match="will not overwrite"):
        check_module.write_invariants(moved)
    assert path.read_text() == before

    # Deleting the entry is the deliberate edit that allows a re-record.
    document = json.loads(before)
    document["traces"].pop(spec.file)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    check_module.write_invariants(moved)
    rewritten = json.loads(path.read_text())["traces"]
    assert rewritten[spec.file]["sha256"] != json.loads(before)["traces"][spec.file]["sha256"]


def test_rendered_case_directories_are_distinct(pack):
    slugs = [case.slug for case in pack.cases]
    assert len(slugs) == len(set(slugs))


def test_golden_keys_all_belong_to_this_pack(pack):
    """Renaming a case must not leave a zombie key behind in the store."""
    expected = {
        compare_module.golden_key(f"{case.variant}/{case.slug}", metric)
        for case in pack.cases
        for metric in metric_names(_schemas(pack))
    }
    for variant_name in pack.variants:
        store = compare_module.golden_store(pack, variant_name)
        if not store.path.is_file():
            continue
        orphans = sorted(set(store.data) - expected)
        assert not orphans, f"{store.path} holds keys no case produces: {orphans[:5]}"


def _schemas(pack):
    declared = pack.acceptance.get("analyzer_schema")
    return {key: int(value) for key, value in declared.items()} if declared else None


# ── rendering ────────────────────────────────────────────────────────────────

def test_render_writes_every_phase_config(pack, tmp_path):
    host = check_module.host_for(pack, None)
    case = pack.cases[0]
    rendered = render_case(pack, case, host, tmp_path, REPO_ROOT)
    variant = pack.variant_of(case)
    for phase in phase_names(variant):
        stem = phase if variant.pass_named(phase) else PHASE_CONFIG_STEMS[phase]
        assert (rendered.directory / f"{stem}.yaml").is_file(), phase
    assert (rendered.directory / "trace.csv").is_file()


def test_render_never_leaks_a_stub_host_into_the_pack(pack, tmp_path):
    """`host_for` invents `/nonexistent/...` so the CPU tier can render without a
    machine. Writing one of those back into the pack would look like a real
    path. Host paths in general are the previous test's job."""
    text = (pack.root / "campaign.yaml").read_text()
    offenders = [line for line in text.splitlines() if "/nonexistent/" in line]
    assert not offenders, f"stub host path in campaign.yaml: {offenders[:2]}"


def test_render_rejects_a_variant_that_sets_a_per_case_field(pack, tmp_path):
    """`max_model_len` and `attn_gpu_memory_gb` are the case's to set. A variant
    that also sets them would produce two sources for one number."""
    import dataclasses

    case = pack.cases[0]
    variant = pack.variant_of(case)
    host = check_module.host_for(pack, None)
    poisoned = dataclasses.replace(variant, arch={**variant.arch, "max_model_len": 999})
    patched = dataclasses.replace(pack, variants={**pack.variants, variant.name: poisoned})
    with pytest.raises(PackError, match="max_model_len"):
        check_module.case_documents  # keep the import meaningful if it moves
        from launcher.alignment_campaign.render import case_documents

        case_documents(patched, case, host, tmp_path, REPO_ROOT)


@pytest.mark.parametrize(
    ("engine", "expected_flag"),
    sorted(CONTEXT_LIMIT_FLAG.items()),
)
def test_context_limit_uses_the_target_engine_cli(pack, engine, expected_flag):
    case = pack.cases[0]
    original = pack.variant_of(case)
    variant = dataclasses.replace(
        original,
        engine=engine,
        server={**original.server, "extra_args": ["--unrelated-flag"]},
    )

    assert _extra_args(variant, case) == [
        "--unrelated-flag",
        expected_flag,
        str(case.max_model_len),
    ]


@pytest.mark.parametrize("engine", sorted(CONTEXT_LIMIT_FLAG))
def test_context_limit_rejects_authored_limits_in_either_engine_spelling(pack, engine):
    case = pack.cases[0]
    original = pack.variant_of(case)
    for reserved in CONTEXT_LIMIT_FLAG.values():
        underscore_alias = f"--{reserved[2:].replace('-', '_')}"
        for argument in (
            reserved,
            f"{reserved}=999",
            underscore_alias,
            f"{underscore_alias}=999",
        ):
            variant = dataclasses.replace(
                original,
                engine=engine,
                server={**original.server, "extra_args": [argument]},
            )
            with pytest.raises(PackError, match="must not set"):
                _extra_args(variant, case)


def test_unknown_engine_has_no_implicit_context_limit_spelling(pack):
    case = pack.cases[0]
    variant = dataclasses.replace(pack.variant_of(case), engine="unknown-engine")

    with pytest.raises(PackError, match="no known context-limit flag"):
        _extra_args(variant, case)


def test_cross_phase_check_uses_the_same_sglang_context_limit_flag(pack, tmp_path):
    case = pack.cases[0]
    original = pack.variant_of(case)
    python_runtime = {
        "packages": [
            {
                "name": "flashinfer-jit-cache",
                "version": "0.6.15.post1",
                "local_version": "cu130",
                "index_url": "https://flashinfer.ai/whl/cu130",
                "required_files": ["flashinfer_jit_cache/jit_cache/module/module.so"],
            }
        ],
        "environment": {"FLASHINFER_DISABLE_JIT": "1"},
    }
    variant = dataclasses.replace(
        original,
        engine="sglang",
        python_runtime=python_runtime,
    )
    patched = dataclasses.replace(
        pack,
        variants={**pack.variants, variant.name: variant},
    )
    host = check_module.host_for(patched, None)
    documents = case_documents(patched, case, host, tmp_path, REPO_ROOT)

    assert not check_module._check_cross_phase(patched, case, documents)
    profile = documents[f"{variant.profile_passes[0].name}.yaml"]
    assert profile["python_runtime"] == python_runtime
    profile["server"]["extra_args"][-2] = "--max-model-len"
    findings = check_module._check_cross_phase(patched, case, documents)
    assert any("--context-length does not match" in item.message for item in findings)


# ── readiness and planning (pure; no subprocess) ─────────────────────────────

def test_plan_reports_missing_run_directory(pack, tmp_path):
    plans = execute.plan_phase(pack, tmp_path, phase_names(pack.variant_of(pack.cases[0]))[0])
    assert all(item.state == "missing" for item in plans)
    assert all("render" in " ".join(item.reasons) for item in plans)


def test_plan_blocks_analysis_kernel_without_its_inputs(pack, tmp_path):
    case = pack.cases[0]
    host = check_module.host_for(pack, None)
    render_case(pack, case, host, tmp_path, REPO_ROOT)
    plans = execute.plan_phase(pack, tmp_path, ANALYSIS_KERNEL_PHASE, cases=[case])
    (plan,) = plans
    assert plan.state == "blocked"
    joined = " ".join(plan.reasons)
    assert "timing_predict" in joined
    assert "kernel_sequences_labeled.json" in joined


def test_plan_skips_a_complete_phase_and_refresh_reselects_it(pack, tmp_path):
    case = pack.cases[0]
    variant = pack.variant_of(case)
    phase = phase_names(variant)[0]
    host = check_module.host_for(pack, None)
    render_case(pack, case, host, tmp_path, REPO_ROOT)

    (plan,) = execute.plan_phase(pack, tmp_path, phase, cases=[case])
    assert plan.state == "ready"

    artifacts = tmp_path / case.slug / phase
    artifacts.mkdir(parents=True)
    for name in execute._artifacts_for(variant, phase):
        (artifacts / name).write_text("{}")
    mark_complete(artifacts)

    (plan,) = execute.plan_phase(pack, tmp_path, phase, cases=[case])
    assert plan.state == "complete"
    (plan,) = execute.plan_phase(pack, tmp_path, phase, cases=[case], refresh=True)
    assert plan.state == "ready"


def test_a_marker_without_artifacts_does_not_suppress_the_rerun(pack, tmp_path):
    """INV-5's marker claims a zero exit, not a finished artifact set. An
    interrupted phase that left a marker must still be re-run."""
    case = pack.cases[0]
    variant = pack.variant_of(case)
    phase = phase_names(variant)[0]
    host = check_module.host_for(pack, None)
    render_case(pack, case, host, tmp_path, REPO_ROOT)
    artifacts = tmp_path / case.slug / phase
    artifacts.mkdir(parents=True)
    mark_complete(artifacts)
    (plan,) = execute.plan_phase(pack, tmp_path, phase, cases=[case])
    assert plan.state == "ready"


def test_a_plan_never_schedules_another_phase(pack, tmp_path):
    """The invariant `alignment/README.md` states: no phase implicitly launches
    the next one. Batching is across cases, never along the pipeline."""
    case = pack.cases[0]
    host = check_module.host_for(pack, None)
    render_case(pack, case, host, tmp_path, REPO_ROOT)
    for phase in phase_names(pack.variant_of(case)):
        plans = execute.plan_phase(pack, tmp_path, phase, cases=[case])
        for plan in plans:
            assert plan.phase == phase
            if plan.command:
                assert str(tmp_path / case.slug) in " ".join(plan.command)


def test_simulation_reads_the_kernel_align_multiplier(pack, tmp_path):
    case = pack.cases[0]
    variant = pack.variant_of(case)
    command = execute.phase_command(
        tmp_path / case.slug, variant, SIMULATION_PHASE, resume=False, refresh=False
    )
    assert "--gpu-time-multiplier-from" in command
    assert command[command.index("--gpu-time-multiplier-from") + 1].endswith(
        ANALYSIS_KERNEL_PHASE
    )


def test_simulation_can_explicitly_reuse_existing_calibration(pack, tmp_path):
    case = pack.cases[0]
    render_case(pack, case, check_module.host_for(pack, None), tmp_path, REPO_ROOT)
    source = tmp_path / "existing_calibration"
    report = source / "reports" / "alignment_iteration_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"meta": {"recommended_gpu_time_multiplier": 1.05}}))
    (original,) = execute.plan_phase(pack, tmp_path, SIMULATION_PHASE, cases=[case])
    assert original.state == "blocked"
    (reused,) = execute.plan_phase(
        pack, tmp_path, SIMULATION_PHASE, cases=[case], gpu_time_multiplier_from=source
    )
    assert reused.state == "ready"
    assert reused.command[reused.command.index("--gpu-time-multiplier-from") + 1] == str(source)
    assert not (tmp_path / case.slug / ANALYSIS_KERNEL_PHASE).exists()
    (tmp_path / case.slug / "simulation.yaml").unlink()
    (missing,) = execute.plan_phase(
        pack, tmp_path, SIMULATION_PHASE, cases=[case], gpu_time_multiplier_from=source
    )
    assert missing.state in {"missing", "blocked"}
    with pytest.raises(ValueError, match="only valid for simulation"):
        execute.plan_phase(pack, tmp_path, ANALYSIS_KERNEL_PHASE, gpu_time_multiplier_from=source)
    report.write_text(json.dumps({"meta": {"recommended_gpu_time_multiplier": 0.9}}))
    with pytest.raises(ValueError, match="no valid recommended"):
        execute.plan_phase(pack, tmp_path, SIMULATION_PHASE, gpu_time_multiplier_from=source)


def test_unknown_phase_is_rejected_with_the_available_list(pack, tmp_path):
    case = pack.cases[0]
    host = check_module.host_for(pack, None)
    render_case(pack, case, host, tmp_path, REPO_ROOT)
    (plan,) = execute.plan_phase(pack, tmp_path, "not_a_phase", cases=[case])
    assert plan.state == "missing"
    assert TIMING_PREDICT_PHASE in " ".join(plan.reasons)


class _ImmediateAsyncContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class _ImmediateScheduler:
    simulation_slot = analysis_slot = staticmethod(_ImmediateAsyncContext)


class _ImmediateLeases:
    @staticmethod
    def run_directory(_):
        return _ImmediateAsyncContext()


def _ready_execution_plan(pack, tmp_path, case_index=0):
    case = pack.cases[case_index]
    variant = pack.variant_of(case)
    case_dir = tmp_path / case.slug
    phase_dir = case_dir / ANALYSIS_KERNEL_PHASE
    phase_dir.mkdir(parents=True)
    plan = execute.PhasePlan(
        case.slug,
        ANALYSIS_KERNEL_PHASE,
        case_dir,
        "ready",
        (),
        ("fake-command", "--case", case.slug),
    )
    return variant, plan, phase_dir


def _process_result(
    plan,
    *,
    exit_code=0,
    leaked_descendants=False,
    termination_reason=None,
    pid=123,
):
    return ProcessResult(
        argv=plan.command,
        pid=pid,
        process_group_id=pid,
        exit_code=exit_code,
        elapsed_seconds=1.25,
        leaked_descendants=leaked_descendants,
        termination_reason=termination_reason,
    )


def _read_stage_journal(plan):
    return json.loads(
        (plan.directory / ".launcher/stages" / f"{plan.phase}.json").read_text()
    )


@pytest.mark.parametrize(
    (
        "exit_code",
        "leaked_descendants",
        "termination_reason",
        "expected_state",
        "effective_code",
    ),
    [
        (0, False, None, "SUCCEEDED", 0),
        (7, False, None, "FAILED", 7),
        (0, True, None, "FAILED", 1),
        (0, False, "timeout", "FAILED", 1),
    ],
)
def test_run_one_journals_the_supervised_process_outcome(
    pack,
    tmp_path,
    monkeypatch,
    exit_code,
    leaked_descendants,
    termination_reason,
    expected_state,
    effective_code,
):
    variant, plan, phase_dir = _ready_execution_plan(pack, tmp_path)
    process_result = _process_result(
        plan,
        exit_code=exit_code,
        leaked_descendants=leaked_descendants,
        termination_reason=termination_reason,
    )

    async def run_process(spec):
        assert tuple(spec.argv) == plan.command
        assert spec.cwd == REPO_ROOT
        return process_result

    monkeypatch.setattr(execute._PROCESS_SUPERVISOR, "run", run_process)
    phase_result = asyncio.run(
        execute._run_one(
            plan,
            variant,
            _ImmediateScheduler(),
            _ImmediateLeases(),
            refresh=False,
        )
    )

    journal = _read_stage_journal(plan)
    assert phase_result.returncode == effective_code
    assert journal["state"] == expected_state
    assert journal["argv"] == list(plan.command)
    assert journal["child_pid"] == 123
    assert journal["process_group_id"] == 123
    assert journal["exit_code"] == exit_code
    assert journal["elapsed_seconds"] == 1.25
    assert journal["termination_reason"] == termination_reason
    assert has_marker(phase_dir) is (expected_state == "SUCCEEDED")


@pytest.mark.parametrize("cancel_at", ["slot", "lease", "process", "lease_exit"])
def test_run_one_journals_cancellation_while_waiting_or_running(
    pack, tmp_path, monkeypatch, cancel_at
):
    class CancelledContext(_ImmediateAsyncContext):
        async def __aenter__(self):
            raise asyncio.CancelledError

    class WaitingScheduler:
        simulation_slot = analysis_slot = staticmethod(CancelledContext)

    class WaitingLeases:
        @staticmethod
        def run_directory(_):
            return CancelledContext()

    class CancelledExitContext(_ImmediateAsyncContext):
        async def __aexit__(self, *_):
            raise asyncio.CancelledError

    class ExitCancelledLeases:
        @staticmethod
        def run_directory(_):
            return CancelledExitContext()

    variant, plan, phase_dir = _ready_execution_plan(pack, tmp_path)

    async def cancel_process(_):
        if cancel_at == "process":
            raise asyncio.CancelledError
        return _process_result(plan)

    monkeypatch.setattr(execute._PROCESS_SUPERVISOR, "run", cancel_process)
    scheduler = WaitingScheduler() if cancel_at == "slot" else _ImmediateScheduler()
    if cancel_at == "lease":
        leases = WaitingLeases()
    elif cancel_at == "lease_exit":
        leases = ExitCancelledLeases()
    else:
        leases = _ImmediateLeases()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            execute._run_one(
                plan,
                variant,
                scheduler,
                leases,
                refresh=False,
            )
        )

    journal = _read_stage_journal(plan)
    assert journal["state"] == "CANCELLED"
    assert journal["argv"] == list(plan.command)
    assert journal["resources"] == ["analysis-slot", "run-directory-lease"]
    if cancel_at == "lease_exit":
        assert journal["child_pid"] == 123
    assert not has_marker(phase_dir)


def test_run_one_contains_spawn_failure_so_other_cases_finish(pack, tmp_path, monkeypatch):
    first_variant, first_plan, first_phase_dir = _ready_execution_plan(pack, tmp_path, 0)
    second_directory = tmp_path / "synthetic_second_case"
    second_phase_dir = second_directory / ANALYSIS_KERNEL_PHASE
    second_phase_dir.mkdir(parents=True)
    second_variant = first_variant
    second_plan = dataclasses.replace(
        first_plan,
        case_slug="synthetic_second_case",
        directory=second_directory,
        command=("fake-command", "--case", "synthetic_second_case"),
    )

    async def run_process(spec):
        if first_plan.case_slug in spec.argv:
            raise OSError("spawn failed")
        return _process_result(second_plan, pid=456)

    monkeypatch.setattr(execute._PROCESS_SUPERVISOR, "run", run_process)

    async def run_both():
        return await asyncio.gather(
            execute._run_one(
                first_plan,
                first_variant,
                _ImmediateScheduler(),
                _ImmediateLeases(),
                refresh=False,
            ),
            execute._run_one(
                second_plan,
                second_variant,
                _ImmediateScheduler(),
                _ImmediateLeases(),
                refresh=False,
            ),
        )

    first_result, second_result = asyncio.run(run_both())
    first_journal = _read_stage_journal(first_plan)
    second_journal = _read_stage_journal(second_plan)
    assert first_result.returncode == 1
    assert first_journal["state"] == "FAILED"
    assert first_journal["error"] == "OSError: spawn failed"
    assert not has_marker(first_phase_dir)
    assert second_result.returncode == 0
    assert second_journal["state"] == "SUCCEEDED"
    assert has_marker(second_phase_dir)


def test_run_one_starts_a_fresh_attempt_after_an_interrupted_exit(
    pack, tmp_path, monkeypatch
):
    variant, plan, phase_dir = _ready_execution_plan(pack, tmp_path)
    old_spec = ProcessSpec(argv=("old-command",), cwd=REPO_ROOT, name="old-attempt")
    old_result = ProcessResult(
        argv=("old-command",),
        pid=999,
        process_group_id=999,
        exit_code=0,
        elapsed_seconds=9.0,
    )
    journal = RunJournal(plan.directory)
    journal.update(plan.phase, StageState.RUNNING, spec=old_spec)
    journal.update(plan.phase, StageState.EXITED, spec=old_spec, result=old_result)

    async def fail_spawn(_):
        raise OSError("new spawn failed")

    monkeypatch.setattr(execute._PROCESS_SUPERVISOR, "run", fail_spawn)
    phase_result = asyncio.run(
        execute._run_one(
            plan,
            variant,
            _ImmediateScheduler(),
            _ImmediateLeases(),
            refresh=False,
        )
    )

    new_record = _read_stage_journal(plan)
    assert phase_result.returncode == 1
    assert new_record["attempt"] == 2
    assert new_record["state"] == "FAILED"
    assert new_record["argv"] == list(plan.command)
    assert new_record["child_pid"] is None
    assert new_record["process_group_id"] is None
    assert new_record["exit_code"] is None
    assert new_record["error"] == "OSError: new spawn failed"
    assert not has_marker(phase_dir)


def test_run_one_does_not_downgrade_programmer_errors(pack, tmp_path, monkeypatch):
    variant, plan, phase_dir = _ready_execution_plan(pack, tmp_path)

    async def broken_supervisor_contract(_):
        raise AssertionError("programmer error")

    monkeypatch.setattr(
        execute._PROCESS_SUPERVISOR,
        "run",
        broken_supervisor_contract,
    )
    with pytest.raises(AssertionError, match="programmer error"):
        asyncio.run(
            execute._run_one(
                plan,
                variant,
                _ImmediateScheduler(),
                _ImmediateLeases(),
                refresh=False,
            )
        )
    assert _read_stage_journal(plan)["state"] == "RUNNING"
    assert not has_marker(phase_dir)


def test_run_one_records_marker_publication_failure(pack, tmp_path, monkeypatch):
    variant, plan, phase_dir = _ready_execution_plan(pack, tmp_path)

    async def run_process(_):
        return _process_result(plan)

    def fail_marker(_):
        raise OSError("marker write failed")

    monkeypatch.setattr(execute._PROCESS_SUPERVISOR, "run", run_process)
    monkeypatch.setattr(execute, "mark_complete", fail_marker)
    phase_result = asyncio.run(
        execute._run_one(
            plan,
            variant,
            _ImmediateScheduler(),
            _ImmediateLeases(),
            refresh=False,
        )
    )

    journal = _read_stage_journal(plan)
    assert phase_result.returncode == 1
    assert journal["state"] == "FAILED"
    assert journal["exit_code"] == 0
    assert journal["error"] == "OSError: marker write failed"
    assert not has_marker(phase_dir)


def test_run_one_releases_resources_before_publishing_success(
    pack, tmp_path, monkeypatch
):
    class ExitFailureContext(_ImmediateAsyncContext):
        async def __aexit__(self, *_):
            raise OSError("lease release failed")

    class ExitFailureLeases:
        @staticmethod
        def run_directory(_):
            return ExitFailureContext()

    variant, plan, phase_dir = _ready_execution_plan(pack, tmp_path)

    async def run_process(_):
        return _process_result(plan)

    monkeypatch.setattr(execute._PROCESS_SUPERVISOR, "run", run_process)
    phase_result = asyncio.run(
        execute._run_one(
            plan,
            variant,
            _ImmediateScheduler(),
            ExitFailureLeases(),
            refresh=False,
        )
    )

    journal = _read_stage_journal(plan)
    assert phase_result.returncode == 1
    assert journal["state"] == "FAILED"
    assert journal["child_pid"] == 123
    assert journal["exit_code"] == 0
    assert journal["error"] == "OSError: lease release failed"
    assert not has_marker(phase_dir)


def test_run_one_leaves_simulation_marker_to_the_simulation_owner(
    pack, tmp_path, monkeypatch
):
    case = pack.cases[0]
    variant = pack.variant_of(case)
    case_dir = tmp_path / case.slug
    phase_dir = case_dir / SIMULATION_PHASE
    phase_dir.mkdir(parents=True)
    plan = execute.PhasePlan(
        case.slug,
        SIMULATION_PHASE,
        case_dir,
        "ready",
        (),
        ("fake-command", "--case", case.slug),
    )

    async def run_process(_):
        return _process_result(plan)

    monkeypatch.setattr(execute._PROCESS_SUPERVISOR, "run", run_process)
    phase_result = asyncio.run(
        execute._run_one(
            plan,
            variant,
            _ImmediateScheduler(),
            _ImmediateLeases(),
            refresh=False,
        )
    )

    assert phase_result.returncode == 0
    assert _read_stage_journal(plan)["state"] == "SUCCEEDED"
    assert not has_marker(phase_dir)


# ── host profiles ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "host_path", sorted(HOST_ROOT.glob("*.yaml")), ids=lambda path: path.stem
)
def test_tracked_host_profiles_load(host_path):
    host = load_host(host_path)
    assert host.checkpoints and host.device_roles


def test_every_pack_case_device_role_is_recorded_with_the_right_width(pack):
    """The recording machine's device split is provenance, so it is in the pack.

    Host profiles are local files — a real one is absolute paths to an HF cache
    and a corpus — so this cannot be checked against the machine itself. What it
    can check is that the split the pack *claims* is internally consistent: a
    role must name exactly `tp_size * dp_size` distinct devices, which is the
    error a copied case would introduce.
    """
    roles = pack.recorded_on.get("device_roles", {})
    assert roles, f"{pack.name} does not record the device split it ran on"
    for case in pack.cases:
        variant = pack.variant_of(case)
        assert case.device_role in roles, f"{case.slug}: role not in recorded_on"
        devices = str(roles[case.device_role]).split(",")
        world = int(variant.server.get("tp_size", 1)) * int(variant.server.get("dp_size", 1))
        assert len(devices) == world, case.slug
        assert len(set(devices)) == len(devices), case.slug


def test_no_recorded_golden_names_the_machine_it_was_recorded_on(pack):
    """The provenance sidecar is the only file the engine writes into the tree
    from a real run, so it is the only way a home directory reaches a commit.

    `recorded_from` and `case_reports` genuinely need to say which experiment
    produced a number -- that is what makes a drift investigable. What they must
    not carry is the absolute prefix in front of it: it names somebody's
    directory and a worktree that will be deleted, and it is worth nothing to
    the next reader. `compare._portable` makes them repo-relative.
    """
    offenders = []
    for path in sorted(GOLDEN_ROOT.glob(f"alignment_{pack.name}/*.provenance.json")):
        document = json.loads(path.read_text())
        candidates = list(document.get("recorded_from", []))
        candidates += [
            value
            for body in document.get("case_reports", {}).values()
            for value in body.values()
        ]
        offenders += [
            f"{path.name}: {value}" for value in candidates if Path(value).is_absolute()
        ]
    assert not offenders, "\n".join(offenders)


def test_no_tracked_alignment_preset_carries_a_host_absolute_path(pack):
    """The property `pack.py` claims and the reason host profiles are local.

    A committed absolute path is both a leak of whoever's box it was and a
    reference that rots — the one this replaced pointed into another user's
    gitignored log directory.
    """
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}"
        for path in sorted(pack.root.rglob("*"))
        if path.is_file() and path.suffix in {".yaml", ".yml", ".json"}
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1)
        if re.search(r"(?<![\w./])/(raid|home|opt|mnt|scratch)/", line)
    ]
    assert not offenders, "\n".join(offenders)


# ── metric formulas ──────────────────────────────────────────────────────────

def test_metric_formulas_are_unique_and_grouped():
    specs = metric_specs()
    assert len({spec.name for spec in specs}) == len(specs)
    assert all(spec.report in SUPPORTED_SCHEMAS for spec in specs)


def test_an_iteration_report_older_than_the_current_schema_is_refused(tmp_path):
    """Schema 1 put an unweighted per-iteration mean where 2 reports a
    duration-weighted error over all iterations. Reading both would mean either
    one golden key whose statistic turns over with the schema, or two names to
    keep aligned forever. The analyzer emits only 2
    (`ALIGNMENT_ITERATION_SCHEMA_VERSION`), so the old shape is named and
    refused instead of measured -- an unavailable case, never a wrong number.
    """
    with pytest.raises(ValueError, match="not supported"):
        metric_specs({"iteration": 1})

    report = tmp_path / REPORT_LOCATIONS["iteration"]
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        "schema_version": 1,
        "total_iteration": {"abs_relative_error_pct": {"mean": 3.5}},
    }))
    measurement = measure_case(tmp_path)
    assert not measurement.available
    assert any("schema_version 1" in issue for issue in measurement.issues)
    assert not measurement.metrics

    # e2e and workload are v1 by current design, not by legacy.
    assert SUPPORTED_SCHEMAS == {"iteration": (2,), "e2e": (1,), "workload": (1,)}


def test_ratio_convention_is_simulated_over_measured():
    from launcher.alignment_campaign.metrics import _ratio_pct

    assert _ratio_pct(110.0, 100.0) == pytest.approx(10.0)
    assert _ratio_pct(90.0, 100.0) == pytest.approx(-10.0)
    assert _ratio_pct(1.0, 0.0) is None


def test_incomplete_case_is_not_available(tmp_path):
    measurement = measure_case(tmp_path)
    assert not measurement.available
    assert measurement.issues


def test_mismatched_request_populations_block_availability(tmp_path):
    for kind, location in REPORT_LOCATIONS.items():
        target = tmp_path / location
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"schema_version": 1, "available": True}))
    e2e = tmp_path / REPORT_LOCATIONS["e2e"]
    e2e.write_text(json.dumps({
        "schema_version": 1, "available": True,
        "meta": {"measured_successful_requests": 10, "simulated_completed_requests": 8},
    }))
    measurement = measure_case(tmp_path)
    assert not measurement.available
    assert any("request populations differ" in issue for issue in measurement.issues)


# ── pack-less mode ───────────────────────────────────────────────────────────

def test_compare_without_a_pack_judges_nothing_and_records_nothing():
    """Pack-less mode is for a one-off alignment: numbers, no policy, no golden.

    It is what makes the tool usable before a pack exists, which is exactly when
    a new model is being aligned. Judging on engine defaults would put a number
    nobody chose in the FAIL column; touching the golden store would record a
    baseline under the placeholder variant.
    """
    document = {
        "schema_version": 1,
        "cases": {
            f"{extract_module.UNSCOPED_VARIANT}/one_off": {
                "available": True,
                "metrics": dict.fromkeys(metric_names(), 1.0),
            }
        },
    }
    comparison = compare_module.compare(document, None)
    assert comparison.judgements
    assert all(item.status == "UNJUDGED" for item in comparison.judgements)
    assert all(item.golden is None for item in comparison.judgements)
    assert not comparison.failures


# ── needs_binary: the arch registry ──────────────────────────────────────────

@pytest.mark.needs_binary
def test_rendered_simulation_preset_passes_the_real_schema(pack, tmp_path):
    from launcher.schema import normalize_params, validate_params
    from launcher.schema.loader import SchemaNotFound, _load_preset, load_schema

    try:
        schema = load_schema("release")
    except SchemaNotFound as exc:
        pytest.skip(str(exc))
    host = check_module.host_for(pack, None)
    for case in pack.cases:
        rendered = render_case(pack, case, host, tmp_path, REPO_ROOT)
        preset = _load_preset(rendered.directory / f"{PHASE_CONFIG_STEMS['simulation']}.yaml")
        errors = validate_params(preset, schema)
        assert not errors, f"{case.slug}: {errors}"
        normalize_params(preset, schema)


@pytest.mark.needs_binary
def test_variant_arch_tag_and_params_exist_in_the_exported_schema(pack):
    """What makes "a new model needs no engine change" true: the pack's arch
    block is validated against the tags and parameters the Rust side exported,
    not against a field list duplicated in Python."""
    from launcher.schema.loader import SchemaNotFound, load_schema

    try:
        schema = load_schema("release")
    except SchemaNotFound as exc:
        pytest.skip(str(exc))
    for name, variant in pack.variants.items():
        tag = variant.arch.get("type")
        contracts = [
            contract for contract, tags in schema.arch_providers.items() if tag in tags
        ]
        assert contracts, f"variants.{name}: unknown arch tag {tag!r}"
        known = {
            item["name"] for item in schema.arch_providers[contracts[0]][tag].get("params", [])
        }
        known |= {item["name"] for item in schema.arch_common}
        unknown = set(variant.arch) - known - {"type", "max_model_len"}
        assert not unknown, f"variants.{name}.arch has unknown params {sorted(unknown)}"
