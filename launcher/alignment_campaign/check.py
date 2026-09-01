"""`alignment-campaign check` — prove a pack is sound without a GPU.

Everything here is pure CPU work on tracked bytes, which is what lets the same
code be both an operator command and the body of the pytest gate. The checks
fall into four groups:

1. **It renders.** Every case is rendered against a stub host and handed to the
   real loaders (`load_profile_config`, `load_timing_predict_config`,
   `load_analyze_config`). A pack that cannot produce a loadable config is
   broken now, not on the machine with the GPU.
2. **The traces are reproducible.** A trace is not committed — every byte of it
   is `shapes` x `repeats` from `campaign.yaml`, so a CSV in the tree would be a
   copy of data already there. What is committed is `traces/invariants.json`:
   the sha256 of each trace plus the derived quantities that say *how* a changed
   one changed. `check` regenerates and compares against that record, which is
   what makes "reproducible from tracked inputs" a claim rather than a hope.

   The record is append-only (see `write_invariants`). Deleting the bytes would
   otherwise make re-baselining cheap — regenerate, refresh the hash, done —
   and a hash refresh is a one-line diff where overwriting a CSV was hundreds.
3. **The cases actually differ.** Two cases may not share an output directory,
   and two rendered config trees may not be identical once `log_dir` is ignored
   — the criterion `launcher/schema/expand.py` already applies to swept runs,
   which catches a copy-pasted case that measures nothing new.
4. **The policy is complete.** Every metric the engine computes has a tolerance,
   every per-case exception states why, and every calibrated value says how it
   was obtained.

The stub host is built in code rather than read from `presets/alignment/hosts/`
so `check` cannot be broken by editing an example file. It deliberately omits
`fork_python`: `load_profile_config` stats that field only when it is non-empty
and never stats `nsys.executable`, so the real loaders run on a host with no
GPU and no engine venv.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ..alignment_config import (
    load_analyze_config,
    load_profile_config,
    load_timing_predict_config,
)
from .pack import Case, Finding, HostProfile, Pack, PackError
from .render import (
    ANALYSIS_E2E_PHASE,
    ANALYSIS_KERNEL_PHASE,
    CONTEXT_LIMIT_FLAG,
    KERNEL_TRACE_NAME,
    PHASE_CONFIG_STEMS,
    SIMULATION_PHASE,
    TIMING_PREDICT_PHASE,
    WORKLOAD_TRACE_NAME,
    case_documents,
    case_traces,
    trace_text,
)

INVARIANTS_SCHEMA_VERSION = 1
INVARIANTS_NAME = "invariants.json"


def stub_host() -> HostProfile:
    """A host profile that resolves without touching the filesystem."""
    return HostProfile(
        path=Path("<stub>"),
        name="stub",
        hf_hub_root="/nonexistent/hub",
        checkpoints={},
        text_corpus="/nonexistent/corpus/enwik9",
        nsys_executable="/nonexistent/nsys",
        fork_python="",
        device_roles={},
        port=8000,
        startup_timeout=900.0,
    )


def host_for(pack: Pack, host: HostProfile | None) -> HostProfile:
    """The stub, widened so every checkpoint and device role the pack names
    resolves. Real hosts are returned unchanged — checking against one is how a
    machine's device map and checkpoint keys get validated before a run."""
    if host is not None:
        return host
    base = stub_host()
    checkpoints = dict(base.checkpoints)
    roles = dict(base.device_roles)
    for variant in pack.variants.values():
        for key in (variant.checkpoint, variant.tokenizer):
            checkpoints.setdefault(key, f"/nonexistent/checkpoints/{key}")
        world_size = int(variant.server.get("tp_size", 1)) * int(variant.server.get("dp_size", 1))
        for case in pack.cases:
            if case.variant != variant.name:
                continue
            roles.setdefault(case.device_role, ",".join(str(index) for index in range(world_size)))
    from dataclasses import replace

    return replace(base, checkpoints=checkpoints, device_roles=roles)


# ── trace invariants ─────────────────────────────────────────────────────────

def trace_invariants(text: str) -> dict[str, Any]:
    """Derived quantities that survive a re-generation of the same trace.

    A sha256 alone would only say "the bytes changed"; these say *how*, which is
    the difference between spotting a widened shape ladder and re-reading a diff.
    """
    lines = text.splitlines()
    header, rows = lines[0], lines[1:]
    if header != "id,input_len,output_len,arrival_time":
        raise PackError(f"unexpected trace header: {header!r}")
    ids: list[str] = []
    inputs: list[int] = []
    outputs: list[int] = []
    arrivals: list[float] = []
    for row in rows:
        request_id, input_len, output_len, arrival = row.split(",")
        ids.append(request_id)
        inputs.append(int(input_len))
        outputs.append(int(output_len))
        arrivals.append(float(arrival))
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "request_count": len(rows),
        "distinct_request_ids": len(set(ids)),
        "distinct_shapes": len(set(zip(inputs, outputs))),
        "input_tokens": sum(inputs),
        "output_tokens": sum(outputs),
        "max_input_len": max(inputs, default=0),
        "max_output_len": max(outputs, default=0),
        "max_total_len": max((a + b for a, b in zip(inputs, outputs)), default=0),
        "arrival_span_ms": (max(arrivals, default=0.0) - min(arrivals, default=0.0)),
    }


def compute_invariants(pack: Pack) -> dict[str, Any]:
    """The full invariants document, derived by regenerating every trace."""
    traces: dict[str, Any] = {}
    for case in pack.cases:
        for spec in (case.workload_trace, case.kernel_trace):
            if spec is None:
                continue
            traces[spec.file] = trace_invariants(trace_text(case, spec))
    return {
        "schema_version": INVARIANTS_SCHEMA_VERSION,
        "note": (
            "The traces themselves are not committed: they are `shapes` x `repeats` "
            "from campaign.yaml, and `alignment-campaign check` regenerates and "
            "compares against this record. Written by `check --update-invariants`, "
            "which only ever ADDS a trace. Re-recording an existing one means "
            "deleting its entry here first, on purpose, in a reviewable diff."
        ),
        "traces": dict(sorted(traces.items())),
    }


def invariants_path(pack: Pack) -> Path:
    return pack.root / "traces" / INVARIANTS_NAME


def write_invariants(pack: Pack) -> Path:
    """Record the invariants of traces that have none yet, and only those.

    Append-only is the whole safeguard. With the CSVs gone, the recorded sha256
    is the only thing tying `campaign.yaml`'s shapes to the workload that
    actually produced the accepted numbers; if this command could refresh it,
    then changing a shape and re-running it would turn a failing check green
    with nothing to review. Removing an entry is a deliberate edit, and that is
    the point at which someone should be asked why.
    """
    path = invariants_path(pack)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = compute_invariants(pack)
    if path.is_file():
        recorded = json.loads(path.read_text()).get("traces", {})
        kept = {
            name: body for name, body in document["traces"].items()
            if name not in recorded
        }
        conflicting = sorted(
            name for name, body in document["traces"].items()
            if name in recorded and recorded[name] != body
        )
        if conflicting:
            raise PackError(
                f"{path} already records {conflicting}, and regenerating them gives "
                "different bytes. This says the matrix no longer describes the traces "
                "the accepted numbers came from. Delete those entries by hand if that "
                "is intended; --update-invariants will not overwrite them."
            )
        document["traces"] = dict(sorted({**recorded, **kept}.items()))
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


# ── the check itself ─────────────────────────────────────────────────────────

def check_pack(
    pack: Pack, *, host: HostProfile | None = None, metric_names: tuple[str, ...] = ()
) -> list[Finding]:
    """Run every CPU check. Returns findings; the caller decides the exit code."""
    findings: list[Finding] = []
    resolved_host = host_for(pack, host)

    findings += _check_identity(pack)
    findings += _check_traces(pack)
    findings += _check_label_rules(pack)
    findings += _check_expert_popularity(pack)
    findings += _check_acceptance(pack, metric_names)
    findings += _check_calibration(pack)
    findings += _check_rendering(pack, resolved_host)
    return findings


def _check_identity(pack: Pack) -> list[Finding]:
    findings: list[Finding] = []
    seen_index: dict[int, str] = {}
    seen_name: dict[str, str] = {}
    for case in pack.cases:
        if case.index in seen_index:
            findings.append(
                Finding("error", f"cases[{case.slug}]",
                        f"index {case.index} already used by {seen_index[case.index]}")
            )
        seen_index[case.index] = case.slug
        if case.name in seen_name:
            findings.append(
                Finding("error", f"cases[{case.slug}]",
                        f"name already used by {seen_name[case.name]}")
            )
        seen_name[case.name] = case.slug
        if case.variant not in pack.variants:
            findings.append(
                Finding("error", f"cases[{case.slug}].variant",
                        f"unknown variant {case.variant!r}; "
                        f"declared: {sorted(pack.variants)}")
            )
    for name, variant in pack.variants.items():
        if variant.raw_overrides:
            findings.append(
                Finding("warn", f"variants.{name}.raw_overrides",
                        f"escape hatch in use for {sorted(variant.raw_overrides)}; "
                        "promote it to a real pack field once a second pack needs it")
            )
    return findings


def _check_traces(pack: Pack) -> list[Finding]:
    findings: list[Finding] = []
    recorded_path = invariants_path(pack)
    recorded: dict[str, Any] = {}
    if recorded_path.is_file():
        recorded = json.loads(recorded_path.read_text()).get("traces", {})
    else:
        findings.append(
            Finding("error", f"traces/{INVARIANTS_NAME}",
                    "missing; run `alignment-campaign check --update-invariants`")
        )

    for case in pack.cases:
        for role, spec in (("workload_trace", case.workload_trace),
                           ("kernel_trace", case.kernel_trace)):
            if spec is None:
                continue
            where = f"cases[{case.slug}].{role}"
            try:
                regenerated = trace_text(case, spec)
            except PackError as exc:
                findings.append(Finding("error", where, str(exc)))
                continue
            actual = trace_invariants(regenerated)
            if actual["request_count"] != spec.request_count:
                findings.append(
                    Finding("error", where,
                            f"{spec.file} regenerates {actual['request_count']} rows, "
                            f"but shapes x repeats says {spec.request_count}")
                )
            if actual["distinct_request_ids"] != actual["request_count"]:
                findings.append(Finding("error", where, f"{spec.file} repeats a request id"))
            if actual["max_total_len"] >= case.max_model_len:
                findings.append(
                    Finding("error", where,
                            f"{spec.file} has a request reaching max_model_len "
                            f"({actual['max_total_len']} >= {case.max_model_len})")
                )
            expected = recorded.get(spec.file)
            if expected is None:
                # With no CSV in the tree, an unrecorded trace has nothing tying
                # it to the run it came from — it is whatever the matrix says
                # today, which is exactly the claim being checked.
                findings.append(
                    Finding("error", where,
                            f"{spec.file} is not recorded in {INVARIANTS_NAME}, so "
                            "nothing ties it to the workload that was measured; run "
                            "`alignment-campaign check --update-invariants`")
                )
            elif expected != actual:
                differing = sorted(
                    key for key in set(expected) | set(actual)
                    if expected.get(key) != actual.get(key)
                )
                findings.append(
                    Finding("error", where,
                            f"{spec.file} invariants differ from {INVARIANTS_NAME} "
                            f"on {differing}")
                )
    stale = set(recorded) - {
        spec.file
        for case in pack.cases
        for spec in (case.workload_trace, case.kernel_trace)
        if spec is not None
    }
    for name in sorted(stale):
        findings.append(
            Finding("error", f"traces/{INVARIANTS_NAME}",
                    f"records {name}, which no case references")
        )
    return findings


def _check_label_rules(pack: Pack) -> list[Finding]:
    from alignment.labeling.rules import load_rules, subsumptions

    findings: list[Finding] = []
    for name, variant in pack.variants.items():
        where = f"variants.{name}.label_rules"
        manifest_path = pack.root / variant.label_rules
        if not manifest_path.is_file():
            findings.append(Finding("error", where, f"missing: {variant.label_rules}"))
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            findings.append(Finding("error", where, f"unreadable manifest: {exc}"))
            continue
        if "ordered_rule_files" in manifest:
            findings.append(Finding(
                "error", where,
                "`ordered_rule_files` means the labels depend on the order the files are "
                "applied in. Merge them into an order-independent set under `rule_files`",
            ))
            continue
        declared = manifest.get("rule_files")
        if not isinstance(declared, list) or not declared:
            findings.append(
                Finding("error", where, "manifest needs a non-empty rule_files list")
            )
            continue
        rules = []
        for entry in declared:
            rule_path = manifest_path.parent / entry
            if not rule_path.is_file():
                findings.append(Finding("error", where, f"rule file missing: {entry}"))
                continue
            try:
                loaded = load_rules(rule_path)
            except (ValueError, KeyError, TypeError) as exc:
                findings.append(Finding("error", where, f"{entry} does not parse: {exc}"))
                continue
            if not loaded:
                findings.append(Finding("warn", where, f"{entry} declares no rules"))
            rules.extend(loaded)
        # The rules are one set, so the files are checked together: splitting a
        # pair across two files is exactly how the order-dependence hid before.
        for line in subsumptions(rules):
            findings.append(Finding("error", where, line))
        present = {
            item.name
            for item in manifest_path.parent.iterdir()
            if item.is_file() and item.suffix == ".json" and item.name != manifest_path.name
        }
        for orphan in sorted(present - set(declared)):
            findings.append(
                Finding("warn", where,
                        f"{orphan} sits beside the rules but the manifest never applies it")
            )
    return findings


def _check_expert_popularity(pack: Pack) -> list[Finding]:
    findings: list[Finding] = []
    for name, variant in pack.variants.items():
        reference = variant.arch.get("expert_popularity_file")
        if not isinstance(reference, str) or not reference:
            continue
        where = f"variants.{name}.arch.expert_popularity_file"
        path = pack.root / reference
        if not path.is_file():
            findings.append(Finding("error", where, f"missing: {reference}"))
            continue
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            findings.append(Finding("error", where, f"unreadable: {exc}"))
            continue
        version = document.get("schema_version")
        if version not in (2, 3):
            findings.append(
                Finding("error", where,
                        f"schema_version must be 2 or 3, got {version!r}")
            )
    return findings


def _check_acceptance(pack: Pack, metric_names: tuple[str, ...]) -> list[Finding]:
    findings: list[Finding] = []
    acceptance = pack.acceptance
    if not acceptance:
        findings.append(Finding("error", "acceptance.yaml", "missing"))
        return findings
    tolerances = acceptance.get("tolerances", {})
    default = tolerances.get("default", {}) if isinstance(tolerances, dict) else {}
    if not isinstance(default, dict) or not default:
        findings.append(Finding("error", "acceptance.yaml", "tolerances.default is required"))
        return findings
    for metric in metric_names:
        if metric not in default:
            findings.append(
                Finding("error", "acceptance.yaml",
                        f"tolerances.default has no entry for metric {metric!r}")
            )
    for metric in sorted(set(default) - set(metric_names)):
        if metric_names:
            findings.append(
                Finding("warn", "acceptance.yaml",
                        f"tolerances.default sets {metric!r}, which the engine does not compute")
            )
    slugs = {case.slug for case in pack.cases}
    per_case = tolerances.get("per_case", {}) or {}
    for slug, body in per_case.items():
        where = f"acceptance.yaml:tolerances.per_case.{slug}"
        if slug not in slugs:
            findings.append(Finding("error", where, "names no case in this pack"))
            continue
        if not isinstance(body, dict) or not body.get("rationale"):
            findings.append(
                Finding("error", where,
                        "every per-case tolerance must carry a rationale; a loosened "
                        "threshold with no stated reason is indistinguishable from a "
                        "silently accepted regression")
            )
            continue
        for metric in body:
            if metric == "rationale":
                continue
            if metric_names and metric not in metric_names:
                findings.append(
                    Finding("error", where, f"overrides unknown metric {metric!r}")
                )
    return findings


def _check_calibration(pack: Pack) -> list[Finding]:
    findings: list[Finding] = []
    for case in pack.cases:
        for name, value in case.calibrated_fields.items():
            where = f"cases[{case.slug}].{name}"
            if value.provisional:
                findings.append(
                    Finding("warn", where,
                            f"still provisional ({value.value!r}); {value.derived_from}")
                )
            elif not value.evidence:
                findings.append(
                    Finding("warn", where,
                            "measured but cites no evidence path; the run that produced "
                            "it cannot be found later")
                )
    return findings


def _check_rendering(pack: Pack, host: HostProfile) -> list[Finding]:
    """Render every case and hand the result to the real phase loaders."""
    findings: list[Finding] = []
    repo_root = Path(__file__).resolve().parents[2]
    signatures: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="campaign-check-") as scratch:
        out_root = Path(scratch)
        for case in pack.cases:
            if case.variant not in pack.variants:
                continue  # already reported by _check_identity
            where = f"cases[{case.slug}]"
            case_dir = out_root / case.slug
            case_dir.mkdir(parents=True, exist_ok=True)
            try:
                documents = case_documents(pack, case, host, case_dir, repo_root)
                traces = case_traces(pack, case)
            except PackError as exc:
                findings.append(Finding("error", where, f"does not render: {exc}"))
                continue
            for name, text in traces.items():
                (case_dir / name).write_text(text, encoding="utf-8")
            for name, document in documents.items():
                (case_dir / name).write_text(
                    yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
                )
            findings += _load_rendered(pack, case, case_dir)
            findings += _check_cross_phase(pack, case, documents)

            key = _config_signature(documents, traces)
            if key in signatures:
                findings.append(
                    Finding("error", where,
                            f"renders the same experiment as {signatures[key]} once per-case "
                            "paths are ignored — one of the two measures nothing new")
                )
            signatures[key] = case.slug
    return findings


def _load_rendered(pack: Pack, case: Case, case_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    where = f"cases[{case.slug}]"
    variant = pack.variant_of(case)
    for profile_pass in variant.profile_passes:
        path = case_dir / f"{profile_pass.name}.yaml"
        try:
            config = load_profile_config(path)
        except ValueError as exc:
            findings.append(Finding("error", f"{where}.{profile_pass.name}", str(exc)))
            continue
        expected = WORKLOAD_TRACE_NAME
        if profile_pass.trace == "kernel" and case.kernel_trace is not None:
            expected = KERNEL_TRACE_NAME
        if Path(config.workload.frontend.path).name != expected:
            findings.append(
                Finding("error", f"{where}.{profile_pass.name}",
                        f"drives {Path(config.workload.frontend.path).name} but its declared "
                        f"trace role {profile_pass.trace!r} means {expected}")
            )
    try:
        load_timing_predict_config(
            case_dir / f"{PHASE_CONFIG_STEMS[TIMING_PREDICT_PHASE]}.yaml"
        )
    except ValueError as exc:
        findings.append(Finding("error", f"{where}.{TIMING_PREDICT_PHASE}", str(exc)))
    for phase in (ANALYSIS_KERNEL_PHASE, ANALYSIS_E2E_PHASE):
        stem = PHASE_CONFIG_STEMS[phase]
        try:
            load_analyze_config(case_dir / f"{stem}.yaml")
        except ValueError as exc:
            findings.append(Finding("error", f"{where}.{phase}", str(exc)))
    return findings


def _check_cross_phase(pack: Pack, case: Case, documents: dict[str, Any]) -> list[Finding]:
    """The same decision is spelled in several documents; make them agree.

    `arrival_mode` uses a hyphen on the profile side and an underscore in a
    simulation preset, and `max_model_len` appears in the arch block, the load
    generator, and a server flag. These are projections of one case field, so a
    disagreement means the rendering, not the pack, has drifted.
    """
    findings: list[Finding] = []
    where = f"cases[{case.slug}]"
    variant = pack.variant_of(case)
    simulation = documents[f"{PHASE_CONFIG_STEMS[SIMULATION_PHASE]}.yaml"]
    group = simulation["pools"]["main"]["groups"][0]

    if group["arch"]["max_model_len"] != case.max_model_len:
        findings.append(Finding("error", f"{where}.simulation", "arch.max_model_len drifted"))
    if simulation["workload"].get("max_concurrency") != case.max_concurrency:
        findings.append(Finding("error", f"{where}.simulation", "max_concurrency drifted"))
    expected_sim_mode = case.profile_arrival_mode.replace("-", "_")
    if simulation["workload"]["arrival_mode"] != expected_sim_mode:
        findings.append(
            Finding("error", f"{where}.simulation",
                    f"arrival_mode {simulation['workload']['arrival_mode']!r} is not the "
                    f"preset spelling of the profile mode {case.profile_arrival_mode!r}")
        )
    for profile_pass in variant.profile_passes:
        document = documents[f"{profile_pass.name}.yaml"]
        workload = document["workload"]
        if workload["max_model_len"] != case.max_model_len:
            findings.append(
                Finding("error", f"{where}.{profile_pass.name}", "workload.max_model_len drifted")
            )
        flags = document["server"]["extra_args"]
        limit_flag = CONTEXT_LIMIT_FLAG.get(variant.engine)
        if limit_flag is None:
            findings.append(
                Finding("error", f"{where}.{profile_pass.name}",
                        f"engine {variant.engine!r} has no known context-limit flag")
            )
        elif flags[-2:] != [limit_flag, str(case.max_model_len)]:
            findings.append(
                Finding("error", f"{where}.{profile_pass.name}",
                        f"server {limit_flag} does not match the case")
            )
        devices = [item for item in document["cuda_visible_devices"].split(",") if item]
        world = int(variant.server.get("tp_size", 1)) * int(variant.server.get("dp_size", 1))
        if len(devices) != world:
            findings.append(
                Finding("error", f"{where}.{profile_pass.name}",
                        f"device role {case.device_role!r} yields {len(devices)} devices "
                        f"but tp*dp is {world}")
            )
    return findings


#: Config keys that name a per-case path or label. Every one of them embeds the
#: case slug, so leaving them in a distinctness signature would make every case
#: trivially unique and the check vacuous. `expand.py::validate_distinct_configs`
#: drops only `log_dir` because a swept run has no other slug-bearing field.
_PER_CASE_KEYS = frozenset({
    "log_dir",
    "name",
    "trace_files",
    "path",
    "simulation_preset",
    "profile_log_dir",
    "workload_profile_log_dir",
    "timing_predict_log_dir",
    "simulation_log_dir",
    "labeled_kernel_sequences_file",
})


def _config_signature(documents: dict[str, Any], traces: dict[str, str]) -> str:
    """What makes one case a distinct experiment.

    The trace *contents* are hashed in rather than the trace paths: cases 03 and
    04 share concurrency, context limit, KV budget and capture window and differ
    only in their shape ladder, so a signature built from the configs alone would
    call two genuinely different experiments duplicates.
    """
    def scrub(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: scrub(value)
                for key, value in sorted(node.items())
                if key not in _PER_CASE_KEYS
            }
        if isinstance(node, list):
            return [scrub(item) for item in node]
        return node

    payload = {
        "configs": scrub(documents),
        # Request ids embed the slug, so hash the shape stream instead of the file.
        "traces": sorted(
            hashlib.sha256(
                "".join(line.split(",", 1)[1] for line in text.splitlines()[1:]).encode()
            ).hexdigest()
            for text in traces.values()
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
