"""Turn one (case, variant, host) into a runnable alignment case directory.

The output is exactly what the five existing phase commands already read — this
module writes configs, it does not introduce a new runtime. Rendering replaces
the two 400-500 line f-string generators the two GLM batches each grew their own
copy of; configs are built as dicts and dumped, so a malformed nesting is a
`yaml.safe_dump` of a wrong tree rather than broken indentation.

## Where each rendered value comes from

| rendered field                       | source                                  |
|--------------------------------------|-----------------------------------------|
| `server.model_path` / `tokenizer`    | host `checkpoints[variant.checkpoint]`  |
| `workload.text_file`                 | host `text_corpus`                      |
| `nsys.executable`                    | host `nsys_executable`                  |
| `fork_python`                        | host `fork_python` (omitted when empty) |
| `cuda_visible_devices`               | host `device_roles[case.device_role]`   |
| `server.port` / `startup_timeout`    | host                                    |
| `server.gpu_memory_utilization`      | case (it decides KV capacity)           |
| `server.extra_args`                  | variant + `--max-model-len` from case   |
| `arch.max_model_len`                 | case                                    |
| `worker.attn_gpu_memory_gb`          | case (calibrated)                       |
| `worker.gpu_time_multiplier`         | always 1.0 — the sim run injects the    |
|                                      | kernel-align value with `--override`    |
| `workload.arrival_mode` / `rate`     | case                                    |

## Two normalizations that are not cosmetic

`analyze_e2e.yaml` is rendered **without** `profile_log_dir`. Every accepted
GLM case carries it, and `load_analyze_config` now rejects it on a workload/e2e
phase ("profile_log_dir does not belong to workload/e2e alignment"). Those
configs predate that rule, which is also why cases 01-10 have
`throughput.measured_server_gpu_*` in their e2e report and later ones do not. A
re-run from this pack produces the current, narrower shape.

Host paths are rendered **absolute**. The accepted configs reach the engine venv
with `../../../../alignment/...`, which only works at one directory depth; a
campaign that may render anywhere cannot rely on that. `fork_python` is anchored
without resolving its final symlink, matching what `load_profile_config` does,
so the venv's `site-packages` discovery is unchanged.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from alignment.load_generator.config import IndependentFrontendConfig, routes_backend

from .pack import Case, HostProfile, Pack, PackError, TraceSpec, Variant

#: Trace file names inside a rendered case directory. Fixed rather than derived
#: from the pack's file names so a run directory reads the same for every pack.
WORKLOAD_TRACE_NAME = "trace.csv"
SPECULATIVE_TRACE_NAME = "trace_speculative.csv"
KERNEL_TRACE_NAME = "trace_nsys.csv"

#: The four phases that follow the variant's profile passes. `--phase` values are
#: artifact directory names throughout; only these two analyze rows have a config
#: stem that differs from their directory, which is pre-existing naming in
#: `alignment/README.md` and so is spelled out rather than left implicit.
TIMING_PREDICT_PHASE = "timing_predict"
ANALYSIS_KERNEL_PHASE = "analysis_kernel"
SIMULATION_PHASE = "simulation"
ANALYSIS_E2E_PHASE = "analysis_e2e"

PHASE_CONFIG_STEMS = {
    TIMING_PREDICT_PHASE: "timing_predict",
    ANALYSIS_KERNEL_PHASE: "analyze_kernel",
    SIMULATION_PHASE: "simulation",
    ANALYSIS_E2E_PHASE: "analyze_e2e",
}

#: Engine CLI spelling for the per-case context limit.
#:
#: Most server fields use the launcher's vLLM-shaped ``ServerConfig`` and are
#: translated by each driver. This value is derived by the campaign and appended
#: directly to ``extra_args``, so it must already use the target engine's CLI.
CONTEXT_LIMIT_FLAG = {
    "vllm": "--max-model-len",
    "sglang": "--context-length",
}

#: Fixed pipeline order after the profile passes. Used for `--phase` validation
#: and for the readiness graph; it is never used to chain phases automatically.
PIPELINE_PHASES = (
    TIMING_PREDICT_PHASE,
    ANALYSIS_KERNEL_PHASE,
    SIMULATION_PHASE,
    ANALYSIS_E2E_PHASE,
)


@dataclass(frozen=True)
class RenderedCase:
    """Where one case's run directory is and what phases it holds."""

    case_slug: str
    directory: Path
    phases: tuple[str, ...]
    files: tuple[Path, ...]


def phase_names(variant: Variant) -> tuple[str, ...]:
    """Every `--phase` value for this variant, in pipeline order."""
    return tuple(item.name for item in variant.profile_passes) + PIPELINE_PHASES


def config_stem(variant: Variant, phase: str) -> str:
    """Config file stem for a phase. A profile pass names its own config; the
    fixed tail uses the table above (`analysis_kernel` reads `analyze_kernel`)."""
    if variant.pass_named(phase) is not None:
        return phase
    try:
        return PHASE_CONFIG_STEMS[phase]
    except KeyError:
        raise PackError(
            f"unknown phase {phase!r}; this variant offers {list(phase_names(variant))}"
        ) from None


# ── traces ───────────────────────────────────────────────────────────────────

def trace_text(case: Case, spec: TraceSpec, *, speculative_acceptance: list[float] | None = None) -> str:
    """Regenerate one trace from its shapes.

    Request ids carry the case slug so two cases replayed into one server are
    never confused, and arrival times are a deterministic 1 req/s baseline that
    `saturated` discards and `trace-timed` rescales by `rate`.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    columns = ["id", "input_len", "output_len", "arrival_time"]
    if speculative_acceptance is not None:
        columns.append("accept_rate")
    writer.writerow(columns)
    rows = [shape for _ in range(spec.repeats) for shape in spec.shapes]
    prefix = case.slug if spec.id_suffix is None else f"{case.slug}-{spec.id_suffix}"
    for index, (input_len, output_len) in enumerate(rows):
        if input_len + output_len >= case.max_model_len:
            raise PackError(
                f"{case.slug} trace row {index} reaches max_model_len: "
                f"{input_len}+{output_len} >= {case.max_model_len}"
            )
        row = [f"{prefix}-{index:04d}", input_len, output_len, float(index * 1000)]
        if speculative_acceptance is not None:
            row.append(json.dumps(speculative_acceptance, separators=(",", ":")))
        writer.writerow(row)
    return buffer.getvalue()


# ── config bodies ────────────────────────────────────────────────────────────

def _extra_args(variant: Variant, case: Case) -> list[str]:
    """Variant flags plus the context limit in the target engine's spelling.

    Both known spellings are reserved, including ``--flag=value`` and vLLM's
    underscore aliases, so a variant cannot introduce a second static limit
    beside the per-case value.
    """
    try:
        flag = CONTEXT_LIMIT_FLAG[variant.engine]
    except KeyError:
        raise PackError(
            f"variants.{variant.name}.engine {variant.engine!r} has no known context-limit "
            f"flag; known engines: {sorted(CONTEXT_LIMIT_FLAG)}"
        ) from None
    args = [str(item) for item in variant.server.get("extra_args", [])]
    for argument in args:
        argument_name = argument.partition("=")[0].replace("_", "-")
        if argument_name in CONTEXT_LIMIT_FLAG.values():
            raise PackError(
                f"variants.{variant.name}.server.extra_args must not set {argument_name}; "
                f"the context limit is derived from each case's max_model_len and "
                f"rendered as {flag} for engine {variant.engine!r}"
            )
    return args + [flag, str(case.max_model_len)]


def _server_block(pack: Pack, variant: Variant, case: Case, host: HostProfile) -> dict[str, Any]:
    body = {
        "model_path": host.checkpoint_path(variant.checkpoint),
        "port": host.port,
    }
    for key, value in variant.server.items():
        if key == "extra_args":
            continue
        body[key] = value
    body["gpu_memory_utilization"] = case.gpu_memory_utilization
    if case.chunk_size is not None:
        body["chunk_size"] = case.chunk_size
    body["startup_timeout"] = host.startup_timeout
    body["extra_args"] = _extra_args(variant, case)
    return body


def _workload_block(
    variant: Variant, case: Case, host: HostProfile, trace_name: str, profile_kind: str
) -> dict[str, Any]:
    backend = variant.backend
    if profile_kind == "token_corpus":
        backend = routes_backend(backend)
    body: dict[str, Any] = {
        "frontend": {"type": IndependentFrontendConfig.type, "path": f"./{trace_name}"},
        "backend": {"type": backend},
        "text_file": host.text_corpus,
        "tokenizer": host.checkpoint_path(variant.tokenizer),
        "token_pool_limit": 1_000_000,
        "arrival_mode": case.profile_arrival_mode,
    }
    if case.max_concurrency is not None:
        body["max_concurrency"] = case.max_concurrency
    if case.rate is not None:
        body["rate"] = case.rate.value
    body["max_model_len"] = case.max_model_len
    return body


def profile_document(
    pack: Pack,
    case: Case,
    variant: Variant,
    host: HostProfile,
    pass_name: str,
    repo_root: Path,
) -> dict[str, Any]:
    """One `alignment profile` config — the `<pass.name>.yaml` of a case."""
    profile_pass = variant.pass_named(pass_name)
    if profile_pass is None:
        raise PackError(f"variants.{variant.name} declares no profile pass {pass_name!r}")
    short = pass_name.removeprefix("profile_")
    prefix = variant.run_name_prefix or pack.name
    trace_name = (
        KERNEL_TRACE_NAME
        if profile_pass.trace == "kernel" and case.kernel_trace is not None
        else WORKLOAD_TRACE_NAME
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "name": f"{prefix}_{case.slug}_{short}",
        "log_dir": f"./{pass_name}",
        "gpu": variant.gpu,
        "cuda_visible_devices": host.devices(case.device_role),
        "profile_kind": profile_pass.kind,
        "engine": variant.engine,
    }
    if host.fork_python:
        document["fork_python"] = _absolute(host.fork_python, repo_root)
    if variant.python_runtime is not None:
        document["python_runtime"] = dict(variant.python_runtime)
    document["server"] = _server_block(pack, variant, case, host)
    if profile_pass.kind == "nsys":
        nsys: dict[str, Any] = {
            "executable": host.nsys_executable,
            "capture_mode": "cuda_profiler_api",
            "capture_duration_seconds": case.capture_seconds,
            "cuda_graph_trace": "node",
            "cuda_event_trace": False,
        }
        if case.analyze_iterations is not None:
            # Omitted unless the case says otherwise, so the profiler's own
            # default stays the default rather than being restated per case.
            start, end = case.analyze_iterations
            nsys["analyze_iteration_start"] = start
            nsys["analyze_iteration_end"] = end
        document["nsys"] = nsys
    document["workload"] = _workload_block(variant, case, host, trace_name, profile_pass.kind)
    if profile_pass.warmup:
        document["workload"]["warmup"] = True
    return document


def timing_predict_document(variant: Variant) -> dict[str, Any]:
    kernel_pass = _kernel_pass(variant)
    return {
        "schema_version": 1,
        "simulation_preset": f"./{PHASE_CONFIG_STEMS[SIMULATION_PHASE]}.yaml",
        "profile_log_dir": f"./{kernel_pass.name}",
        "log_dir": f"./{TIMING_PREDICT_PHASE}",
        "input_builder": dict(variant.input_builder),
    }


def analyze_kernel_document(variant: Variant) -> dict[str, Any]:
    kernel_pass = _kernel_pass(variant)
    return {
        "schema_version": 1,
        "profile_log_dir": f"./{kernel_pass.name}",
        "timing_predict_log_dir": f"./{TIMING_PREDICT_PHASE}",
        "log_dir": f"./{ANALYSIS_KERNEL_PHASE}",
        "iteration": {
            "enabled": True,
            "labeled_kernel_sequences_file": "./kernel_sequences_labeled.json",
        },
    }


def analyze_e2e_document(variant: Variant) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "simulation_log_dir": f"./{SIMULATION_PHASE}",
        "workload_profile_log_dir": f"./{_workload_pass(variant).name}",
        "log_dir": f"./{ANALYSIS_E2E_PHASE}",
        "workload": {"enabled": True},
        "e2e": {"enabled": True, "throughput_bins": 20},
    }


def simulation_document(
    pack: Pack, case: Case, variant: Variant, case_dir: Path, repo_root: Path
) -> dict[str, Any]:
    """The ordinary ServingStudioSim preset.

    Unlike the four launcher-owned phase configs, a preset's paths are
    repo-root-relative, not config-relative — so this is the one document that
    cannot be written once and moved. Paths are emitted repo-relative when the
    run directory is inside the repository and absolute otherwise.
    """
    speculative = variant.arch.get("type") == "glm52_vllm_nvfp4_dsa_moe_speculative"
    if speculative != (case.speculative_acceptance is not None):
        raise PackError(f"{case.slug}: speculative architecture and acceptance must be configured together")
    if speculative:
        depth = variant.arch.get("draft_tokens", 5)
        if len(case.speculative_acceptance.value) != depth or variant.worker.get("draft_tokens", 5) != depth:
            raise PackError(f"{case.slug}: acceptance length, worker and architecture draft_tokens must agree")
    trace_name = SPECULATIVE_TRACE_NAME if speculative else WORKLOAD_TRACE_NAME
    workload: dict[str, Any] = {
        "trace_files": [_preset_path(case_dir / trace_name, repo_root)],
        "input_file_format": IndependentFrontendConfig.input_file_format,
        "arrival_mode": case.simulation_arrival_mode,
    }
    if speculative:
        workload["input_file_tags"] = ["speculative"]
    if case.max_concurrency is not None:
        workload["max_concurrency"] = case.max_concurrency
    if case.rate is not None:
        workload["request_rate"] = case.rate.value
    workload["session_dependency"] = "independent"
    workload["run_to_end"] = True

    arch = dict(variant.arch)
    if "max_model_len" in arch:
        raise PackError(
            f"variants.{variant.name}.arch must not set max_model_len; it is per-case"
        )
    arch["max_model_len"] = case.max_model_len
    # Both measured-routing artifacts are named pack-relative in a variant and
    # must reach the simulator as paths it can resolve from the repository root.
    # A corpus manifest is resolved relative to the process, so leaving it
    # pack-relative fails the build after the capture has already run.
    for key in ("expert_popularity_file", "token_corpus_file"):
        reference = arch.get(key)
        if isinstance(reference, str) and reference and not reference.startswith("hf://"):
            arch[key] = _preset_path(pack.root / reference, repo_root)

    worker = dict(variant.worker)
    for derived in ("attn_gpu_memory_gb", "gpu_time_multiplier"):
        if derived in worker:
            raise PackError(
                f"variants.{variant.name}.worker must not set {derived}; "
                "attn_gpu_memory_gb is per-case calibrated and gpu_time_multiplier "
                "is injected by the simulation phase from the kernel-align result"
            )
    worker["attn_gpu_memory_gb"] = case.attn_gpu_memory_gb.value
    if case.chunk_size is not None:
        worker["max_batch_tokens"] = case.chunk_size
    # Rendered as the neutral 1.0 on purpose: `alignment sim` overrides it with
    # the multiplier read out of the completed kernel-align artifact, so a preset
    # that already carried one would make the source of the number ambiguous.
    worker["gpu_time_multiplier"] = 1.0

    document = {
        "deployment": variant.deployment,
        "workload": workload,
        "io": {"log_dir": _preset_path(case_dir / SIMULATION_PHASE, repo_root)},
        "pools": {
            "main": {
                "groups": [
                    {
                        "gpu": variant.gpu,
                        "replicas": variant.replicas,
                        "arch": arch,
                        "worker": worker,
                    }
                ]
            }
        },
    }
    for key, value in variant.raw_overrides.items():
        document[key] = value
    return document


def _kernel_pass(variant: Variant):
    for item in variant.profile_passes:
        if item.kind == "nsys":
            return item
    raise PackError(
        f"variants.{variant.name} has no nsys profile pass; kernel alignment needs one"
    )


def _workload_pass(variant: Variant):
    for item in variant.profile_passes:
        if item.kind == "workload_metrics":
            return item
    raise PackError(
        f"variants.{variant.name} has no workload_metrics profile pass; "
        "workload/e2e alignment needs one"
    )


def _absolute(value: str, base: Path) -> str:
    """Anchor a host path against the repository, without resolving its final
    symlink — a venv's `bin/python` must keep its lexical path or
    `site-packages` discovery moves. A relative host path means "inside this
    repository", which is how `fork_python` reaches the engine venv."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return os.path.abspath(os.path.normpath(candidate))


def _preset_path(path: Path, repo_root: Path) -> str:
    path = Path(os.path.abspath(path))
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


# ── writing ──────────────────────────────────────────────────────────────────

def case_documents(
    pack: Pack, case: Case, host: HostProfile, case_dir: Path, repo_root: Path
) -> dict[str, dict[str, Any]]:
    """Every phase config for one case, keyed by file name. Pure — the CPU tier
    renders into a temp dir and the migration diff renders into memory."""
    variant = pack.variant_of(case)
    documents: dict[str, dict[str, Any]] = {}
    for profile_pass in variant.profile_passes:
        documents[f"{profile_pass.name}.yaml"] = profile_document(
            pack, case, variant, host, profile_pass.name, repo_root
        )
    documents[f"{PHASE_CONFIG_STEMS[TIMING_PREDICT_PHASE]}.yaml"] = timing_predict_document(variant)
    documents[f"{PHASE_CONFIG_STEMS[ANALYSIS_KERNEL_PHASE]}.yaml"] = analyze_kernel_document(
        variant
    )
    documents[f"{PHASE_CONFIG_STEMS[ANALYSIS_E2E_PHASE]}.yaml"] = analyze_e2e_document(variant)
    documents[f"{PHASE_CONFIG_STEMS[SIMULATION_PHASE]}.yaml"] = simulation_document(
        pack, case, variant, case_dir, repo_root
    )
    return documents


def case_traces(pack: Pack, case: Case) -> dict[str, str]:
    """Trace file name → contents for one rendered case directory."""
    traces = {WORKLOAD_TRACE_NAME: trace_text(case, case.workload_trace)}
    if case.speculative_acceptance is not None:
        traces[SPECULATIVE_TRACE_NAME] = trace_text(
            case, case.workload_trace, speculative_acceptance=case.speculative_acceptance.value
        )
    if case.kernel_trace is not None:
        traces[KERNEL_TRACE_NAME] = trace_text(case, case.kernel_trace)
    return traces


def render_case(
    pack: Pack, case: Case, host: HostProfile, out_root: Path, repo_root: Path
) -> RenderedCase:
    """Write one case's run directory. Overwrites configs and traces; never
    touches artifact directories, so re-rendering after a config fix does not
    discard an expensive capture."""
    case_dir = Path(out_root).resolve() / case.slug
    case_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in case_traces(pack, case).items():
        target = case_dir / name
        target.write_text(text, encoding="utf-8")
        written.append(target)
    for name, document in case_documents(pack, case, host, case_dir, repo_root).items():
        target = case_dir / name
        target.write_text(
            yaml.safe_dump(document, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        written.append(target)
    return RenderedCase(
        case_slug=case.slug,
        directory=case_dir,
        phases=phase_names(pack.variant_of(case)),
        files=tuple(written),
    )
