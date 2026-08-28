"""Strict launcher-owned schemas for the three alignment-specific phases.

The ordinary simulation phase keeps using the normal VibeSim preset schema.
Every path parsed here is resolved relative to the file that declared it, so a
dated experiment directory can be moved as one self-contained unit. Runtime
modules receive typed values and never parse YAML or JSON themselves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from alignment.load_generator.config import LoadGeneratorConfig
from alignment.profiler.config import IdleWaitConfig, NsysConfig, ProfileConfig, ServerConfig
from alignment.timing_predict_input import EngineTextInputSpec

from .schema.loader import PresetError, _load_preset

CONFIG_SUFFIXES = frozenset({".json", ".yaml", ".yml"})


@dataclass(frozen=True)
class TimingPredictPhaseConfig:
    """Sim preset plus profile artifacts and builder policy for one offline
    prediction phase.

    Timing-predict is kernel-only and multiplier-independent, so it reads the
    simulation *preset* (for gpu / arch / backends) rather than a completed run.
    That lets it run before the simulation, so the kernel-align analysis can
    derive the gpu_time_multiplier that the simulation then bakes in.
    """

    simulation_preset: Path
    profile_log_dir: Path
    log_dir: Path
    input_builder: EngineTextInputSpec


# Every subject defaults to disabled so a phase is opted into explicitly: a
# kernel-align config names only `iteration`, an e2e-align config names only
# `workload`/`e2e`. An absent block therefore means "not this phase", which is
# what keeps the two phases from silently mixing.
@dataclass(frozen=True)
class IterationAnalysisPolicy:
    enabled: bool = False
    labeled_kernel_sequences_file: Path | None = None


@dataclass(frozen=True)
class E2EAnalysisPolicy:
    enabled: bool = False
    throughput_bins: int = 20


@dataclass(frozen=True)
class WorkloadAnalysisPolicy:
    """Enable scheduler-workload comparison by each run's iteration ids."""

    enabled: bool = False


@dataclass(frozen=True)
class AnalyzePhaseConfig:
    """Completed artifacts and analyzer policy for one alignment analysis pass.

    `simulation_log_dir` is optional: the kernel-align pass (iteration only) runs
    before any simulation and needs none. It is required only when a sim-consuming
    subject (workload, e2e) is enabled.
    """

    simulation_log_dir: Path | None
    profile_log_dir: Path | None
    workload_profile_log_dir: Path | None
    timing_predict_log_dir: Path | None
    log_dir: Path
    iteration: IterationAnalysisPolicy
    workload: WorkloadAnalysisPolicy
    e2e: E2EAnalysisPolicy

    @property
    def subjects(self) -> list[str]:
        selected = []
        if self.iteration.enabled:
            selected.append("alignment-iteration")
            # Same inputs, same measured reduction; it keeps the per-kernel
            # timestamps `alignment-iteration` reduces away, so the two lanes can
            # be drawn on one axis. Never useful without its sibling.
            selected.append("alignment-timeline")
        if self.workload.enabled:
            selected.append("alignment-workload")
        if self.e2e.enabled:
            selected.append("alignment-e2e")
        return selected


def load_profile_config(path: Path) -> ProfileConfig:
    """Load the profile-only schema and normalize its filesystem inputs."""
    raw = _phase_document(path, "profile")
    base = path.resolve().parent
    try:
        server_raw = _pop_mapping(raw, "server", "profile")
        workload_raw = _pop_mapping(raw, "workload", "profile")
        idle_raw = _pop_optional_mapping(raw, "idle", "profile")
        nsys_raw = _pop_optional_mapping(raw, "nsys", "profile")
        server = ServerConfig(**server_raw)
        workload = LoadGeneratorConfig.from_mapping(workload_raw)
        workload = replace(
            workload,
            frontend=replace(
                workload.frontend,
                path=str(_config_path(base, workload.frontend.path, "workload.frontend.path")),
            ),
            text_file=str(_config_path(base, workload.text_file, "workload.text_file")),
        )
        idle = IdleWaitConfig(**idle_raw)
        nsys = NsysConfig(**nsys_raw)
        nsys.validate()
        log_dir = _config_path(base, raw.pop("log_dir"), "log_dir")
        fork_python = raw.pop("fork_python", "")
        if fork_python:
            fork_python = str(_config_path_preserving_symlink(base, fork_python, "fork_python"))
        driver_compat_lib_dir = raw.pop("driver_compat_lib_dir", "")
        if driver_compat_lib_dir:
            driver_compat_lib_dir = str(
                _config_path(base, driver_compat_lib_dir, "driver_compat_lib_dir")
            )
        config = ProfileConfig(
            log_dir=str(log_dir),
            server=server,
            workload=workload,
            idle=idle,
            nsys=nsys,
            fork_python=fork_python,
            driver_compat_lib_dir=driver_compat_lib_dir,
            **raw,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid profile config: {exc}") from exc
    _require_nonempty(config.name, "name")
    _require_nonempty(config.gpu, "gpu")
    if config.server.tp_size <= 0:
        raise ValueError("invalid profile config: server.tp_size must be > 0")
    if config.server.dp_size <= 0:
        raise ValueError("invalid profile config: server.dp_size must be > 0")
    if config.engine not in {"vllm", "sglang"}:
        raise ValueError(
            f"invalid profile config: engine must be 'vllm' or 'sglang', got {config.engine!r}"
        )
    if config.profile_kind not in {"nsys", "expert_popularity", "workload_metrics"}:
        raise ValueError(
            "invalid profile config: profile_kind must be 'nsys', "
            f"'expert_popularity', or 'workload_metrics', got {config.profile_kind!r}"
        )
    visible_devices = [
        device.strip() for device in config.cuda_visible_devices.split(",") if device.strip()
    ]
    world_size = config.server.tp_size * config.server.dp_size
    if len(visible_devices) != world_size:
        raise ValueError(
            "invalid profile config: cuda_visible_devices must contain exactly "
            f"tp_size * dp_size={world_size} devices; found {visible_devices}"
        )
    if len(set(visible_devices)) != len(visible_devices):
        raise ValueError("invalid profile config: cuda_visible_devices contains duplicates")
    if config.fork_python and not Path(config.fork_python).is_file():
        raise ValueError(
            f"invalid profile config: fork_python does not exist: {config.fork_python}"
        )
    if config.driver_compat_lib_dir:
        compat_library = Path(config.driver_compat_lib_dir) / "libcuda.so.1"
        if not compat_library.is_file():
            raise ValueError(
                "invalid profile config: driver_compat_lib_dir holds no libcuda.so.1: "
                f"{config.driver_compat_lib_dir}"
            )
    return config


def load_timing_predict_config(path: Path) -> TimingPredictPhaseConfig:
    """Load one timing-input build and offline-predict phase config."""
    raw = _phase_document(path, "timing-predict")
    base = path.resolve().parent
    try:
        builder_raw = _pop_mapping(raw, "input_builder", "timing-predict")
        builder_type = builder_raw.pop("type", None)
        # `vllm_text` is the pre-SGLang spelling of the same builder, kept so
        # configs written against the single-engine pipeline still load.
        if builder_type not in {"engine_text", "vllm_text"}:
            raise ValueError(
                f"unsupported input_builder.type {builder_type!r}; "
                "available: ['engine_text'] (legacy alias: 'vllm_text')"
            )
        builder = EngineTextInputSpec(**builder_raw)
        builder.validate()
        config = TimingPredictPhaseConfig(
            simulation_preset=_config_path(base, raw.pop("simulation_preset"), "simulation_preset"),
            profile_log_dir=_config_path(base, raw.pop("profile_log_dir"), "profile_log_dir"),
            log_dir=_config_path(base, raw.pop("log_dir"), "log_dir"),
            input_builder=builder,
        )
        _reject_extra(raw, "timing-predict")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid timing-predict config: {exc}") from exc
    _require_distinct(
        {
            "profile_log_dir": config.profile_log_dir,
            "log_dir": config.log_dir,
        }
    )
    return config


def load_analyze_config(path: Path) -> AnalyzePhaseConfig:
    """Load the final analyzer phase without consulting an earlier config file."""
    raw = _phase_document(path, "analyze")
    base = path.resolve().parent
    try:
        iteration_raw = _pop_optional_mapping(raw, "iteration", "analyze")
        workload_raw = _pop_optional_mapping(raw, "workload", "analyze")
        e2e_raw = _pop_optional_mapping(raw, "e2e", "analyze")
        sequences_text = iteration_raw.pop("labeled_kernel_sequences_file", None)
        iteration = IterationAnalysisPolicy(
            labeled_kernel_sequences_file=(
                _config_path(
                    base,
                    sequences_text,
                    "iteration.labeled_kernel_sequences_file",
                )
                if sequences_text is not None
                else None
            ),
            **iteration_raw,
        )
        workload = WorkloadAnalysisPolicy(**workload_raw)
        e2e = E2EAnalysisPolicy(**e2e_raw)
        if iteration.enabled and iteration.labeled_kernel_sequences_file is None:
            raise ValueError(
                "iteration.labeled_kernel_sequences_file is required when iteration.enabled is true"
            )
        if e2e.throughput_bins <= 0:
            raise ValueError("e2e.throughput_bins must be > 0")
        # One analyze config is exactly one phase. kernel-align (iteration) and
        # e2e-align (workload/e2e) read disjoint inputs and write distinct
        # manifest shapes, so mixing them in one config is rejected rather than
        # papered over with optional fields.
        if iteration.enabled and (workload.enabled or e2e.enabled):
            raise ValueError(
                "an analyze config is one phase: enable iteration (kernel-align) "
                "or workload/e2e (e2e-align), not both"
            )
        # Only the e2e-align phase consumes a completed simulation; the
        # kernel-align pass runs before the sim and omits it.
        simulation_raw = raw.pop("simulation_log_dir", None)
        if (workload.enabled or e2e.enabled) and simulation_raw is None:
            raise ValueError(
                "simulation_log_dir is required when the workload or e2e subject is enabled"
            )
        simulation_log_dir = (
            _config_path(base, simulation_raw, "simulation_log_dir")
            if simulation_raw is not None
            else None
        )
        profile_raw = raw.pop("profile_log_dir", None)
        workload_profile_raw = raw.pop("workload_profile_log_dir", None)
        if iteration.enabled and profile_raw is None:
            raise ValueError("profile_log_dir is required for kernel alignment")
        if (workload.enabled or e2e.enabled) and workload_profile_raw is None:
            raise ValueError(
                "workload_profile_log_dir is required for workload or e2e alignment"
            )
        if iteration.enabled and workload_profile_raw is not None:
            raise ValueError(
                "workload_profile_log_dir does not belong to kernel alignment"
            )
        if (workload.enabled or e2e.enabled) and profile_raw is not None:
            raise ValueError("profile_log_dir does not belong to workload/e2e alignment")
        profile_log_dir = (
            _config_path(base, profile_raw, "profile_log_dir")
            if profile_raw is not None
            else None
        )
        workload_profile_log_dir = (
            _config_path(base, workload_profile_raw, "workload_profile_log_dir")
            if workload_profile_raw is not None
            else None
        )
        timing_predict_raw = raw.pop("timing_predict_log_dir", None)
        if iteration.enabled and timing_predict_raw is None:
            raise ValueError(
                "timing_predict_log_dir is required when the iteration subject is enabled"
            )
        config = AnalyzePhaseConfig(
            simulation_log_dir=simulation_log_dir,
            profile_log_dir=profile_log_dir,
            workload_profile_log_dir=workload_profile_log_dir,
            timing_predict_log_dir=(
                _config_path(base, timing_predict_raw, "timing_predict_log_dir")
                if timing_predict_raw is not None
                else None
            ),
            log_dir=_config_path(base, raw.pop("log_dir"), "log_dir"),
            iteration=iteration,
            workload=workload,
            e2e=e2e,
        )
        _reject_extra(raw, "analyze")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid analyze config: {exc}") from exc
    if not config.subjects:
        raise ValueError("invalid analyze config: at least one analysis subject must be enabled")
    distinct = {"log_dir": config.log_dir}
    if config.profile_log_dir is not None:
        distinct["profile_log_dir"] = config.profile_log_dir
    if config.workload_profile_log_dir is not None:
        distinct["workload_profile_log_dir"] = config.workload_profile_log_dir
    if config.timing_predict_log_dir is not None:
        distinct["timing_predict_log_dir"] = config.timing_predict_log_dir
    if config.simulation_log_dir is not None:
        distinct["simulation_log_dir"] = config.simulation_log_dir
    _require_distinct(distinct)
    return config


def _validate_iteration_list(value: Any, context: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, int) and item >= 0 for item in value)
    ):
        raise ValueError(f"{context} must be a non-empty list of non-negative integers")
    return value


def _labeled_sequence_positions(
    sequence: dict[str, Any], context: str, schema_version: int
) -> set[tuple[int | None, int]]:
    """The measured positions one labeled sequence claims.

    Schema 4 stores a union catalog, so a sequence names the exact devices that
    executed it; earlier schemas stored one representative sequence covering
    every device, leaving the iteration as the whole position.
    """
    if schema_version < 4:
        return {
            (None, iteration)
            for iteration in _validate_iteration_list(
                sequence["iterations"], f"{context}.iterations"
            )
        }

    occurrences = sequence["occurrences"]
    if not isinstance(occurrences, list) or not occurrences:
        raise ValueError(f"{context}.occurrences must be a non-empty list")
    positions: set[tuple[int | None, int]] = set()
    seen_devices: set[int] = set()
    for occurrence_index, occurrence in enumerate(occurrences):
        occurrence_context = f"{context}.occurrences[{occurrence_index}]"
        if not isinstance(occurrence, dict) or set(occurrence) != {"device_id", "iterations"}:
            raise ValueError(f"{occurrence_context} must hold only device_id and iterations")
        device_id = occurrence["device_id"]
        if not isinstance(device_id, int) or device_id < 0:
            raise ValueError(f"{occurrence_context}.device_id must be a non-negative integer")
        if device_id in seen_devices:
            raise ValueError(f"{occurrence_context} repeats device {device_id}")
        seen_devices.add(device_id)
        for iteration in _validate_iteration_list(
            occurrence["iterations"], f"{occurrence_context}.iterations"
        ):
            positions.add((device_id, iteration))
    return positions


def load_labeled_kernel_sequences(path: Path) -> dict[str, Any]:
    """Validate a self-contained compact or occurrence-addressable inventory."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"labeled kernel sequences must be JSON: {path}")
    raw = _document(path, "labeled kernel sequences")
    schema_version = raw.get("schema_version")
    required = {"schema_version", "encoding", "source_parsed", "folding_policy", "phases"}
    if schema_version == 3:
        required.update({"device_ids", "representative_device_id"})
    if schema_version in {4, 5}:
        required.add("device_ids")
    extra = set(raw) - required
    missing = required - set(raw)
    if extra or missing:
        raise ValueError(
            "labeled kernel sequence keys mismatch: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    if schema_version not in {2, 3, 4, 5} or raw["encoding"] not in {
        "folded-v1",
        "folded-v2",
        "literal-v1",
    }:
        raise ValueError(
            "labeled kernel sequences require schema_version 2, 3, 4 or 5 and encoding "
            "folded-v1, folded-v2 or literal-v1"
        )
    if schema_version in {3, 4, 5}:
        device_ids = raw["device_ids"]
        if (
            not isinstance(device_ids, list)
            or not device_ids
            or any(not isinstance(device_id, int) or device_id < 0 for device_id in device_ids)
            or len(set(device_ids)) != len(device_ids)
            or device_ids != sorted(device_ids)
        ):
            raise ValueError(
                f"schema-v{schema_version} device_ids must be sorted unique nonnegative integers"
            )
    if schema_version == 3 and raw["representative_device_id"] != raw["device_ids"][0]:
        raise ValueError("schema-v3 representative_device_id must be the first device_ids entry")
    phases = raw["phases"]
    if not isinstance(phases, dict) or not phases:
        raise ValueError("labeled kernel sequences phases must be a non-empty mapping")

    operation_signatures: dict[str, tuple[tuple[str, ...], str, str]] = {}
    slots: dict[str, set[str]] = {}
    for phase_name, phase in phases.items():
        _require_nonempty(phase_name, "phase name")
        if not isinstance(phase, dict) or set(phase) != {"unique_sequences"}:
            raise ValueError(f"phase {phase_name!r} must contain only unique_sequences")
        sequences = phase["unique_sequences"]
        if not isinstance(sequences, list) or not sequences:
            raise ValueError(f"phase {phase_name!r} unique_sequences must be non-empty")
        sequence_ids: set[str] = set()
        # One measured position — a (device, iteration) pair — executes exactly one
        # sequence, so it may be claimed once. Before schema 4 the inventory held a
        # single representative sequence applied to every device, so the position
        # was the iteration alone.
        assigned_positions: set[tuple[int | None, int]] = set()
        position_label = (
            "(device, iteration) pairs" if schema_version >= 4 else "iterations"
        )
        occurrence_key = "occurrences" if schema_version >= 4 else "iterations"
        for index, sequence in enumerate(sequences):
            context = f"phases.{phase_name}.unique_sequences[{index}]"
            # Schema 5 splits a sequence into concurrent tracks. Older documents
            # carry the single implicit track as a bare `program`.
            program_key = "tracks" if schema_version >= 5 else "program"
            if not isinstance(sequence, dict) or set(sequence) != {
                "sequence_id",
                occurrence_key,
                "expanded_kernel_count",
                program_key,
            }:
                raise ValueError(f"{context} has invalid keys")
            sequence_id = sequence["sequence_id"]
            _require_nonempty(sequence_id, f"{context}.sequence_id")
            if sequence_id in sequence_ids:
                raise ValueError(f"{context} duplicates sequence_id {sequence_id!r}")
            sequence_ids.add(sequence_id)
            positions = _labeled_sequence_positions(sequence, context, schema_version)
            overlap = assigned_positions.intersection(positions)
            if overlap:
                raise ValueError(
                    f"phase {phase_name!r} assigns {position_label} twice: {sorted(overlap)}"
                )
            assigned_positions.update(positions)
            if schema_version >= 5:
                expanded = _validate_labeled_tracks(
                    sequence["tracks"], context, operation_signatures, slots
                )
            else:
                expanded = _validate_labeled_program(
                    sequence["program"], context, operation_signatures, slots
                )
            if sequence["expanded_kernel_count"] != expanded:
                raise ValueError(
                    f"{context}.expanded_kernel_count "
                    f"{sequence['expanded_kernel_count']} != {expanded}"
                )
    return raw


def _validate_labeled_tracks(
    tracks: Any,
    context: str,
    operations: dict[str, tuple[tuple[str, ...], str, str]],
    slots: dict[str, set[str]],
) -> int:
    """Validate a schema-5 sequence's concurrent tracks and total its kernels.

    Track indices must be dense and in order: they are the coordinate a labeling
    decision is addressed by, and a gap would make two inventories of the same
    capture disagree on where a label belongs.
    """
    if not isinstance(tracks, list) or not tracks:
        raise ValueError(f"{context}.tracks must be a non-empty list")
    expanded = 0
    for track_index, track in enumerate(tracks):
        track_context = f"{context}.tracks[{track_index}]"
        if not isinstance(track, dict) or set(track) != {
            "track_index",
            "stream_role",
            "kernel_count",
            "program",
        }:
            raise ValueError(f"{track_context} has invalid keys")
        if track["track_index"] != track_index:
            raise ValueError(
                f"{track_context}.track_index {track['track_index']} != {track_index}"
            )
        if track["stream_role"] not in {"primary", "concurrent"}:
            raise ValueError(f"{track_context}.stream_role must be primary or concurrent")
        track_expanded = _validate_labeled_program(
            track["program"], track_context, operations, slots
        )
        if track["kernel_count"] != track_expanded:
            raise ValueError(
                f"{track_context}.kernel_count {track['kernel_count']} != {track_expanded}"
            )
        expanded += track_expanded
    return expanded


def _validate_labeled_program(
    program: Any,
    context: str,
    operations: dict[str, tuple[tuple[str, ...], str, str]],
    slots: dict[str, set[str]],
) -> int:
    if not isinstance(program, list) or not program:
        raise ValueError(f"{context}.program must be non-empty")
    expanded = 0
    for node_index, node in enumerate(program):
        node_context = f"{context}.program[{node_index}]"
        if not isinstance(node, dict) or len(node) != 1:
            raise ValueError(f"{node_context} must be one tagged node")
        if "kernels" in node:
            kernels = node["kernels"]
            count = 1
        elif "repeat" in node:
            repeat = node["repeat"]
            if not isinstance(repeat, dict) or set(repeat) != {"count", "body"}:
                raise ValueError(f"{node_context}.repeat has invalid keys")
            count = repeat["count"]
            if not isinstance(count, int) or count < 2:
                raise ValueError(f"{node_context}.repeat.count must be >= 2")
            body = repeat["body"]
            if not isinstance(body, dict) or set(body) != {"kernels"}:
                raise ValueError(f"{node_context}.repeat.body must contain only kernels")
            kernels = body["kernels"]
        else:
            raise ValueError(f"{node_context} must contain kernels or repeat")
        if not isinstance(kernels, list) or not kernels:
            raise ValueError(f"{node_context} kernels must be non-empty")
        for kernel_index, kernel in enumerate(kernels):
            _validate_labeled_kernel(
                kernel,
                f"{node_context}.kernels[{kernel_index}]",
                operations,
                slots,
            )
        expanded += len(kernels) * count
    return expanded


def _validate_labeled_kernel(
    kernel: Any,
    context: str,
    operations: dict[str, tuple[tuple[str, ...], str, str]],
    slots: dict[str, set[str]],
) -> None:
    if not isinstance(kernel, dict) or set(kernel) != {"name", "suggested_category", "label"}:
        raise ValueError(f"{context} must contain name, suggested_category, and label")
    _require_nonempty(kernel["name"], f"{context}.name")
    _require_nonempty(kernel["suggested_category"], f"{context}.suggested_category")
    label = kernel["label"]
    if not isinstance(label, dict) or "status" not in label:
        raise ValueError(f"{context}.label must have explicit status")
    # cross_rank is an optional per-kernel reduction class, valid on both mapped
    # and unmapped labels; the analyzer consumes it independently of mapping.
    cross_rank = label.get("cross_rank")
    if cross_rank is not None and cross_rank not in {"synchronizing", "independent"}:
        raise ValueError(f"{context}.label.cross_rank must be synchronizing or independent")
    mapping_keys = set(label) - {"cross_rank"}
    if label["status"] == "unmapped":
        if mapping_keys != {"status"}:
            raise ValueError(f"{context} unmapped label cannot contain mapping fields")
        if cross_rank is None:
            raise ValueError(f"{context} unmapped label must declare cross_rank")
        return
    if label["status"] != "mapped":
        raise ValueError(f"{context}.label.status must be mapped or unmapped")
    expected = {"status", "operation", "simulated_slots", "type", "role"}
    if mapping_keys != expected:
        raise ValueError(f"{context} mapped label must contain {sorted(expected)}")
    for field in expected - {"status", "simulated_slots"}:
        _require_nonempty(label[field], f"{context}.label.{field}")
    simulated_slots = label["simulated_slots"]
    if (
        not isinstance(simulated_slots, list)
        or not simulated_slots
        or not all(isinstance(slot, str) and slot.strip() for slot in simulated_slots)
    ):
        raise ValueError(f"{context}.label.simulated_slots must be a non-empty string list")
    if len(set(simulated_slots)) != len(simulated_slots):
        raise ValueError(f"{context}.label.simulated_slots cannot contain duplicates")
    operation = label["operation"]
    signature = (tuple(simulated_slots), label["type"], label["role"])
    old_signature = operations.setdefault(operation, signature)
    if old_signature != signature:
        raise ValueError(f"operation {operation!r} has inconsistent label metadata")
    # A slot may be declared by several operations (a fused aggregate boundary
    # and an unfused split boundary share the same tp_allreduce slot); the
    # analyzer resolves the owner per iteration from the operations present.
    for simulated_slot in simulated_slots:
        slots.setdefault(simulated_slot, set()).add(operation)


def _phase_document(path: Path, role: str) -> dict[str, Any]:
    raw = _document(path, f"{role} config")
    version = raw.pop("schema_version", None)
    if version != 1:
        raise ValueError(f"{role} config schema_version must be 1")
    return raw


def _document(path: Path, role: str) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() not in CONFIG_SUFFIXES:
        raise ValueError(f"{role} must be YAML or JSON: {path}")
    try:
        return dict(_load_preset(path))
    except (OSError, PresetError) as exc:
        raise ValueError(str(exc)) from exc


def _pop_mapping(raw: dict, field: str, role: str) -> dict:
    value = raw.pop(field)
    if not isinstance(value, dict):
        raise ValueError(f"{role}.{field} must be a mapping")
    return dict(value)


def _pop_optional_mapping(raw: dict, field: str, role: str) -> dict:
    value = raw.pop(field, {})
    if not isinstance(value, dict):
        raise ValueError(f"{role}.{field} must be a mapping")
    return dict(value)


def _config_path(base: Path, value: Any, field: str) -> Path:
    _require_nonempty(value, field)
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _config_path_preserving_symlink(base: Path, value: Any, field: str) -> Path:
    """Anchor an executable path without escaping its virtual environment.

    A venv's ``bin/python`` is normally a symlink to the base interpreter.
    Resolving that final symlink changes how Python discovers site-packages, so
    ``fork_python`` must retain the lexical venv path passed to the subprocess.
    """
    _require_nonempty(value, field)
    path = Path(value)
    anchored = path if path.is_absolute() else base / path
    return Path(os.path.abspath(os.path.normpath(anchored)))


def _require_nonempty(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _reject_extra(raw: dict, role: str) -> None:
    if raw:
        raise ValueError(f"unknown {role} config keys: {sorted(raw)}")


def _require_distinct(paths: dict[str, Path]) -> None:
    by_path: dict[Path, list[str]] = {}
    for name, path in paths.items():
        by_path.setdefault(path, []).append(name)
    overlaps = [names for names in by_path.values() if len(names) > 1]
    if overlaps:
        raise ValueError(f"phase artifact directories must be distinct: {overlaps}")
