"""The declarative half of an alignment campaign: packs and host profiles.

A **pack** is one `model x precision x GPU` alignment matrix, stored as data:

    presets/alignment/<pack>/
      campaign.yaml           variants (topology) + cases (the matrix itself)
      acceptance.yaml         tolerances + rationale (formulas live in metrics.py)
      expert_popularity.json  measured routing demand the numbers depend on
      label_rules/            the order-independent rule set
      traces/                 the workload/kernel traces + invariants.json

A **host profile** (`presets/alignment/hosts/<host>.yaml`) carries everything
that is true of a machine rather than of the workload: checkpoint locations, the
nsys binary, the instrumented-fork interpreter, which physical devices a role
maps to. No tracked preset in this repo contains a host absolute path, and a
pack keeps that property — the absolute paths appear only in rendered configs,
which live under `logs/` and are not tracked.

Host profiles are therefore **local files**: only `hosts/example.yaml` is
tracked, and a real one is gitignored. Its paths are an HF cache and a 1 GB
corpus, which cannot be committed and which name whoever's directory they happen
to sit in. What is genuinely reproducible about the recording machine — the
checkpoint snapshot hash, the profiler version, the device split — is not a path
and does not need a runnable config, so it is recorded as the pack's
`recorded_on` block instead.

The split that matters inside a case is **authored** versus **calibrated**.
`max_concurrency` is a design choice; `attn_gpu_memory_gb` is read off a
workload-only startup log and `rate` off a saturation sweep. A calibrated field
is a mapping (`value` + `status` + `derived_from`) rather than a bare scalar, so
re-rendering on a different machine cannot silently reuse a stale KV budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schema.loader import PresetError, _load_preset

#: Repository root. Rendered configs anchor to it and recorded provenance is
#: made relative to it, so neither names the machine it was produced on.
REPO_ROOT = Path(__file__).resolve().parents[2]

PACK_SCHEMA_VERSION = 1
HOST_SCHEMA_VERSION = 1

#: `profile_kind` values `launcher.alignment_config.load_profile_config` accepts.
PROFILE_KINDS = frozenset({"nsys", "expert_popularity", "workload_metrics"})

#: Which of a case's two traces a profile pass drives. `kernel` is the bounded
#: NSYS capture; `workload` is the full run that e2e alignment consumes. A case
#: without a kernel trace uses its workload trace for both.
TRACE_ROLES = frozenset({"kernel", "workload"})

#: `status` values a calibrated field may declare. `measured` was read off a
#: preflight run; `provisional` is a placeholder that preflight has not replaced
#: yet. Recording a golden over provisional inputs is possible but must be said
#: out loud (`compare --record --accept-provisional`).
CALIBRATION_STATUSES = frozenset({"measured", "provisional"})


class PackError(ValueError):
    """A pack or host profile that cannot be loaded at all."""


@dataclass(frozen=True)
class Finding:
    """One `check` result. `error` makes the pack unusable; `warn` is a smell
    (an unacknowledged provisional value, a used escape hatch) that still runs."""

    level: str  # "error" | "warn"
    where: str  # pack-relative locus, e.g. "cases[09_rate_knee].rate"
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


@dataclass(frozen=True)
class Calibrated:
    """A case value obtained by measurement rather than authored."""

    value: Any
    status: str
    derived_from: str
    evidence: str = ""

    @property
    def provisional(self) -> bool:
        return self.status == "provisional"


@dataclass(frozen=True)
class TraceSpec:
    """One request trace, as the recipe that produces it.

    `shapes` x `repeats` *is* the trace: `render` generates the CSV into the run
    directory and `check` regenerates it to compare against the sha256 in
    `traces/invariants.json`. Committing the rows too would be committing the
    same data twice, and `file` names the entry in that record rather than a
    file in the tree. `id_suffix` exists because the two source generators
    disagreed: the first wrote `<slug>-0000`, the second `<slug>-full-0000`.
    That difference is in the bytes the accepted runs consumed, so it is data,
    not something to normalize away.
    """

    file: str
    shapes: tuple[tuple[int, int], ...]
    repeats: int
    id_suffix: str | None = None

    @property
    def request_count(self) -> int:
        return len(self.shapes) * self.repeats


@dataclass(frozen=True)
class ProfilePass:
    """One measured pass. `name` is simultaneously the config stem, the artifact
    directory, and the `--phase` value, so adding a pass needs no engine edit."""

    kind: str
    name: str
    trace: str
    warmup: bool = False


@dataclass(frozen=True)
class Variant:
    """One topology. DP+EP adds a sibling entry; it is not a new pack."""

    name: str
    deployment: str
    gpu: str
    checkpoint: str  # key into the host profile's `checkpoints`
    tokenizer: str  # key into the host profile's `checkpoints`
    engine: str
    backend: str
    server: dict[str, Any]
    arch: dict[str, Any]
    worker: dict[str, Any]
    profile_passes: tuple[ProfilePass, ...]
    input_builder: dict[str, Any]
    label_rules: str
    python_runtime: dict[str, Any] | None = None
    replicas: int = 1
    #: Stem of each rendered profile config's `name:`. Purely an identity label,
    #: but pinning it lets a migrated pack reproduce an accepted config byte-wise.
    run_name_prefix: str = ""
    raw_overrides: dict[str, Any] = field(default_factory=dict)

    def pass_named(self, name: str) -> ProfilePass | None:
        for item in self.profile_passes:
            if item.name == name:
                return item
        return None


@dataclass(frozen=True)
class Case:
    """One workload in the matrix."""

    index: int
    name: str
    variant: str
    purpose: str
    coverage: tuple[str, ...]
    max_model_len: int
    max_concurrency: int | None
    profile_arrival_mode: str
    simulation_arrival_mode: str
    gpu_memory_utilization: float
    capture_seconds: float
    device_role: str
    #: `[start, end]` over the per-worker forward index that kernel alignment
    #: analyzes when a stable numbered excerpt, rather than the full capture,
    #: is the intended evidence population. `None` keeps the whole capture.
    analyze_iterations: tuple[int, int] | None
    workload_trace: TraceSpec
    kernel_trace: TraceSpec | None
    attn_gpu_memory_gb: Calibrated
    rate: Calibrated | None
    provenance: dict[str, Any]
    chunk_size: int | None = None
    speculative_acceptance: Calibrated | None = None

    @property
    def slug(self) -> str:
        """`01_micro_throughput_c256` — the run directory, the golden key, and
        the prefix of every request id in this case's traces."""
        return f"{self.index:02d}_{self.name}"

    @property
    def calibrated_fields(self) -> dict[str, Calibrated]:
        found = {"attn_gpu_memory_gb": self.attn_gpu_memory_gb}
        if self.rate is not None:
            found["rate"] = self.rate
        if self.speculative_acceptance is not None:
            found["speculative_acceptance"] = self.speculative_acceptance
        return found

    def trace_for(self, role: str) -> TraceSpec:
        """The trace a pass with this role drives. A case with no bounded kernel
        trace profiles kernels from its full workload trace, which is what cases
        01-10 do."""
        if role == "kernel" and self.kernel_trace is not None:
            return self.kernel_trace
        return self.workload_trace


@dataclass(frozen=True)
class Pack:
    root: Path
    name: str
    description: str
    variants: dict[str, Variant]
    cases: tuple[Case, ...]
    acceptance: dict[str, Any]
    recorded_on: dict[str, Any]
    """Facts about the machine that produced the accepted numbers. Free-form on
    purpose: it is read by people and by `--record`'s provenance, never to
    render anything, so it must not grow into a second host profile."""

    def case_named(self, slug_or_index: str) -> Case | None:
        for case in self.cases:
            if slug_or_index in (case.slug, case.name, f"{case.index:02d}", str(case.index)):
                return case
        return None

    def variant_of(self, case: Case) -> Variant:
        return self.variants[case.variant]

    @property
    def provisional_fields(self) -> list[str]:
        """`<slug>.<field>` for every calibrated value still awaiting preflight."""
        return [
            f"{case.slug}.{name}"
            for case in self.cases
            for name, value in case.calibrated_fields.items()
            if value.provisional
        ]


@dataclass(frozen=True)
class HostProfile:
    """Machine-bound inputs. Never part of a pack: the same matrix must render
    on a second B200 host by swapping this file alone."""

    path: Path
    name: str
    hf_hub_root: str
    checkpoints: dict[str, str]
    text_corpus: str
    nsys_executable: str
    fork_python: str
    device_roles: dict[str, str]
    port: int
    startup_timeout: float

    def checkpoint_path(self, key: str) -> str:
        try:
            relative = self.checkpoints[key]
        except KeyError:
            raise PackError(
                f"host {self.name}: no checkpoint named {key!r} "
                f"(declared: {sorted(self.checkpoints)})"
            ) from None
        candidate = Path(relative)
        if candidate.is_absolute():
            return str(candidate)
        return str(Path(self.hf_hub_root) / candidate)

    def devices(self, role: str) -> str:
        try:
            return self.device_roles[role]
        except KeyError:
            raise PackError(
                f"host {self.name}: no device role {role!r} "
                f"(declared: {sorted(self.device_roles)})"
            ) from None


# ── loading ──────────────────────────────────────────────────────────────────

def _document(path: Path, role: str) -> dict[str, Any]:
    try:
        raw = _load_preset(path)
    except (OSError, PresetError) as exc:
        raise PackError(f"{role} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PackError(f"{role} {path}: must be a mapping")
    return dict(raw)


def _require(raw: dict, key: str, where: str) -> Any:
    if key not in raw:
        raise PackError(f"{where}: missing required key {key!r}")
    return raw.pop(key)


def _reject_extra(raw: dict, where: str) -> None:
    if raw:
        raise PackError(f"{where}: unknown key(s) {sorted(raw)}")


def _typed(value: Any, kind: type | tuple[type, ...], where: str) -> Any:
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        names = kind.__name__ if isinstance(kind, type) else "/".join(k.__name__ for k in kind)
        raise PackError(f"{where}: expected {names}, got {value!r}")
    return value


def _calibrated(raw: Any, where: str) -> Calibrated:
    if not isinstance(raw, dict):
        raise PackError(
            f"{where}: a calibrated field must be a mapping with 'value', 'status' and "
            f"'derived_from' (got {raw!r}); a bare scalar would hide where it came from"
        )
    body = dict(raw)
    value = _require(body, "value", where)
    status = _require(body, "status", where)
    derived_from = _require(body, "derived_from", where)
    evidence = body.pop("evidence", "")
    _reject_extra(body, where)
    if status not in CALIBRATION_STATUSES:
        raise PackError(
            f"{where}.status must be one of {sorted(CALIBRATION_STATUSES)}, got {status!r}"
        )
    if not isinstance(derived_from, str) or not derived_from.strip():
        raise PackError(f"{where}.derived_from must say how the value was obtained")
    return Calibrated(value=value, status=status, derived_from=derived_from, evidence=evidence)


def _trace_spec(raw: Any, where: str) -> TraceSpec:
    if not isinstance(raw, dict):
        raise PackError(f"{where}: must be a mapping")
    body = dict(raw)
    file_name = _typed(_require(body, "file", where), str, f"{where}.file")
    shapes_raw = _require(body, "shapes", where)
    repeats = _typed(body.pop("repeats", 1), int, f"{where}.repeats")
    id_suffix = body.pop("id_suffix", None)
    _reject_extra(body, where)
    if not isinstance(shapes_raw, list) or not shapes_raw:
        raise PackError(f"{where}.shapes must be a non-empty list of [input_len, output_len]")
    shapes: list[tuple[int, int]] = []
    for index, pair in enumerate(shapes_raw):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in pair)
        ):
            raise PackError(f"{where}.shapes[{index}] must be [input_len, output_len] integers")
        shapes.append((int(pair[0]), int(pair[1])))
    if repeats < 1:
        raise PackError(f"{where}.repeats must be >= 1")
    if id_suffix is not None and (not isinstance(id_suffix, str) or not id_suffix):
        raise PackError(f"{where}.id_suffix must be a non-empty string when present")
    return TraceSpec(file=file_name, shapes=tuple(shapes), repeats=repeats, id_suffix=id_suffix)


def _variant(name: str, raw: Any) -> Variant:
    where = f"variants.{name}"
    if not isinstance(raw, dict):
        raise PackError(f"{where}: must be a mapping")
    body = dict(raw)
    passes_raw = _require(body, "profile_passes", where)
    if not isinstance(passes_raw, list) or not passes_raw:
        raise PackError(f"{where}.profile_passes must be a non-empty list")
    passes: list[ProfilePass] = []
    for index, item in enumerate(passes_raw):
        pass_where = f"{where}.profile_passes[{index}]"
        if not isinstance(item, dict):
            raise PackError(f"{pass_where}: must be a mapping")
        entry = dict(item)
        kind = _typed(_require(entry, "kind", pass_where), str, f"{pass_where}.kind")
        pass_name = _typed(_require(entry, "name", pass_where), str, f"{pass_where}.name")
        trace = _typed(_require(entry, "trace", pass_where), str, f"{pass_where}.trace")
        warmup = _typed(entry.pop("warmup", False), bool, f"{pass_where}.warmup")
        _reject_extra(entry, pass_where)
        if kind not in PROFILE_KINDS:
            raise PackError(
                f"{pass_where}.kind must be one of {sorted(PROFILE_KINDS)}, got {kind!r}"
            )
        if trace not in TRACE_ROLES:
            raise PackError(
                f"{pass_where}.trace must be one of {sorted(TRACE_ROLES)}, got {trace!r}"
            )
        passes.append(ProfilePass(kind=kind, name=pass_name, trace=trace, warmup=warmup))

    # `tokenizer` defaults to `checkpoint`: for every alignment run so far the
    # tokenizer ships inside the checkpoint directory, and repeating the key
    # invites the two to drift apart.
    checkpoint = _typed(_require(body, "checkpoint", where), str, f"{where}.checkpoint")
    tokenizer = _typed(body.pop("tokenizer", None) or checkpoint, str, f"{where}.tokenizer")

    variant = Variant(
        name=name,
        deployment=_typed(_require(body, "deployment", where), str, f"{where}.deployment"),
        gpu=_typed(_require(body, "gpu", where), str, f"{where}.gpu"),
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        engine=_typed(body.pop("engine", "vllm"), str, f"{where}.engine"),
        backend=_typed(_require(body, "backend", where), str, f"{where}.backend"),
        server=dict(_typed(_require(body, "server", where), dict, f"{where}.server")),
        arch=dict(_typed(_require(body, "arch", where), dict, f"{where}.arch")),
        worker=dict(_typed(_require(body, "worker", where), dict, f"{where}.worker")),
        profile_passes=tuple(passes),
        input_builder=dict(
            _typed(_require(body, "input_builder", where), dict, f"{where}.input_builder")
        ),
        label_rules=_typed(_require(body, "label_rules", where), str, f"{where}.label_rules"),
        python_runtime=(
            None
            if (python_runtime := body.pop("python_runtime", None)) is None
            else dict(_typed(python_runtime, dict, f"{where}.python_runtime"))
        ),
        replicas=_typed(body.pop("replicas", 1), int, f"{where}.replicas"),
        run_name_prefix=_typed(body.pop("run_name_prefix", ""), str, f"{where}.run_name_prefix"),
        raw_overrides=dict(body.pop("raw_overrides", {}) or {}),
    )
    _reject_extra(body, where)
    if len({item.name for item in variant.profile_passes}) != len(variant.profile_passes):
        raise PackError(f"{where}.profile_passes has duplicate names")
    return variant


def _case(raw: Any, index_hint: int) -> Case:
    if not isinstance(raw, dict):
        raise PackError(f"cases[{index_hint}]: must be a mapping")
    body = dict(raw)
    case_index = _typed(_require(body, "index", f"cases[{index_hint}]"), int,
                        f"cases[{index_hint}].index")
    name = _typed(_require(body, "name", f"cases[{index_hint}]"), str, f"cases[{index_hint}].name")
    where = f"cases[{case_index:02d}_{name}]"

    max_concurrency_raw = body.pop("max_concurrency", None)
    chunk_size_raw = body.pop("chunk_size", None)
    acceptance_raw = body.pop("speculative_acceptance", None)
    rate_raw = body.pop("rate", None)
    kernel_raw = body.pop("kernel_trace", None)
    window_raw = body.pop("analyze_iterations", None)
    window: tuple[int, int] | None = None
    if window_raw is not None:
        if (
            not isinstance(window_raw, (list, tuple))
            or len(window_raw) != 2
            or not all(isinstance(item, int) and not isinstance(item, bool)
                       for item in window_raw)
            or window_raw[0] >= window_raw[1]
        ):
            raise PackError(
                f"{where}.analyze_iterations must be [start, end] integers with start < end"
            )
        window = (int(window_raw[0]), int(window_raw[1]))

    case = Case(
        index=case_index,
        name=name,
        variant=_typed(_require(body, "variant", where), str, f"{where}.variant"),
        purpose=_typed(_require(body, "purpose", where), str, f"{where}.purpose"),
        coverage=tuple(body.pop("coverage", []) or ()),
        max_model_len=_typed(
            _require(body, "max_model_len", where), int, f"{where}.max_model_len"
        ),
        max_concurrency=(
            None
            if max_concurrency_raw is None
            else _typed(max_concurrency_raw, int, f"{where}.max_concurrency")
        ),
        profile_arrival_mode=_typed(
            _require(body, "profile_arrival_mode", where), str, f"{where}.profile_arrival_mode"
        ),
        simulation_arrival_mode=_typed(
            _require(body, "simulation_arrival_mode", where),
            str,
            f"{where}.simulation_arrival_mode",
        ),
        gpu_memory_utilization=float(
            _typed(
                _require(body, "gpu_memory_utilization", where),
                (int, float),
                f"{where}.gpu_memory_utilization",
            )
        ),
        capture_seconds=float(
            _typed(_require(body, "capture_seconds", where), (int, float),
                   f"{where}.capture_seconds")
        ),
        device_role=_typed(_require(body, "device_role", where), str, f"{where}.device_role"),
        analyze_iterations=window,
        workload_trace=_trace_spec(
            _require(body, "workload_trace", where), f"{where}.workload_trace"
        ),
        kernel_trace=(
            None if kernel_raw is None else _trace_spec(kernel_raw, f"{where}.kernel_trace")
        ),
        attn_gpu_memory_gb=_calibrated(
            _require(body, "attn_gpu_memory_gb", where), f"{where}.attn_gpu_memory_gb"
        ),
        rate=(None if rate_raw is None else _calibrated(rate_raw, f"{where}.rate")),
        provenance=dict(body.pop("provenance", {}) or {}),
        chunk_size=None if chunk_size_raw is None else _typed(chunk_size_raw, int, f"{where}.chunk_size"),
        speculative_acceptance=None if acceptance_raw is None else _calibrated(
            acceptance_raw, f"{where}.speculative_acceptance"
        ),
    )
    _reject_extra(body, where)
    if case.chunk_size is not None and case.chunk_size <= 0:
        raise PackError(f"{where}.chunk_size must be positive")
    if case.speculative_acceptance is not None:
        probabilities = case.speculative_acceptance.value
        if not isinstance(probabilities, list) or not probabilities or any(
            type(value) not in (int, float) or not 0 <= value <= 1 for value in probabilities
        ):
            raise PackError(f"{where}.speculative_acceptance.value must be a non-empty probability list")

    # The two arrival vocabularies are separate spellings of one decision, and a
    # config that disagrees with itself produces a simulation of a different
    # workload than was measured. Bind them here rather than at render time.
    profile_mode, sim_mode = case.profile_arrival_mode, case.simulation_arrival_mode
    if {profile_mode, sim_mode} not in ({"saturated"}, {"trace-timed", "trace_timed"}):
        raise PackError(
            f"{where}: arrival modes disagree — profile {profile_mode!r} vs simulation "
            f"{sim_mode!r}; the pairs are ('saturated','saturated') and "
            "('trace-timed','trace_timed')"
        )
    if profile_mode == "saturated" and case.rate is not None:
        raise PackError(
            f"{where}: rate rescales the trace arrival timeline, which arrival_mode "
            "'saturated' discards; set one or the other"
        )
    if profile_mode == "trace-timed" and case.rate is None:
        raise PackError(f"{where}: trace-timed arrival needs a rate")
    return case


def load_pack(root: Path) -> Pack:
    """Load `<root>/campaign.yaml` + `<root>/acceptance.yaml` into typed values.

    Structural errors raise; policy findings (a provisional value, a used escape
    hatch) are `check_pack`'s job, so a pack with warnings still renders.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise PackError(f"pack {root} is not a directory")
    raw = _document(root / "campaign.yaml", "campaign")
    where = "campaign.yaml"
    version = _require(raw, "schema_version", where)
    if version != PACK_SCHEMA_VERSION:
        raise PackError(f"{where}: schema_version must be {PACK_SCHEMA_VERSION}, got {version!r}")
    name = _typed(_require(raw, "pack", where), str, f"{where}.pack")
    description = _typed(raw.pop("description", ""), str, f"{where}.description")
    recorded_on = _typed(raw.pop("recorded_on", {}), dict, f"{where}.recorded_on")
    variants_raw = _typed(_require(raw, "variants", where), dict, f"{where}.variants")
    cases_raw = _require(raw, "cases", where)
    _reject_extra(raw, where)

    variants = {key: _variant(key, value) for key, value in variants_raw.items()}
    if not variants:
        raise PackError(f"{where}.variants must declare at least one topology")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise PackError(f"{where}.cases must be a non-empty list")
    cases = tuple(_case(item, position) for position, item in enumerate(cases_raw))

    acceptance_path = root / "acceptance.yaml"
    acceptance = _document(acceptance_path, "acceptance") if acceptance_path.is_file() else {}
    return Pack(
        root=root,
        name=name,
        description=description,
        variants=variants,
        cases=cases,
        acceptance=acceptance,
        recorded_on=recorded_on,
    )


def load_host(path: Path) -> HostProfile:
    """Load one host profile.

    `fork_python` may be empty. `load_profile_config` only stats that field when
    it is non-empty and never stats `nsys.executable`, so a stub host lets the
    real loaders validate a rendered config on a machine with no GPU and no
    engine venv — which is what the CPU tier does.
    """
    path = Path(path).resolve()
    raw = _document(path, "host profile")
    where = f"host {path.name}"
    version = _require(raw, "schema_version", where)
    if version != HOST_SCHEMA_VERSION:
        raise PackError(f"{where}: schema_version must be {HOST_SCHEMA_VERSION}, got {version!r}")
    host = HostProfile(
        path=path,
        name=_typed(_require(raw, "host", where), str, f"{where}.host"),
        hf_hub_root=_typed(raw.pop("hf_hub_root", ""), str, f"{where}.hf_hub_root"),
        checkpoints=dict(_typed(_require(raw, "checkpoints", where), dict,
                                f"{where}.checkpoints")),
        text_corpus=_typed(_require(raw, "text_corpus", where), str, f"{where}.text_corpus"),
        nsys_executable=_typed(
            _require(raw, "nsys_executable", where), str, f"{where}.nsys_executable"
        ),
        fork_python=_typed(raw.pop("fork_python", ""), str, f"{where}.fork_python"),
        device_roles=dict(
            _typed(_require(raw, "device_roles", where), dict, f"{where}.device_roles")
        ),
        port=_typed(raw.pop("port", 8000), int, f"{where}.port"),
        startup_timeout=float(
            _typed(raw.pop("startup_timeout", 900.0), (int, float), f"{where}.startup_timeout")
        ),
    )
    _reject_extra(raw, where)
    if not host.checkpoints:
        raise PackError(f"{where}.checkpoints must declare at least one checkpoint")
    if not host.device_roles:
        raise PackError(f"{where}.device_roles must declare at least one role")
    return host
