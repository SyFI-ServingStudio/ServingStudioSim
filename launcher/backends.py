"""Per-kernel backend selection — dry-run enumerate → skeleton + validation.

The Rust `emit-backends` subcommand builds the cost-tree *structure* of a concrete
config (no GPU, no profiling) and prints one JSON record per `Kernel::build`:
`{pool, name, kind, gpu, compute_dtype, kv_dtype, config, backends}`. This module
turns those records into

  1. the `backends` skeleton (`launcher … --emit-backends`), one entry per distinct
     kernel role, and
  2. a validation pass over a user backend map — unknown role key, missing strict
     coverage, or a backend incompatible with the role's dtype / GPU.

Capability + the dtype/gpu-filtered `options` come from the single source of truth,
`profiling.db.registry` (`supported_backends` / `backend_supports`). The dtype and
GPU are TYPED fields on the record — Rust emits them from the config (dtype in wire
form `"bf16"` / `"fp8_e4m3"`, matching the Python `DType`), so this module never
parses the `describe_config` string for them; only the informational `shape`
annotation is derived from `config`.
"""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml

from profiling.db.args import DType
from profiling.db.registry import backend_supports, known_backends, supported_backends

from .exec import REPO_ROOT, _build_subprocess_env, binary_path
from .process import ProcessSpec, ProcessSupervisor

_PROCESS_SUPERVISOR = ProcessSupervisor()

# describe_config tokens that are NOT geometry: the candidate set, the GPU
# identity, the dtype columns (surfaced separately as `dtype=…`), and the
# per-expert routing load (a workload signal, and 16+ ints of noise). Whatever
# remains is the kernel's shape.
_SHAPE_NOISE_KEYS = frozenset(
    {"backends", "gpu_name", "dtype", "q_dtype", "kv_dtype", "o_dtype", "local_ppm"}
)


class BackendEnumError(RuntimeError):
    """`simulator emit-backends` failed (bad config / build cascade error)."""


def _dtype(wire: str | None) -> DType | None:
    """Map a record's serialized dtype literal (`"bf16"` / `"fp8_e4m3"`, exactly as
    the Rust `DType` emits it) to the Python `DType`. `None` for a dtype-agnostic
    kernel (size-keyed comm, byte-keyed elementwise) → all its backends are options.

    Both sides share the enum wire form, so this is a direct construction — the
    launcher never scrapes `describe_config` for dtype; Rust emits it typed."""
    return DType(wire) if wire else None


def _display_config_value(value: object) -> str:
    if isinstance(value, dict) and "value" in value:
        expression = value.get("expression")
        return f"{expression}={value['value']}" if expression else str(value["value"])
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _parse_shape(config: dict[str, object]) -> str:
    """The kernel's geometry from structured `describe_config` — every field except
    the candidate set, GPU identity, dtype columns, and routing load (see
    `_SHAPE_NOISE_KEYS`). E.g. attention → `num_qo_heads=16 num_kv_heads=1
    head_dim=128`, a GEMM → `n=6144 k=4096`, comm → `num_gpus=4 fabric=Nvlink`.

    This is THIS concrete config's shape, *post* tp/ep split — it changes across a
    tp/ep sweep. The backend map keys on the shape-invariant role NAME, never on
    this, so shape is informational only (a skeleton annotation), never a key."""
    return " ".join(
        f"{key}={_display_config_value(value)}"
        for key, value in config.items()
        if key not in _SHAPE_NOISE_KEYS
    )


@dataclass
class Role:
    """One distinct kernel role (deduplicated across build sites)."""

    pool: str
    name: str  # dotted role name, pool prefix stripped
    kind: str
    compute: DType | None
    kv: DType | None
    default: list[str]
    shape: str = ""  # geometry of the first build site (post tp/ep) — annotation only
    gpu: str | None = None  # run GPU (nvidia-smi name) — capability GPU-axis filter
    count: int = 1  # occurrences (folded layers / unrolled experts)

    @property
    def key(self) -> str:
        return f"{self.pool}/{self.name}"

    @property
    def options(self) -> list[str]:
        """Registered backends compatible with this role's dtype AND GPU — the
        dry-run `options` column, already dtype- and gpu-filtered (so e.g. trt is
        dropped on a non-Blackwell run)."""
        return supported_backends(self.kind, self.compute, self.kv, self.gpu)


def dedup_roles(records: list[dict]) -> list[Role]:
    """Group raw `emit-backends` records by `(pool, name)`, counting occurrences
    (reused sites collapse to one knob). First-seen order is preserved (the
    cost-tree build order, which reads naturally)."""
    by_key: dict[tuple[str, str], Role] = {}
    order: list[tuple[str, str]] = []
    for r in records:
        k = (r["pool"], r["name"])
        if k in by_key:
            by_key[k].count += 1
            continue
        by_key[k] = Role(
            pool=r["pool"],
            name=r["name"],
            kind=r["kind"],
            compute=_dtype(r.get("compute_dtype")),
            kv=_dtype(r.get("kv_dtype")),
            default=list(r["backends"]),
            shape=_parse_shape(r["config"]),
            gpu=r.get("gpu") or None,
        )
        order.append(k)
    return [by_key[k] for k in order]


def enumerate_kernels(config: dict, build_type: str = "debug") -> list[dict]:
    """Run `simulator emit-backends` on a concrete config (its `backends` block +
    launcher-internal `_`-keys stripped) and return the JSON records. Structural
    build only — no GPU, no `profile.db`."""
    binary = binary_path(build_type)
    stripped = {k: v for k, v in config.items() if k != "backends" and not k.startswith("_")}
    with tempfile.TemporaryDirectory() as td:
        # Redirect the build's log_dir into the throwaway temp dir: building the
        # flow writes cost_manifest / cost_log sidecars, which must NOT land in
        # the run's real log_dir on a read-only enumerate.
        io = dict(stripped.get("io") or {})
        io["log_dir"] = td
        stripped["io"] = io
        cfg_path = Path(td) / "emit.yaml"
        cfg_path.write_text(yaml.safe_dump(stripped, default_flow_style=False, sort_keys=False))
        result = _PROCESS_SUPERVISOR.run_sync(
            ProcessSpec(
                argv=[str(binary), "emit-backends", str(cfg_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                env=_build_subprocess_env(),
                name="emit-backends",
            )
        )
    if not result.succeeded:
        raise BackendEnumError(result.output.strip() or "emit-backends failed")
    return json.loads(result.output)


def _merge_role_variants(variants: list[tuple[str, list[Role]]]) -> list[Role]:
    """Reconcile the enumerated roles of every swept structure into one skeleton.

    `variants` is `(label, roles)` per distinct structure (the label — a log_dir —
    is used in the reject message). One `backends` file keys on role NAMES, so:

      - if the structures do NOT all share the same role-NAME set, a single file
        cannot cover them → HARD ERROR (BackendEnumError), naming the divergent
        role per variant. Shape differences alone never trigger this.
      - otherwise the first variant's roles are returned, with each role whose
        `shape` is NOT constant across the structures marked ` (varies)` — the
        skeleton shows run-0's shape, the mark flags that it moves under the sweep.
    """
    ref_label, ref_roles = variants[0]
    ref_names = {r.key for r in ref_roles}

    diffs: list[str] = []
    for label, roles in variants[1:]:
        names = {r.key for r in roles}
        if names == ref_names:
            continue
        parts = []
        if ref_names - names:
            parts.append("missing " + ", ".join(sorted(ref_names - names)))
        if names - ref_names:
            parts.append("extra " + ", ".join(sorted(names - ref_names)))
        diffs.append(f"  {label}: {len(names)} roles ({'; '.join(parts)})")
    if diffs:
        raise BackendEnumError(
            "sweep produces distinct kernel role sets — one backends file cannot "
            f"cover all:\n  {ref_label}: {len(ref_names)} roles (reference)\n"
            + "\n".join(diffs)
            + "\nSplit the divergent runs into separate presets / backends files."
        )

    # Same role set everywhere; flag the roles whose geometry moves under the sweep.
    if len(variants) > 1:
        shapes: dict[str, set[str]] = {}
        for _, roles in variants:
            for r in roles:
                shapes.setdefault(r.key, set()).add(r.shape)
        for r in ref_roles:
            if r.shape and len(shapes.get(r.key, {r.shape})) > 1:
                r.shape += " (varies)"
    return ref_roles


def emit_roles(candidates: list[dict], build_type: str = "debug") -> list[Role]:
    """Enumerate roles across ALL swept candidates for the skeleton. Groups by
    structure (one subprocess per distinct shape/quant — a backend-only sweep
    collapses to one), then reconciles via `_merge_role_variants`: rejects a sweep
    whose structures have different role sets, else returns run-0's roles with
    `(varies)` shape marks. Candidates must already be `normalize_params`-ed."""
    from .schema.loader import log_dir_of

    groups: dict[str, list[dict]] = {}
    for cand in candidates:
        groups.setdefault(_structure_key(cand), []).append(cand)
    representatives = [members[0] for members in groups.values()]
    variants = [
        (log_dir_of(representative), roles)
        for representative, roles in zip(
            representatives, _enumerate_structures(representatives, build_type)
        )
    ]
    return _merge_role_variants(variants)


def _enumerate_structures(representatives: list[dict], build_type: str) -> list[list[Role]]:
    """`dedup_roles(enumerate_kernels(c))` for each distinct structure, in order.

    Each enumeration is its own `simulator emit-backends` process, which spends
    most of its ~0.1 s starting the embedded interpreter and importing the kernel
    registry. Run them side by side rather than one after another.
    """
    if len(representatives) <= 1:
        return [dedup_roles(enumerate_kernels(c, build_type)) for c in representatives]
    workers = min(len(representatives), os.cpu_count() or 1, 32)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda c: dedup_roles(enumerate_kernels(c, build_type)), representatives)
        )


# ── skeleton rendering (`--emit-backends`) ───────────────────────────────────

_SKELETON_HEADER = """\
# GENERATED by:  python -m launcher <preset> --emit-backends
#
# One entry per DISTINCT kernel role (deduplicated by name). EVERY kernel is a
# knob — even where only one backend is valid today. Each value is the current
# default; edit to a subset or to a ${var} declared under the preset's `sweep:`.
#   value semantics: [fa2, fa3] best-of-N · [fa3] force · ${name} sweep var
#   dtype = the precision this kernel runs at · options = backends valid there
#   xN    = role occurs N times (folded layers / unrolled experts) → one knob
#   shape = the FIRST swept run's kernel geometry, post tp/ep — INFORMATIONAL
#           only (the map keys on the shape-invariant role name); `(varies)`
#           flags a shape that changes across the sweep. One map still spans it:
#           best-of-N re-picks per actual shape at eval time"""


def render_skeleton(roles: list[Role], pool_arch: dict[str, str] | None = None) -> str:
    """Render the annotated `backends:` skeleton YAML from enumerated roles.

    Only per-pool roles are editable knobs. Deployment-level kernels (built
    outside any pool — e.g. the AFD/PD cross-pool transfer, `pool == ""`) are not
    routable by the per-pool override map, so they are listed in a trailing
    comment (transparency, not a silent drop) rather than as entries."""
    pool_arch = pool_arch or {}
    pooled = [r for r in roles if r.pool]
    deployment = [r for r in roles if not r.pool]

    # Every comment field is a left-justified column so the `|` separators line
    # up; only `shape:` (the longest, most variable field) trails unpadded at the
    # end, where raggedness reads cleanly. `x{N}` gets a column only if some role
    # is reused (else it would be dead whitespace on every line).
    def _val(r: Role) -> str:
        return "[" + ", ".join(r.default) + "]"

    def _dtype(r: Role) -> str:
        return f"dtype={r.compute.value if r.compute else '-'}"

    def _xn(r: Role) -> str:
        return f"x{r.count}" if r.count > 1 else ""

    def _opts(r: Role) -> str:
        return " ".join(r.options) or "-"

    key_w = max((len(r.key) + 1 for r in pooled), default=0)
    val_w = max((len(_val(r)) for r in pooled), default=0)
    kind_w = max((len(r.kind) for r in pooled), default=0)
    dtype_w = max((len(_dtype(r)) for r in pooled), default=0)
    xn_w = max((len(_xn(r)) for r in pooled), default=0)
    opts_w = max((len(_opts(r)) for r in pooled), default=0)

    lines = [_SKELETON_HEADER, "", "backends:"]
    last_pool: str | None = None
    for role in pooled:
        if role.pool != last_pool:
            arch = pool_arch.get(role.pool)
            suffix = f"   (arch {arch})" if arch else ""
            lines.append(f"\n  # ── pool: {role.pool}{suffix} ──")
            last_pool = role.pool
        cols = [f"{role.kind:<{kind_w}}", f"{_dtype(role):<{dtype_w}}"]
        if xn_w:
            cols.append(f"{_xn(role):<{xn_w}}")
        cols.append(f"options: {_opts(role):<{opts_w}}")
        comment = " | ".join(cols)
        if role.shape:
            comment += f" | shape: {role.shape}"
        key = (role.key + ":").ljust(key_w)
        lines.append(f"  {key} {_val(role):<{val_w}} # {comment}")
    if deployment:
        names = ", ".join(
            f"{r.name} ({r.kind}{', ' + r.shape if r.shape else ''})" for r in deployment
        )
        lines.append(f"\n# deployment-level kernels (not per-pool overridable): {names}")
    return "\n".join(lines) + "\n"


# ── validation (compiler-front-end gate) ─────────────────────────────────────


def validate_backend_map(nested: dict, roles: list[Role]) -> list[str]:
    """Validate a nested `pool -> role -> backends` map against the enumerated
    roles. Reports: an unknown role key; a role left unassigned (strict coverage);
    an unknown backend; a backend incompatible with the role's dtype."""
    errors: list[str] = []
    # Only per-pool roles are override targets; deployment-level kernels (pool
    # "") aren't routable by the per-pool map, so they're exempt from coverage.
    pooled = [r for r in roles if r.pool]
    role_by = {(r.pool, r.name): r for r in pooled}
    seen: set[tuple[str, str]] = set()

    for pool, submap in nested.items():
        if not isinstance(submap, dict):
            errors.append(f"backends[{pool!r}] must be a role -> backends mapping")
            continue
        for name, backends in submap.items():
            role = role_by.get((pool, name))
            if role is None:
                errors.append(
                    f"unknown backend role {pool}/{name} — not a kernel in this "
                    "config (run --emit-backends for the exact role keys)"
                )
                continue
            seen.add((pool, name))
            if not isinstance(backends, list) or not backends:
                errors.append(f"{pool}/{name}: backends must be a non-empty list")
                continue
            known = known_backends(role.kind)
            for be in backends:
                if be not in known:
                    errors.append(
                        f"{pool}/{name}: unknown backend {be!r} for {role.kind} "
                        f"(registered: {known})"
                    )
                elif not backend_supports(role.kind, be, role.compute, role.kv, role.gpu):
                    at = f"dtype={role.compute.value if role.compute else 'any'}"
                    if role.gpu:
                        at += f", gpu={role.gpu}"
                    errors.append(
                        f"{pool}/{name}: backend {be!r} unsupported for {role.kind} "
                        f"at {at} (options: {role.options})"
                    )

    missing = [r.key for r in pooled if (r.pool, r.name) not in seen]
    if missing:
        shown = ", ".join(missing[:6]) + (" …" if len(missing) > 6 else "")
        errors.append(
            f"backend map is missing {len(missing)} role(s) (strict coverage — "
            f"every kernel must be assigned): {shown}. Start from --emit-backends."
        )
    return errors


#: Top-level sections the `emit-backends` build never reads when it builds kernels.
#: A deployment's `build` takes its pools (arch, worker, pool policy) and only
#: logging switches from `io`; the workload is replayed later, by the run. So
#: a sweep over traces, rates or logging enumerates once.
_NOT_STRUCTURAL = frozenset({"backends", "io", "workload"})


def _structure_key(candidate: dict) -> str:
    """A candidate's identity ignoring the sections that cannot change its
    kernels (`_NOT_STRUCTURAL`) and launcher-internal `_`-keys — so runs that
    differ only in backend values, workload or log paths share one structure and
    are enumerated once, while an fp8 / quant sweep (which changes kernel dtypes)
    re-enumerates."""
    tree = {
        k: v for k, v in candidate.items() if k not in _NOT_STRUCTURAL and not k.startswith("_")
    }
    return json.dumps(tree, sort_keys=True, default=str)


def validate_backends_for_candidates(
    candidates: list[dict], build_type: str = "debug"
) -> list[str]:
    """Validate the `backends` map of every candidate that has one. Groups by
    structure so the (subprocess) role enumeration runs once per distinct config
    shape, then checks each member's map against that shape's roles. Returns a
    flat list of `log_dir: message` errors (empty = all valid)."""
    from .schema.loader import log_dir_of

    groups: dict[str, list[dict]] = {}
    for cand in candidates:
        if cand.get("backends"):
            groups.setdefault(_structure_key(cand), []).append(cand)

    errors: list[str] = []
    members_by_structure = list(groups.values())
    roles_by_structure = _enumerate_structures(
        [members[0] for members in members_by_structure], build_type
    )
    for members, roles in zip(members_by_structure, roles_by_structure):
        for cand in members:
            for err in validate_backend_map(cand["backends"], roles):
                errors.append(f"{log_dir_of(cand)}: {err}")
    return errors


def pool_arch_map(preset: dict) -> dict[str, str]:
    """`pool -> arch type` from a preset tree (for the skeleton's pool headers)."""
    out: dict[str, str] = {}
    pools = preset.get("pools")
    if isinstance(pools, dict):
        for pool, spec in pools.items():
            groups = spec.get("groups") if isinstance(spec, dict) else None
            if isinstance(groups, list) and groups and isinstance(groups[0], dict):
                arch = groups[0].get("arch")
                if isinstance(arch, dict) and "type" in arch:
                    out[pool] = arch["type"]
    return out
