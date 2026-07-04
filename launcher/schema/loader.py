"""Load the Rust-authoritative deployment schema (new-interface-design §11).

Per L7 design / INV-7, per-param data lives in the Rust binary, not in Python.
`cargo_build()` (see `launcher.exec`) runs `simulator list-params` after a
successful compile and writes the JSON to
`target/<build_type>/deployment_schema.json`. This module only *reads* that
file; it never declares param data.

`list-params` no longer emits a flat per-deployment param list. It publishes the
*structure* of a config tree (it is NOT a cartesian product of deployment × arch
× worker):

    {
      "deployments": { "unified": {"pools": {"main": "iter_wise"}}, ... },
      "providers": {
        "arch":   {"iter_wise": {"llama3_dense": {"params": [...]}, ...}, ...},
        "worker": {"iter_wise": {"barebone": {"params": [...]}, ...}, ...}
      },
      "arch_common": [ ...ParamDef... ],   # fields every arch tag carries
      "group_common": [ ...ParamDef... ],  # flat fields every group carries (gpu/replicas)
      "pool_common": [ ...ParamDef... ],   # flat fields every pool carries (placement)
      "common": {"workload": [...], "io": [...]}   # run-global params
    }

`Registry` parses that document; the structural walk (`iter_slots`,
`unknown_keys`) walks a concrete config tree against it, resolving each leaf to
its governing `ParamDef`. The fixed skeleton (`workload` / `io` / `pools.<role>.
{placement, groups[].{gpu, replicas, arch, worker}}`) is launcher knowledge — the
registry only supplies the per-field ParamDef metadata (type / default / choices
/ affects_cache).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# launcher/schema/loader.py -> parents: [0]=schema, [1]=launcher, [2]=repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


class PresetError(ValueError):
    """A preset file is malformed (bad root type, duplicate keys, parse error)."""


def _reject_duplicate_pairs(pairs: list[tuple]) -> dict:
    """`object_pairs_hook` / mapping builder that rejects duplicate keys instead of
    silently keeping the last (both `json` and PyYAML default to last-wins)."""
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise PresetError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _load_preset(path: Path) -> dict:
    """Read a preset file into a mapping. `.json` uses the JSON parser; everything
    else (`.yaml` / `.yml`) uses YAML (a JSON superset), matching the Rust binary.
    Strict: duplicate keys are rejected (not last-wins) and the root must be a
    mapping. Raises `PresetError` on any of these."""
    text = path.read_text()
    if path.suffix == ".json":
        try:
            data = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
        except json.JSONDecodeError as exc:
            raise PresetError(str(exc)) from exc
    else:
        import yaml

        class _StrictLoader(yaml.SafeLoader):
            pass

        _StrictLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            lambda loader, node: _reject_duplicate_pairs(
                [
                    (loader.construct_object(k), loader.construct_object(v))
                    for k, v in node.value
                ]
            ),
        )
        try:
            data = yaml.load(text, Loader=_StrictLoader)
        except yaml.YAMLError as exc:
            raise PresetError(str(exc)) from exc

    if not isinstance(data, dict):
        raise PresetError(
            f"top-level must be a mapping, got {type(data).__name__}"
        )
    return data


# Top-level preset keys that are launcher control blocks, not config tree nodes.
# `backends` is the per-kernel backend override map (`pool/role -> candidate
# list`); `backends_file` points at an external file holding it. Both are
# schema-exempt (not walked by `iter_slots` / not flagged by `unknown_keys`), but
# `backends` still takes part in `${}` placeholder substitution (it is an ordinary
# sweep participant) and is written through to the Rust config — see
# `expand.expand_sweep_params` and `__main__._merge_backends_file`.
CONTROL_KEYS = frozenset(
    {"sweep", "compound", "derived", "constraints", "backends", "backends_file"}
)


def _unflatten_backends(flat: dict, kind: str) -> dict:
    """Un-flatten a `pool/role -> value` map into Rust's nested `pool -> {role ->
    value}` (`RunConfig.backends`). Every key must be `pool/role` — the
    pool-prefixed dotted role the dry-run emits — so a key without `/` (which
    can't be routed to a pool) is an error. `kind` names the source (`backends` /
    a file path) for the message."""
    nested: dict = {}
    for key, value in flat.items():
        if not isinstance(key, str) or "/" not in key:
            raise PresetError(
                f"{kind} key {key!r} must be `pool/role` (pool-prefixed, e.g. "
                "`ffn/afd.moe_expert_compute.gate_up`); run `--emit-backends` for "
                "the exact keys"
            )
        pool, role = key.split("/", 1)
        nested.setdefault(pool, {})[role] = value
    return nested


def _merge_backends_file(preset: dict, source: str) -> dict:
    """Resolve a `backends_file:` pointer + any inline `backends:` block into one
    nested `backends` map (Rust's `RunConfig.backends`), keyed `pool -> role ->
    value`.

    Both the file and the inline block use flat pool-prefixed `pool/role:` keys —
    exactly what `--emit-backends` writes — which are un-flattened here. `${var}`
    values are left intact for `expand_sweep_params` to substitute per combo. The
    file (the more specific pointer) wins on a key collision with the inline
    block. Mutates + returns the preset (dropping `backends_file`). Raises
    `PresetError` on a malformed block / missing file / non-`pool/role` key."""
    flat: dict = {}
    inline = preset.get("backends")
    if inline is not None:
        if not isinstance(inline, dict):
            raise PresetError("`backends` must be a mapping of `pool/role -> backends`")
        flat.update(inline)

    file_ref = preset.pop("backends_file", None)
    if file_ref is not None:
        if not isinstance(file_ref, str):
            raise PresetError("`backends_file` must be a path string")
        path = Path(file_ref)
        if not path.is_absolute():
            path = Path(source).parent / path  # resolve relative to the preset
        if not path.is_file():
            raise PresetError(f"backends_file {file_ref!r} not found")
        doc = _load_preset(path)
        file_map = doc.get("backends")
        if not isinstance(file_map, dict):
            raise PresetError(
                f"backends_file {file_ref!r}: expected a top-level `backends:` "
                "mapping (the shape `--emit-backends` writes)"
            )
        flat.update(file_map)

    if flat:
        preset["backends"] = _unflatten_backends(flat, "backends")
    return preset

# Fixed skeleton key sets (the uniform pool shape — new-interface-design §8).
_POOL_KEYS = frozenset({"placement", "groups"})
_GROUP_FLAT = ("gpu", "replicas")  # group_common fields (looked up on each group)
_GROUP_KEYS = frozenset({"gpu", "replicas", "arch", "worker"})


class SchemaNotFound(FileNotFoundError):
    """Raised when `deployment_schema.json` is absent — i.e. the binary has not
    been built (or the build failed) since schema discovery is part of
    `cargo_build()`. Carries build instructions, never a Python fallback."""


def schema_path(build_type: str = "debug") -> Path:
    return REPO_ROOT / "target" / build_type / "deployment_schema.json"


@dataclass(frozen=True)
class Slot:
    """One expected scalar leaf in a config tree, bound to its governing
    ParamDef. `container[key]` is the leaf's storage (mutate in place to coerce /
    fill defaults); `path` is for human-readable error messages."""

    container: dict
    key: str
    pdef: dict
    path: tuple[str, ...]

    @property
    def present(self) -> bool:
        return self.key in self.container and self.container[self.key] is not None

    @property
    def value(self) -> Any:
        return self.container.get(self.key)


@dataclass(frozen=True)
class Registry:
    """Parsed `deployment_schema.json`: the config-tree grammar + per-field
    ParamDef dictionaries. Holds only Rust-authoritative data; the structural
    knowledge (how the tree is shaped) lives in the walk functions below."""

    # deployment name -> {"pools": {role -> contract class}}
    # (e.g. {"pools": {"main": "iter_wise"}}).
    deployments: dict[str, dict[str, dict[str, str]]]
    # contract -> arch tag -> {"params": [ParamDef]}; same for workers.
    arch_providers: dict[str, dict[str, dict[str, list[dict]]]]
    worker_providers: dict[str, dict[str, dict[str, list[dict]]]]
    # ParamDefs every arch tag / group / pool / the run-global blocks carry.
    arch_common: list[dict]
    group_common: list[dict]
    pool_common: list[dict]
    workload_common: list[dict]
    io_common: list[dict]

    def roles(self, deployment: str) -> dict[str, str]:
        """role -> contract for one deployment (e.g. {"main": "iter_wise"})."""
        return self.deployments[deployment]["pools"]

    def arch_params(self, contract: str, arch_type: str | None) -> list[dict]:
        return self.arch_providers.get(contract, {}).get(arch_type or "", {}).get("params", [])

    def worker_params(self, contract: str, worker_type: str | None) -> list[dict]:
        return self.worker_providers.get(contract, {}).get(worker_type or "", {}).get("params", [])

    def arch_tags(self, contract: str) -> list[str]:
        return sorted(self.arch_providers.get(contract, {}))

    def worker_tags(self, contract: str) -> list[str]:
        return sorted(self.worker_providers.get(contract, {}))


def _pdef(params: list[dict], name: str) -> dict | None:
    return next((p for p in params if p["name"] == name), None)


def schema_from_dict(raw: dict[str, Any]) -> Registry:
    """Build a `Registry` from an already-parsed `list-params` document. Exposed
    so tests can inject a schema without a built binary on disk."""
    providers = raw.get("providers", {})
    common = raw.get("common", {})
    return Registry(
        deployments=raw.get("deployments", {}),
        arch_providers=providers.get("arch", {}),
        worker_providers=providers.get("worker", {}),
        arch_common=raw.get("arch_common", []),
        group_common=raw.get("group_common", []),
        pool_common=raw.get("pool_common", []),
        workload_common=common.get("workload", []),
        io_common=common.get("io", []),
    )


def load_schema(build_type: str = "debug") -> Registry:
    """Read `target/<build_type>/deployment_schema.json`. Raises `SchemaNotFound`
    with build instructions if absent — there is no hand-maintained fallback."""
    path = schema_path(build_type)
    if not path.is_file():
        raise SchemaNotFound(
            f"deployment schema unavailable at {path}.\n"
            "The schema is generated by `cargo build` + `simulator list-params` "
            "(see launcher.exec.cargo_build). Build the simulator first, e.g.:\n"
            f"  uv run cargo build && ./target/{build_type}/simulator list-params "
            f"> {path}"
        )
    return schema_from_dict(json.loads(path.read_text()))


# ── structural walk over a concrete config tree ──────────────────────────────


def iter_slots(registry: Registry, config: dict, *, create: bool = False) -> Iterator[Slot]:
    """Yield one `Slot` per *expected* scalar leaf of `config`, in tree order.

    Expected = the fixed skeleton (workload / io / each role's pool + groups +
    arch + worker) crossed with the registry's ParamDefs. A leaf is yielded even
    when absent (so `normalize` can fill its default); `Slot.present` says
    whether the value is there.

    `create=True` materializes missing skeleton containers (workload/io/pools/
    group dicts) so filling a default actually lands in the tree — use it only on
    a tree you own (a normalize copy). With `create=False` a missing container
    yields no slots for its subtree (read-only walks tolerate partial trees)."""
    deployment = config.get("deployment")
    if deployment not in registry.deployments:
        return

    def _child(parent: dict, key: str) -> dict | None:
        if key in parent and isinstance(parent[key], dict):
            return parent[key]
        if create:
            parent[key] = {}
            return parent[key]
        return None

    workload = _child(config, "workload")
    if workload is not None:
        for pdef in registry.workload_common:
            yield Slot(workload, pdef["name"], pdef, ("workload", pdef["name"]))

    io = _child(config, "io")
    if io is not None:
        for pdef in registry.io_common:
            yield Slot(io, pdef["name"], pdef, ("io", pdef["name"]))

    pools = _child(config, "pools")
    if pools is None:
        return
    for role, contract in registry.roles(deployment).items():
        pool = _child(pools, role)
        if pool is None:
            continue
        placement = _pdef(registry.pool_common, "placement")
        if placement is not None:
            yield Slot(pool, "placement", placement, ("pools", role, "placement"))
        groups = pool.get("groups")
        if not isinstance(groups, list):
            continue
        for gi, group in enumerate(groups):
            if not isinstance(group, dict):
                continue
            base = ("pools", role, "groups", str(gi))
            for fld in _GROUP_FLAT:
                pdef = _pdef(registry.group_common, fld)
                if pdef is not None:
                    yield Slot(group, fld, pdef, base + (fld,))
            arch = _child(group, "arch")
            if arch is not None:
                arch_type = arch.get("type")
                for pdef in registry.arch_common:
                    yield Slot(arch, pdef["name"], pdef, base + ("arch", pdef["name"]))
                for pdef in registry.arch_params(contract, arch_type):
                    yield Slot(arch, pdef["name"], pdef, base + ("arch", pdef["name"]))
            worker = _child(group, "worker")
            if worker is not None:
                worker_type = worker.get("type")
                for pdef in registry.worker_params(contract, worker_type):
                    yield Slot(worker, pdef["name"], pdef, base + ("worker", pdef["name"]))


def unknown_keys(registry: Registry, config: dict) -> list[str]:
    """Report keys that are not part of the expected config tree for this
    deployment — the authoritative payload-typo guard (a Rust tagged-enum variant
    silently ignores `deny_unknown_fields`, so arch/worker payload typos must be
    caught here, see plan G1). Returns dotted-path strings."""
    deployment = config.get("deployment")
    if deployment not in registry.deployments:
        return []
    errors: list[str] = []

    def _check(node: dict, allowed: set[str], path: str) -> None:
        for key in node:
            # Only top-level `_`-keys are launcher internals (_env, _sweep_labels).
            # A nested `_`-key is NOT an internal — report it, else an arch/worker
            # payload typo like `arch._model_config` slips past unflagged.
            if key.startswith("_") and path == "":
                continue
            if key not in allowed:
                where = f"{path}.{key}" if path else key
                errors.append(where)

    root_allowed = {"deployment", "workload", "io", "pools"} | CONTROL_KEYS
    _check(config, root_allowed, "")

    workload = config.get("workload")
    if isinstance(workload, dict):
        _check(workload, {p["name"] for p in registry.workload_common}, "workload")
    io = config.get("io")
    if isinstance(io, dict):
        _check(io, {p["name"] for p in registry.io_common}, "io")

    pools = config.get("pools")
    if not isinstance(pools, dict):
        return errors
    roles = registry.roles(deployment)
    _check(pools, set(roles), "pools")
    for role, contract in roles.items():
        pool = pools.get(role)
        if not isinstance(pool, dict):
            continue
        _check(pool, set(_POOL_KEYS), f"pools.{role}")
        groups = pool.get("groups")
        if not isinstance(groups, list):
            continue
        for gi, group in enumerate(groups):
            if not isinstance(group, dict):
                continue
            gpath = f"pools.{role}.groups[{gi}]"
            _check(group, set(_GROUP_KEYS), gpath)
            arch = group.get("arch")
            if isinstance(arch, dict):
                arch_type = arch.get("type")
                allowed = (
                    {"type"}
                    | {p["name"] for p in registry.arch_common}
                    | {p["name"] for p in registry.arch_params(contract, arch_type)}
                )
                _check(arch, allowed, f"{gpath}.arch")
            worker = group.get("worker")
            if isinstance(worker, dict):
                worker_type = worker.get("type")
                allowed = {"type"} | {
                    p["name"] for p in registry.worker_params(contract, worker_type)
                }
                _check(worker, allowed, f"{gpath}.worker")
    return errors


# ── log_dir accessor (it lives under io now, not at the root) ────────────────


def log_dir_of(config: dict, default: str = "logs") -> str:
    """The run's output directory — `config["io"]["log_dir"]` in the tree model."""
    io = config.get("io")
    if isinstance(io, dict) and io.get("log_dir") is not None:
        return str(io["log_dir"])
    return default
