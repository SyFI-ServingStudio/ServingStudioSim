"""The public deployments, built once when the service starts.

A deployment is one member of a public preset (:mod:`public_api.preset`): the
preset's id (``<checkpoint>/<arch type>``, its path under ``presets/public``)
and one value of each axis it sweeps. Every member is built structure-only by
``simulator cost-trees --kernel-configs``, one call per preset, in parallel,
with its capture references (``hf://...``) resolved to the local Hugging Face
cache. The index keeps, per member, its cost tree and the shape of a
``timing-predict`` case, and per kernel config any member asks profile.db for,
its grid and the members and roles that ask.

The documents name a config by ``id``: a hash of its kind, GPU and identity
(``KernelConfig::identity``: every field but ``gpu_name`` and ``backends``,
each ``Dim`` at its value). A leaf's config is found from the leaf's own Rust
config the same way, so a tree's leaf and the Kernels page name one config.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from launcher.corpus import resolve_hf_references
from launcher.schema.loader import schema_from_dict
from profiling.db.storage import canonical_json
from public_api import preset as public_preset

REPO_ROOT = public_preset.REPO_ROOT

# How a `Dim` serializes in a manifest's `kernel_config`.
_COVERAGE_FLAGS = (("extrapolated", 1 << 0), ("jit", 1 << 1), ("no_coverage", 1 << 2))


class UnknownDeployment(LookupError):
    """No public preset has this id."""


class BadMember(ValueError):
    """Axis values that name no member of the preset: ``choices`` lists those that do."""

    def __init__(self, message: str, choices: list[dict]) -> None:
        super().__init__(message)
        self.choices = choices


def config_id(kind: str, gpu: str, identity: dict) -> str:
    """The id a config is published under."""
    text = canonical_json({"kind": kind, "gpu": gpu, "identity": identity})
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def scalar_identity(identity: dict) -> tuple[dict, list[str]]:
    """The scalar fields of an identity, and the names of the structured ones
    (an MoE config's expert demand runs to a hundred kilobytes)."""
    scalars, structured = {}, []
    for name, value in identity.items():
        if isinstance(value, (dict, list)):
            structured.append(name)
        else:
            scalars[name] = value
    return scalars, structured


def coverage_flags(bits: int) -> list[str]:
    """``CoverageFlags`` (simulator ``timing``) as names."""
    return [name for name, bit in _COVERAGE_FLAGS if bits & bit]


def _text(value: Any) -> str:
    """How an axis value is spelled in a query string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _repo_path(value: Any) -> Any:
    """A path under this checkout as a repo-relative path; anything else as is."""
    if isinstance(value, str) and value.startswith(str(REPO_ROOT) + "/"):
        return value[len(str(REPO_ROOT)) + 1 :]
    return value


@dataclass
class Member:
    """One deployment: a preset's member, built."""

    preset: str
    params: dict[str, Any]
    gpu: str
    # As the preset writes it, every param filled: capture references stay
    # `hf://`, the model config is repo-relative.
    arch: dict[str, Any]
    # As the binary reads it: captures resolved to local files.
    block: dict[str, Any]
    contract: str | None = None
    gpus_per_replica: int | None = None
    predict: dict | None = None
    sections: list[dict] = field(default_factory=list)
    error: str | None = None
    # profile.db rows its kernels lack, {kind: count} and {config id: count}
    # (:meth:`DeploymentIndex.check`); None until checked.
    missing: dict[str, int] | None = None
    missing_configs: dict[str, int] | None = None

    def summary(self) -> dict:
        slots = [slot for section in self.sections for slot in section["slots"]]
        return {
            "params": self.params,
            "gpus_per_replica": self.gpus_per_replica,
            "predict": self.predict,
            "leaves": len(slots),
            "configs": len({slot["config"] for slot in slots}),
            "error": self.error,
            "missing": self.missing,
        }


@dataclass
class Config:
    """One kernel config some member asks profile.db for."""

    id: str
    kind: str
    profile_kind: str
    gpu: str
    identity: dict
    grid: dict
    # {(preset, member index): sorted roles}
    uses: dict[tuple[str, int], list[str]] = field(default_factory=dict)


@dataclass
class Preset:
    id: str
    checkpoint: str
    arch: str
    gpu: str
    axes: list[dict]
    members: list[Member]


def _axes(preset: dict) -> list[dict]:
    """The axes a preset sweeps, in its order: ``{name, values}``, and for a
    compound group the fields each row binds (its nulls left out)."""
    axes = [
        {"name": name, "values": list(values)} for name, values in preset.get("sweep", {}).items()
    ]
    for name, rows in preset.get("compound", {}).items():
        axes.append(
            {
                "name": name,
                "values": list(rows),
                "rows": {
                    label: {k: v for k, v in row.items() if v is not None}
                    for label, row in rows.items()
                },
            }
        )
    return axes


def _section(section: dict, configs: list[str | None]) -> dict:
    """One manifest section's kernel slots, ``{section, slots}``; ``configs``
    is each slot's config id. A prediction gives the tree itself, as the
    Analyzer reads it."""
    slots = [
        {
            "name": slot["name"],
            "kernel": slot["kind"],
            "config": config,
            "backends": slot["kernel_config"].get("backends", []),
        }
        for slot, config in zip(section["slots"], configs, strict=True)
    ]
    return {"section": section["section"], "slots": slots}


class DeploymentIndex:
    """Every public deployment and kernel config, built from the presets."""

    def __init__(
        self,
        sim_commit: str | None,
        case_fields: dict,
        models: dict,
        arch_names: dict[str, str] | None = None,
    ) -> None:
        self.sim_commit = sim_commit
        self.case_fields = case_fields
        self.models = models
        # Each arch type's reader-facing name (model/arch_catalog.yaml).
        self.arch_names = arch_names or {}
        self.presets: dict[str, Preset] = {}
        self.configs: dict[str, Config] = {}

    # -- building ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        cost_trees: Callable[[list[dict]], list[dict]],
        list_params: dict,
        *,
        sim_commit: str | None,
        paths: list[Path] | None = None,
        jobs: int = 8,
    ) -> DeploymentIndex:
        """Load each preset, expand and complete its members against the schema,
        resolve their captures and build them, one ``cost_trees`` call per preset."""
        catalog = yaml.safe_load(public_preset.MODEL_CATALOG.read_text())
        registry = schema_from_dict(list_params)
        arch_catalog = yaml.safe_load(public_preset.ARCH_CATALOG.read_text()) or {}
        arch_names = {arch: entry["name"] for arch, entry in arch_catalog.items()}
        index = cls(sim_commit, list_params.get("predict_cases", {}), catalog, arch_names)
        for path in public_preset.preset_paths() if paths is None else paths:
            preset = public_preset.load(path, catalog)
            preset_id = f"{path.parent.name}/{path.stem}"
            members = [
                Member(
                    preset=preset_id,
                    params=member["labels"],
                    gpu=member["gpu"],
                    arch={k: _repo_path(v) for k, v in member["arch"].items()},
                    block={"gpu": member["gpu"], "arch": resolve_hf_references(member["arch"])},
                )
                for member in public_preset.members(preset, registry)
            ]
            if preset["arch"]["type"] not in arch_names:
                raise ValueError(
                    f"{path}: arch {preset['arch']['type']!r} has no name in "
                    f"{public_preset.ARCH_CATALOG.name}"
                )
            index.presets[preset_id] = Preset(
                id=preset_id,
                checkpoint=preset["checkpoint"],
                arch=preset["arch"]["type"],
                gpu=preset["gpu"],
                axes=_axes(preset),
                members=members,
            )
        presets = list(index.presets.values())
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            builds = pool.map(lambda p: cost_trees([m.block for m in p.members]), presets)
            for preset, built in zip(presets, builds, strict=True):
                for position, (member, result) in enumerate(
                    zip(preset.members, built, strict=True)
                ):
                    index._add(preset.id, position, member, result)
        return index

    def check(self, missing_specs: Callable[[Member], dict[str, int]], *, jobs: int = 8) -> None:
        """Ask profile.db, per built member, which rows its kernels lack.
        ``missing_specs`` counts them by the kernel's dotted role, the role its
        config lists in ``uses``."""
        built = [
            (preset.id, position, member)
            for preset in self.presets.values()
            for position, member in enumerate(preset.members)
            if not member.error
        ]
        role_configs: dict[tuple[str, int], dict[str, Config]] = {}
        for config in self.configs.values():
            for key, roles in config.uses.items():
                role_configs.setdefault(key, {}).update(dict.fromkeys(roles, config))
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            reports = pool.map(missing_specs, [member for _, _, member in built])
            for (preset_id, position, member), by_role in zip(built, reports, strict=True):
                configs = role_configs.get((preset_id, position), {})
                member.missing, member.missing_configs = {}, {}
                for role, count in by_role.items():
                    config = configs.get(role)
                    if config is None:
                        raise RuntimeError(f"{preset_id} {member.params}: no config uses {role}")
                    member.missing[config.kind] = member.missing.get(config.kind, 0) + count
                    member.missing_configs[config.id] = (
                        member.missing_configs.get(config.id, 0) + count
                    )

    def _add(self, preset_id: str, position: int, member: Member, built: dict) -> None:
        member.contract = built["contract"] or None
        member.error = built["error"]
        if member.error:
            return
        member.gpus_per_replica = built["gpus_per_replica"]
        member.predict = built["predict"]
        ids = [
            config_id(record["kind"], record["gpu_name"], record["identity"])
            for record in built["kernel_configs"]["configs"]
        ]
        # The simulator names the record each slot's kernel reads.
        member.sections = [
            _section(section, [None if i is None else ids[i] for i in slots])
            for section, slots in zip(
                built["cost_manifest"]["sections"], built["slot_configs"], strict=True
            )
        ]
        roles: dict[str, set[str]] = {}
        for cid, record in zip(ids, built["kernel_configs"]["configs"], strict=True):
            config = self.configs.setdefault(
                cid,
                Config(
                    id=cid,
                    kind=record["kind"],
                    profile_kind=record["profile_kind"],
                    gpu=record["gpu_name"],
                    identity=record["identity"],
                    grid=record["grid"],
                ),
            )
            roles.setdefault(cid, set()).update(use["role"] for use in record["uses"])
            config.uses[(preset_id, position)] = sorted(roles[cid])

    # -- lookups -------------------------------------------------------------------

    def preset(self, preset_id: str) -> Preset:
        if preset_id not in self.presets:
            raise UnknownDeployment(preset_id)
        return self.presets[preset_id]

    def member(self, preset_id: str, params: dict[str, Any]) -> tuple[int, Member]:
        """The member whose axis values are ``params``: each axis, no other
        name, compared as a query string spells them."""
        preset = self.preset(preset_id)
        names = [axis["name"] for axis in preset.axes]
        choices = [m.params for m in preset.members]
        given = {name: _text(value) for name, value in params.items()}
        missing = [name for name in names if name not in given]
        unknown = sorted(set(given) - set(names))
        if missing or unknown:
            raise BadMember(
                f"{preset_id} takes exactly its axes {names}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unknown {unknown}" if unknown else ""),
                choices,
            )
        for position, member in enumerate(preset.members):
            if all(_text(member.params[name]) == given[name] for name in names):
                return position, member
        raise BadMember(f"{preset_id} has no member {given}", choices)

    # -- documents -----------------------------------------------------------------

    def catalog(self) -> dict:
        """Every checkpoint, in the model catalog's order, with its presets."""
        by_checkpoint: dict[str, list[Preset]] = {}
        for preset in self.presets.values():
            by_checkpoint.setdefault(preset.checkpoint, []).append(preset)
        return {
            "sim_commit": self.sim_commit,
            "case_fields": self.case_fields,
            "checkpoints": [
                {
                    "checkpoint": checkpoint,
                    **entry,
                    "presets": [
                        {
                            "id": preset.id,
                            "arch": preset.arch,
                            "arch_name": self.arch_names.get(preset.arch),
                            "contract": next(
                                (m.contract for m in preset.members if m.contract), None
                            ),
                            "gpu": preset.gpu,
                            "axes": preset.axes,
                            "members": [m.summary() for m in preset.members],
                        }
                        for preset in by_checkpoint.get(checkpoint, [])
                    ],
                }
                for checkpoint, entry in self.models.items()
            ],
        }

    def tree(self, preset_id: str, params: dict[str, Any]) -> dict:
        """One member's kernels: each section's slots and the configs they
        read. A prediction's tree names its leaves by these slot indices."""
        _, member = self.member(preset_id, params)
        configs = {}
        for section in member.sections:
            for slot in section["slots"]:
                config = self.configs[slot["config"]]
                scalars, structured = scalar_identity(config.identity)
                configs[config.id] = {
                    "kind": config.kind,
                    "identity": scalars,
                    "structured": structured,
                    # profile.db rows this member asks of it and lacks; None until checked.
                    "missing": (
                        None
                        if member.missing_configs is None
                        else member.missing_configs.get(config.id, 0)
                    ),
                }
        return {
            "sim_commit": self.sim_commit,
            "preset": preset_id,
            "params": member.params,
            "gpu": member.gpu,
            "arch": member.arch,
            "arch_name": self.arch_names.get(self.presets[preset_id].arch),
            "contract": member.contract,
            "gpus_per_replica": member.gpus_per_replica,
            "predict": member.predict,
            "error": member.error,
            "missing": member.missing,
            "sections": member.sections,
            "configs": configs,
        }

    def kind_configs(self, kind: str) -> list[Config]:
        return [config for config in self.configs.values() if config.kind == kind]

    def uses(self, config: Config) -> list[dict]:
        """Who asks for a config: ``{preset, params, roles}`` per member."""
        return [
            {
                "preset": preset_id,
                "params": self.presets[preset_id].members[position].params,
                "roles": roles,
            }
            for (preset_id, position), roles in config.uses.items()
        ]
