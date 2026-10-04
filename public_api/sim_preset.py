"""Public sim presets: the deployments the public site can simulate.

A sim preset (``presets/public_sim/<checkpoint>/<name>.yaml``, under the same
checkpoint directory name as ``presets/public``) is a run config without its
workload: the ``deployment`` and its ``pools``, with every supported value of
its parameters in the launcher's sweep language (``sweep`` / ``compound`` /
``derived`` / ``constraints``)::

    deployment: pd
    pools:
      prefill:
        groups:
          - replicas: ${prefill_replicas}
            arch: {preset: llama3_dense_tp, tp_size: ${prefill_tp}}
            worker: {type: pd_prefill}
      decode: ...
    sweep:
      prefill_tp: [1, 2, 4, 8]
      ...

A group names its arch by reference: ``preset`` is an arch preset of the same
checkpoint (``presets/public/<checkpoint>/<preset>.yaml``) and the other keys
are that preset's axes, every one but ``workload``. The group's GPU and its
complete arch block are that arch member's, so a pool is a deployment the
public site already predicts, and what the arch presets prove (it builds, its
rows are measured) carries over. The expanded list is the support list: a
combination is supported when a sim preset has a member for it.

The workload is the request's (:mod:`public_api.simulate`). Today it is a
capture: one ``workload`` row of the pools' arch preset, whose routing file the
arch reads and whose ``trace.csv`` the run replays. An arch that routes no
experts (a dense model) has no capture rows; its member replays the trace of
any published capture, with nothing read from that capture's routing.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from launcher.schema.expand import expand_sweep_params
from public_api import preset as public_preset
from public_api.deployments import DeploymentIndex, Member, _axes, _text

REPO_ROOT = public_preset.REPO_ROOT
SIM_PRESET_ROOT = REPO_ROOT / "presets" / "public_sim"

_CONTROL = ("sweep", "compound", "derived", "constraints")
# Routing a capture row binds; `uniform` / `random` rows are synthetic, not captures.
_CAPTURE_FIELDS = {"popularity": "expert_popularity_file", "corpus": "token_corpus_file"}


class SimPresetError(ValueError):
    """A sim preset that does not describe a supported deployment."""


def sim_preset_paths(root: Path = SIM_PRESET_ROOT) -> list[Path]:
    return sorted(root.glob("*/*.yaml"))


def load(path: Path) -> dict:
    """Read one sim preset and check its shape: pools of one group each, every
    arch a reference to an arch preset of the same checkpoint."""
    preset = yaml.safe_load(path.read_text())
    if not isinstance(preset, dict):
        raise SimPresetError(f"{path}: not a mapping")
    unknown = sorted(set(preset) - {"deployment", "pools", *_CONTROL})
    if unknown:
        raise SimPresetError(f"{path}: unknown keys {unknown}; a sim preset has no workload")
    if not isinstance(preset.get("deployment"), str):
        raise SimPresetError(f"{path}: deployment must be literal")
    pools = preset.get("pools")
    if not isinstance(pools, dict) or not pools:
        raise SimPresetError(f"{path}: pools must be a non-empty mapping")
    arch_dir = public_preset.PRESET_ROOT / path.parent.name
    for role, pool in pools.items():
        if not isinstance(pool, dict) or set(pool) - {"placement", "groups"}:
            raise SimPresetError(f"{path}: pool {role} takes placement and groups only")
        groups = pool.get("groups")
        if not isinstance(groups, list) or len(groups) != 1:
            raise SimPresetError(f"{path}: pool {role} needs exactly one group")
        group = groups[0]
        if not isinstance(group, dict) or set(group) != {"replicas", "arch", "worker"}:
            raise SimPresetError(
                f"{path}: pool {role}'s group takes replicas, arch and worker; "
                "its GPU is its arch preset's"
            )
        arch = group["arch"]
        name = arch.get("preset") if isinstance(arch, dict) else None
        if not isinstance(name, str) or "${" in name:
            raise SimPresetError(f"{path}: pool {role}'s arch must name a literal `preset`")
        if not (arch_dir / f"{name}.yaml").is_file():
            raise SimPresetError(f"{path}: pool {role} names no arch preset {arch_dir.name}/{name}")
        if public_preset.WORKLOAD in arch:
            raise SimPresetError(
                f"{path}: pool {role} fixes `workload`; the capture is the request's"
            )
        worker = group["worker"]
        if not isinstance(worker, dict) or not isinstance(worker.get("type"), str):
            raise SimPresetError(f"{path}: pool {role}'s worker needs a literal type")
    return preset


def members(preset: dict) -> list[dict]:
    """Every member of a sim preset as ``{labels, deployment, pools}``: per role,
    ``{placement?, replicas, arch_preset, arch_params, worker}``. ``labels``
    names the member by its swept values, a compound group by its row label."""
    tree = {key: preset[key] for key in ("deployment", "pools")}
    control = {key: preset[key] for key in _CONTROL if key in preset}
    swept = [*preset.get("sweep", {}), *preset.get("compound", {})]
    out = []
    for candidate in expand_sweep_params(tree | control, registry=None):
        env = candidate["_env"]
        labels = candidate.get("_sweep_labels", {})
        pools = {}
        for role, pool in candidate["pools"].items():
            group = pool["groups"][0]
            arch = dict(group["arch"])
            pools[role] = {
                **({"placement": pool["placement"]} if "placement" in pool else {}),
                "replicas": group["replicas"],
                "arch_preset": arch.pop("preset"),
                "arch_params": arch,
                "worker": group["worker"],
            }
        out.append(
            {
                "labels": {name: labels.get(name, env[name]) for name in swept},
                "deployment": candidate["deployment"],
                "pools": pools,
            }
        )
    return out


@dataclass(frozen=True)
class Capture:
    """One recorded workload a member can replay: its trace, and the routing the
    arch reads from it (none for an arch that routes no experts)."""

    # The arch preset's `workload` row label; for a dense member, which takes
    # any capture, the capture's directory in the dataset repo.
    name: str
    # `hf://datasets/...@<sha>/<dir>/trace.csv`.
    trace: str
    routing: str | None = None
    routing_file: str | None = None
    # The arch preset row this capture is; None for a dense member's.
    arch_row: str | None = None

    def document(self) -> dict:
        return {"name": self.name, "trace": self.trace, "routing": self.routing}


def _capture_dir(reference: str) -> str:
    return reference.rsplit("/", 1)[0]


def _row_captures(rows: dict) -> list[Capture]:
    """An arch preset's capture rows, in its order (the first is its default)."""
    out = []
    for label, row in rows.items():
        field_name = _CAPTURE_FIELDS.get(row.get("routing"))
        if field_name is None:
            continue
        reference = row[field_name]
        out.append(
            Capture(
                name=label,
                trace=f"{_capture_dir(reference)}/trace.csv",
                routing=row["routing"],
                routing_file=reference,
                arch_row=label,
            )
        )
    return out


def _all_traces(index: DeploymentIndex) -> list[Capture]:
    """Every published capture's trace once, for a member that routes nothing:
    named by its directory in the dataset repo, in path order."""
    traces = {}
    for preset in index.presets.values():
        for axis in preset.axes:
            if axis["name"] != public_preset.WORKLOAD:
                continue
            for capture in _row_captures(axis["rows"]):
                name = capture.trace.split("@", 1)[1].split("/", 1)[1].removesuffix("/trace.csv")
                traces.setdefault(name, Capture(name=name, trace=capture.trace))
    return [traces[name] for name in sorted(traces)]


@dataclass
class SimMember:
    """One sim preset member, bound to the arch members its pools run."""

    preset: str
    params: dict[str, Any]
    deployment: str
    # role -> {placement?, replicas, arch_preset (`<checkpoint>/<arch>`), arch_params, worker}
    pools: dict[str, dict]
    captures: list[Capture]
    # Which captures it was checked with: one for a dense member (its build does
    # not depend on the trace), every one otherwise.
    checked: list[Capture] = field(default_factory=list)
    # Why it cannot run, for every capture; None when it builds.
    error: str | None = None
    # Per capture name: the build error, or the profile.db rows it lacks
    # ({kernel role: count}). Filled by `SimIndex.check`; a capture missing here
    # is covered by `checked[0]` (dense).
    failures: dict[str, str] = field(default_factory=dict)
    missing: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def dense(self) -> bool:
        return all(capture.arch_row is None for capture in self.captures)

    def capture(self, name: str | None) -> Capture:
        if name is None:
            return self.captures[0]
        for capture in self.captures:
            if capture.name == name:
                return capture
        raise KeyError(name)

    def runnable(self, capture: Capture) -> str | None:
        """Why this member cannot run ``capture``; None when it can."""
        if self.error:
            return self.error
        key = self.checked[0].name if self.dense and self.checked else capture.name
        if key in self.failures:
            return self.failures[key]
        if self.missing.get(key):
            rows = ", ".join(f"{k} {n}" for k, n in sorted(self.missing[key].items()))
            return f"lacks profile.db rows: {rows}"
        if key not in self.missing:
            return "not checked"
        return None

    def arch_member(self, index: DeploymentIndex, role: str, capture: Capture) -> Member:
        pool = self.pools[role]
        params = dict(pool["arch_params"])
        if capture.arch_row is not None:
            params[public_preset.WORKLOAD] = capture.arch_row
        return index.member(pool["arch_preset"], params)[1]

    def summary(self, index: DeploymentIndex) -> dict:
        gpus = 0
        pools = {}
        for role, pool in self.pools.items():
            member = self.arch_member(index, role, self.captures[0]) if not self.error else None
            per_replica = member.gpus_per_replica if member else None
            if per_replica is not None:
                gpus += per_replica * pool["replicas"]
            pools[role] = {
                "replicas": pool["replicas"],
                "arch_params": pool["arch_params"],
                "gpus_per_replica": per_replica,
                "worker": pool["worker"],
            }
        return {
            "params": self.params,
            "gpus": gpus or None,
            "pools": pools,
            "error": self.error,
            "unavailable": {
                capture.name: reason
                for capture in self.captures
                if (reason := self.runnable(capture)) is not None
            },
        }


@dataclass
class SimPreset:
    id: str
    checkpoint: str
    deployment: str
    # role -> {arch_preset, worker type, gpu}
    pools: dict[str, dict]
    axes: list[dict]
    captures: list[Capture]
    members: list[SimMember]


class SimIndex:
    """Every public sim preset, expanded and bound to the arch presets' members."""

    def __init__(self, index: DeploymentIndex) -> None:
        self.index = index
        self.presets: dict[str, SimPreset] = {}

    @classmethod
    def build(cls, index: DeploymentIndex, paths: list[Path] | None = None) -> SimIndex:
        """Load and expand each sim preset. A pool whose arch member the index
        lacks or could not build makes its member an error, not the preset."""
        sims = cls(index)
        for path in sim_preset_paths() if paths is None else paths:
            preset = load(path)
            preset_id = f"{path.parent.name}/{path.stem}"
            expanded = members(preset)
            roles = {
                role: f"{path.parent.name}/{pool['groups'][0]['arch']['preset']}"
                for role, pool in preset["pools"].items()
            }
            captures = sims._captures(preset_id, set(roles.values()))
            sim_members = []
            for member in expanded:
                pools = {
                    role: {**pool, "arch_preset": roles[role]}
                    for role, pool in member["pools"].items()
                }
                sim_member = SimMember(
                    preset=preset_id,
                    params=member["labels"],
                    deployment=member["deployment"],
                    pools=pools,
                    captures=captures,
                )
                sim_member.error = sims._bind(sim_member)
                sim_members.append(sim_member)
            arch_presets = {role: index.preset(arch_id) for role, arch_id in roles.items()}
            sims.presets[preset_id] = SimPreset(
                id=preset_id,
                checkpoint=next(iter(arch_presets.values())).checkpoint,
                deployment=preset["deployment"],
                pools={
                    role: {
                        "arch_preset": roles[role],
                        "arch": arch_presets[role].arch,
                        "gpu": arch_presets[role].gpu,
                        "worker": preset["pools"][role]["groups"][0]["worker"]["type"],
                    }
                    for role in roles
                },
                axes=_axes(preset),
                captures=captures,
                members=sim_members,
            )
        return sims

    def _captures(self, preset_id: str, arch_ids: set[str]) -> list[Capture]:
        """The captures a preset's members replay: the arch preset's capture rows,
        which every pool must share; for a dense arch, every published trace."""
        rows = []
        for arch_id in sorted(arch_ids):
            axes = {axis["name"]: axis for axis in self.index.preset(arch_id).axes}
            workload = axes.get(public_preset.WORKLOAD)
            if workload is None:
                rows.append(None)
                continue
            captures = _row_captures(workload["rows"])
            if not captures:
                # Its only rows are synthetic routing; a simulation never
                # defaults to it.
                raise SimPresetError(f"{preset_id}: arch preset {arch_id} has no capture")
            rows.append(captures)
        if any(row != rows[0] for row in rows):
            raise SimPresetError(f"{preset_id}: its pools' arch presets have different captures")
        return rows[0] if rows[0] is not None else _all_traces(self.index)

    def _bind(self, member: SimMember) -> str | None:
        """Find each pool's arch member for every capture; the reason it cannot
        run any capture, or None."""
        for capture in member.captures:
            for role in member.pools:
                try:
                    arch = member.arch_member(self.index, role, capture)
                except Exception as error:  # noqa: BLE001 — BadMember, UnknownDeployment
                    return f"pool {role}: {error}"
                if arch.error:
                    return f"pool {role}: its arch does not build: {arch.error}"
        return None

    def check(
        self, dry_run: Callable[[SimMember, Capture], dict[str, int]], *, jobs: int = 8
    ) -> None:
        """Build every member through the deployment (``dry_run``: the
        simulator's ``dry-run`` of its run config) and record the profile.db
        rows each lacks. A dense member is built once; an MoE member once per
        capture, whose routing changes its kernels."""
        work = []
        for preset in self.presets.values():
            for member in preset.members:
                if member.error:
                    continue
                member.checked = member.captures[:1] if member.dense else list(member.captures)
                work.extend((member, capture) for capture in member.checked)

        def one(item):
            member, capture = item
            try:
                return member, capture, dry_run(member, capture), None
            except Exception as error:  # noqa: BLE001 — the build's own message
                return member, capture, None, str(error)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for member, capture, missing, error in pool.map(one, work):
                if error is not None:
                    member.failures[capture.name] = error
                else:
                    member.missing[capture.name] = missing

    def preset(self, preset_id: str) -> SimPreset:
        if preset_id not in self.presets:
            from public_api.deployments import UnknownDeployment

            raise UnknownDeployment(preset_id)
        return self.presets[preset_id]

    def member(self, preset_id: str, params: dict[str, Any]) -> SimMember:
        """The member whose axis values are ``params``, compared as a query
        string spells them."""
        from public_api.deployments import BadMember

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
        for member in preset.members:
            if all(_text(member.params[name]) == given[name] for name in names):
                return member
        raise BadMember(f"{preset_id} has no member {given}", choices)

    def catalog(self) -> list[dict]:
        return [
            {
                "id": preset.id,
                "checkpoint": preset.checkpoint,
                "deployment": preset.deployment,
                "pools": preset.pools,
                "axes": preset.axes,
                "captures": [capture.document() for capture in preset.captures],
                "members": [member.summary(self.index) for member in preset.members],
            }
            for preset in self.presets.values()
        ]
