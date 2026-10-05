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

The workload is the request's (:mod:`public_api.simulate`): a capture's
requests, requests of the reader's shape, or the reader's file. The routing is
always a capture's: one ``workload`` row of the pools' arch preset, whose
routing file the arch reads and whose ``trace.csv`` a capture run replays. An arch that routes no
experts (a dense model) has no capture rows; its member replays the requests
of any published capture (each distinct trace once), with nothing read from
that capture's routing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from launcher.corpus import ROUTING_FILES, resolve_reference
from public_api import preset as public_preset
from public_api.deployments import DeploymentIndex, Member, _axes, find_member

REPO_ROOT = public_preset.REPO_ROOT
SIM_PRESET_ROOT = REPO_ROOT / "presets" / "public_sim"

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
    unknown = sorted(set(preset) - {"deployment", "pools", *public_preset.CONTROL})
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
    out = []
    for candidate, labels in public_preset.expand(preset, tree):
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
                "labels": labels,
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
    # any capture's requests, the workload label (`_all_traces`).
    name: str
    # `hf://datasets/...@<sha>/<dir>/trace.csv`.
    trace: str
    routing: str | None = None
    # The arch preset row this capture is; None for a dense member's.
    arch_row: str | None = None

    def document(self) -> dict:
        return {"name": self.name, "trace": self.trace, "routing": self.routing}


def _capture_dir(reference: str) -> str:
    return reference.rsplit("/", 1)[0]


def _row_captures(rows: dict) -> list[Capture]:
    """An arch preset's capture rows, in its order (the first is its default):
    the rows whose routing names a capture (:data:`ROUTING_FILES`)."""
    out = []
    for label, row in rows.items():
        field_name = ROUTING_FILES.get(row.get("routing"))
        if field_name is None:
            continue
        reference = row[field_name]
        out.append(
            Capture(
                name=label,
                trace=f"{_capture_dir(reference)}/trace.csv",
                routing=row["routing"],
                arch_row=label,
            )
        )
    return out


def _content(reference: str) -> str:
    """A digest of the requests the trace ``reference`` names (a column a tag
    adds, such as a speculative capture's acceptance, is not a request)."""
    from public_api.workloads import requests_digest

    return requests_digest(Path(resolve_reference(reference)))


def _all_traces(index: DeploymentIndex) -> list[Capture]:
    """Every published request list once, for a member that routes nothing.

    Captures of one workload on several models often record the same requests
    (a speculative capture adds its acceptance, which is not a request); each
    list is kept once, named by its workload label (the directory after
    `<model>/<backend>/`), with its first model's directory added when one label
    has several lists. The list the most captures record comes first, the
    default."""
    by_content: dict[str, list[str]] = {}
    for preset in index.presets.values():
        for axis in preset.axes:
            if axis["name"] != public_preset.WORKLOAD:
                continue
            for capture in _row_captures(axis["rows"]):
                traces = by_content.setdefault(_content(capture.trace), [])
                if capture.trace not in traces:
                    traces.append(capture.trace)

    def path(trace: str) -> list[str]:
        return trace.split("@", 1)[1].split("/")[1:]

    lists = [sorted(traces) for traces in by_content.values()]
    labels = Counter(path(traces[0])[2] for traces in lists)
    captures = []
    for traces in sorted(lists, key=lambda traces: (-len(traces), path(traces[0])[2], traces[0])):
        model, _, label = path(traces[0])[:3]
        name = label if labels[label] == 1 else f"{label}/{model}"
        captures.append(Capture(name=name, trace=traces[0]))
    return captures


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
    # Per capture name: how its requests do not fit the pools as a run replays
    # them (`simulate.check_capture`). Checked for every capture, dense or not.
    misfits: dict[str, dict] = field(default_factory=dict)
    # Its pools' request bounds as the simulator's build gives them
    # (`dry-run --report-json` `pools`: role, max_model_len, draft_tokens).
    bounds: list[dict] | None = None

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

    def blocker(self, capture: Capture, *, replayed: bool = True) -> dict | None:
        """Why this member cannot run ``capture``: ``{"error": <the build's
        message>}``, ``{"misfit": <how its requests do not fit>}`` or
        ``{"missing": {<kernel role>: <profile.db rows it lacks>}}``; None when
        it can. The build and its rows follow the capture's routing, so they
        block every run of it; a misfit is of the capture's own requests, so it
        blocks only a run that replays them (``replayed``), not one that takes
        only its routing, and it is named only when nothing else blocks."""
        if self.error:
            return {"error": self.error}
        if capture.name in self.failures:
            return {"error": self.failures[capture.name]}
        key = self.checked[0].name if self.dense and self.checked else capture.name
        if key in self.failures:
            return {"error": self.failures[key]}
        if self.missing.get(key):
            return {"missing": self.missing[key]}
        if key not in self.missing:
            return {"error": "not checked"}
        if replayed and capture.name in self.misfits:
            return {"misfit": self.misfits[capture.name]}
        return None

    def runnable(self, capture: Capture, *, replayed: bool = True) -> str | None:
        """:meth:`blocker` as one sentence; None when it can run ``capture``."""
        blocker = self.blocker(capture, replayed=replayed)
        if blocker is None or "error" in blocker:
            return blocker and blocker["error"]
        if "misfit" in blocker:
            return blocker["misfit"]["reason"]
        rows = ", ".join(f"{k} {n}" for k, n in sorted(blocker["missing"].items()))
        return f"lacks profile.db rows: {rows}"

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
                capture.name: blocker
                for capture in self.captures
                if (blocker := self.blocker(capture)) is not None
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
        captures = rows[0] if rows[0] is not None else _all_traces(self.index)
        if not captures:
            # A member's first capture is its default and names its GPUs.
            raise SimPresetError(f"{preset_id}: no published trace to replay")
        return captures

    def _bind(self, member: SimMember) -> str | None:
        """Find each pool's arch member for every capture. A capture whose arch
        does not build is that capture's failure; the reason it cannot run any
        capture, or None."""
        for capture in member.captures:
            for role in member.pools:
                try:
                    arch = member.arch_member(self.index, role, capture)
                except Exception as error:  # noqa: BLE001 — BadMember, UnknownDeployment
                    member.failures[capture.name] = f"pool {role}: {error}"
                    break
                if arch.error:
                    member.failures[capture.name] = (
                        f"pool {role}: its arch does not build: {arch.error}"
                    )
                    break
        if len(member.failures) == len(member.captures):
            return member.failures[member.captures[0].name]
        return None

    def check(
        self,
        check: Callable[[SimMember, Capture, bool], tuple],
        *,
        jobs: int = 8,
    ) -> None:
        """Check every member's captures as a run would (``check``:
        :func:`public_api.simulate.check_capture`), recording the profile.db
        rows each lacks, its pools' bounds and each capture whose requests do
        not fit. The build is asked once of a dense member and once per capture
        of an MoE member, whose routing changes its kernels (``check``'s third
        argument); whether the requests fit, of every capture, after the builds
        that give the bounds they are fitted to."""
        builds, fits = [], []
        for preset in self.presets.values():
            for member in preset.members:
                if member.error:
                    continue
                member.checked = member.captures[:1] if member.dense else list(member.captures)
                for capture in member.captures:
                    if capture.name not in member.failures:
                        built = capture in member.checked
                        (builds if built else fits).append((member, capture, built))

        def one(item):
            member, capture, build = item
            try:
                return member, capture, check(member, capture, build), None
            except Exception as error:  # noqa: BLE001 — the build's own message
                return member, capture, None, str(error)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for work in (builds, fits):
                for member, capture, answer, error in pool.map(one, work):
                    if error is not None:
                        member.failures[capture.name] = error
                        continue
                    missing, misfit, bounds = answer
                    if missing is not None:
                        member.missing[capture.name] = missing
                        member.bounds = bounds
                    if misfit is not None:
                        member.misfits[capture.name] = misfit

    def preset(self, preset_id: str) -> SimPreset:
        if preset_id not in self.presets:
            from public_api.deployments import UnknownDeployment

            raise UnknownDeployment(preset_id)
        return self.presets[preset_id]

    def member(self, preset_id: str, params: dict[str, Any]) -> SimMember:
        """The member whose axis values are ``params``, compared as a query
        string spells them."""
        preset = self.preset(preset_id)
        return find_member(preset_id, preset.axes, preset.members, params)[1]

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
