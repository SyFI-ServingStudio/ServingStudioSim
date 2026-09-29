"""The Models page's documents: every arch, one arch's parameters and the
parameter sets it supports, and one parameter set's cost tree.

Every field comes from a source that lives with the code:

- which parameter sets an arch supports: its ``#[supported(...)]`` rows
  (``simulator list-params``), grouped the way the kernel library groups them
  (``public_api.kernel.library.row_entry``): one entry per row, GPU and model
  config, one member per combination of the values the row lists;
- a parameter's type, default, choices and description: ``list-params``;
- the cost tree of each combination: the arch block of a run or prediction
  the kernel-config registry records for it (``_kernel_config_source``), built
  structure-only by ``simulator supported-cost-trees --archs`` (no GPU,
  profile.db or Python) with that run's params, the best-measured run first;
  a set no registered run matches falls back to ``supported-cost-trees``, which
  builds it with every other param at its schema default;
- a leaf's kernel config hash: the leaf's Rust config reduced to its identity,
  the same ``content_hash`` profile.db's kernel-config registry keys configs by
  with their kind; a tree keys a config by both (:func:`config_key`);
- whether a leaf's config is measured: the registry and the rows it reads
  (``KernelLibrary._summaries``);
- names: ``model/arch_catalog.yaml`` for archs, ``model/catalog.yaml`` for
  models, the kind docs for kernels.

Nothing here predicts a time. A tree says how leaf costs compose; the cost of a
leaf depends on the batch, which a prediction supplies.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any

import yaml

from launcher.alignment_campaign.check import ROUTING_ARTIFACTS
from launcher.corpus import HF_SCHEME, CorpusError, resolve_reference
from profiling.db import kernel_data
from profiling.db.kernel_config import (
    CONFIG_TABLE,
    SOURCE_TABLE,
    USE_TABLE,
    USES_SQL,
    canonical_json,
    content_hash,
)
from profiling.db.storage import load_blobs, unpack_identity
from public_api.kernel import demand
from public_api.kernel import library as kernel_library
from public_api.kernel.library import (
    KernelLibrary,
    _model_config,
    _source_archs,
    row_entry,
    row_members,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCH_CATALOG = REPO_ROOT / "model" / "arch_catalog.yaml"

# Query names of a cost tree besides the params the arch's rows name.
GPU, MODEL = "gpu", "model"

#: The arch params that name a routing's file, and the params that together
#: name one routing: the routing, its file and its seed. A picker offers them
#: as one choice.
ARTIFACT_KEYS = tuple(key for key, _ in ROUTING_ARTIFACTS.values())
ROUTING_KEYS = (demand.PARAM, *ARTIFACT_KEYS, "routing_seed")

#: How a registry source recorded a run, in the order a tie between two
#: equally measured runs is broken: a preset first, then an alignment case,
#: then a prediction config.
SOURCE_KINDS = ("preset", "alignment", "timing_predict")


class UnknownArch(LookupError):
    """No arch tag has this name."""


class BadParams(ValueError):
    """A cost-tree query leaves out a param, names one the arch's rows do not,
    or gives a value of the wrong type. ``choices`` are the valid queries."""

    def __init__(self, message: str, choices: list[dict]) -> None:
        super().__init__(message)
        self.choices = choices


class UnsupportedParams(LookupError):
    """A well-formed cost-tree query that no ``#[supported]`` row covers, or
    whose run params no registered run of its set used."""

    def __init__(self, message: str, choices: list[dict]) -> None:
        super().__init__(message)
        self.choices = choices


class UnknownFile(LookupError):
    """No input file a prediction reads has this path."""


class _Unavailable(Exception):
    """A registered run's block names a file this machine does not have."""


def _public_error(text: str) -> str:
    """A build error with every absolute path cut to its file name."""

    return re.sub(r"(?:/[^/\s\"'()]+)+/([^/\s\"'()]+)", r"\1", text)


def _public_path(path: str | None, tracked: set[str]) -> str | None:
    """``path`` as this repository names it when its git tracks it (a file, or
    a directory holding tracked files), else None: a path on the machine that
    ran it (``host:/...`` or an absolute one) is that machine's."""

    if not path or ":" in path:
        return None
    relative = demand._repo_relative(path)
    if relative and (
        relative in tracked or any(t.startswith(relative.rstrip("/") + "/") for t in tracked)
    ):
        return relative
    return None


def _source_paths(record: dict) -> list[str]:
    source = record["source"]
    path = (
        source["alignment"].get("pack")
        if "alignment" in source
        else source.get("timing_predict") or source.get("preset")
    )
    return [p for p in [path] if isinstance(p, str) and ":" not in p and demand._repo_relative(p)]


def _source_entry(record: dict, tracked: set[str]) -> dict:
    """How one registry source recorded a run: ``kind`` (``preset``,
    ``alignment`` with its pack, variant and cases, or ``timing_predict``),
    ``path`` when this checkout tracks it, and ``name``, the path or else the
    file name. ``id`` is the source hash's first 12 digits."""

    source = record["source"]
    extra: dict[str, Any] = {}
    if "alignment" in source:
        kind, path = "alignment", source["alignment"].get("pack")
        extra = {
            "variant": source["alignment"].get("variant"),
            "cases": list(source["alignment"].get("cases") or ()),
        }
    elif "timing_predict" in source:
        kind, path = "timing_predict", source["timing_predict"]
    else:
        kind, path = "preset", source.get("preset")
    public = _public_path(path, tracked)
    name = public or (Path(path.split(":", 1)[-1]).name if path else None)
    return {"id": record["hash"][:12], "kind": kind, "path": public, "name": name, **extra}


def _artifact_value(path: str, name: dict | None, tracked: set[str]) -> str:
    """A routing file as a reader names it: the ``hf://`` reference, the repo
    path when git tracks it, else the file name and the fingerprint of the
    demand it produced (``demand.demand_name``), or the file name alone when
    the run could not be built here."""

    if path.startswith(HF_SCHEME) or _public_path(path, tracked):
        return _public_path(path, tracked) or path
    if name is not None:
        return f"{Path(path).name} · {name['fingerprint']}"
    return Path(path.split(":", 1)[-1]).name


def _values_only(value: Any) -> Any:
    """``value`` with every rich ``Dim`` (``{value, expression, bindings}``)
    reduced to its value, as Rust's ``dims::values_only`` serializes it."""

    if isinstance(value, dict):
        if value.keys() == {"value", "expression", "bindings"}:
            return value["value"]
        return {k: _values_only(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_values_only(v) for v in value]
    return value


def config_identity(kernel_config: dict) -> dict:
    """A leaf's kernel config as ``KernelConfig::identity`` gives it: every
    field but ``gpu_name`` and ``backends``, with ``Dim`` values only. Its
    ``content_hash`` is the config hash profile.db's registry keys it by."""

    identity = _values_only(kernel_config)
    identity.pop("gpu_name", None)
    identity.pop("backends", None)
    return identity


def config_key(kind: str, config_hash: str) -> str:
    """A tree's key for a leaf config, ``<kind>:<config_hash>``. The registry
    keys a config by kind and hash: an identity leaves the kind out, so two
    kinds of one shape (a prefill and a decode attention) share a hash."""

    return f"{kind}:{config_hash}"


def _common_path(paths: list[str | None]) -> str | None:
    """The longest dotted prefix every path shares: the role a composite covers."""

    split = [p.split(".") for p in paths if p]
    if not split:
        return None
    common = []
    for parts in zip(*split):
        if len(set(parts)) > 1:
            break
        common.append(parts[0])
    return ".".join(common) or None


def _section_tree(manifest: dict, configs: dict[str, dict]) -> dict:
    """One ``cost_manifest`` section as a nested tree, in the Analyzer's node
    shape: ``{kind: sum | max | scale | leaf, label?, children}``, ``max`` with
    its ``overlap`` divisor and ``scale`` with its repeat ``n``. Every node
    keeps its flat index as ``id`` and each leaf its slot ``index``, the orders
    a cost log and a prediction report use. A composite's ``path`` is the
    dotted role its leaves share. Adds each leaf's config to ``configs`` under
    its :func:`config_key`, which the leaf's slot names."""

    slots, nodes, labels = manifest["slots"], manifest["nodes"], manifest["node_labels"]

    def build(i: int) -> tuple[dict, str | None]:
        ((op, body),) = nodes[i].items()
        if op == "Leaf":
            slot = slots[body]
            identity = config_identity(slot["kernel_config"])
            config_hash = content_hash(identity)
            key = config_key(slot["kind"], config_hash)
            config = configs.setdefault(
                key,
                {
                    "kind": slot["kind"],
                    "config_hash": config_hash,
                    "identity": identity,
                    "backends": [],
                },
            )
            # Leaves of one config may run different backends: the simulator
            # fits every one a leaf names.
            for backend in slot["kernel_config"].get("backends") or ():
                if backend not in config["backends"]:
                    config["backends"].append(backend)
            node = {
                "id": i,
                "kind": "leaf",
                "slot": {
                    "index": body,
                    "name": slot["name"],
                    "kind": slot["kind"],
                    "backends": list(slot["kernel_config"].get("backends") or ()),
                    "config_hash": config_hash,
                    "config_key": key,
                },
            }
            path = slot["name"]
        else:
            span = body["children"]
            built = [build(c) for c in range(span["start"], span["end"])]
            path = _common_path([p for _, p in built])
            node = {"id": i, "kind": op.lower()}
            if op == "Max":
                node["overlap"] = body["overlap"]
            if op == "Scale":
                node["n"] = body["n"]
            node["path"] = path
            node["children"] = [child for child, _ in built]
        if labels[i]:
            node["label"] = labels[i]
        return node, path

    return build(0)[0]


class ArchLibrary:
    """Builds the arch documents from the kernel library's sources."""

    def __init__(self, kernels: KernelLibrary) -> None:
        self.kernels = kernels
        self.sources = kernels.sources
        self.sources.watch(ARCH_CATALOG)

    @property
    def names(self) -> dict[str, dict]:
        """``model/arch_catalog.yaml``, reread when it changes."""

        return self.sources.cached_by_db(
            "arch-catalog", lambda: yaml.safe_load(ARCH_CATALOG.read_text()) or {}
        )

    # -- sources -------------------------------------------------------------------

    def _providers(self) -> dict[str, tuple[str, dict]]:
        """``{arch tag: (contract, list-params entry)}`` in ``list-params`` order."""

        schema = self.sources.deployment_schema()
        return {
            tag: (contract, provider)
            for contract, providers in schema["providers"]["arch"].items()
            for tag, provider in providers.items()
        }

    def _provider(self, arch: str) -> tuple[str, dict]:
        providers = self._providers()
        if arch not in providers:
            raise UnknownArch(arch)
        return providers[arch]

    def _builds(self) -> dict[str, dict]:
        """Every supported combination's structure-only build, keyed like the
        kernel library keys a deployment (arch, GPU, model config and the
        params the rows name, defaults filled). The nested trees and their
        configs are kept; the raw kernel configs are not."""

        def compute() -> dict[str, dict]:
            out: dict[str, dict] = {}
            for build in self.sources.supported_cost_trees():
                labeled = self.kernels._label(
                    build["gpu"], {"type": build["arch"], **build["params"]}
                )
                key = self.kernels._deployment_key(labeled)
                # Rust builds every row's combinations; the first row that
                # covers one owns it, as the launcher checks a config.
                if key in out:
                    continue
                out[key] = _kept(build)
            return out

        return self.sources.cached_by_binary("arch-builds", compute)

    def _registered(self, kind: str) -> dict[tuple[str, str], dict]:
        """``{(config_hash, gpu): {cells, infeasible, measured, usable}}`` over
        the registry's configs of ``kind``: how much of each grid profile.db
        holds, per backend (``KernelLibrary._summaries``)."""

        def compute() -> dict[tuple[str, str], dict]:
            return {
                (s["config"].config_hash, s["config"].gpu_name): {
                    "cells": len(s["config"].grid.cells),
                    "infeasible": len(s["config"].grid.infeasible),
                    "measured": s["measured"],
                    "usable": s["usable"],
                }
                for s in self.kernels._summaries(kind)
            }

        return self.sources.cached_by_db(("arch-registered", kind), compute)

    def _config_status(self, build: dict, gpu: str) -> dict[str, dict | None]:
        """Each of a build's configs' registry status on ``gpu``, by its
        :func:`config_key`, or None when the registry does not hold it."""

        return {
            key: (
                self._registered(config["kind"]).get((config["config_hash"], gpu))
                if config["kind"] in self.kernels.specs
                else None
            )
            for key, config in build["configs"].items()
        }

    # -- parameter sets ------------------------------------------------------------

    def _rows(self, arch: str) -> list[dict]:
        return self._provider(arch)[1].get("supported", [])

    def _row_names(self, arch: str) -> list[str]:
        """The params the arch's rows name besides the GPU and model config:
        with ``gpu`` and ``model``, the names of a cost-tree query."""

        rows = self._rows(arch)
        names = dict.fromkeys(n for row in rows for n in row)
        return [n for n in names if n not in ("gpu", "model_config")]

    def _sets(self, arch: str) -> list[dict]:
        """The arch's supported parameter sets: one entry per ``#[supported]``
        row, GPU and model config, in the shape the kernel library gives a
        deployment entry, each member with its label and cost-tree ``query``."""

        def compute() -> list[dict]:
            rows = self._rows(arch)
            groups: dict[tuple, dict[str, dict]] = {}
            for index, row in enumerate(rows):
                names = [n for n in row if n not in ("gpu", "model_config")]
                for gpu, model, values in itertools.product(
                    row.get("gpu", ()),
                    row.get("model_config", ()),
                    itertools.product(*(row[n] for n in names)),
                ):
                    block = {"type": arch, "model_config": model, **dict(zip(names, values))}
                    deployment = self.kernels._deployment(gpu, block)
                    covering = self.kernels._covering_row(deployment)
                    if covering is None or covering[0] != index:
                        continue
                    key = self.kernels._deployment_key(deployment)
                    groups.setdefault((index, gpu, model), {})[key] = deployment

            entries = []
            for (index, gpu, model), deployments in groups.items():
                members = row_members(rows[index], list(deployments.values()))
                entry = row_entry(rows[index], members)
                for listed, deployment in zip(entry["members"], members, strict=True):
                    listed.update(
                        {
                            "label": deployment["label"],
                            "query": {GPU: gpu, MODEL: model, **deployment["params"]},
                        }
                    )
                entries.append(entry)
            entries.sort(
                key=lambda e: (self.kernels._model_rank(e["model_config"]), e["gpu"], e["label"])
            )
            return entries

        return self.sources.cached_by_db_and_binary(("arch-sets", arch), compute)

    def _param_sets(self, arch: str) -> list[dict]:
        """:meth:`_sets`, each member with its replica width, build error, how
        much of its tree profile.db measures, and ``run``: which registered run
        its tree is built as (``basis`` ``registry``, with the run's params and
        sources) or ``defaults`` when no registered run matches it."""

        def compute() -> list[dict]:
            entries = copy.deepcopy(self._sets(arch))
            for entry in entries:
                for member in entry["members"]:
                    key = self._member_key(arch, member["query"])
                    combos = self._runs(arch).get(key, {}).get("combinations", [])
                    if combos:
                        best = combos[0]
                        member.update(
                            {
                                "gpus_per_replica": best["gpus_per_replica"],
                                "error": None,
                                "counts": best["counts"],
                                "run": {
                                    "basis": "registry",
                                    "params": best["params"],
                                    "sources": [src["name"] for src in best["sources"]],
                                    "combinations": len(combos),
                                },
                            }
                        )
                    else:
                        member.update(
                            {
                                **self._member_status(self._builds().get(key), entry["gpu"]),
                                "run": {"basis": "defaults", "combinations": 0},
                            }
                        )
            return entries

        return self.sources.cached_by_db_and_binary(("arch-param-sets", arch), compute)

    def _member_key(self, arch: str, query: dict) -> str:
        """The deployment key of the member a set query names."""

        block = {
            "type": arch,
            "model_config": query[MODEL],
            **{n: v for n, v in query.items() if n not in (GPU, MODEL)},
        }
        return self.kernels._deployment_key(self.kernels._deployment(query[GPU], block))

    def _member_status(self, build: dict | None, gpu: str) -> dict:
        if build is None:
            return {"gpus_per_replica": None, "error": "not built", "counts": None}
        return {
            "gpus_per_replica": build["gpus_per_replica"],
            "error": build["error"],
            "counts": self._counts(build, gpu),
        }

    def _counts(self, build: dict, gpu: str) -> dict:
        """How many of a build's configs the registry holds, has any row for
        (``measured``), and could predict from (``predictable``: every
        backend its leaves run has a usable row at every feasible cell, as a
        kernel-data bridge fits it)."""

        status = self._config_status(build, gpu)

        def predictable(key: str) -> bool:
            s = status[key]
            if s is None:
                return False
            feasible = s["cells"] - s["infeasible"]
            backends = build["configs"][key]["backends"]
            return all(s["usable"].get(b, 0) == feasible for b in backends)

        return {
            "leaves": build["leaves"],
            "configs": len(status),
            "registered": sum(s is not None for s in status.values()),
            "measured": sum(bool(s and any(s["measured"].values())) for s in status.values()),
            "predictable": sum(predictable(key) for key in status),
        }

    # -- registered runs -------------------------------------------------------------

    def _arch_params(self, arch: str) -> list[dict]:
        schema = self.sources.deployment_schema()
        return [*schema["arch_common"], *self._provider(arch)[1].get("params", ())]

    def _routing_default(self, tag: str | None) -> str | None:
        params = self.kernels._arch_provider(tag).get("params", [])
        return next((p.get("default") for p in params if p["name"] == demand.PARAM), None)

    def _run_blocks(self) -> dict[str, list[dict]]:
        """``{deployment key: [{order, hash, source, gpu, arch}]}``: the arch
        block of every run and prediction the kernel-config registry records,
        by the deployment it names (``KernelLibrary._deployment``: arch, GPU,
        model config and the params the arch's rows choose, defaults filled),
        in registration order. A ``supported`` source is left out: its block
        is the schema defaults a fallback tree is built with, not a run. The
        registry's configs are not read: a run registers only the configs it
        had measured rows for, so its count is no denominator."""

        def compute() -> dict[str, list[dict]]:
            with self.sources.connect() as conn:
                tables = {
                    row[0]
                    for row in conn.execute("select name from sqlite_master where type = 'table'")
                }
                if SOURCE_TABLE not in tables:
                    return {}
                records = conn.execute(
                    f"select id, source_hash, source from {SOURCE_TABLE} order by id"
                ).fetchall()
            out: dict[str, list[dict]] = {}
            for order, source_hash, text in records:
                source = json.loads(text)
                if "supported" in source:
                    continue
                for gpu, block in _source_archs(source):
                    key = self.kernels._deployment_key(self.kernels._deployment(gpu, block))
                    out.setdefault(key, []).append(
                        {
                            "order": order,
                            "hash": source_hash,
                            "source": source,
                            "gpu": gpu,
                            "arch": block,
                        }
                    )
            return out

        return self.sources.cached_by_db_and_binary("arch-run-blocks", compute)

    def _repo_copies(self) -> dict[str, str]:
        """``{path a run recorded: repo path}``: the routing files this
        checkout holds a copy of, so that a run registered from another
        machine's log builds, and reads as the run the copy's preset records.
        A copy's provenance sidecar (``<name>.provenance.json`` beside it)
        names the paths it was copied from, ``copied_from``, and its
        ``sha256``; a file whose content no longer matches is no copy."""

        def compute() -> dict[str, str]:
            out: dict[str, str] = {}
            for sidecar in sorted((REPO_ROOT / "presets").rglob("*.provenance.json")):
                try:
                    meta = json.loads(sidecar.read_text())
                except ValueError:
                    continue
                copied = meta.get("copied_from") if isinstance(meta, dict) else None
                target = sidecar.with_name(sidecar.name.removesuffix(".provenance.json") + ".json")
                if not copied or not target.is_file():
                    continue
                if hashlib.sha256(target.read_bytes()).hexdigest() != meta.get("sha256"):
                    continue
                out.update({path: str(target.relative_to(REPO_ROOT)) for path in copied})
            return out

        return self.sources.cached_by_db_and_binary("arch-repo-copies", compute)

    def _source_configs(self) -> dict[str, dict]:
        """``{source hash: {configs, routed}}``: the ``(kind, config hash,
        GPU)`` of every config each registry source asked for, and whether one
        of them folds an MoE routing (its identity names ``expert_demand``)."""

        def compute() -> dict[str, dict]:
            with self.sources.connect() as conn:
                tables = {
                    row[0]
                    for row in conn.execute("select name from sqlite_master where type = 'table'")
                }
                if not {CONFIG_TABLE, USE_TABLE} <= tables:
                    return {}
                identities = {
                    (kind, config_hash, gpu): text
                    for kind, config_hash, gpu, text in conn.execute(
                        f"select kind, config_hash, gpu_name, identity from {CONFIG_TABLE}"
                    )
                }
                blobs = load_blobs(conn)
                records = {
                    (source_hash, kind, config_hash, gpu)
                    for kind, config_hash, gpu, _, source_hash, *_ in conn.execute(USES_SQL)
                }
            out: dict[str, dict] = {}
            for source_hash, kind, config_hash, gpu in records:
                entry = out.setdefault(source_hash, {"configs": set(), "routed": False})
                entry["configs"].add((kind, config_hash, gpu))
                identity = unpack_identity(identities[kind, config_hash, gpu], blobs)
                entry["routed"] = entry["routed"] or demand.FIELD in identity
            return out

        return self.sources.cached_by_db("arch-source-configs", compute)

    def _run_params(self, arch: str, block: dict) -> dict:
        """A run block's params besides its type, model config and the set's
        params, in schema order: each schema param the block names, else its
        default, except a param set when predicting (MoE routing), which a
        block that leaves it out did not record. A null for a param with no
        default is left out, as the binary reads it: a config recorded as
        written names ``num_layers: null`` where another leaves it out, and
        both are one run. A name the schema no longer has is kept; the build
        then rejects the block."""

        chosen = {"type", "model_config", *self._row_names(arch)}
        unset = set()
        out: dict[str, Any] = {}
        for param in self._arch_params(arch):
            name = param["name"]
            if name in chosen:
                continue
            if block.get(name) is None and "default" not in param:
                unset.add(name)
            elif name in block:
                out[name] = block[name]
            elif "default" in param and not param.get("set_when_predicting"):
                out[name] = param["default"]
        out.update({n: v for n, v in block.items() if n not in chosen | unset and n not in out})
        return out

    @staticmethod
    def _local_block(arch: str, block: dict) -> dict:
        """The block as this process's binary opens it: the model config and
        every routing file as local paths, an ``hf://`` reference only when the
        local hub cache already holds it (nothing is downloaded). Raises
        :class:`_Unavailable` for a file this machine does not have."""

        stem = _model_config(block)
        model = REPO_ROOT / "model" / "config" / f"{stem}.json"
        if not model.is_file():
            raise _Unavailable(f"model config {stem} is not in this checkout")
        local = {**block, "type": arch, "model_config": str(model)}
        for key in ARTIFACT_KEYS:
            value = block.get(key)
            if not isinstance(value, str) or not value:
                continue
            if value.startswith(HF_SCHEME):
                try:
                    local[key] = resolve_reference(value, local_only=True)
                except CorpusError:
                    raise _Unavailable(f"{value} is not in this machine's hub cache") from None
                continue
            path = Path(value)
            if not path.is_absolute():
                path = REPO_ROOT / path
            if not path.is_file():
                name = Path(value.split(":", 1)[-1]).name
                raise _Unavailable(f"{key} {name} is not on this machine")
            local[key] = str(path)
        return local

    def _runs(self, arch: str) -> dict[str, dict]:
        """``{member key: {combinations, skipped}}`` for every supported set of
        the arch: the distinct run params (:meth:`_run_params`) of the
        registered runs that match the set, each built structure-only with
        that run's block and counted as the defaults tree is. ``combinations``
        are the ones built here, best first: most configs measured, then the
        higher measured share, then a preset over an alignment case over a
        prediction, then registration order. ``skipped`` are the ones that name
        a file this machine lacks or that fail to build, with the reason."""

        def compute() -> dict[str, dict]:
            index = self._run_blocks()
            copies = self._repo_copies()
            members: dict[str, tuple[str, dict]] = {}
            for entry in self._sets(arch):
                for member in entry["members"]:
                    key = self._member_key(arch, member["query"])
                    members[key] = (entry["gpu"], {})
            for key, (gpu, combos) in members.items():
                for record in index.get(key, ()):
                    block = {
                        n: copies.get(v, v) if n in ARTIFACT_KEYS and isinstance(v, str) else v
                        for n, v in record["arch"].items()
                    }
                    run = self._run_params(arch, block)
                    combo = combos.setdefault(
                        canonical_json(run),
                        {"run": run, "block": block, "gpu": gpu, "records": []},
                    )
                    combo["records"].append(record)
            every = [c for _, combos in members.values() for c in combos.values()]
            if not every:
                return {}
            paths = demand.artifact_paths(c["block"] for c in every)
            paths += [p for c in every for r in c["records"] for p in _source_paths(r)]
            tracked = self.sources.tracked(sorted(set(paths)))

            requests: list[dict] = []
            where: dict[str, int] = {}
            predicting = [
                p["name"] for p in self._arch_params(arch) if p.get("set_when_predicting")
            ]
            for combo in every:
                # A run that did not record its routing built at the schema
                # default (uniform), which is no routing a run chose.
                unrecorded = [n for n in predicting if n not in combo["run"]]
                if unrecorded:
                    combo["unrecorded"] = unrecorded
                    combo["error"] = f"the run did not record its {', '.join(unrecorded)}"
                    continue
                try:
                    local = self._local_block(arch, combo["block"])
                except _Unavailable as error:
                    combo["error"] = str(error)
                    continue
                request = {"gpu": combo["gpu"], "arch": local}
                combo["build"] = where.setdefault(canonical_json(request), len(requests))
                if combo["build"] == len(requests):
                    requests.append(request)
            built = [_kept(b) for b in self.sources.arch_cost_trees(requests)]

            out = {}
            for key, (gpu, combos) in members.items():
                merged: dict[str, dict] = {}
                for combo in combos.values():
                    doc = self._combination(arch, combo, built, tracked)
                    same = merged.setdefault(canonical_json(doc["params"]), doc)
                    if same is not doc:
                        # Two blocks a reader names alike (one file by two
                        # paths): one choice, the better-built one's tree.
                        keep, other = sorted([same, doc], key=_rank)
                        keep["sources"] = sorted(
                            {s["id"]: s for s in keep["sources"] + other["sources"]}.values(),
                            key=lambda s: (SOURCE_KINDS.index(s["kind"]), s["_order"]),
                        )
                        merged[canonical_json(doc["params"])] = keep
                built_here = sorted((d for d in merged.values() if not d["error"]), key=_rank)
                skipped = []
                for doc in (d for d in merged.values() if d["error"]):
                    left = []
                    for source in doc["sources"]:
                        same = (
                            None
                            if doc["_unrecorded"]
                            else self._recorded_again(source, built_here, gpu)
                        )
                        if same is None:
                            left.append(source)
                            continue
                        same["sources"] = sorted(
                            {s["id"]: s for s in [*same["sources"], source]}.values(),
                            key=lambda s: (SOURCE_KINDS.index(s["kind"]), s["_order"]),
                        )
                    if left:
                        skipped.append({**doc, "sources": left})
                built_here.sort(key=_rank)
                out[key] = {"combinations": built_here, "skipped": skipped}
            return out

        return self.sources.cached_by_db_and_binary(("arch-runs", arch), compute)

    def _recorded_again(self, source: dict, built: list[dict], gpu: str) -> dict | None:
        """The best built run that ``source``, a record of a run this machine
        cannot build, is a second record of, or None. It is when the registry
        shows the record asked for nothing that run's tree lacks, and for at
        least one config folding an MoE routing: a config's hash covers the
        routing it folds, so the two ran the same routing. That is how a run
        registered again from the repository, after it named a file in
        another machine's logs or recorded a value its YAML read another way,
        shows as the one run it is. A record whose routed configs were not
        measured, and so not registered, cannot be told apart and stays
        skipped."""

        record = self._source_configs().get(source["_hash"])
        if record is None or not record["routed"]:
            return None
        for run in built:
            tree = {(c["kind"], c["config_hash"], gpu) for c in run["_build"]["configs"].values()}
            if record["configs"] <= tree:
                return run
        return None

    def _combination(self, arch: str, combo: dict, built: list[dict], tracked: set[str]) -> dict:
        """One distinct run of a set as the documents give it: its params as a
        reader names them (a routing file by :func:`_artifact_value`), the
        sources that recorded it, its routing's name, and its build and counts
        (none for a skipped one)."""

        build = built[combo["build"]] if "build" in combo else None
        error = combo.get("error")
        if error is None and build is not None and build["error"]:
            error = _public_error(build["error"])
        block = {**combo["block"], "type": arch}
        name = None
        if build is not None and not error and demand.PARAM in combo["block"]:
            routed = next(
                (
                    c["identity"][demand.FIELD]
                    for c in build["configs"].values()
                    if demand.FIELD in c["identity"]
                ),
                None,
            )
            if routed is not None:
                name = demand.demand_name(
                    routed, [block], self._routing_default, tracked, self.kernels.routing_names()
                )
        params = {
            n: _artifact_value(v, name, tracked) if n in ARTIFACT_KEYS and isinstance(v, str) else v
            for n, v in combo["run"].items()
        }
        sources = sorted(
            {
                r["hash"]: {**_source_entry(r, tracked), "_order": r["order"], "_hash": r["hash"]}
                for r in combo["records"]
            }.values(),
            key=lambda s: (SOURCE_KINDS.index(s["kind"]), s["_order"]),
        )
        chosen = {"model_config", *self._row_names(arch)}
        unchosen = [p for p in self._arch_params(arch) if p["name"] not in chosen]
        return {
            "params": params,
            "sources": sources,
            "routing": name,
            "omitted": [
                p["name"]
                for p in unchosen
                if p["name"] not in combo["block"] and p["name"] in params
            ],
            "gpus_per_replica": build["gpus_per_replica"] if build and not error else None,
            "counts": self._counts(build, combo["gpu"]) if build and not error else None,
            "error": error,
            "_unrecorded": "unrecorded" in combo,
            "_build": build if not error else None,
        }

    def _choices(self, arch: str) -> list[dict]:
        return [m["query"] for entry in self._sets(arch) for m in entry["members"]]

    # -- documents -----------------------------------------------------------------

    def warm(self) -> None:
        """Build the list, every arch and every supported tree once. An arch or
        tree that fails here fails the same way when asked for."""

        self.catalog()
        for arch in self._providers():
            try:
                self.arch(arch)
                for query in self._choices(arch):
                    self.cost_tree(arch, query_text(query))
            except Exception:  # noqa: BLE001 - reported again on request
                continue

    def _name(self, arch: str) -> dict:
        entry = self.names.get(arch) or {}
        return {"name": entry.get("name"), "summary": entry.get("summary")}

    def _model(self, stem: str) -> dict:
        return {
            "model_config": stem,
            **(self.kernels.models.get(stem) or {"name": None, "family": None, "checkpoint": None}),
        }

    def _arch_models(self, arch: str) -> list[str]:
        stems = dict.fromkeys(m for row in self._rows(arch) for m in row.get("model_config", ()))
        return sorted(stems, key=self.kernels._model_rank)

    def _arch_gpus(self, arch: str) -> list[str]:
        return list(dict.fromkeys(g for row in self._rows(arch) for g in row.get("gpu", ())))

    def catalog(self) -> dict:
        """Every arch tag with its name, the models and GPUs its rows run, the
        params its rows choose between, and how many parameter sets it has.
        Ordered by the catalog order of the first model each arch runs; an arch
        without rows comes last."""

        archs = []
        for order, (arch, (contract, _)) in enumerate(self._providers().items()):
            stems = self._arch_models(arch)
            sets = self._sets(arch)
            families = dict.fromkeys(self._model(s)["family"] for s in stems)
            archs.append(
                {
                    "arch": arch,
                    "contract": contract,
                    **self._name(arch),
                    "models": stems,
                    "families": [f for f in families if f],
                    "gpus": self._arch_gpus(arch),
                    "params": self._row_names(arch),
                    "param_sets": len(sets),
                    "combinations": sum(len(e["members"]) for e in sets),
                    "_order": (self.kernels._model_rank(stems[0] if stems else None), order),
                }
            )
        archs.sort(key=lambda a: a.pop("_order"))
        stems = sorted({s for a in archs for s in a["models"]}, key=self.kernels._model_rank)
        return {"archs": archs, "models": [self._model(s) for s in stems]}

    def arch(self, arch: str) -> dict:
        """One arch: its params (type, default, choices and description, and
        the values its rows list for the ones they name) and its supported
        parameter sets, each member with the query of its cost tree."""

        contract, provider = self._provider(arch)
        schema = self.sources.deployment_schema()
        rows = self._rows(arch)
        listed = {
            n: list(dict.fromkeys(v for row in rows for v in row.get(n, ())))
            for n in ("model_config", *self._row_names(arch))
        }
        params = [
            {**param, "values": listed.get(param["name"])}
            for param in (*schema["arch_common"], *provider.get("params", ()))
        ]
        return {
            "arch": arch,
            "contract": contract,
            **self._name(arch),
            "models": [self._model(s) for s in self._arch_models(arch)],
            "gpus": self._arch_gpus(arch),
            "params": params,
            "query": [GPU, MODEL, *self._row_names(arch)],
            "param_sets": self._param_sets(arch),
        }

    def _parse(self, arch: str, query: dict[str, str]) -> tuple[dict, dict[str, str]]:
        """The cost-tree query's set params, each typed by its param: the GPU,
        the model config stem and every param the arch's rows name, all
        required; and the rest as given, the run params :meth:`cost_tree`
        checks against its set's registered runs."""

        names = [GPU, MODEL, *self._row_names(arch)]
        choices = self._choices(arch)
        missing = [n for n in names if n not in query]
        if missing:
            raise BadParams(f"{arch} needs {', '.join(missing)}", choices)
        types = {p["name"]: p["type"] for p in self._arch_params(arch)}
        out: dict[str, Any] = {}
        for name in names:
            text, kind = query[name], types.get(name, "string")
            if kind == "int":
                try:
                    out[name] = int(text)
                except ValueError:
                    raise BadParams(f"{name} must be an integer, got {text!r}", choices) from None
            elif kind == "bool":
                if text.lower() not in ("true", "false"):
                    raise BadParams(f"{name} must be true or false, got {text!r}", choices)
                out[name] = text.lower() == "true"
            else:
                out[name] = text
        return out, {n: v for n, v in query.items() if n not in names}

    def _select(self, arch: str, wanted: dict, run: dict[str, str], runs: dict) -> dict | None:
        """The registered run of the set the tree is built as: the best of the
        ones whose params hold every run param the query gives, or None when
        the set has none (the defaults tree). Raises :class:`BadParams` for a
        name no run of the set chooses and :class:`UnsupportedParams` for
        values no run used."""

        combos = runs.get("combinations", [])
        observed = list(dict.fromkeys(n for c in combos for n in c["params"]))
        unknown = [n for n in run if n not in observed]
        if unknown:
            names = [GPU, MODEL, *self._row_names(arch)]
            if combos:
                tail = f"; the registered runs of this set also choose {', '.join(observed)}"
            else:
                predicting = [
                    p["name"] for p in self._arch_params(arch) if p.get("set_when_predicting")
                ]
                tail = "; every other param takes its default" + (
                    f", except {', '.join(predicting)}, set when predicting" if predicting else ""
                )
            raise BadParams(
                f"{arch} does not choose {', '.join(unknown)}: its supported sets are chosen by "
                f"{', '.join(names)}{tail}",
                self._choices(arch),
            )
        matching = [
            c
            for c in combos
            if all(query_text({n: c["params"].get(n)})[n] == v for n, v in run.items())
        ]
        if run and not matching:
            raise UnsupportedParams(
                f"no registered run of this {arch} set used {canonical_json(run)}",
                [{**query_text(wanted), **query_text(c["params"])} for c in combos],
            )
        return (matching or combos or [None])[0]

    def cost_tree(self, arch: str, query: dict[str, str]) -> dict:
        """One supported parameter set's cost tree, with each leaf's kernel and
        config and whether profile.db holds that config's rows. The tree is
        built as the set's best-measured registered run (``run``), or as the
        run the query's run params pick; with no registered run, at the schema
        defaults. Raises :class:`BadParams` for a missing, unknown or mistyped
        param and :class:`UnsupportedParams` for a set no ``#[supported]`` row
        covers or run params no registered run used."""

        contract, provider = self._provider(arch)
        wanted, run_query = self._parse(arch, query)
        member = next(
            (
                (entry, m)
                for entry in self._sets(arch)
                for m in entry["members"]
                if m["query"] == wanted
            ),
            None,
        )
        if member is None:
            raise UnsupportedParams(
                f"no #[supported] row of {arch} covers {canonical_json(wanted)}",
                self._choices(arch),
            )
        entry, listed = member
        key = self._member_key(arch, wanted)
        runs = self._runs(arch).get(key, {})
        chosen = self._select(arch, wanted, run_query, runs)

        def compute() -> dict:
            gpu = wanted[GPU]
            deployment = self.kernels._deployment(
                gpu,
                {
                    "type": arch,
                    "model_config": wanted[MODEL],
                    **{n: v for n, v in wanted.items() if n not in (GPU, MODEL)},
                },
            )
            if chosen is not None:
                build = chosen["_build"]
                defaults = {n: chosen["params"][n] for n in chosen["omitted"]}
                # The run recorded every param set when predicting.
                predicting = []
                status = {k: chosen[k] for k in ("gpus_per_replica", "counts")}
                status["error"] = None
            else:
                build = self._builds().get(key)
                # A param the rows leave open takes its schema default, except
                # a traffic param (MoE routing): its default is no real choice,
                # so the document names it as set when predicting instead.
                unchosen = [
                    p for p in self._arch_params(arch) if p["name"] not in deployment["params"]
                ]
                defaults = {
                    p["name"]: p["default"]
                    for p in unchosen
                    if "default" in p and not p.get("set_when_predicting")
                }
                predicting = [p["name"] for p in unchosen if p.get("set_when_predicting")]
                status = self._member_status(build, gpu)
            configs, kernels = {}, {}
            config_status = self._config_status(build, gpu) if build else {}
            for ref, config in (build or {"configs": {}})["configs"].items():
                args, omitted = kernel_library._config_args(config["identity"])
                configs[ref] = {
                    "kind": config["kind"],
                    "config_hash": config["config_hash"],
                    "args": args,
                    "args_omitted": omitted,
                    "registry": config_status.get(ref),
                }
                if config["kind"] not in kernels:
                    doc = kernel_library.kernel_doc(config["kind"])
                    kernels[config["kind"]] = {
                        "documented": doc is not None,
                        "title": doc.title if doc else None,
                        "category": doc.category if doc else None,
                    }
            return {
                "arch": arch,
                "contract": contract,
                **self._name(arch),
                "gpu": gpu,
                "model_config": deployment["model_config"],
                "model": deployment["model"],
                "params": deployment["params"],
                "label": deployment["label"],
                "query": listed["query"],
                "defaults": defaults,
                "set_when_predicting": predicting,
                **status,
                "run": {
                    **self._run_document(wanted, runs, chosen),
                    "predict": self._predict_input(arch, wanted, chosen),
                },
                "sections": build["sections"] if build else [],
                "configs": configs,
                "kernels": kernels,
            }

        return self.sources.cached_by_db_and_binary(
            ("arch-cost-tree", arch, canonical_json(wanted), canonical_json(run_query)), compute
        )

    def _predict_input(self, arch: str, wanted: dict, chosen: dict | None) -> dict | None:
        """``run.predict``: what the in-browser simulator needs to predict this
        run, or None for a tree built at the defaults (no registered run, so no
        data set). ``arch`` is the run's arch block as a run config carries it
        (the predictor picks its contract from ``type``), ``gpu`` its GPU,
        ``files`` the paths ``/files`` serves for it (the model config and a
        named routing's file), ``complete`` whether profile.db measured every
        config the tree reads, and ``error`` why it cannot be predicted."""

        if chosen is None:
            return None
        block = {
            "type": arch,
            "model_config": f"model/config/{wanted[MODEL]}.json",
            **{n: v for n, v in wanted.items() if n not in (GPU, MODEL)},
            **chosen["params"],
        }
        files, error = [block["model_config"]], None
        named = self._routing_files()
        for key in ARTIFACT_KEYS:
            value = block.get(key)
            if not isinstance(value, str):
                continue
            if value in named:
                files.append(value)
            else:
                error = f"its {key} is no named routing this repository holds"
        counts = chosen["counts"]
        return {
            "arch": block,
            "gpu": wanted[GPU],
            "files": files,
            "complete": bool(counts) and counts["predictable"] == counts["configs"],
            "error": error or chosen["error"],
        }

    def _routing_files(self) -> set[str]:
        """The ``file`` of every routing ``routings.yaml`` names: tracked
        popularity paths and hub corpus references."""

        def compute() -> set[str]:
            table = yaml.safe_load(demand.NAMES.read_text()) or {}
            return {
                entry["file"]
                for kind in ("corpus", "popularity")
                for entry in table.get(kind) or ()
                if entry.get("file")
            }

        return self.sources.cached_by_db("routing-files", compute)

    def input_file(self, path: str) -> str:
        """The text of one file a prediction reads (``run.predict.files``): a
        model config this repository tracks, or a named routing's file, a hub
        corpus's manifest from this machine's hub cache (never its payload, and
        nothing is downloaded). Raises :class:`UnknownFile` for anything else."""

        if path in self._routing_files():
            if path.startswith(HF_SCHEME):
                try:
                    return Path(resolve_reference(path, local_only=True)).read_text()
                except CorpusError:
                    raise UnknownFile(path) from None
            return (REPO_ROOT / path).read_text()
        stem = re.fullmatch(r"model/config/([\w.-]+)\.json", path)
        if stem and path in self.sources.tracked([path]):
            return (REPO_ROOT / path).read_text()
        raise UnknownFile(path)

    def kernel_data(self, arch: str, query: dict[str, str]) -> dict:
        """The registry's config document (``profiling.db.kernel_data``) of
        every config the cost tree of ``query`` reads, as a simulator
        kernel-data bridge reads them: ``{format, gpu, configs, missing}``.
        ``missing`` lists the configs profile.db does not register."""

        tree = self.cost_tree(arch, query)
        gpu = tree["gpu"]

        def compute() -> dict:
            documents, missing = [], []
            with self.sources.connect() as conn:
                for ref, config in sorted(tree["configs"].items()):
                    document = kernel_data.config_document(
                        conn, self.sources.db_path, config["kind"], config["config_hash"], gpu
                    )
                    if document is None:
                        missing.append(ref)
                    else:
                        documents.append(document)
            return {
                "format": kernel_data.FORMAT,
                "gpu": gpu,
                "configs": documents,
                "missing": missing,
            }

        refs = canonical_json(sorted(tree["configs"]))
        return self.sources.cached_by_db(("arch-kernel-data", gpu, refs), compute)

    def _run_document(self, wanted: dict, runs: dict, chosen: dict | None) -> dict:
        """``run``: which registered run a tree is built as and how to pick
        another. ``basis`` is ``registry`` or ``defaults``; ``params`` the run
        params the tree was built with; ``sources`` the registry sources that
        recorded that run; ``routing`` its routing's name; ``query`` the full
        cost-tree query of this run; ``pickers`` one per run param (the
        routing and its file as one), with the values the set's runs used;
        ``skipped`` the runs this machine could not build, and why."""

        combos = runs.get("combinations", [])
        skipped = [
            {"params": d["params"], "sources": _public_sources(d["sources"]), "error": d["error"]}
            for d in runs.get("skipped", [])
        ]
        if chosen is None:
            return {
                "basis": "defaults",
                "params": {},
                "sources": [],
                "routing": None,
                "query": None,
                "pickers": [],
                "combinations": 0,
                "skipped": skipped,
            }
        return {
            "basis": "registry",
            "params": chosen["params"],
            "sources": _public_sources(chosen["sources"]),
            "routing": chosen["routing"],
            "query": {**wanted, **chosen["params"]},
            "pickers": _pickers(combos, chosen),
            "combinations": len(combos),
            "skipped": skipped,
        }


def query_text(query: dict) -> dict[str, str]:
    """A cost-tree query as a URL spells it: ``true``/``false`` for a bool."""

    return {k: json.dumps(v) if isinstance(v, bool) else str(v) for k, v in query.items()}


def _kept(build: dict) -> dict:
    """What a build's documents keep: its replica width, error, nested trees,
    their configs and leaf count; not the raw kernel configs."""

    configs: dict[str, dict] = {}
    sections = [
        {"section": s["section"], "root": _section_tree(s, configs)}
        for s in (build["cost_manifest"] or {}).get("sections", ())
    ]
    return {
        "gpus_per_replica": build["gpus_per_replica"],
        "error": build["error"],
        "sections": sections,
        "configs": configs,
        "leaves": sum(_leaf_count(s["root"]) for s in sections),
    }


def _public_sources(sources: list[dict]) -> list[dict]:
    return [{k: v for k, v in s.items() if not k.startswith("_")} for s in sources]


def _picker_groups(combos: list[dict]) -> list[list[str]]:
    """The run params the set's runs name, as pickers: the routing params as
    one, where the first of them falls; every other param alone."""

    names = list(dict.fromkeys(n for c in combos for n in c["params"]))
    routing = [n for n in ROUTING_KEYS if n in names]
    groups: list[list[str]] = []
    for name in names:
        if name in routing:
            if routing not in groups:
                groups.append(routing)
        else:
            groups.append([name])
    return groups


def _pickers(combos: list[dict], chosen: dict) -> list[dict]:
    """One picker per run-param group (:func:`_picker_groups`). Its options
    are the values the set's runs used, each ``selected`` when the tree's run
    has it, ``compatible`` when a run has it and every other picker's
    selection, and with the counts of the run choosing it lands on (that run
    when compatible, else the best run that has it). A picker with one option
    is ``fixed``."""

    groups = _picker_groups(combos)

    def values(combo: dict, group: list[str]) -> dict:
        return {n: combo["params"][n] for n in group if n in combo["params"]}

    out = []
    for group in groups:
        others = [g for g in groups if g is not group]
        options: dict[str, dict] = {}
        for combo in combos:
            value = values(combo, group)
            key = canonical_json(value)
            exact = all(values(combo, g) == values(chosen, g) for g in others)
            option = options.get(key)
            if option is None or (exact and not option["compatible"]):
                options[key] = {
                    "value": value,
                    "selected": value == values(chosen, group),
                    "compatible": exact,
                    "counts": combo["counts"],
                    "routing": combo["routing"] if group[0] == demand.PARAM else None,
                }
        ordered = sorted(options.values(), key=lambda o: _option_order(group, o))
        out.append(
            {"name": group[0], "keys": group, "fixed": len(ordered) == 1, "options": ordered}
        )
    return out


def _option_order(group: list[str], option: dict) -> tuple:
    value = option["value"]
    if group[0] == demand.PARAM:
        routing = option["routing"] or {}
        label = routing.get("label")
        return (demand._preference(value.get(demand.PARAM, "")), label is None, label or "")
    scalar = value.get(group[0])
    if isinstance(scalar, (bool, int, float)):
        return (0, float(scalar), "")
    return (1, 0.0, str(scalar))


def _rank(run: dict) -> tuple:
    """A registered run's place among its set's: its routing's category
    (``demand.PREFERENCE``: measured before synthetic, so uniform never leads
    a routed arch), then most configs a browser can predict from, then most
    measured, then the higher measured share, then how it was recorded, then
    registration."""

    counts = run["counts"]
    kind = min(SOURCE_KINDS.index(s["kind"]) for s in run["sources"])
    order = min(s["_order"] for s in run["sources"])
    if counts is None:
        return (1, 0, 0, 0, 0, kind, order)
    # The run's own routing param: a kernel that folds no `expert_demand` (an
    # FFN pool routing uniformly or at random) still ran one.
    chosen = run["params"].get(demand.PARAM)
    routing = demand._preference(chosen) if isinstance(chosen, str) else 0
    share = counts["measured"] / counts["configs"] if counts["configs"] else 0.0
    return (0, routing, -counts["predictable"], -counts["measured"], -share, kind, order)


def _leaf_count(node: dict) -> int:
    if node["kind"] == "leaf":
        return 1
    return sum(_leaf_count(child) for child in node["children"])
