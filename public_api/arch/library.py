"""The Models page's documents: every arch, one arch's parameters and the
parameter sets it supports, and one parameter set's cost tree.

Every field comes from a source that lives with the code:

- which parameter sets an arch supports: its ``#[supported(...)]`` rows
  (``simulator list-params``), grouped the way the kernel library groups them
  (``public_api.kernel.library.row_entry``): one entry per row, GPU and model
  config, one member per combination of the values the row lists;
- a parameter's type, default, choices and description: ``list-params``;
- the cost tree of each combination: ``simulator supported-cost-trees``, which
  builds every ``#[supported]`` combination structure-only (no GPU, profile.db
  or Python), with every other param at its schema default;
- a leaf's kernel config hash: the leaf's Rust config reduced to its identity,
  the same ``content_hash`` profile.db's kernel-config registry keys configs by;
- whether a leaf's config is measured: the registry and the rows it reads
  (``KernelLibrary._summaries``);
- names: ``model/arch_catalog.yaml`` for archs, ``model/catalog.yaml`` for
  models, the kind docs for kernels.

Nothing here predicts a time. A tree says how leaf costs compose; the cost of a
leaf depends on the batch, which a prediction supplies.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from profiling.db.kernel_config import canonical_json, content_hash
from public_api.kernel import library as kernel_library
from public_api.kernel.library import KernelLibrary, row_entry, row_members

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCH_CATALOG = REPO_ROOT / "model" / "arch_catalog.yaml"

# Query names of a cost tree besides the params the arch's rows name.
GPU, MODEL = "gpu", "model"


class UnknownArch(LookupError):
    """No arch tag has this name."""


class BadParams(ValueError):
    """A cost-tree query leaves out a param, names one the arch's rows do not,
    or gives a value of the wrong type. ``choices`` are the valid queries."""

    def __init__(self, message: str, choices: list[dict]) -> None:
        super().__init__(message)
        self.choices = choices


class UnsupportedParams(LookupError):
    """A well-formed cost-tree query that no ``#[supported]`` row covers."""

    def __init__(self, message: str, choices: list[dict]) -> None:
        super().__init__(message)
        self.choices = choices


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
    dotted role its leaves share. Adds each leaf's config to ``configs``."""

    slots, nodes, labels = manifest["slots"], manifest["nodes"], manifest["node_labels"]

    def build(i: int) -> tuple[dict, str | None]:
        ((op, body),) = nodes[i].items()
        if op == "Leaf":
            slot = slots[body]
            identity = config_identity(slot["kernel_config"])
            config_hash = content_hash(identity)
            configs.setdefault(config_hash, {"kind": slot["kind"], "identity": identity})
            node = {
                "id": i,
                "kind": "leaf",
                "slot": {
                    "index": body,
                    "name": slot["name"],
                    "kind": slot["kind"],
                    "backends": list(slot["kernel_config"].get("backends") or ()),
                    "config_hash": config_hash,
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
                configs: dict[str, dict] = {}
                sections = [
                    {"section": s["section"], "root": _section_tree(s, configs)}
                    for s in (build["cost_manifest"] or {}).get("sections", ())
                ]
                out[key] = {
                    "gpus_per_replica": build["gpus_per_replica"],
                    "error": build["error"],
                    "sections": sections,
                    "configs": configs,
                    "leaves": sum(_leaf_count(s["root"]) for s in sections),
                }
            return out

        return self.sources.cached_by_binary("arch-builds", compute)

    def _registered(self, kind: str) -> dict[tuple[str, str], dict]:
        """``{(config_hash, gpu): {cells, infeasible, measured}}`` over the
        registry's configs of ``kind``: how much of each grid profile.db holds."""

        def compute() -> dict[tuple[str, str], dict]:
            return {
                (s["config"].config_hash, s["config"].gpu_name): {
                    "cells": len(s["config"].grid.cells),
                    "infeasible": len(s["config"].grid.infeasible),
                    "measured": s["measured"],
                }
                for s in self.kernels._summaries(kind)
            }

        return self.sources.cached_by_db(("arch-registered", kind), compute)

    def _config_status(self, build: dict, gpu: str) -> dict[str, dict | None]:
        """Each of a build's configs' registry status on ``gpu``, or None when
        the registry does not hold it."""

        return {
            config_hash: (
                self._registered(config["kind"]).get((config_hash, gpu))
                if config["kind"] in self.kernels.specs
                else None
            )
            for config_hash, config in build["configs"].items()
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

    def _param_sets(self, arch: str) -> list[dict]:
        """The arch's supported parameter sets: one entry per ``#[supported]``
        row, GPU and model config, in the shape the kernel library gives a
        deployment entry, each member with its cost-tree ``query``, label,
        replica width and how much of its tree profile.db measures."""

        def compute() -> list[dict]:
            rows = self._rows(arch)
            builds = self._builds()
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
                    build = builds.get(self.kernels._deployment_key(deployment))
                    listed.update(
                        {
                            "label": deployment["label"],
                            "query": {GPU: gpu, MODEL: model, **deployment["params"]},
                            **self._member_status(build, gpu),
                        }
                    )
                entries.append(entry)
            entries.sort(
                key=lambda e: (self.kernels._model_rank(e["model_config"]), e["gpu"], e["label"])
            )
            return entries

        return self.sources.cached_by_db_and_binary(("arch-param-sets", arch), compute)

    def _member_status(self, build: dict | None, gpu: str) -> dict:
        if build is None:
            return {"gpus_per_replica": None, "error": "not built", "counts": None}
        return {
            "gpus_per_replica": build["gpus_per_replica"],
            "error": build["error"],
            "counts": self._counts(build, gpu),
        }

    def _counts(self, build: dict, gpu: str) -> dict:
        status = self._config_status(build, gpu)
        return {
            "leaves": build["leaves"],
            "configs": len(status),
            "registered": sum(s is not None for s in status.values()),
            "measured": sum(bool(s and any(s["measured"].values())) for s in status.values()),
        }

    def _choices(self, arch: str) -> list[dict]:
        return [m["query"] for entry in self._param_sets(arch) for m in entry["members"]]

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
            sets = self._param_sets(arch)
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

    def _parse(self, arch: str, query: dict[str, str]) -> dict:
        """The cost-tree query with each value typed by its param: the GPU, the
        model config stem and every param the arch's rows name, all required."""

        names = [GPU, MODEL, *self._row_names(arch)]
        choices = self._choices(arch)
        schema = self.sources.deployment_schema()
        params = (*schema["arch_common"], *self._provider(arch)[1].get("params", ()))
        unknown = [n for n in query if n not in names]
        if unknown:
            predicting = [p["name"] for p in params if p.get("set_when_predicting")]
            raise BadParams(
                f"{arch} does not choose {', '.join(unknown)}: its supported sets are chosen by "
                f"{', '.join(names)}; every other param takes its default"
                + (f", except {', '.join(predicting)}, set when predicting" if predicting else ""),
                choices,
            )
        missing = [n for n in names if n not in query]
        if missing:
            raise BadParams(f"{arch} needs {', '.join(missing)}", choices)
        types = {p["name"]: p["type"] for p in params}
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
        return out

    def cost_tree(self, arch: str, query: dict[str, str]) -> dict:
        """One supported parameter set's cost tree, with each leaf's kernel and
        config and whether profile.db holds that config's rows. Raises
        :class:`BadParams` for a missing, unknown or mistyped param and
        :class:`UnsupportedParams` for a set no ``#[supported]`` row covers."""

        contract, provider = self._provider(arch)
        wanted = self._parse(arch, query)
        member = next(
            (
                (entry, m)
                for entry in self._param_sets(arch)
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
            build = self._builds().get(self.kernels._deployment_key(deployment))
            schema = self.sources.deployment_schema()
            # A param the rows leave open takes its schema default, except a
            # traffic param (MoE routing): its default is no real choice, so
            # the document names it as set when predicting instead.
            unchosen = [
                p
                for p in (*schema["arch_common"], *provider.get("params", ()))
                if p["name"] not in deployment["params"]
            ]
            defaults = {
                p["name"]: p["default"]
                for p in unchosen
                if "default" in p and not p.get("set_when_predicting")
            }
            predicting = [p["name"] for p in unchosen if p.get("set_when_predicting")]
            configs, kernels = {}, {}
            status = self._config_status(build, gpu) if build else {}
            for config_hash, config in (build or {"configs": {}})["configs"].items():
                args, omitted = kernel_library._config_args(
                    kernel_library._without_local_paths(config["identity"])
                )
                configs[config_hash] = {
                    "kind": config["kind"],
                    "args": args,
                    "args_omitted": omitted,
                    "registry": status.get(config_hash),
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
                "gpus_per_replica": listed["gpus_per_replica"],
                "error": listed["error"],
                "counts": listed["counts"],
                "sections": build["sections"] if build else [],
                "configs": configs,
                "kernels": kernels,
            }

        return self.sources.cached_by_db_and_binary(
            ("arch-cost-tree", arch, canonical_json(wanted)), compute
        )


def query_text(query: dict) -> dict[str, str]:
    """A cost-tree query as a URL spells it: ``true``/``false`` for a bool."""

    return {k: json.dumps(v) if isinstance(v, bool) else str(v) for k, v in query.items()}


def _leaf_count(node: dict) -> int:
    if node["kind"] == "leaf":
        return 1
    return sum(_leaf_count(child) for child in node["children"])
