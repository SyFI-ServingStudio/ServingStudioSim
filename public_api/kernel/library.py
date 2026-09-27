"""The kernel library's documents: the catalog, one kind's detail, measured rows.

Every field comes from a source that lives with the code; nothing here guesses:

- prose, including how the kind is timed, from the kind module (``DOC``,
  ``arg(...)``, ``BackendDoc``) and the mechanical facts from the profiling
  registry (argument order, backends, supported dtypes and GPUs, metric family,
  subprocess environment);
- the models that run a kind, from the cost tree of every ``#[supported]`` arch
  deployment on its declared GPU, and from every registered kernel config's uses
  (the arch block of the preset, prediction or supported row that built it);
- which arguments a model sweeps and which it fixes, and the profile.db columns
  each model's shape reads, from the kernel's own ``enumerate`` applied to those
  cost tree leaves and from the registered grids;
- a deployment's model and label from its arch block: the model config's file
  stem (as ``#[supported]`` rows spell it) named by ``model/catalog.yaml``, and
  the arch params the arch's ``#[supported]`` rows name (``list-params``);
- a row's precision, from the column the Rust kernel names as its compute dtype;
- GPU peaks from ``gpu/spec.json`` and model names from ``model/catalog.yaml``;
- measurements from profile.db, and the kernel configs that read them from its
  kernel-config registry (``profiling/db/kernel_config.py``).
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import typing
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from launcher.schema.validate import supported_value
from profiling.db.args import DType
from profiling.db.doc import CATEGORIES, SUBCATEGORIES, arg_docs, kernel_doc
from profiling.db.doc import METRICS as METRIC_DOCS
from profiling.db.kernel_config import (
    CONFIG_TABLE,
    SOURCE_TABLE,
    USE_TABLE,
    ConfigUse,
    RegisteredConfig,
    canonical_json,
    cell_keys,
    pack_cells,
    registered_configs,
)
from profiling.db.registry import MetricFamily, iter_kernel_profiler_specs
from profiling.db.table import STANDARD_COLUMNS, Table
from profiling.gpu_catalog import GpuSpecResolution, resolve_gpu_spec
from profiling.runners.metrics import CommMetrics, ComputeMetrics
from public_api.kernel.sources import KernelSources

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG = REPO_ROOT / "model" / "catalog.yaml"

# What each row records about the run that measured it: every standard column
# but the row's key (GPU, backend) and the outlier check's flag.
PROVENANCE = tuple(c for c in STANDARD_COLUMNS if c not in ("gpu_name", "backend", "verified"))
METRICS = {
    MetricFamily.COMPUTE: [f.name for f in fields(ComputeMetrics)],
    MetricFamily.COMM: [f.name for f in fields(CommMetrics)],
}
# Bound values per statement when matching grid cells to rows; SQLite allows 32766.
_SQL_VARIABLES = 30000


class UnknownKind(LookupError):
    """No registered kernel kind has this name."""


class UnknownConfig(LookupError):
    """No registered kernel config of this kind has this hash."""


class BadQuery(ValueError):
    """A rows filter names an unknown column or a value of the wrong type."""


def _arg_type(annotation: Any) -> str:
    """How the page treats an argument: a dtype, a number, a list or a label."""

    if annotation is DType:
        return "dtype"
    if annotation in (int, float):
        return "number"
    if typing.get_origin(annotation) is tuple:
        return "list"
    return "label"


def _peaks(spec: GpuSpecResolution, dtypes: list[str]) -> dict[str, dict]:
    """The spec-sheet ceiling of each metric that has one, from gpu/spec.json:
    ``{metric: {"value" or "by_dtype", "note"}}``. Throughput depends on the
    compute dtype; bus bandwidth is one direction of the catalog's bidirectional
    link bandwidth."""

    out: dict[str, dict] = {}
    by_dtype = {d: v for d in dtypes if (v := spec.dense_peak_tflops(d)) is not None}
    if by_dtype:
        out["tflops"] = {"by_dtype": by_dtype, "note": "dense"}
    if spec.mem_bandwidth_gbps is not None:
        out["memory_bandwidth_gbps"] = {"value": spec.mem_bandwidth_gbps, "note": "HBM"}
    if spec.interconnect_one_way_gbps is not None:
        out["busbw_gbps"] = {
            "value": spec.interconnect_one_way_gbps,
            "note": f"{spec.interconnect}, one direction",
        }
    return out


def _quoted(columns: list[str]) -> str:
    return ", ".join(f'"{c}"' for c in columns)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _source_via(source: dict) -> dict:
    """What built a registered config, from the source forms the launcher writes
    (``launcher/kernel_configs.py``): a ``#[supported]`` deployment, a
    timing-predict config, a run of an alignment pack's cases, or a preset run."""

    if "supported" in source:
        return {"type": "supported", "ref": None}
    if "timing_predict" in source:
        return {"type": "timing_predict", "ref": source["timing_predict"]}
    if source.get("alignment"):
        alignment = source["alignment"]
        return {
            "type": "alignment",
            "ref": alignment["pack"],
            "variant": alignment.get("variant"),
            "cases": len(alignment.get("cases") or ()),
        }
    return {"type": "preset", "ref": source.get("preset")}


def _source_archs(source: dict) -> list[tuple[str, dict]]:
    """``(gpu, arch block)`` of every arch a source names. A run's
    deployment-level source (pool ``""``, AFD's attention-to-FFN transfer)
    names none."""

    if "supported" in source:
        row = source["supported"]
        return [(row["gpu"], {"type": row["arch"], **row["params"]})]
    if "timing_predict" in source:
        return [(source["gpu"], arch) for arch in source["arch"].values()]
    return [(group["gpu"], group["arch"]) for group in source.get("groups") or ()]


def _model_config(arch: dict) -> str | None:
    """The arch's model config as ``#[supported]`` rows and ``model/catalog.yaml``
    spell it: the file stem of a ``model/config/`` file, else the value as given."""

    value = arch.get("model_config")
    return supported_value("model_config", value) if value is not None else None


def _axes(config: RegisteredConfig) -> list[list[float]]:
    return [list(axis) for axis in config.grid.axes]


def _config_args(identity: dict) -> tuple[dict, list[str]]:
    """The identity's scalar fields, and the names of the structured ones (an
    expert-demand table can run to megabytes); ``/configs/{hash}`` has them whole."""

    scalar = {k: v for k, v in identity.items() if not isinstance(v, (dict, list))}
    return scalar, [k for k in identity if k not in scalar]


class KernelLibrary:
    """Builds the kernel library documents from :class:`KernelSources`."""

    def __init__(self, sources: KernelSources) -> None:
        self.sources = sources
        self.specs: dict[str, list] = {}
        for spec in iter_kernel_profiler_specs():
            self.specs.setdefault(spec.kernel_kind, []).append(spec)
        sources.watch(MODEL_CATALOG)
        self.sim_commit = _git_commit()

    @property
    def models(self) -> dict[str, dict]:
        """``model/catalog.yaml``, reread when it changes."""

        return self.sources.cached_by_db(
            "model-catalog", lambda: yaml.safe_load(MODEL_CATALOG.read_text()) or {}
        )

    # -- per-kind facts ------------------------------------------------------------

    def _specs(self, kind: str) -> list:
        if kind not in self.specs:
            raise UnknownKind(kind)
        return self.specs[kind]

    def _args(self, kind: str) -> list[str]:
        return [f.name for f in fields(self._specs(kind)[0].args_schema)]

    def _metrics(self, kind: str) -> list[str]:
        return METRICS[self._specs(kind)[0].metric_family]

    def _precision_column(self, kind: str) -> str | None:
        """The profile.db column holding the kernel's compute dtype, if it has one."""

        entry = next((e for e in self.sources.kernel_list() if e["kind"] == kind), None)
        column = entry and entry["compute_dtype"]
        return column if column in self._args(kind) else None

    def _column_types(self, kind: str) -> dict[str, str]:
        """``{column: declared type}`` of the kind's table; empty when it has none."""

        def compute() -> dict[str, str]:
            with self.sources.connect() as conn:
                info = conn.execute(f'pragma table_info("{kind}")').fetchall()
            return {row[1]: (row[2] or "").upper() for row in info}

        return self.sources.cached_by_db(("columns", kind), compute)

    def _coverage(self, kind: str) -> list[dict]:
        """Row counts per GPU, backend and precision, with the latest measurement."""

        def compute() -> list[dict]:
            if not self._column_types(kind):
                return []
            precision = self._precision_column(kind)
            precision_sql = f'"{precision}"' if precision else "null"
            query = (
                f"select gpu_name, backend, {precision_sql}, count(*), max(profiler_run_at) "
                f'from "{kind}" group by 1, 2, 3 order by 1, 2, 3'
            )
            with self.sources.connect() as conn:
                records = conn.execute(query).fetchall()
            return [
                {"gpu": g, "backend": b, "precision": p, "rows": n, "last_measured_at": t}
                for g, b, p, n, t in records
            ]

        return self.sources.cached_by_db(("coverage", kind), compute)

    # -- models and deployments ------------------------------------------------------

    def _arch_provider(self, tag: str | None) -> dict:
        """The arch tag's ``list-params`` entry: ``params`` and ``supported`` rows.
        A registry source names the tag but not its contract (``iter_wise``,
        ``layer_wise_attn``, ...), and each tag lives under one contract."""

        for providers in self.sources.deployment_schema()["providers"]["arch"].values():
            if tag in providers:
                return providers[tag]
        return {}

    def _deployment(self, gpu: str, arch: dict) -> dict:
        """One arch block on one GPU as the site names it: the model it runs and
        a label of the arch and its deployment params.

        The params are the names the arch's ``#[supported]`` rows give besides
        ``gpu`` and ``model_config`` (the sizes a supported deployment is chosen
        by). An arch without rows yet falls back to its own params that change
        its kernel configs (``affects_cache``) and take a number, a bool or one
        of listed choices; free-form strings such as file paths stay out of the
        label. A param the block leaves out takes its schema default."""

        deployment = self.sources.cached_by_binary(
            ("deployment", canonical_json([gpu, arch])),
            lambda: self._label(gpu, arch),
        )
        # name, family, checkpoint; null for a model not in model/catalog.yaml.
        # Looked up here, not in the binary-keyed label: the catalog changes on
        # its own.
        return {**deployment, "model": self.models.get(deployment["model_config"])}

    def _label(self, gpu: str, arch: dict) -> dict:
        tag = arch.get("type")
        provider = self._arch_provider(tag)
        own = provider.get("params", [])
        rows = provider.get("supported", [])
        if rows:
            names = [n for n in dict.fromkeys(n for row in rows for n in row)]
            names = [n for n in names if n not in ("gpu", "model_config")]
        else:
            names = [
                p["name"]
                for p in own
                if p.get("affects_cache") and (p["type"] in ("int", "bool") or p.get("choices"))
            ]
        schema = self.sources.deployment_schema()
        defaults = {p["name"]: p.get("default") for p in (*schema["arch_common"], *own)}
        params = {}
        for name in names:
            value = arch.get(name, defaults.get(name))
            if value is not None:
                params[name] = value
        model_config = _model_config(arch)
        text = (json.dumps(v) if isinstance(v, bool) else str(v) for v in params.values())
        return {
            "arch": tag,
            "gpu": gpu,
            "model_config": model_config,
            "params": params,
            "label": ", ".join([tag, *(f"{n} {v}" for n, v in zip(params, text))]),
        }

    def _source_deployments(self, source: dict, gpu: str) -> list[dict]:
        """The deployments a registry source describes on ``gpu`` (the config's
        GPU picks the group that built it)."""

        return [self._deployment(g, arch) for g, arch in _source_archs(source) if g == gpu]

    @staticmethod
    def _deployment_key(deployment: dict) -> str:
        return canonical_json([deployment[k] for k in ("arch", "gpu", "model_config", "params")])

    def _model_rank(self, model_config: str | None) -> tuple[int, str]:
        """Catalog order first, then models the catalog does not name, by stem."""

        order = list(self.models)
        stem = model_config or ""
        return (order.index(stem) if stem in self.models else len(order), stem)

    def _registry_models(self) -> dict[str, set[str]]:
        """``{kind: model configs}`` over every registered config's uses."""

        def compute() -> dict[str, set[str]]:
            with self.sources.connect() as conn:
                tables = {
                    row[0]
                    for row in conn.execute("select name from sqlite_master where type = 'table'")
                }
                if not {CONFIG_TABLE, SOURCE_TABLE, USE_TABLE} <= tables:
                    return {}
                records = conn.execute(
                    f"""
                    select distinct c.profile_kind, s.source from {USE_TABLE} u
                    join {SOURCE_TABLE} s on s.source_hash = u.source_hash
                    join {CONFIG_TABLE} c on c.kind = u.kind and c.config_hash = u.config_hash
                        and c.gpu_name = u.gpu_name
                    """
                ).fetchall()
            kinds = {spec.table_name: kind for kind, specs in self.specs.items() for spec in specs}
            out: dict[str, set[str]] = {}
            for table, source in records:
                for _, arch in _source_archs(json.loads(source)):
                    model = _model_config(arch)
                    if model and table in kinds:
                        out.setdefault(kinds[table], set()).add(model)
            return out

        return self.sources.cached_by_db("registry-models", compute)

    def _used_by(self, kind: str) -> tuple[list[dict], set[str]]:
        """The deployments that run ``kind``, each with the shapes it asks for, and
        the columns those shapes sweep, from the kernel-config registry: every
        registered config's uses (the presets, predictions, alignment packs and
        ``#[supported]`` deployments that built it). A shape is the profile.db
        columns one leaf fixes, with the config that holds them."""

        deployments: dict[str, dict] = {}
        swept: set[str] = set()
        for summary in self._summaries(kind):
            config = summary["config"]
            swept.update(summary["swept"])
            for use in config.uses:
                via = _source_via(use.source)["type"]
                for deployment in self._source_deployments(use.source, config.gpu_name):
                    key = self._deployment_key(deployment)
                    target = deployments.setdefault(
                        key, {**deployment, "sources": set(), "shapes": {}}
                    )
                    target["sources"].add(via)
                    # One leaf per role and shape: rank copies and repeat
                    # registrations of it are listed once.
                    target["shapes"].setdefault(
                        canonical_json([use.role, config.config_hash]),
                        {
                            "layer": use.role,
                            "pool": use.pool,
                            "db": summary["fixed"],
                            "config_hash": config.config_hash,
                        },
                    )

        out = [
            {**d, "sources": sorted(d["sources"]), "shapes": list(d["shapes"].values())}
            for d in deployments.values()
        ]
        out.sort(key=lambda d: (self._model_rank(d["model_config"]), d["gpu"], d["label"]))
        return out, swept

    # -- documents -----------------------------------------------------------------

    def catalog(self) -> dict:
        """Every registered kind with its coverage and the models that run it.

        A kind's ``used_by`` is the model configs (keys of ``models``) of every
        registered config that reads its rows, in ``models`` order. ``models`` is
        ``model/catalog.yaml`` in its order, then any model a kind names that
        the catalog does not, with ``name`` null."""

        models_by_kind = self._registry_models()
        named = set().union(*models_by_kind.values()) if models_by_kind else set()
        stems = sorted(set(self.models) | named, key=self._model_rank)

        kernels = []
        gpu_rows: dict[str, int] = {}
        for kind, specs in sorted(self.specs.items()):
            doc = kernel_doc(kind)
            coverage = self._coverage(kind)
            for entry in coverage:
                gpu_rows[entry["gpu"]] = gpu_rows.get(entry["gpu"], 0) + entry["rows"]
            kernels.append(
                {
                    "kind": kind,
                    "documented": doc is not None,
                    "title": doc.title if doc else None,
                    "summary": doc.summary if doc else None,
                    "category": doc.category if doc else None,
                    "subcategory": doc.subcategory if doc else None,
                    "metric_family": specs[0].metric_family.value,
                    "backends": [spec.backend for spec in specs],
                    "precisions": sorted({e["precision"] for e in coverage if e["precision"]}),
                    "coverage": [
                        {k: e[k] for k in ("gpu", "backend", "precision", "rows")} for e in coverage
                    ],
                    "rows": sum(e["rows"] for e in coverage),
                    "used_by": sorted(models_by_kind.get(kind, ()), key=self._model_rank),
                }
            )

        precisions = sorted({p for k in kernels for p in k["precisions"]})
        last_measured = [
            e["last_measured_at"]
            for kind in self.specs
            for e in self._coverage(kind)
            if e["last_measured_at"]
        ]
        return {
            "snapshot": {
                "sim_commit": self.sim_commit,
                "last_measured_at": max(last_measured, default=None),
            },
            "categories": list(CATEGORIES),
            "subcategories": {
                category: [asdict(sub) for sub in subs] for category, subs in SUBCATEGORIES.items()
            },
            "precisions": precisions,
            "gpus": [
                {"name": gpu, "rows": rows, "peaks": self._peaks(gpu, precisions)}
                for gpu, rows in sorted(gpu_rows.items(), key=lambda item: -item[1])
            ],
            "models": [
                {
                    "model_config": stem,
                    **(self.models.get(stem) or {"name": None, "family": None, "checkpoint": None}),
                }
                for stem in stems
            ],
            "kernels": kernels,
        }

    @staticmethod
    def _peaks(gpu: str, dtypes: list[str]) -> dict[str, dict]:
        spec = resolve_gpu_spec(gpu)
        return _peaks(spec, dtypes) if spec else {}

    def kernel(self, kind: str) -> dict:
        """What one kind computes, how it is measured, its backends and its models."""

        specs = self._specs(kind)
        doc = kernel_doc(kind)
        deployments, swept = self._used_by(kind)
        precision = self._precision_column(kind)
        schema = specs[0].args_schema
        types = typing.get_type_hints(schema)
        prose = asdict(doc) if doc else {}
        reference = prose.pop("reference", None)

        return {
            "kind": kind,
            "documented": doc is not None,
            **prose,
            "metric_family": specs[0].metric_family.value,
            "metrics": [
                {"name": name, **asdict(METRIC_DOCS[name])} for name in self._metrics(kind)
            ],
            "args": [
                {
                    "name": name,
                    "type": _arg_type(types[name]),
                    # Unknown until a registered config runs the kind.
                    "role": ("sweep" if name in swept else "config") if deployments else None,
                    # The column holding the compute dtype, which picks the throughput peak.
                    "precision": name == precision,
                    **entry,
                }
                for name, entry in arg_docs(schema).items()
            ],
            "backends": {
                spec.backend: {
                    **(asdict(spec.doc) if spec.doc else {"summary": None, "url": None}),
                    "supports": {
                        "compute": sorted(spec.supports.compute) if spec.supports.compute else None,
                        "kv": sorted(spec.supports.kv) if spec.supports.kv else None,
                        "gpus": sorted(spec.supports.gpus) if spec.supports.gpus else None,
                    },
                    "env": spec.subprocess_env or "default",
                }
                for spec in specs
            },
            "reference": self._reference(reference),
            "used_by": deployments,
        }

    @staticmethod
    def _reference(module: str | None) -> dict | None:
        if not module:
            return None
        path = Path(*module.split(".")).with_suffix(".py")
        return {"path": str(path), "source": (REPO_ROOT / path).read_text()}

    def rows(self, kind: str, filters: dict[str, str]) -> dict:
        """Measured rows, column-oriented, filtered by equality on any argument,
        ``gpu`` or ``backend``. Provenance repeats across rows, so each row points
        into a ``provenance`` table instead of carrying it."""

        args, metrics = self._args(kind), self._metrics(kind)
        types = self._column_types(kind)
        where, params = [], []
        for name, value in filters.items():
            column = "gpu_name" if name == "gpu" else name
            if name not in ("gpu", "backend", *args):
                raise BadQuery(f"{kind} has no column {name!r}")
            where.append(f'"{column}" = ?')
            params.append(self._typed(name, value, types.get(column, "")))
        columns = ["gpu_name", "backend", *args, *metrics, *PROVENANCE]
        index: dict[tuple, int] = {}
        provenance: list[dict] = []
        rows = []
        if types:
            query = (
                f'select {_quoted(columns)} from "{kind}"'
                + (f" where {' and '.join(where)}" if where else "")
                + f" order by {_quoted(columns[: 2 + len(args)])}"
            )
            with self.sources.connect() as conn:
                records = conn.execute(query, params).fetchall()
            width = 2 + len(args) + len(metrics)
            for record in records:
                prov = tuple(record[width:])
                if prov not in index:
                    index[prov] = len(provenance)
                    provenance.append(dict(zip(PROVENANCE, prov)))
                rows.append([*record[:width], index[prov]])
        return {
            "kind": kind,
            "columns": ["gpu", "backend", *args, *metrics, "provenance"],
            "rows": rows,
            "provenance": provenance,
        }

    # -- registered kernel configs --------------------------------------------------

    def _registered(self, kind: str, gpu: str | None = None) -> list[RegisteredConfig]:
        """The registered configs whose grids read ``kind``'s rows (every backend's
        table, though a kind's specs share one)."""

        tables = dict.fromkeys(spec.table_name for spec in self._specs(kind))
        with self.sources.connect() as conn:
            return [
                config
                for table in tables
                for config in registered_configs(conn, table, gpu_name=gpu)
            ]

    def _cell_rows(self, kind: str, config: RegisteredConfig) -> list[dict[str, dict]]:
        """Per cell, the row each backend measured for it: ``{backend: row}``.

        The same match as ``kernel_config.cell_row_ids`` (in SQL, so each column's
        declared type converts what it compares), one statement per chunk of
        cells instead of one per cell: the cells are a ``values`` table joined to
        the kind's rows on the config's GPU."""

        spec = next(s for s in self._specs(kind) if s.table_name == config.profile_kind)
        table = Table(spec, self.sources.db_path)
        args = list(table.args_columns)
        keys = cell_keys(table, config.grid.cells)
        metrics = ["backend", *self._metrics(kind), "is_outlier"]
        out: list[dict[str, dict]] = [{} for _ in keys]
        width = 1 + len(args)
        chunk = max(1, (_SQL_VARIABLES - 2) // width)
        cell_columns = _quoted(["_cell", *args])
        # Joining through the GPU's backends lets the (gpu_name, backend, args)
        # unique index answer each cell with one lookup per backend.
        backends = f'b(backend) as (select distinct backend from "{table.name}" where gpu_name = ?)'
        on = " and ".join(
            ["t.gpu_name = ?", "t.backend = b.backend", *(f't."{a}" = c."{a}"' for a in args)]
        )
        select = ", ".join(f't."{m}"' for m in metrics)
        with self.sources.connect() as conn:
            for start in range(0, len(keys), chunk):
                part = keys[start : start + chunk]
                values = ", ".join([f"({', '.join('?' * width)})"] * len(part))
                query = (
                    f"with c({cell_columns}) as (values {values}), {backends} "
                    f'select c."_cell", {select} from c cross join b join "{table.name}" t on {on}'
                )
                bound = [v for i, key in enumerate(part, start) for v in (i, *key)]
                gpu = config.gpu_name
                for cell, *record in conn.execute(query, [*bound, gpu, gpu]):
                    row = dict(zip(metrics, record))
                    out[cell][row.pop("backend")] = row
        return out

    def _summaries(self, kind: str) -> list[dict]:
        """Each registered config of ``kind`` with its grid's fixed and swept
        profile.db columns and the cells each backend measured."""

        def compute() -> list[dict]:
            out = []
            for config in self._registered(kind):
                packed = pack_cells(config.grid.cells)
                measured: dict[str, int] = {}
                for cell in self._cell_rows(kind, config):
                    for backend in cell:
                        measured[backend] = measured.get(backend, 0) + 1
                out.append(
                    {
                        "config": config,
                        "fixed": packed["fixed"],
                        "swept": list(packed["swept"]),
                        "measured": measured,
                    }
                )
            return out

        return self.sources.cached_by_db(("summaries", kind), compute)

    @staticmethod
    def _grid(summary: dict) -> dict:
        """What the list and the detail both say about one config's grid."""

        config = summary["config"]
        config_args, omitted = _config_args(config.identity)
        return {
            "config_hash": config.config_hash,
            "kind": config.kind,
            "gpu": config.gpu_name,
            "cache_coords": list(config.grid.cache_coords),
            "shape": [len(axis) for axis in config.grid.axes],
            "cells": len(config.grid.cells),
            "infeasible": len(config.grid.infeasible),
            "measured": summary["measured"],
            # profile.db args every cell shares, and the ones the cache axes move
            "fixed": summary["fixed"],
            "swept": summary["swept"],
            # the Rust config's own scalar values, some of which name no DB column
            "config_args": config_args,
            "config_args_omitted": omitted,
        }

    def _use(self, use: ConfigUse, gpu: str) -> dict:
        """One use of a config: what built it, and the deployment and model."""

        return {
            "pool": use.pool,
            "role": use.role,
            "via": _source_via(use.source),
            "deployments": self._source_deployments(use.source, gpu),
        }

    def configs(self, kind: str) -> dict:
        """The kernel configs registered as reading ``kind``'s rows: one per
        config and GPU, with its fixed and swept args, how many of its grid
        cells each backend measured, and the uses that built it.

        Uses repeat their source and deployment across configs, so each use
        points into the ``sources`` and ``deployments`` tables; the full
        identity and source of a config are on ``/configs/{config_hash}``."""

        def compute() -> dict:
            sources: dict[str, dict] = {}
            deployments: dict[str, dict] = {}
            out = []
            for summary in self._summaries(kind):
                config = summary["config"]
                uses = []
                for use in config.uses:
                    labeled = self._use(use, config.gpu_name)
                    source = sources.setdefault(
                        canonical_json(labeled["via"]), {"id": len(sources), **labeled["via"]}
                    )
                    ids = []
                    for deployment in labeled["deployments"]:
                        key = self._deployment_key(deployment)
                        ids.append(
                            deployments.setdefault(key, {"id": len(deployments), **deployment})[
                                "id"
                            ]
                        )
                    uses.append(
                        {
                            "pool": use.pool,
                            "role": use.role,
                            "source": source["id"],
                            "deployments": ids,
                        }
                    )
                out.append({**self._grid(summary), "axes": _axes(config), "uses": uses})
            return {
                "kind": kind,
                "sources": list(sources.values()),
                "deployments": list(deployments.values()),
                "configs": out,
            }

        return self.sources.cached_by_db(("configs", kind), compute)

    def config(self, kind: str, config_hash: str, gpu: str | None) -> dict:
        """One registered config's grid on the Rust cache axes: every cell's
        coordinates, its profile.db args, whether the kernel can run it, and the
        metrics each backend measured there (absent where nothing was); plus
        the full identity and each use with its source, deployment and model."""

        matches = [c for c in self._registered(kind, gpu) if c.config_hash == config_hash]
        if not matches:
            raise UnknownConfig(config_hash)
        if len(matches) > 1:
            gpus = sorted(c.gpu_name for c in matches)
            raise BadQuery(f"config {config_hash} is registered on {gpus}; pass gpu")
        [config] = matches
        [summary] = [
            s
            for s in self._summaries(kind)
            if (s["config"].config_hash, s["config"].gpu_name) == (config_hash, config.gpu_name)
        ]
        grid = config.grid
        metrics = self._metrics(kind)
        points = [
            {
                "coords": list(grid.coords(i)),
                "feasible": i not in grid.infeasible,
                "args": cell,
                "measured": {
                    backend: {
                        **{m: row[m] for m in metrics},
                        "outlier": bool(row["is_outlier"]),
                    }
                    for backend, row in measured.items()
                },
            }
            for i, (cell, measured) in enumerate(
                zip(grid.cells, self._cell_rows(kind, config), strict=True)
            )
        ]
        return {
            **self._grid(summary),
            "identity": config.identity,
            "axes": _axes(config),
            "metrics": metrics,
            "points": points,
            "uses": [
                {"source": use.source, **self._use(use, config.gpu_name)} for use in config.uses
            ],
        }

    @staticmethod
    def _typed(name: str, value: str, declared: str) -> Any:
        try:
            if "INT" in declared:
                return int(value)
            if any(t in declared for t in ("REAL", "FLOA", "DOUB")):
                return float(value)
        except ValueError:
            raise BadQuery(f"{name} must be a number, got {value!r}") from None
        return value

    @staticmethod
    def rows_csv(document: dict) -> str:
        """The rows document as CSV, one provenance field per column."""

        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow([*document["columns"][:-1], *PROVENANCE])
        for row in document["rows"]:
            prov = document["provenance"][row[-1]]
            writer.writerow([*row[:-1], *(prov[k] for k in PROVENANCE)])
        return out.getvalue()
