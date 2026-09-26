"""The kernel library's documents: the catalog, one kind's detail, measured rows.

Every field comes from a source that lives with the code; nothing here guesses:

- prose, including how the kind is timed, from the kind module (``DOC``,
  ``arg(...)``, ``BackendDoc``) and the mechanical facts from the profiling
  registry (argument order, backends, supported dtypes and GPUs, metric family,
  subprocess environment);
- the models that run a kind, from the cost tree of every ``#[supported]`` arch
  deployment on its declared GPU;
- which arguments a model sweeps and which it fixes, and the profile.db columns
  each model's shape reads, from the kernel's own ``enumerate`` applied to those
  cost tree leaves;
- a row's precision, from the column the Rust kernel names as its compute dtype;
- GPU peaks from ``gpu/spec.json`` and model names from ``model/catalog.yaml``;
- measurements from profile.db, and the kernel configs that read them from its
  kernel-config registry (``profiling/db/kernel_config.py``).
"""

from __future__ import annotations

import csv
import io
import subprocess
import typing
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from profiling.db.args import DType
from profiling.db.doc import CATEGORIES, SUBCATEGORIES, arg_docs, kernel_doc
from profiling.db.doc import METRICS as METRIC_DOCS
from profiling.db.kernel_config import RegisteredConfig, cell_row_ids, registered_configs
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


class KernelLibrary:
    """Builds the kernel library documents from :class:`KernelSources`."""

    def __init__(self, sources: KernelSources) -> None:
        self.sources = sources
        self.specs: dict[str, list] = {}
        for spec in iter_kernel_profiler_specs():
            self.specs.setdefault(spec.kernel_kind, []).append(spec)
        self.models = yaml.safe_load(MODEL_CATALOG.read_text()) or {}
        self.sim_commit = _git_commit()

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

    def _slots(self):
        """``(build, section, slot)`` for every leaf of every supported cost tree."""

        for build in self.sources.supported_builds():
            if build["error"]:
                raise RuntimeError(f"{build['arch']} {build['params']}: {build['error']}")
            for section in build["cost_manifest"]["sections"]:
                for slot in section["slots"]:
                    yield build, section, slot

    def _model_name(self, model_config: str) -> str:
        return (self.models.get(model_config) or {}).get("name", model_config)

    def _used_by(self, kind: str) -> tuple[list[dict], set[str]]:
        """The supported deployments whose cost tree holds ``kind``, each with the
        shapes it asks for, and the columns those leaves sweep.

        A shape is the profile.db columns its leaf fixes (``rows``'s ``fixed``,
        without ``backend``), plus the expression and bindings behind each config
        value. Rank copies of one leaf share name and config and are listed once.
        """

        deployments: dict[int, dict] = {}
        swept: set[str] = set()
        for build, section, slot in self._slots():
            if slot["kind"] != kind:
                continue
            config = slot["kernel_config"]
            report = self.sources.rows_report(kind, config)
            swept.update(report["swept"])
            db = {k: v for k, v in report["fixed"].items() if k != "backend"}
            deployment = deployments.get(id(build))
            if deployment is None:
                params = dict(build["params"])
                model = params.pop("model_config", None)
                deployment = deployments[id(build)] = {
                    "arch": build["arch"],
                    "model_config": model,
                    # name, family, checkpoint; null for a model not in model/catalog.yaml
                    "model": self.models.get(model),
                    "gpu": build["gpu"],
                    "gpus_per_replica": build["gpus_per_replica"],
                    "parallel": params,
                    "shapes": {},
                }
            deployment["shapes"].setdefault(
                repr((slot["name"], sorted(db.items()))),
                {
                    "layer": slot["name"],
                    "section": section["section"],
                    "db": db,
                    "why": {
                        k: {"expression": v["expression"], "bindings": v["bindings"]}
                        for k, v in config.items()
                        if isinstance(v, dict) and v.get("expression")
                    },
                },
            )
        out = [{**d, "shapes": list(d["shapes"].values())} for d in deployments.values()]
        return out, swept

    # -- documents -----------------------------------------------------------------

    def catalog(self) -> dict:
        """Every registered kind with its coverage and the models that run it."""

        models_by_kind: dict[str, list[str]] = {}
        for build, _, slot in self._slots():
            names = models_by_kind.setdefault(slot["kind"], [])
            name = self._model_name(build["params"]["model_config"])
            if name not in names:
                names.append(name)

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
                    "used_by": models_by_kind.get(kind, []),
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
            "models": [{"model_config": stem, **entry} for stem, entry in self.models.items()],
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
                    # Unknown until a supported deployment runs the kind.
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
        """Per cell, the row each backend measured for it: ``{backend: row}``."""

        spec = next(s for s in self._specs(kind) if s.table_name == config.profile_kind)
        columns = ["id", "backend", *self._metrics(kind), "is_outlier"]
        with self.sources.connect() as conn:
            ids = cell_row_ids(
                conn, Table(spec, self.sources.db_path), config.gpu_name, config.grid.cells
            )
            wanted = sorted({i for cell in ids for i in cell})
            rows = {}
            for start in range(0, len(wanted), 500):
                chunk = wanted[start : start + 500]
                query = (
                    f'select {_quoted(columns)} from "{config.profile_kind}" '
                    f"where id in ({', '.join('?' * len(chunk))})"
                )
                for record in conn.execute(query, chunk):
                    rows[record[0]] = dict(zip(columns[1:], record[1:]))
        return [{rows[i]["backend"]: rows[i] for i in cell} for cell in ids]

    def configs(self, kind: str) -> dict:
        """The kernel configs registered as reading ``kind``'s rows: one per
        config and GPU, with how many of its grid cells each backend measured and
        the deployments or predictions that use it."""

        def compute() -> dict:
            out = []
            for config in self._registered(kind):
                measured: dict[str, int] = {}
                for cell in self._cell_rows(kind, config):
                    for backend in cell:
                        measured[backend] = measured.get(backend, 0) + 1
                out.append(
                    {
                        "config_hash": config.config_hash,
                        "kind": config.kind,
                        "gpu": config.gpu_name,
                        "identity": config.identity,
                        "cache_coords": list(config.grid.cache_coords),
                        "shape": [len(axis) for axis in config.grid.axes],
                        "cells": len(config.grid.cells),
                        "infeasible": len(config.grid.infeasible),
                        "measured": measured,
                        "uses": [asdict(use) for use in config.uses],
                    }
                )
            return {"kind": kind, "configs": out}

        return self.sources.cached_by_db(("configs", kind), compute)

    def config(self, kind: str, config_hash: str, gpu: str | None) -> dict:
        """One registered config's grid on the Rust cache axes: every cell's
        coordinates, its profile.db args, whether the kernel can run it, and the
        metrics each backend measured there (absent where nothing was)."""

        matches = [c for c in self._registered(kind, gpu) if c.config_hash == config_hash]
        if not matches:
            raise UnknownConfig(config_hash)
        if len(matches) > 1:
            gpus = sorted(c.gpu_name for c in matches)
            raise BadQuery(f"config {config_hash} is registered on {gpus}; pass gpu")
        [config] = matches
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
            "config_hash": config.config_hash,
            "kind": config.kind,
            "gpu": config.gpu_name,
            "identity": config.identity,
            "cache_coords": list(grid.cache_coords),
            "axes": [list(axis) for axis in grid.axes],
            "metrics": metrics,
            "points": points,
            "uses": [asdict(use) for use in config.uses],
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
