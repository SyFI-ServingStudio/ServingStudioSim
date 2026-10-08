"""The kernel library's documents: the catalog, one kind's detail, measured rows.

Every field comes from a source that lives with the code; nothing here guesses:

- prose, including how the kind is timed, from the kind module (``DOC``,
  ``arg(...)``, ``BackendDoc``) and the mechanical facts from the profiling
  registry (argument order, backends, supported dtypes and GPUs, metric family,
  subprocess environment);
- a row's precision, from the column the Rust kernel names as its compute dtype;
- GPU peaks from ``gpu/spec.json`` and model names from ``model/catalog.yaml``;
- measurements from profile.db;
- the kernel configs, from the public deployments (:mod:`public_api.deployments`):
  each config a public member asks profile.db for, its grid, and who asks.
"""

from __future__ import annotations

import csv
import io
import subprocess
import typing
from dataclasses import asdict, fields
from functools import cache
from pathlib import Path
from typing import Any

from profiling.db.args import DType
from profiling.db.doc import CATEGORIES, SUBCATEGORIES, arg_docs, kernel_doc, kind_vocabulary
from profiling.db.doc import METRICS as METRIC_DOCS
from profiling.db.registry import BackendSupport, MetricFamily, iter_kernel_profiler_specs
from profiling.db.storage import CREATED_AT_FORMAT, RUN_AT_FORMAT, RUN_TABLE, iso_sql
from profiling.db.table import STANDARD_COLUMNS, Table
from profiling.gpu_catalog import GpuSpecResolution, load_gpu_catalog, resolve_gpu_spec
from profiling.runners.metrics import CommMetrics, ComputeMetrics
from public_api.deployments import Config, DeploymentIndex, scalar_identity
from public_api.sources import Sources

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    """No public deployment asks for a config with this id."""


class BadQuery(ValueError):
    """A rows filter names an unknown column or a value of the wrong type."""


def _supports(supports: BackendSupport) -> dict:
    """A backend's dtypes and device requirement. ``gpus`` names the catalog
    GPUs the device gate admits — NVIDIA by compute capability, AMD by
    ``arch_targets`` architecture; None when the backend has no device gate
    (any GPU)."""
    capability = supports.min_compute_capability
    device = (
        capability is not None
        or supports.sm_targets is not None
        or supports.arch_targets is not None
    )
    return {
        "compute": sorted(supports.compute) if supports.compute else None,
        "kv": sorted(supports.kv) if supports.kv else None,
        "min_compute_capability": ".".join(map(str, capability)) if capability else None,
        "sm_targets": sorted(supports.sm_targets) if supports.sm_targets else None,
        "arch_targets": sorted(supports.arch_targets) if supports.arch_targets else None,
        "gpus": [
            gpu.canonical_name
            for gpu in _catalog_gpus()
            if supports.allows_device(gpu.canonical_name)
        ]
        if device
        else None,
    }


@cache
def _catalog_gpus() -> list[GpuSpecResolution]:
    """Every resolvable GPU of ``gpu/spec.json``, in catalog order. The device
    gate (``allows_device``) decides which a backend admits."""
    gpus = (resolve_gpu_spec(gpu["name"]) for gpu in load_gpu_catalog())
    return [gpu for gpu in gpus if gpu is not None]


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


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


class KernelLibrary:
    """Builds the kernel library documents from :class:`Sources`."""

    def __init__(self, sources: Sources, index: DeploymentIndex) -> None:
        self.sources = sources
        self.index = index
        self.specs: dict[str, list] = {}
        for spec in iter_kernel_profiler_specs():
            self.specs.setdefault(spec.kernel_kind, []).append(spec)
        self.sim_commit = git_commit()

    @property
    def models(self) -> dict[str, dict]:
        """``model/catalog.yaml``, as the deployments read it at start."""

        return self.index.models

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
            last = iso_sql("max(profiler_run_at)", RUN_AT_FORMAT)
            query = (
                f"select gpu_name, backend, {precision_sql}, count(*), {last} "
                f'from "{kind}" group by 1, 2, 3 order by 1, 2, 3'
            )
            with self.sources.connect() as conn:
                records = conn.execute(query).fetchall()
            return [
                {"gpu": g, "backend": b, "precision": p, "rows": n, "last_measured_at": t}
                for g, b, p, n, t in records
            ]

        return self.sources.cached_by_db(("coverage", kind), compute)

    # -- documents -----------------------------------------------------------------

    @staticmethod
    def kinds() -> dict:
        """Each kind's title and category from its DOC, as the Analyzer's
        ``/api/analyzer/v1/kernel-kinds`` serves them
        (:func:`profiling.db.doc.kind_vocabulary`)."""

        return kind_vocabulary()

    def catalog(self) -> dict:
        """Every registered kind with its coverage, the GPUs with their peaks, and
        the checkpoints ``model/catalog.yaml`` names, in its order."""

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
                    # The presets with a member that builds a config of this kind.
                    "used_by": sorted(
                        {
                            preset
                            for config in self.index.kind_configs(kind)
                            for preset, _ in config.uses
                        }
                    ),
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
            "models": [{"checkpoint": name, **model} for name, model in self.models.items()],
            "kernels": kernels,
        }

    @staticmethod
    def _peaks(gpu: str, dtypes: list[str]) -> dict[str, dict]:
        spec = resolve_gpu_spec(gpu)
        return _peaks(spec, dtypes) if spec else {}

    def kernel(self, kind: str) -> dict:
        """What one kind computes, how it is measured, and its backends."""

        specs = self._specs(kind)
        doc = kernel_doc(kind)
        precision = self._precision_column(kind)
        schema = specs[0].args_schema
        types = typing.get_type_hints(schema)
        prose = asdict(doc) if doc else {}
        reference = prose.pop("reference", None)
        view = prose.pop("view", None)

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
                    # The column holding the compute dtype, which picks the throughput peak.
                    "precision": name == precision,
                    **entry,
                }
                for name, entry in arg_docs(schema).items()
            ],
            "backends": {
                spec.backend: {
                    **(asdict(spec.doc) if spec.doc else {"summary": None, "url": None}),
                    "supports": _supports(spec.supports),
                    "env": spec.subprocess_env or "default",
                }
                for spec in specs
            },
            "reference": self._reference(reference),
            # a chart over several configs, declared by the kind (ConfigView)
            "view": view,
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
            where.append(f't."{column}" = ?')
            params.append(self._typed(name, value, types.get(column, "")))
        keyed = ["gpu_name", "backend", *args]
        stored = {
            "profiler_run_at": iso_sql("t.profiler_run_at", RUN_AT_FORMAT),
            "created_at": iso_sql("t.created_at", CREATED_AT_FORMAT),
        }
        select = [
            *(f't."{c}"' for c in [*keyed, *metrics]),
            *(stored.get(c, f'r."{c}"') for c in PROVENANCE),
        ]
        index: dict[tuple, int] = {}
        provenance: list[dict] = []
        rows = []
        if types:
            query = (
                f'select {", ".join(select)} from "{kind}" t '
                f"join {RUN_TABLE} r on r.run_key = t.run_key"
                + (f" where {' and '.join(where)}" if where else "")
                + f" order by {', '.join(select[: len(keyed)])}"
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

    # -- kernel configs ------------------------------------------------------------

    def _measured(self, kind: str, profile_kind: str, gpu: str) -> dict[bytes, dict[str, dict]]:
        """``{args_hash: {backend: metrics}}`` of a table's rows on one GPU."""

        def compute() -> dict[bytes, dict[str, dict]]:
            metrics = self._metrics(kind)
            select = ", ".join(f'"{m}"' for m in metrics)
            out: dict[bytes, dict[str, dict]] = {}
            with self.sources.connect() as conn:
                if not conn.execute(
                    "select 1 from sqlite_master where type = 'table' and name = ?", [profile_kind]
                ).fetchone():
                    return out
                query = (
                    f'select args_hash, backend, is_outlier, {select} from "{profile_kind}" '
                    "where gpu_name = ?"
                )
                for args_hash, backend, outlier, *values in conn.execute(query, [gpu]):
                    out.setdefault(args_hash, {})[backend] = {
                        **dict(zip(metrics, values, strict=True)),
                        "outlier": bool(outlier),
                    }
            return out

        return self.sources.cached_by_db(("measured", profile_kind, gpu), compute)

    def _cell_rows(self, config: Config) -> list[dict[str, dict]]:
        """Per grid cell, the row each backend measured there."""

        spec = next(s for s in self._specs(config.kind) if s.table_name == config.profile_kind)
        table = Table(spec, self.sources.db_path)
        measured = self._measured(config.kind, config.profile_kind, config.gpu)
        return [
            measured.get(table.args_hash(table.db_key(spec.args_schema(**cell))), {})
            for cell in config.grid["cells"]
        ]

    def _summary(self, config: Config) -> dict:
        """What the list and the detail both say about one config."""

        cells = config.grid["cells"]
        columns = list(cells[0]) if cells else []
        fixed = {c: cells[0][c] for c in columns if all(cell[c] == cells[0][c] for cell in cells)}
        counts: dict[str, int] = {}
        for backends in self._cell_rows(config):
            for backend in backends:
                counts[backend] = counts.get(backend, 0) + 1
        scalars, structured = scalar_identity(config.identity)
        return {
            "id": config.id,
            "kind": config.kind,
            "gpu": config.gpu,
            "cache_coords": config.grid["cache_coords"],
            "axes": config.grid["axes"],
            "shape": [len(axis) for axis in config.grid["axes"]],
            "cells": len(cells),
            "infeasible": len(config.grid["infeasible"]),
            # cells each backend measured
            "measured": counts,
            # profile.db args every cell shares, and the ones the cache axes move
            "fixed": fixed,
            "swept": [c for c in columns if c not in fixed],
            # the Rust config's scalar fields, and the names of its structured ones
            "identity": scalars,
            "structured": structured,
            "uses": self.index.uses(config),
        }

    def configs(self, kind: str) -> dict:
        """Every config a public deployment asks ``kind``'s rows for."""

        self._specs(kind)
        return {
            "kind": kind,
            "configs": [self._summary(config) for config in self.index.kind_configs(kind)],
        }

    def config(self, kind: str, config_id: str) -> dict:
        """One config's grid on the simulator's cache axes: each cell's coordinates,
        profile.db args, whether the kernel can run it and what each backend
        measured there; plus the whole identity."""

        config = self.index.configs.get(config_id)
        if config is None or config.kind != kind:
            raise UnknownConfig(config_id)
        axes = config.grid["axes"]
        infeasible = set(config.grid["infeasible"])

        def coords(index: int) -> list[float]:
            out = []
            for axis in reversed(axes):
                index, at = divmod(index, len(axis))
                out.append(axis[at])
            return out[::-1]

        return {
            **self._summary(config),
            "identity": config.identity,
            "metrics": self._metrics(kind),
            "points": [
                {
                    "coords": coords(i),
                    "feasible": i not in infeasible,
                    "args": cell,
                    "measured": measured,
                }
                for i, (cell, measured) in enumerate(
                    zip(config.grid["cells"], self._cell_rows(config), strict=True)
                )
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
