"""Every kernel kind documents everything a reader sees.

Each registered kind has a complete module-level ``DOC``, every Args field
declared through ``arg``, and a ``BackendDoc`` on every backend.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import fields

import pytest

from launcher.exec import _build_subprocess_env
from profiling.db import doc as kernel_docs
from profiling.db.args import DType
from profiling.db.doc import CATEGORIES, SUBCATEGORIES, arg_docs, kernel_doc
from profiling.db.registry import MetricFamily, iter_kernel_profiler_specs
from profiling.runners.metrics import CommMetrics, ComputeMetrics

METRICS = {
    MetricFamily.COMPUTE: {f.name for f in fields(ComputeMetrics)},
    MetricFamily.COMM: {f.name for f in fields(CommMetrics)},
}


def _kinds() -> dict[str, list]:
    kinds: dict[str, list] = {}
    for spec in iter_kernel_profiler_specs():
        kinds.setdefault(spec.kernel_kind, []).append(spec)
    return kinds


KINDS = _kinds()
DOCUMENTED_NOW = sorted(kind for kind in KINDS if kernel_doc(kind) is not None)


def test_every_kind_is_documented() -> None:
    assert DOCUMENTED_NOW == sorted(KINDS)


@pytest.mark.parametrize("kind", DOCUMENTED_NOW)
def test_kernel_doc_is_complete(kind: str) -> None:
    doc = kernel_doc(kind)
    specs = KINDS[kind]
    for text in (doc.title, doc.summary, doc.description):
        assert text.strip()
    assert doc.formula
    assert doc.category in CATEGORIES
    names = [sub.name for sub in SUBCATEGORIES.get(doc.category, ())]
    assert doc.subcategory in (None, *names)
    assert doc.default_metric in METRICS[specs[0].metric_family]


@pytest.mark.parametrize("kind", DOCUMENTED_NOW)
def test_every_argument_is_documented(kind: str) -> None:
    schema = KINDS[kind][0].args_schema
    types = {f.name: f.type for f in fields(schema)}
    for name, entry in arg_docs(schema).items():
        assert entry["doc"], f"{kind}.{name} has no doc"
        # A count or size has a unit; a dtype, enum or flag does not.
        numeric = str(types[name]) in ("int", "float", "tuple[int, ...]")
        assert (entry["unit"] is not None) == numeric, f"{kind}.{name} unit"


@pytest.mark.parametrize("kind", DOCUMENTED_NOW)
def test_every_backend_is_documented(kind: str) -> None:
    for spec in KINDS[kind]:
        assert spec.doc is not None, f"{kind}/{spec.backend} has no BackendDoc"
        assert spec.doc.summary.strip()


def test_subcategories_are_summarized() -> None:
    for category, subs in SUBCATEGORIES.items():
        assert category in CATEGORIES
        assert len({sub.name for sub in subs}) == len(subs)
        for sub in subs:
            assert sub.name.strip() and sub.summary.strip()


def test_every_metric_is_labeled() -> None:
    assert set(kernel_docs.METRICS) == set().union(*METRICS.values())


def test_dtype_arguments_have_no_unit() -> None:
    schema = KINDS["single_gemm"][0].args_schema
    assert fields(schema)[3].type in ("DType", DType)
    assert arg_docs(schema)["dtype"]["unit"] is None


@pytest.mark.parametrize("kind", DOCUMENTED_NOW)
def test_subdivided_category_names_a_subcategory(kind: str) -> None:
    # The library lists a subdivided category only through its subcategories,
    # so a kind without one would not be shown.
    doc = kernel_doc(kind)
    if doc.category in SUBCATEGORIES:
        assert doc.subcategory is not None, f"{kind} has no {doc.category} subcategory"


VIEWED = [kind for kind in DOCUMENTED_NOW if kernel_doc(kind).view is not None]


@pytest.fixture(scope="module")
def rust_kernels(sim_bin) -> list[dict]:
    """``simulator kernel-list``: every Rust kernel kind with its config fields."""

    result = subprocess.run(
        [str(sim_bin), "kernel-list"],
        capture_output=True,
        text=True,
        env=_build_subprocess_env(),
        check=True,
    )
    return json.loads(result.stdout)


def test_some_kind_declares_a_view() -> None:
    assert VIEWED


@pytest.mark.needs_binary
@pytest.mark.parametrize("kind", VIEWED)
def test_view_fields_are_rust_config_fields(kind: str, rust_kernels: list[dict]) -> None:
    # A view is drawn by config values the registry records, so each field it
    # names must be a field of every Rust config that reads the kind's rows.
    tables = {spec.table_name for spec in KINDS[kind]}
    configs = [e["config"] for e in rust_kernels if e["profile_kind"] in tables]
    assert configs, f"no Rust kernel reads {kind}'s rows"
    view = kernel_doc(kind).view
    for view_field in (view.series, view.workload):
        for config in configs:
            assert view_field.field in config, f"{kind}: {view_field.field} is not a config field"
        assert view_field.label.strip() and view_field.doc.strip()
    assert view.title.strip() and view.summary.strip()
