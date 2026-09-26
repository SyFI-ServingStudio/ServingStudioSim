"""A documented kernel kind documents everything a reader sees.

Documentation is opt-in per kind while it is being written: a kind with a
module-level ``DOC`` must be complete, and the kinds in ``DOCUMENTED`` must
have one. Once every kind is documented, ``DOCUMENTED`` becomes all kinds.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from profiling.db import doc as kernel_docs
from profiling.db.args import DType
from profiling.db.doc import CATEGORIES, SUBCATEGORIES, arg_docs, kernel_doc
from profiling.db.registry import MetricFamily, iter_kernel_profiler_specs
from profiling.runners.metrics import CommMetrics, ComputeMetrics

DOCUMENTED: set[str] = set()

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


def test_listed_kinds_are_documented() -> None:
    assert DOCUMENTED <= set(DOCUMENTED_NOW)


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
