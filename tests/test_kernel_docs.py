"""Every kernel kind documents everything a reader sees.

Each registered kind has a complete module-level ``DOC``, every Args field
declared through ``arg``, and a ``BackendDoc`` on every backend. A kernel
describes an operation, not a model: no kind, backend or documentation text
names a model from ``model/catalog.yaml`` (which models run a kernel is
published from the kernel-config registry instead).
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

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
CATALOG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "model" / "catalog.yaml").read_text()
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _joined(text: str) -> str:
    return "".join(_tokens(text))


def _model_terms() -> tuple[frozenset[str], frozenset[str]]:
    """Model names from the catalog, normalized so spellings compare equal.

    Returns ``(prefixes, phrases)``. A word starting with a prefix names a model:
    each family ("glm" catches GLM-5.2, glm52 and GLM's) and each family's
    initials plus a model's version ("DeepSeek" + "V4" -> "dsv4"). A phrase is a
    whole name, config stem or checkpoint, matched on text with case and
    punctuation removed ("deepseek_v4_flash_0731", "DeepSeek V4 Flash 0731").
    """
    prefixes: set[str] = set()
    phrases: set[str] = set()
    for stem, entry in CATALOG.items():
        family = entry["family"]
        prefixes.add(_joined(family))
        initials = "".join(re.findall(r"[A-Z]", family)).lower()
        after_family = _tokens(entry["name"])[len(_tokens(family)) :]
        if len(initials) > 1 and after_family:
            prefixes.add(initials + after_family[0])
        for text in (stem, entry["name"], entry["checkpoint"].split("/")[-1]):
            phrases.add(_joined(text))
    return frozenset(prefixes), frozenset(phrases)


MODEL_PREFIXES, MODEL_PHRASES = _model_terms()


def _model_names_in(text: str) -> set[str]:
    words = _tokens(text)
    joined = "".join(words)
    found = {w for w in words for p in MODEL_PREFIXES if w.startswith(p)}
    return found | {p for p in MODEL_PHRASES if p in joined}


def _reader_texts(kind: str) -> dict[str, str]:
    """Every name and sentence the Kernel Library shows for a kind, except URLs,
    which may legitimately point into an upstream model directory."""
    doc = kernel_doc(kind)
    texts = {
        "kind": kind,
        "title": doc.title,
        "summary": doc.summary,
        "description": doc.description,
        "method": doc.method,
        "formula": " ".join(doc.formula),
        "caveats": " ".join(doc.caveats),
        "reference": doc.reference or "",
    }
    for name, entry in arg_docs(KINDS[kind][0].args_schema).items():
        texts[f"arg {name}"] = entry["doc"] or ""
    for spec in KINDS[kind]:
        texts[f"backend {spec.backend}"] = spec.backend
        texts[f"backend {spec.backend} summary"] = spec.doc.summary if spec.doc else ""
    return texts


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


def test_model_names_are_recognized_in_any_spelling() -> None:
    # The terms come from the catalog; check they catch the usual spellings and
    # leave kernel vocabulary alone.
    for text in (
        "GLM-5.2's DSA indexer",
        "glm52",
        "DeepSeek V4",
        "deepseek_v4_x",
        "DSV4",
        "Qwen3.6",
    ):
        assert _model_names_in(text), text
    for text in ("DSA indexer", "DeepGEMM", "FlashMLA", "vllm_cutedsl", "mHC head"):
        assert not _model_names_in(text), text


@pytest.mark.parametrize("kind", DOCUMENTED_NOW)
def test_kernel_names_no_model(kind: str) -> None:
    named = {
        field: sorted(found)
        for field, text in _reader_texts(kind).items()
        if (found := _model_names_in(text))
    }
    assert not named, f"{kind} names a model: {named}"


@pytest.mark.parametrize("kind", DOCUMENTED_NOW)
def test_subdivided_category_names_a_subcategory(kind: str) -> None:
    # The library lists a subdivided category only through its subcategories,
    # so a kind without one would not be shown.
    doc = kernel_doc(kind)
    if doc.category in SUBCATEGORIES:
        assert doc.subcategory is not None, f"{kind} has no {doc.category} subcategory"
