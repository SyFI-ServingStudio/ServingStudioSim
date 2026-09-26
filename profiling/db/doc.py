"""Reader-facing documentation for kernel kinds and their backends.

The Kernel Library (the ServingStudio site's kernel pages) shows what each
kernel computes, what its arguments mean and which backends measure it. Only the
prose lives here; everything mechanical (argument order and types, supported
dtypes and GPUs, metric family, which arguments a run sweeps) is read from the
registry and the Rust kernel specs, so it cannot drift from the code.

A kind module documents itself in three places, all reviewed with the code:

- each ``<Kind>Args`` field through :func:`arg` (unit and meaning);
- a module-level ``DOC = KernelDoc(...)``;
- each ``KernelProfilerSpec`` through ``doc=BackendDoc(...)``.

``tests/test_kernel_docs.py`` checks that a documented kind is complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

#: Kernel categories, in the order the library lists them.
CATEGORIES: tuple[str, ...] = (
    "GEMM",
    "Attention",
    "MoE",
    "Communication",
    "Normalization",
    "Quantization",
    "Other",
)


@dataclass(frozen=True)
class Subcategory:
    """One subdivision of a category, with a one-line summary for the reader."""

    name: str
    summary: str


#: Subcategories per category, in the order the library lists them. Attention
#: is divided by the attention a model layer runs.
SUBCATEGORIES: dict[str, tuple[Subcategory, ...]] = {
    "Attention": (
        Subcategory(
            "MHA / GQA",
            "Dense attention over a per-head KV cache, with query heads sharing KV "
            "heads under GQA.",
        ),
        Subcategory(
            "MLA",
            "Multi-head latent attention: every head reads one compressed KV latent per token.",
        ),
        Subcategory(
            "DSA",
            "Sparse MLA: an indexer scores the cache and each query attends only to "
            "its top-k tokens.",
        ),
        Subcategory(
            "Gated DeltaNet",
            "Linear attention: a gated delta-rule state stands in for the KV cache.",
        ),
    ),
}


@dataclass(frozen=True)
class MetricDoc:
    """How the library labels one measured metric."""

    label: str
    unit: str


#: Every field of ``ComputeMetrics`` and ``CommMetrics``
#: (``profiling/runners/metrics.py``), by name.
METRICS: dict[str, MetricDoc] = {
    "time_ms": MetricDoc("Time", "ms"),
    "tflops": MetricDoc("Throughput", "TFLOPS"),
    "memory_bandwidth_gbps": MetricDoc("Memory bandwidth", "GB/s"),
    "algbw_gbps": MetricDoc("Algorithm bandwidth", "GB/s"),
    "busbw_gbps": MetricDoc("Bus bandwidth", "GB/s"),
    "energy_j": MetricDoc("Energy", "J"),
}


def arg(*, doc: str, unit: str | None = None) -> Any:
    """Declare a documented ``KernelArgs`` field.

    A plain ``dataclasses.field`` with no default, so the field stays required
    and the declaration order, which is the DB column order, is unchanged.
    ``unit`` names what one step of a numeric argument counts (tokens, elements,
    heads, bytes, GPUs); a dtype or enum argument has none.
    """

    return field(metadata={"doc": doc, "unit": unit})


def arg_docs(args_schema: type) -> dict[str, dict[str, str | None]]:
    """``{field: {"doc", "unit"}}`` for every field of an Args schema, in order.

    A field declared without :func:`arg` maps to ``{"doc": None, "unit": None}``.
    """

    return {
        f.name: {"doc": f.metadata.get("doc"), "unit": f.metadata.get("unit")}
        for f in fields(args_schema)
    }


@dataclass(frozen=True)
class BackendDoc:
    """What one backend measures, for a reader choosing between backends."""

    summary: str
    url: str | None = None


#: How ``Timer.cupti`` measures with its defaults (``profiling/profilers/timer.py``).
CUPTI_METHOD = (
    "GPU kernel time from CUPTI activity records, averaged over repeated launches "
    "in one capture. The L2 cache is flushed before each launch, so the numbers are "
    "cold-cache times."
)


@dataclass(frozen=True)
class KernelDoc:
    """What one kernel kind computes and how to read its numbers.

    ``method`` says how the kind's runners time it. A CUPTI-timed kind opens
    with :data:`CUPTI_METHOD` and adds only what is particular to it.
    ``reference`` is the module path of a PyTorch reference implementation, or
    None when a measured backend is itself the reference.
    """

    title: str
    summary: str
    description: str
    category: str
    formula: tuple[str, ...]
    default_metric: str
    method: str
    subcategory: str | None = None
    caveats: tuple[str, ...] = ()
    reference: str | None = None


def kernel_doc(kernel_kind: str) -> KernelDoc | None:
    """The ``DOC`` of ``profiling/kernels/<kernel_kind>.py``, or None if it has none."""

    import importlib

    try:
        module = importlib.import_module(f"profiling.kernels.{kernel_kind}")
    except ModuleNotFoundError:
        return None
    return getattr(module, "DOC", None)
