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
#: is divided by the attention a model layer runs, MoE by the step of the layer
#: and Normalization by what is normalized. In a category listed here every
#: kind names one of its subcategories: the library shows a subdivided category
#: only through them.
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
            "Compressed sparse MLA",
            "Sparse MLA over a compressed KV cache: every few tokens are pooled into "
            "one key, and each query attends to a sliding window of recent tokens "
            "plus compressed keys.",
        ),
        Subcategory(
            "Gated DeltaNet",
            "Linear attention: a gated delta-rule state stands in for the KV cache.",
        ),
    ),
    "MoE": (
        Subcategory(
            "Expert compute",
            "The routed experts' matrix multiplies and the activation between them, "
            "as fused MoE calls or grouped GEMMs.",
        ),
        Subcategory(
            "Routing and combine",
            "Choose each token's experts, group tokens by expert for the GEMMs, and "
            "sum the expert outputs back into one row per token.",
        ),
    ),
    "Normalization": (
        Subcategory(
            "RMSNorm",
            "Root-mean-square normalization of hidden rows, alone or fused with a "
            "residual add, a gate or a second input.",
        ),
        Subcategory(
            "Hyper-connections",
            "mHC: each token keeps several residual streams, mixed before and after "
            "every block; these calls fuse the mixing with RMSNorm.",
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


#: How ``Timer.rocprof`` measures on ROCm (``profiling/profilers/timer.py``).
ROCPROF_METHOD = (
    "GPU kernel time from rocprofv3 kernel-dispatch durations (end - start per "
    "dispatch, from the rocpd database), summed per logical launch and averaged "
    "over a fixed launch count. The ROCm counterpart of the CUPTI method."
)


@dataclass(frozen=True)
class ViewField:
    """One Rust config field a :class:`ConfigView` is drawn by.

    ``field`` is the name ``simulator kernel-list`` gives it among the kind's
    config fields; ``label`` is what the page calls it; ``doc`` says what its
    values mean.
    """

    field: str
    label: str
    doc: str


@dataclass(frozen=True)
class ConfigView:
    """A chart drawn from several registered configs of a kind at once.

    The configs a deployment builds for one leaf that differ only in
    ``series`` are one chart: one line per config, over the config's sweep
    axis. ``series`` values are positions in an order, 0 first; line ``n``
    reads ``label n+1`` and position 0, which leads the order, is drawn
    strongest. ``workload`` is picked with a selector; the public API names
    each config's value (``config_labels``). The page knows only these roles,
    never the kind.
    """

    title: str
    summary: str
    series: ViewField
    workload: ViewField


#: Expert compute split over expert-parallel ranks. The simulator folds each
#: step's routed demand onto the EP ranks and orders the ranks by active
#: experts, then routed rows (``fold_layerwise_expert_counts`` in
#: ``simulator/src/timing/routing.rs``); it builds one config per position in
#: that order (``folded_rank_position``), all from one ``expert_demand``.
EP_RANKS_BY_LOAD = ConfigView(
    title="EP ranks by load",
    summary=(
        "One line per expert-parallel rank of one routing. The simulator draws "
        "each step's routed tokens from the routing, folds them onto the ranks and "
        "orders the ranks by load, so a line is a position in that order, not a "
        "physical GPU."
    ),
    series=ViewField(
        field="folded_rank_position",
        label="Load rank",
        doc=(
            "Load rank 1 holds the most active experts, ties going to the most "
            "routed rows; the fold orders each layer's ranks this way and "
            "averages the layers position by position."
        ),
    ),
    workload=ViewField(
        field="expert_demand",
        label="Routing",
        doc=(
            "Where the routed demand comes from: a recorded token corpus, an "
            "expert-popularity file, or synthetic uniform or random routing."
        ),
    ),
)


@dataclass(frozen=True)
class KernelDoc:
    """What one kernel kind computes and how to read its numbers.

    ``method`` says how the kind's runners time it. A CUPTI-timed kind opens
    with :data:`CUPTI_METHOD` and adds only what is particular to it.
    ``reference`` is the module path of a PyTorch reference implementation, or
    None when a measured backend is itself the reference. ``view`` declares a
    chart over several of the kind's configs (:class:`ConfigView`), for a kind
    whose Rust config carries both of its fields.
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
    view: ConfigView | None = None


def kernel_doc(kernel_kind: str) -> KernelDoc | None:
    """The ``DOC`` of ``profiling/kernels/<kernel_kind>.py``, or None if it has none."""

    import importlib

    try:
        module = importlib.import_module(f"profiling.kernels.{kernel_kind}")
    except ModuleNotFoundError:
        return None
    return getattr(module, "DOC", None)
