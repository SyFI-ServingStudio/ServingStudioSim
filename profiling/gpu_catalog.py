"""GPU spec catalog resolution — the profiling side of ``gpu/spec.json``.

Mirror of the analyzer's ``hardware`` resolution (``analyzer/rust/src/ui_service/
hardware.rs``): a name maps to a canonical SKU only via the catalog's exact
case-insensitive ``name`` / ``aliases`` list. No fuzzy matching and no web
fallback, so a unknown GPU is cleanly ``unmatched`` and never silently mistaken for
a default SKU.

The profiling artifact path uses this to (a) verify that a measured job's
requested DB cache key and the worker-observed physical GPU resolve to the same
canonical SKU — the job fails otherwise — and (b) stamp the resolved canonical
name into resource metadata. Backend-capability validation
(``profiling.db.registry.BackendSupport``) also reads a GPU's CUDA compute
capability from it. Of the catalog, only ``compute_capability`` reaches the
timing path: the simulator reads it too (``simulator/src/common/gpu.rs``), for
capability-keyed rules such as vLLM's FlashInfer all-reduce workspace budget.
Kernel times themselves come from ``profile.db`` rows keyed by the requested
``gpu_name`` string; the simulator never reads peak TFLOPs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

# `profiling/` sits one level below the git root, which owns `gpu/spec.json`.
_REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SPEC_FIELDS = (
    "name",
    "mem_bandwidth_gbps",
    "fp16_tflops",
    "bf16_tflops",
    "fp8_tflops",
    "fp32_tflops",
    "int8_tops",
    "interconnect",
    "interconnect_bandwidth_gbps",
)


def repo_root() -> Path:
    """The git root that owns ``gpu/spec.json``. Exposed so tests can pin a
    fixture catalog without changing process cwd."""
    return _REPO_ROOT


@dataclass(frozen=True)
class GpuSpecResolution:
    """One matched catalog entry. Numeric fields are ``None`` when the catalog
    declares ``null`` for that GPU (never fabricated zeros)."""

    canonical_name: str
    matched_alias: str
    provenance: str = "catalog"
    mem_bandwidth_gbps: float | None = None
    fp16_tflops: float | None = None
    bf16_tflops: float | None = None
    fp8_tflops: float | None = None
    # Optional in the catalog: only FP4-capable GPUs (Blackwell) declare it.
    fp4_tflops: float | None = None
    fp32_tflops: float | None = None
    int8_tops: float | None = None
    interconnect: str | None = None
    interconnect_bandwidth_gbps: float | None = None
    # CUDA ``(major, minor)``; ``None`` for a non-NVIDIA part.
    compute_capability: tuple[int, int] | None = None
    # Hardware vendor (e.g. ``"NVIDIA"``, ``"AMD"``) and micro-architecture
    # (e.g. ``"CDNA3"``), verbatim from the catalog. The AMD device gate reads
    # ``architecture`` the way the CUDA gate reads ``compute_capability``.
    vendor: str | None = None
    architecture: str | None = None

    @property
    def interconnect_one_way_gbps(self) -> float | None:
        if self.interconnect_bandwidth_gbps is None:
            return None
        # Catalog bandwidth is BIDIRECTIONAL (bytes/s); a transfer is one-way ≈ half.
        return self.interconnect_bandwidth_gbps / 2.0

    def dense_peak_tflops(self, dtype: str) -> float | None:
        """Dense tensor-core peak for a dtype string, or ``None`` when the dtype
        is unrecognized or the GPU lacks that dtype — callers must not draw a fake
        line for ``None``."""
        normalized = dtype.strip().lower()
        # Before the fp8 tokens: NVFP4 carries fp8 group scales, and its compute
        # dtype must never read as fp8.
        if any(token in normalized for token in ("nvfp4", "e2m1", "fp4")):
            return self.fp4_tflops
        if any(token in normalized for token in ("fp8", "e4m3", "e5m2")):
            return self.fp8_tflops
        if normalized in ("int8",):
            return self.int8_tops
        if any(token in normalized for token in ("fp16", "half", "float16")):
            return self.fp16_tflops
        if any(token in normalized for token in ("fp32", "float32", "tf32")):
            return self.fp32_tflops
        if any(token in normalized for token in ("bf16", "bfloat16")):
            return self.bf16_tflops
        return None


@lru_cache(maxsize=1)
def load_gpu_catalog(root: Path | None = None) -> list[dict]:
    """Parse ``gpu/spec.json`` into the raw ``gpus[]`` list. Returns ``[]`` when
    the file is missing or unparseable (callers decide the failure policy)."""
    spec_path = (root or repo_root()) / "gpu" / "spec.json"
    try:
        document = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    gpus = document.get("gpus")
    return gpus if isinstance(gpus, list) else []


def resolve_gpu_spec(name: str, root: Path | None = None) -> GpuSpecResolution | None:
    """Resolve a canonical catalog SKU from an exact case-insensitive ``name`` or
    alias match. ``None`` = unmatched/unavailable — never a default GPU."""
    if not name or not name.strip():
        return None
    target = name.strip().lower()
    for gpu in load_gpu_catalog(root):
        canonical = gpu.get("name")
        if not isinstance(canonical, str):
            continue
        candidates = [canonical, *(gpu.get("aliases") or [])]
        matched_alias = next(
            (
                str(candidate).strip()
                for candidate in candidates
                if isinstance(candidate, str) and candidate.strip().lower() == target
            ),
            None,
        )
        if matched_alias is None:
            continue
        return GpuSpecResolution(
            canonical_name=canonical,
            matched_alias=matched_alias,
            mem_bandwidth_gbps=_number(gpu, "mem_bandwidth_gbps"),
            fp16_tflops=_number(gpu, "fp16_tflops"),
            bf16_tflops=_number(gpu, "bf16_tflops"),
            fp8_tflops=_number(gpu, "fp8_tflops"),
            fp4_tflops=_number(gpu, "fp4_tflops"),
            fp32_tflops=_number(gpu, "fp32_tflops"),
            int8_tops=_number(gpu, "int8_tops"),
            interconnect=_string_or_none(gpu, "interconnect"),
            interconnect_bandwidth_gbps=_number(gpu, "interconnect_bandwidth_gbps"),
            compute_capability=_compute_capability(gpu),
            vendor=_string_or_none(gpu, "vendor"),
            architecture=_string_or_none(gpu, "architecture"),
        )
    return None


@cache
def gpu_compute_capability(name: str) -> tuple[int, int] | None:
    """CUDA compute capability of a GPU named by catalog name or alias (e.g.
    ``"NVIDIA H200"`` -> ``(9, 0)``). ``None`` when the name is not in the
    catalog or the part has no CUDA compute capability."""
    spec = resolve_gpu_spec(name)
    return spec.compute_capability if spec is not None else None


@cache
def gpu_vendor(name: str) -> str | None:
    """Hardware vendor of a catalog GPU (e.g. ``"NVIDIA"``, ``"AMD"``). Falls
    back to ``"NVIDIA"`` for a matched part that declares a CUDA compute
    capability but no explicit ``vendor``. ``None`` when the name is not in the
    catalog or its vendor cannot be determined."""
    spec = resolve_gpu_spec(name)
    if spec is None:
        return None
    if spec.vendor is not None:
        return spec.vendor
    return "NVIDIA" if spec.compute_capability is not None else None


@cache
def gpu_architecture(name: str) -> str | None:
    """Micro-architecture of a catalog GPU (e.g. ``"CDNA3"``). ``None`` when the
    name is not in the catalog or the entry declares no ``architecture``."""
    spec = resolve_gpu_spec(name)
    return spec.architecture if spec is not None else None


def same_canonical_sku(left: str, right: str, root: Path | None = None) -> bool:
    """True when both names resolve to the same canonical SKU. False when either
    side is unmatched (the job must fail, never guess a SKU)."""
    left_spec = resolve_gpu_spec(left, root)
    right_spec = resolve_gpu_spec(right, root)
    return (
        left_spec is not None
        and right_spec is not None
        and left_spec.canonical_name == right_spec.canonical_name
    )


def _number(gpu: dict, key: str) -> float | None:
    value = gpu.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _compute_capability(gpu: dict) -> tuple[int, int] | None:
    value = gpu.get("compute_capability")
    if not isinstance(value, str):
        return None
    major, _, minor = value.partition(".")
    if not (major.isdigit() and minor.isdigit()):
        raise ValueError(
            f"{gpu.get('name')}: compute_capability must be 'major.minor', got {value!r}"
        )
    return int(major), int(minor)


def _string_or_none(gpu: dict, key: str) -> str | None:
    value = gpu.get(key)
    return str(value) if isinstance(value, str) and value else None
