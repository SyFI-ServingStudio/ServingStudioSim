"""L1b registry: the authority for ``(KernelKind, backend)`` profiler specs.

Per-kernel modules under ``profiling/kernels/`` call ``register(...)`` at
import time to add their ``KernelProfilerSpec`` rows. Lookup functions
(``iter_kernel_profiler_specs``, ``find_kernel_profiler_spec``, ...) trigger
``_ensure_loaded()`` on first access, which imports ``profiling.kernels`` to
run those side effects exactly once, then runs ``_validate_registry`` once on
the accumulated specs.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from profiling.db.args import DType, KernelArgs
from profiling.db.kind import KernelKind
from profiling.db.outlier import BatchOutlierPolicy
from profiling.runners.metrics import Metrics, RunnerResult

ProfileFn = Callable[..., Metrics]
# A list runner takes the whole (homogeneous) chunk's coerced kwargs and returns
# one RunnerResult per spec, in order. This is the contract the worker calls.
ListRunnerFn = Callable[[list[dict]], list[RunnerResult]]


class MetricFamily(StrEnum):
    """Metric-column family owned by one profiling table."""

    COMPUTE = "compute"
    COMM = "comm"


@dataclass(frozen=True)
class RunnerRef:
    """Lazy runner reference.

    Registry rows must not import env-specific runner modules in the main
    process. The execution backend loads this reference only after selecting
    ``subprocess_env`` and spawning the worker interpreter.
    """

    module_name: str
    function_name: str

    def load(self, module_name: str | None = None) -> ProfileFn:
        loaded_module = importlib.import_module(module_name or self.module_name)
        runner = getattr(loaded_module, self.function_name)
        if not callable(runner):
            raise TypeError(f"{loaded_module.__name__}.{self.function_name} is not callable")
        return runner


@dataclass(frozen=True)
class BackendSupport:
    """Which dtypes / GPU a ``(kernel_kind, backend)`` can actually run.

    The single source of truth for backend-selection validation and the dry-run
    ``options`` column — NOT ``profile.db`` row presence. A cold cache has no
    measured rows yet, but the combo is still *supported* (it can be JIT-filled);
    conversely a combo the kernel does not support (e.g. ``fa2`` with an fp8
    *query*) must be rejected before it ever reaches profiling.

    Two dtype axes, because for attention they are genuinely independent:

    - ``compute`` — the compute/activation precision (attention's ``q_dtype``, or
      the single ``dtype`` of a GEMM/norm). ``None`` = compute-dtype-agnostic
      (the comm kernels are size-keyed; elementwise is byte-keyed).
    - ``kv`` — the KV-cache precision (attention only; ``None`` = not applicable).
      fp8 KV with a bf16 *query* is broadly supported — e.g. ``fa2`` runs bf16-q /
      fp8-kv, which is a current production config. Only fp8 *compute* needs
      ``fa3``. Keeping ``compute`` and ``kv`` separate is what lets ``fa2`` stay
      valid for fp8-KV while ``cudnn`` (bf16-kv only) is correctly excluded.
    - ``gpus`` — ``None`` = any GPU; a set restricts to those ``gpu_name`` values.
    - ``compute_gpu_pairs`` — optional non-Cartesian refinement for backends
      whose verified dtype set differs by GPU. When both dtype and GPU are
      known, the pair must be present in this set in addition to passing the
      independent axes.

    The profile.db cache still keys on the full dtype tuple; this type only gates
    *which backends are legal*. Fine, per-request shape constraints (e.g. cudnn
    causal + prefix_len>0) are NOT modeled here — they stay a JIT/profile-time
    concern.
    """

    compute: frozenset[DType] | None
    kv: frozenset[DType] | None = None
    gpus: frozenset[str] | None = None
    compute_gpu_pairs: frozenset[tuple[DType, str]] | None = None

    def allows(
        self,
        compute_dtype: DType,
        kv_dtype: DType | None = None,
        gpu: str | None = None,
    ) -> bool:
        if self.compute is not None and compute_dtype not in self.compute:
            return False
        if self.kv is not None and kv_dtype is not None and kv_dtype not in self.kv:
            return False
        if self.gpus is not None and gpu is not None and gpu not in self.gpus:
            return False
        if (
            self.compute_gpu_pairs is not None
            and gpu is not None
            and (compute_dtype, gpu) not in self.compute_gpu_pairs
        ):
            return False
        return True


# Permissive default for specs that do not declare a capability (test fixtures);
# every production kernel below sets `supports=` explicitly.
_ANY_SUPPORT = BackendSupport(compute=None)


@dataclass(frozen=True)
class KernelProfilerSpec:
    kernel_kind: KernelKind
    backend: str
    runner_ref: RunnerRef
    table_name: str
    args_schema: type[KernelArgs]
    metric_family: MetricFamily
    batch_outlier_policy: BatchOutlierPolicy
    supports: BackendSupport = _ANY_SUPPORT
    subprocess_env: str | None = None
    subprocess_module: str | None = None
    gpu_count_fn: Callable[[dict[str, Any]], int] | None = None
    # When True the referenced runner already implements the list contract
    # (``list[dict] -> list[RunnerResult]``) natively — used by the multi-GPU
    # comm runners so they spawn their rank group once per chunk instead of per
    # spec. When False (the default) the single-spec runner is wrapped by
    # ``batched`` to satisfy the same contract.
    list_native: bool = False
    # Environment variables the execution backend sets on this row's worker
    # process, on top of the inherited environment. For measurement policy that
    # must hold before the framework is imported and that a runner therefore
    # cannot set itself -- e.g. ``TRITON_CACHE_AUTOTUNING=0``, so a Triton
    # ``@autotune`` selection never leaks between worker processes through
    # ``TRITON_CACHE_DIR``. Runners must not mutate ``os.environ`` for this.
    worker_env: tuple[tuple[str, str], ...] = ()
    # Optional lazy ``(**schema_kwargs) -> str | None`` called in the worker
    # right after the chunk runs, once per successful spec. A returned note is
    # appended to the row's ``backend_version`` so worker-process state that
    # changes the number (e.g. the pinned autotune configs) is recorded with it.
    row_provenance_ref: RunnerRef | None = None

    @property
    def runner_module(self) -> str:
        return self.subprocess_module or self.runner_ref.module_name

    def load_runner(self) -> ProfileFn:
        return self.runner_ref.load(self.runner_module)

    def load_list_runner(self) -> ListRunnerFn:
        """Load the runner as a list runner: native if ``list_native``, else the
        single-spec runner wrapped by ``batched``. This is what the worker calls
        so its dispatch is uniform (one call, no per-kernel-kind branch)."""
        from profiling.runners.batched import batched  # lazy: worker-subprocess only

        runner = self.runner_ref.load(self.runner_module)
        return runner if self.list_native else batched(runner)

    def load_row_provenance(self) -> Callable[..., str | None] | None:
        return None if self.row_provenance_ref is None else self.row_provenance_ref.load()


@dataclass(frozen=True)
class _TableContract:
    # kernel_kind is part of the contract because `profiling.facade` derives the
    # public `get_<stem>_times` / `count_missing_<stem>` names from
    # `table_name` and refuses to map one stem onto two kinds. Keeping the kind
    # here lets the validator catch that conflict at registration time instead
    # of at facade build time.
    kernel_kind: KernelKind
    args_schema: type[KernelArgs]
    metric_family: MetricFamily


_REGISTRY: list[KernelProfilerSpec] = []
_loaded: bool = False


def register(spec: KernelProfilerSpec) -> None:
    """Append a profiler spec. Called by ``profiling/kernels/<kind>.py`` modules
    at import time. During the initial barrel import, validation is deferred to
    ``_ensure_loaded`` so per-kernel modules don't need to know about peers;
    after the registry is loaded, late additions re-run validation against the
    candidate registry first and only commit on success, so a conflicting row
    cannot leave the registry in a poisoned state.
    """
    if _loaded:
        _validate_registry(_REGISTRY + [spec])
    _REGISTRY.append(spec)


def _ensure_loaded() -> None:
    """Trigger per-kernel module imports on first registry access. Imports
    ``profiling.kernels`` exactly once; that package's ``__init__`` chains the
    per-kernel modules whose ``register(...)`` calls populate ``_REGISTRY``.

    ``_loaded`` flips only after both the import and the validation succeed,
    so a failed first attempt does not leave the registry stuck in a
    partially-loaded, never-revalidated state.
    """
    global _loaded
    if _loaded:
        return
    importlib.import_module("profiling.kernels")
    _validate_registry(_REGISTRY)
    _loaded = True


def _validate_registry(registry: list[KernelProfilerSpec]) -> None:
    # Tables may be shared across backend variants of the same kernel kind,
    # but their schema/family/kind ownership must remain single-valued so
    # Table never creates a mixed compute/comm DB and `facade.py` can keep
    # using `table_name` as the public-function stem.
    table_contracts: dict[str, _TableContract] = {}
    registered_keys: set[tuple[KernelKind, str]] = set()
    for profiler_spec in registry:
        # The Rust bridge derives its perf_api call name as `get_{kernel_kind}_times`
        # (format! over `KernelSpec::KIND`), while `facade.py` derives the Python
        # function name from `table_name`. The cross-language call only resolves
        # when the two strings are identical, so a per-kernel module must keep them
        # equal. This also subsumes the "one table_name maps to two kinds" hazard:
        # distinct kinds now necessarily own distinct table_names.
        if profiler_spec.table_name != profiler_spec.kernel_kind:
            raise ValueError(
                f"table_name {profiler_spec.table_name!r} must equal kernel_kind "
                f"{profiler_spec.kernel_kind!r}: the Rust bridge calls "
                f"get_{{kernel_kind}}_times while Python exposes get_{{table_name}}_times, "
                f"so the cross-language facade name resolves only when they match"
            )

        key = (profiler_spec.kernel_kind, profiler_spec.backend)
        if key in registered_keys:
            raise ValueError(
                f"duplicate profiler spec for {profiler_spec.kernel_kind}:{profiler_spec.backend}"
            )
        registered_keys.add(key)

        # An empty dtype set means "supports nothing", which is never intended —
        # dtype-agnostic axes use `None`.
        supports = profiler_spec.supports
        for axis, allowed in (("compute", supports.compute), ("kv", supports.kv)):
            if allowed is not None and not allowed:
                raise ValueError(
                    f"{profiler_spec.kernel_kind}:{profiler_spec.backend} declares an "
                    f"empty {axis} dtype set; use None for a {axis}-agnostic axis"
                )
        if supports.compute_gpu_pairs is not None:
            if not supports.compute_gpu_pairs:
                raise ValueError(
                    f"{profiler_spec.kernel_kind}:{profiler_spec.backend} declares an "
                    "empty compute_gpu_pairs set; use None when no pair refinement is needed"
                )
            if supports.compute is not None and any(
                dtype not in supports.compute for dtype, _gpu in supports.compute_gpu_pairs
            ):
                raise ValueError(
                    f"{profiler_spec.kernel_kind}:{profiler_spec.backend} declares a "
                    "compute_gpu_pairs dtype outside its compute axis"
                )
            if supports.gpus is not None and any(
                gpu not in supports.gpus for _dtype, gpu in supports.compute_gpu_pairs
            ):
                raise ValueError(
                    f"{profiler_spec.kernel_kind}:{profiler_spec.backend} declares a "
                    "compute_gpu_pairs GPU outside its gpus axis"
                )

        _validate_worker_env(profiler_spec)

        expected_contract = table_contracts.get(profiler_spec.table_name)
        actual_contract = _TableContract(
            kernel_kind=profiler_spec.kernel_kind,
            args_schema=profiler_spec.args_schema,
            metric_family=profiler_spec.metric_family,
        )
        if expected_contract is None:
            table_contracts[profiler_spec.table_name] = actual_contract
            continue
        if expected_contract != actual_contract:
            raise ValueError(
                f"conflicting table contract for {profiler_spec.table_name}: "
                f"{expected_contract} vs {actual_contract}"
            )


# The execution backend composes these itself (GPU selection, the env's import
# and library paths); a row overriding them would silently undo that.
_BACKEND_OWNED_WORKER_ENV = frozenset({"CUDA_VISIBLE_DEVICES", "PYTHONPATH", "LD_LIBRARY_PATH"})


def _validate_worker_env(profiler_spec: KernelProfilerSpec) -> None:
    label = f"{profiler_spec.kernel_kind}:{profiler_spec.backend}"
    names = [name for name, _value in profiler_spec.worker_env]
    if len(set(names)) != len(names):
        raise ValueError(f"{label} sets a worker_env variable twice: {names}")
    for name, value in profiler_spec.worker_env:
        if not name or not isinstance(value, str):
            raise ValueError(f"{label} worker_env entries must be (name, str value) pairs")
        if name in _BACKEND_OWNED_WORKER_ENV:
            raise ValueError(f"{label} worker_env must not set {name}; the backend owns it")


def iter_kernel_profiler_specs(
    kernel_kind: KernelKind | None = None,
) -> Iterator[KernelProfilerSpec]:
    _ensure_loaded()
    for profiler_spec in _REGISTRY:
        if kernel_kind is None or profiler_spec.kernel_kind == kernel_kind:
            yield profiler_spec


def known_backends(kernel_kind: KernelKind) -> list[str]:
    return [profiler_spec.backend for profiler_spec in iter_kernel_profiler_specs(kernel_kind)]


def backend_supports(
    kernel_kind: KernelKind,
    backend: str,
    compute_dtype: DType,
    kv_dtype: DType | None = None,
    gpu: str | None = None,
) -> bool:
    """Whether ``backend`` can run ``kernel_kind`` at these dtypes on ``gpu`` — the
    capability gate for the backend-selection validator (rejects e.g. torch@fp8,
    or fa2 with an fp8 *query*, while allowing fa2 with fp8 KV + bf16 query)."""
    return find_kernel_profiler_spec(kernel_kind, backend).supports.allows(
        compute_dtype, kv_dtype, gpu
    )


def supported_backends(
    kernel_kind: KernelKind,
    compute_dtype: DType,
    kv_dtype: DType | None = None,
    gpu: str | None = None,
) -> list[str]:
    """Registered backends of ``kernel_kind`` that support these dtypes on ``gpu``
    — the dry-run ``options`` column (already filtered to the kernel's dtypes)."""
    return [
        profiler_spec.backend
        for profiler_spec in iter_kernel_profiler_specs(kernel_kind)
        if profiler_spec.supports.allows(compute_dtype, kv_dtype, gpu)
    ]


def resolve_spec_backend(kernel_kind: KernelKind, spec: dict[str, Any]) -> str:
    backends = known_backends(kernel_kind)
    if "backend" in spec:
        backend = str(spec["backend"])
        if backend not in backends:
            raise ValueError(
                f"unknown backend {backend!r} for {kernel_kind}; known backends: {backends}"
            )
        return backend
    if len(backends) == 1:
        return backends[0]
    raise ValueError(f"backend is required for {kernel_kind}; known backends: {backends}")


def find_kernel_profiler_spec(
    kernel_kind: KernelKind,
    backend: str,
) -> KernelProfilerSpec:
    for profiler_spec in iter_kernel_profiler_specs(kernel_kind):
        if profiler_spec.backend == backend:
            return profiler_spec
    raise KeyError(f"no profiler spec registered for {kernel_kind}:{backend}")


def load_runner(kernel_kind: KernelKind, backend: str) -> ProfileFn:
    return find_kernel_profiler_spec(kernel_kind, backend).load_runner()


def find_table(kernel_kind: KernelKind, backend: str) -> str:
    return find_kernel_profiler_spec(kernel_kind, backend).table_name


def find_args_schema(kernel_kind: KernelKind, backend: str) -> type[KernelArgs]:
    return find_kernel_profiler_spec(kernel_kind, backend).args_schema
