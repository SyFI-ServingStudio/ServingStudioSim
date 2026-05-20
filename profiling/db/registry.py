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

from profiling.db.args import KernelArgs
from profiling.db.kind import KernelKind
from profiling.db.outlier import BatchOutlierPolicy
from profiling.runners.metrics import Metrics

ProfileFn = Callable[..., Metrics]


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
class KernelProfilerSpec:
    kernel_kind: KernelKind
    backend: str
    runner_ref: RunnerRef
    table_name: str
    args_schema: type[KernelArgs]
    metric_family: MetricFamily
    batch_outlier_policy: BatchOutlierPolicy
    subprocess_env: str | None = None
    subprocess_module: str | None = None
    gpu_count_fn: Callable[[dict[str, Any]], int] | None = None

    @property
    def runner_module(self) -> str:
        return self.subprocess_module or self.runner_ref.module_name

    def load_runner(self) -> ProfileFn:
        return self.runner_ref.load(self.runner_module)


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
                f"duplicate profiler spec for {profiler_spec.kernel_kind}:"
                f"{profiler_spec.backend}"
            )
        registered_keys.add(key)

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


def iter_kernel_profiler_specs(
    kernel_kind: KernelKind | None = None,
) -> Iterator[KernelProfilerSpec]:
    _ensure_loaded()
    for profiler_spec in _REGISTRY:
        if kernel_kind is None or profiler_spec.kernel_kind == kernel_kind:
            yield profiler_spec


def known_backends(kernel_kind: KernelKind) -> list[str]:
    return [profiler_spec.backend for profiler_spec in iter_kernel_profiler_specs(kernel_kind)]


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
