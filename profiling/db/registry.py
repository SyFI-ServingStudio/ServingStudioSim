"""L1b registry: the authority for ``(KernelKind, backend)`` profiler specs."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from profiling.db.args import KernelArgs, SingleGemmArgs
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
    args_schema: type[KernelArgs]
    metric_family: MetricFamily


REGISTRY: tuple[KernelProfilerSpec, ...] = (
    KernelProfilerSpec(
        kernel_kind=KernelKind.GEMM_SINGLE,
        backend="torch",
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.torch",
            function_name="profile_single_gemm",
        ),
        table_name="single_gemm",
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    ),
)


def _validate_registry(registry: tuple[KernelProfilerSpec, ...]) -> None:
    # Tables may be shared across backend variants, but their schema ownership
    # must remain single-family so Table never creates a mixed compute/comm DB.
    table_contracts: dict[str, _TableContract] = {}
    registered_keys: set[tuple[KernelKind, str]] = set()
    for profiler_spec in registry:
        key = (profiler_spec.kernel_kind, profiler_spec.backend)
        if key in registered_keys:
            raise ValueError(
                f"duplicate profiler spec for {profiler_spec.kernel_kind.value}:"
                f"{profiler_spec.backend}"
            )
        registered_keys.add(key)

        expected_contract = table_contracts.get(profiler_spec.table_name)
        actual_contract = _TableContract(
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


_validate_registry(REGISTRY)


def iter_kernel_profiler_specs(
    kernel_kind: KernelKind | None = None,
) -> Iterator[KernelProfilerSpec]:
    for profiler_spec in REGISTRY:
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
                f"unknown backend {backend!r} for {kernel_kind.value}; known backends: {backends}"
            )
        return backend
    if len(backends) == 1:
        return backends[0]
    raise ValueError(f"backend is required for {kernel_kind.value}; known backends: {backends}")


def find_kernel_profiler_spec(
    kernel_kind: KernelKind,
    backend: str,
) -> KernelProfilerSpec:
    for profiler_spec in iter_kernel_profiler_specs(kernel_kind):
        if profiler_spec.backend == backend:
            return profiler_spec
    raise KeyError(f"no profiler spec registered for {kernel_kind.value}:{backend}")


def load_runner(kernel_kind: KernelKind, backend: str) -> ProfileFn:
    return find_kernel_profiler_spec(kernel_kind, backend).load_runner()


def find_table(kernel_kind: KernelKind, backend: str) -> str:
    return find_kernel_profiler_spec(kernel_kind, backend).table_name


def find_args_schema(kernel_kind: KernelKind, backend: str) -> type[KernelArgs]:
    return find_kernel_profiler_spec(kernel_kind, backend).args_schema
