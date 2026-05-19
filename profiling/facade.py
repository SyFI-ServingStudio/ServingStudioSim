"""Registry-driven builders for public perf_api facade functions.

Future kernel additions should not add hand-written wrappers to
``profiling.perf_api``. Add a ``KernelProfilerSpec`` row with a stable table stem;
this module generates the documented public names from that registry row.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

from profiling.db.args import KernelArgs
from profiling.db.batch import args_to_spec, coerce_args, run_profile_batch
from profiling.db.kind import KernelKind
from profiling.db.registry import find_kernel_profiler_spec, iter_kernel_profiler_specs
from profiling.db.table import MissingEntry, Table
from profiling.runners.metrics import Metrics

# Public callable shapes exposed from profiling.perf_api.
#
# The concrete symbols are generated dynamically, but these protocols give
# tests, IDEs, and future typed aliases a single place to describe the contract.
SpecT = TypeVar("SpecT", bound=KernelArgs)


class GetTimesFn(Protocol[SpecT]):
    def __call__(
        self,
        specs: Sequence[Mapping[str, Any] | SpecT],
        *,
        backend: str,
        gpu_name: str | None = None,
        force: bool = False,
    ) -> list[Metrics | MissingEntry]: ...


class CountMissingFn(Protocol[SpecT]):
    def __call__(
        self,
        specs: Sequence[Mapping[str, Any] | SpecT],
        *,
        backend: str,
        gpu_name: str | None = None,
    ) -> int: ...


DbPathFn = Callable[[], Path]
JitEnabledFn = Callable[[], bool]


# Public installer. perf_api.py calls this once and publishes the returned
# functions on its own module globals; callers still enter through perf_api.py.
def build_kind_facades(
    *,
    db_path: DbPathFn,
    jit_enabled: JitEnabledFn,
) -> dict[str, Callable[..., object]]:
    """Build public get/count facade functions from ``REGISTRY``.

    The generated names stay on ``profiling.perf_api``; this module only owns
    the mechanical factory so adding a new ``KernelProfilerSpec`` row does not also
    require repetitive public wrapper bodies.
    """

    generated: dict[str, Callable[..., object]] = {}
    installed: dict[str, KernelKind] = {}
    for profiler_spec in iter_kernel_profiler_specs():
        # table_name is the function stem by design:
        #   "single_gemm" -> get_single_gemm_times / count_missing_single_gemm
        # Keep this stable because Rust bridge calls these names through PyO3.
        stem = profiler_spec.table_name
        if not stem.isidentifier():
            raise ValueError(f"table_name {stem!r} cannot be used as a perf_api function stem")
        if stem in installed:
            if installed[stem] != profiler_spec.kernel_kind:
                raise ValueError(f"perf_api function stem {stem!r} maps to multiple KernelKinds")
            continue

        get_name = f"get_{stem}_times"
        count_name = f"count_missing_{stem}"
        generated[get_name] = _make_get_times(
            profiler_spec.kernel_kind,
            get_name,
            db_path,
            jit_enabled,
        )
        generated[count_name] = _make_count_missing(
            profiler_spec.kernel_kind,
            count_name,
            db_path,
        )
        installed[stem] = profiler_spec.kernel_kind
    return generated


# Function factories. They close over KernelKind only; backend remains an
# explicit public argument so a single KernelKind can support multiple backends.
def _make_get_times(
    kernel_kind: KernelKind,
    public_name: str,
    db_path: DbPathFn,
    jit_enabled: JitEnabledFn,
) -> GetTimesFn[KernelArgs]:
    def wrapper(
        specs: Sequence[Mapping[str, Any] | KernelArgs],
        *,
        backend: str,
        gpu_name: str | None = None,
        force: bool = False,
    ) -> list[Metrics | MissingEntry]:
        args_list = _coerce_input_specs(kernel_kind, backend, specs)
        return _get_times(
            kernel_kind,
            args_list,
            backend=backend,
            gpu_name=gpu_name,
            db_path=db_path(),
            jit_enabled=jit_enabled(),
            force=force,
        )

    wrapper.__name__ = public_name
    wrapper.__qualname__ = public_name
    wrapper.__doc__ = f"Batch query facade for {kernel_kind}."
    return wrapper


def _make_count_missing(
    kernel_kind: KernelKind,
    public_name: str,
    db_path: DbPathFn,
) -> CountMissingFn[KernelArgs]:
    def wrapper(
        specs: Sequence[Mapping[str, Any] | KernelArgs],
        *,
        backend: str,
        gpu_name: str | None = None,
    ) -> int:
        args_list = _coerce_input_specs(kernel_kind, backend, specs)
        return _count_missing(
            kernel_kind,
            args_list,
            backend=backend,
            gpu_name=gpu_name,
            db_path=db_path(),
        )

    wrapper.__name__ = public_name
    wrapper.__qualname__ = public_name
    wrapper.__doc__ = f"Dry-run missing-count facade for {kernel_kind}."
    return wrapper


# Query implementations. These are the only functions here that touch the DB
# table layer or schedule JIT profiling. Runners remain behind run_profile_batch.
def _get_times(
    kernel_kind: KernelKind,
    args_list: list[KernelArgs],
    *,
    backend: str,
    gpu_name: str | None,
    db_path: Path,
    jit_enabled: bool,
    force: bool,
) -> list[Metrics | MissingEntry]:
    resolved_gpu = _resolve_gpu_name(gpu_name)
    profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)
    table = Table(profiler_spec, db_path)
    if force:
        # Force refresh is still a public facade policy, not runner behavior:
        # skip the initial DB query, re-profile every requested spec, then
        # return the freshly persisted rows through the same table path.
        profile_specs = [dict(args_to_spec(args), backend=backend) for args in args_list]
        if profile_specs:
            run_profile_batch(kernel_kind, profile_specs, db_path=db_path)
        return table.query(args_list, backend=backend, gpu_name=resolved_gpu)

    results = table.query(args_list, backend=backend, gpu_name=resolved_gpu)

    missing = [result.args for result in results if isinstance(result, MissingEntry)]
    if missing and jit_enabled:
        jit_input_specs = [dict(args_to_spec(args), backend=backend) for args in missing]
        run_profile_batch(kernel_kind, jit_input_specs, db_path=db_path)
        results = table.query(args_list, backend=backend, gpu_name=resolved_gpu)
    return results


# Dry-run implementation. This must stay read-only: no JIT, no runner, no pool.
def _count_missing(
    kernel_kind: KernelKind,
    args_list: list[KernelArgs],
    *,
    backend: str,
    gpu_name: str | None,
    db_path: Path,
) -> int:
    resolved_gpu = _resolve_gpu_name(gpu_name)
    profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)
    table = Table(profiler_spec, db_path)
    return table.exists(args_list, backend=backend, gpu_name=resolved_gpu).count(False)


# Input normalization. Public APIs accept dicts for ergonomics, but the table
# layer and registry reason in typed KernelArgs dataclasses.
def _coerce_input_specs(
    kernel_kind: KernelKind,
    backend: str,
    input_specs: Sequence[Mapping[str, Any] | KernelArgs],
) -> list[KernelArgs]:
    schema = find_kernel_profiler_spec(kernel_kind, backend).args_schema
    args_list = []
    for input_spec in input_specs:
        if isinstance(input_spec, schema):
            args_list.append(input_spec)
        elif isinstance(input_spec, Mapping):
            spec_backend = input_spec.get("backend")
            if spec_backend is not None and str(spec_backend) != backend:
                raise ValueError(
                    f"spec backend {spec_backend!r} does not match requested backend {backend!r}"
                )
            args_list.append(
                coerce_args(schema, {k: v for k, v in input_spec.items() if k != "backend"})
            )
        else:
            raise TypeError(
                f"expected dict or {schema.__name__}, got {type(input_spec).__name__}"
            )
    return args_list


# gpu_name is a DB key. CUDA auto-detect is only a convenience for local use;
# bridge/build paths should pass it explicitly for non-current or remote GPUs.
def _resolve_gpu_name(gpu_name: str | None) -> str:
    if gpu_name:
        return gpu_name
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except ImportError:
        pass
    raise ValueError("gpu_name is required when CUDA is unavailable")
