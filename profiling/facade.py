"""Registry-driven builders for public perf_api facade functions.

Future kernel additions should not add hand-written wrappers to
``profiling.perf_api``. Add a ``KernelProfilerSpec`` row with a stable table stem;
this module generates the documented public names from that registry row.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from profiling.db.args import KernelArgs
from profiling.db.batch import (
    ProfileBatchOutcome,
    ProfileProvenance,
    args_to_spec,
    coerce_args,
    execute_profile_batch,
)
from profiling.db.kind import KernelKind
from profiling.db.registry import find_kernel_profiler_spec, iter_kernel_profiler_specs
from profiling.db.table import MissingEntry, Table
from profiling.plan import active_collector
from profiling.profilers.energy import require_measured_energy
from profiling.runners.metrics import ComputeMetrics, Metrics


class UnmeasuredEnergyError(RuntimeError):
    """A run asked for energy and got rows that were never measured for it."""


@dataclass(frozen=True)
class KindTimesResult:
    """Typed per-invocation result of one kind's query/profile path.

    Both the public ``get_<kind>_times`` facade (which returns only ``results`` to
    keep its documented signature stable) and the CLI's clearly-named internal entry
    read this. GPU ``provenance`` is measured for this exact call — never an ambient
    process-global or thread-local side channel.
    """

    results: list[Metrics | MissingEntry]
    provenance: ProfileProvenance


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


def run_kind_times(
    kernel_kind: KernelKind,
    specs: Sequence[Mapping[str, Any] | KernelArgs],
    *,
    backend: str,
    gpu_name: str | None = None,
    db_path: Path,
    jit_enabled: bool,
    force: bool = False,
    persist: bool = True,
) -> KindTimesResult:
    """Internal typed entry the profiling CLI calls for one kind's run.

    Returns results AND this invocation's typed GPU provenance so the artifact writer
    never reads an ambient channel. Shares the exact same core ``_get_times`` as the
    generated public facade, so public and CLI behavior cannot drift.

    ``persist=False`` is the CLI's ``--fresh``: measure every spec, hand the rows
    back, and leave ``profile.db`` untouched. It is only defined together with
    ``force``.
    """
    args_list = _coerce_input_specs(kernel_kind, backend, specs)
    return _get_times(
        kernel_kind,
        args_list,
        backend=backend,
        gpu_name=gpu_name,
        db_path=db_path,
        jit_enabled=jit_enabled,
        force=force,
        persist=persist,
    )


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
        ).results

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


# Query implementation. This is the only function here that touches the DB table
# layer or schedules JIT profiling; runners remain behind execute_profile_batch.
# Results and the execution-layer GPU provenance are returned together as a typed
# ``KindTimesResult`` for this exact invocation.
def _get_times(
    kernel_kind: KernelKind,
    args_list: list[KernelArgs],
    *,
    backend: str,
    gpu_name: str | None,
    db_path: Path,
    jit_enabled: bool,
    force: bool,
    persist: bool = True,
) -> KindTimesResult:
    if not persist and not force:
        # The cache-read half and the measured half would have to be stitched back
        # together by hand, and nothing asks for that. Keep the unpersisted mode to
        # the one shape the CLI exposes rather than inventing a second policy here.
        raise ValueError("persist=False is only defined together with force=True")
    resolved_gpu = _resolve_gpu_name(gpu_name)
    profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)
    table = Table(profiler_spec, db_path)
    outcome: ProfileBatchOutcome | None = None
    if force:
        # Force refresh is still a public facade policy, not runner behavior:
        # skip the initial DB query, re-profile every requested spec, then
        # return the freshly persisted rows through the same table path.
        profile_specs = [dict(args_to_spec(args), backend=backend) for args in args_list]
        if profile_specs:
            outcome = execute_profile_batch(
                kernel_kind,
                profile_specs,
                db_path=db_path if persist else None,
                gpu_name=resolved_gpu,
            )
        results = (
            table.query(args_list, backend=backend, gpu_name=resolved_gpu)
            if persist
            else _unpersisted_results(outcome, args_list, kernel_kind, backend, resolved_gpu)
        )
    else:
        results = table.query(args_list, backend=backend, gpu_name=resolved_gpu)

        missing = [result.args for result in results if isinstance(result, MissingEntry)]
        if missing and jit_enabled:
            jit_input_specs = [dict(args_to_spec(args), backend=backend) for args in missing]
            outcome = execute_profile_batch(
                kernel_kind,
                jit_input_specs,
                db_path=db_path,
                gpu_name=resolved_gpu,
            )
            results = table.query(args_list, backend=backend, gpu_name=resolved_gpu)
    _reject_unmeasured_energy(results, kernel_kind, backend, resolved_gpu)
    provenance = (
        outcome.provenance
        if outcome is not None and outcome.provenance is not None
        else ProfileProvenance(source="cache_key", requested_gpu_name=resolved_gpu)
    )
    return KindTimesResult(results=results, provenance=provenance)


def _reject_unmeasured_energy(
    results: Sequence[Metrics | MissingEntry],
    kernel_kind: KernelKind,
    backend: str,
    gpu_name: str,
) -> None:
    """Fail the run when energy was asked for and a row does not carry it.

    ``energy_j = 0.0`` means "not measured" on every path that can produce it --
    the NVML window skipped, pynvml missing, the handle unresolvable, or a
    column default -- so nothing downstream can tell it from a real zero. The
    simulator writes the value straight into its output parquet, where a cache
    filled without the energy window would read as a run that drew no power.
    Raise here, at the one core both the public facade and the CLI share, rather
    than let it become a plausible number in an artifact.

    Off unless ``VIBESIM_REQUIRE_ENERGY`` says otherwise, because a row measured
    without energy is perfectly valid input for a timing-only simulation; it is
    only wrong when the caller asked for energy.

    ``CommMetrics`` is out of scope, and has to be excluded by type rather than
    by value. It declares ``energy_j: float = 0.0``, but no comm runner has ever
    measured it -- the rank-group runners build their metrics without the field
    -- so a zero there is a structural constant, not a skipped measurement.
    Judging it by the same rule made ``energy: true`` fail on the first
    all-reduce of any tp>1 run, with a message telling the user to re-profile
    with energy on, which could never succeed.
    """

    if not require_measured_energy():
        return
    offenders = [result for result in results if _is_unmeasured_compute_energy(result)]
    if not offenders:
        return
    raise UnmeasuredEnergyError(
        f"{len(offenders)} of {len(results)} {kernel_kind}:{backend} rows on "
        f"{gpu_name} have energy_j = 0.0, which means not measured. The run asked "
        f"for energy, so these rows cannot answer it. Re-profile them with energy "
        f"on (launcher `energy: true`, or `python -m profiling run --energy ...`), "
        f"or set `energy: false` to run timing-only."
    )


def _is_unmeasured_compute_energy(result: Metrics | MissingEntry) -> bool:
    return isinstance(result, ComputeMetrics) and result.energy_j == 0.0


def _unpersisted_results(
    outcome: ProfileBatchOutcome | None,
    args_list: list[KernelArgs],
    kernel_kind: KernelKind,
    backend: str,
    resolved_gpu: str,
) -> list[Metrics | MissingEntry]:
    """Return the just-measured rows directly, because no row was written to read back.

    The persisted path re-queries the table so the caller sees exactly what landed in
    the cache. With nothing inserted there is nothing to re-query, so the ``Metrics``
    already in hand are the answer, and a spec whose runner failed becomes the same
    ``MissingEntry`` the query path would have reported for an absent row.
    """

    batch_results = outcome.results if outcome is not None else []
    return [
        metrics
        if metrics is not None
        else MissingEntry(
            kernel_kind=kernel_kind,
            backend=backend,
            gpu_name=resolved_gpu,
            args=args,
        )
        for metrics, args in zip(batch_results, args_list, strict=True)
    ]


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
    present = table.exists(args_list, backend=backend, gpu_name=resolved_gpu)
    collector = active_collector()
    if collector is not None:
        # The simulator's dry-run mode walks the whole build cascade calling only
        # `count_missing`, never fitting and never profiling. That walk already
        # visits every cost-tree node, so it is the collect pass: record what is
        # absent here and one `issue` can measure all of it, instead of 55
        # separate demand-driven calls each fanning across four cards.
        collector.record(
            kernel_kind,
            backend,
            [args_to_spec(args) for args, found in zip(args_list, present) if not found],
            # The key the miss was found under, so the fill writes back to the
            # same place. `resolved_gpu`, not `gpu_name`: this is the key
            # `table.exists` just used.
            gpu_name=resolved_gpu,
        )
    return present.count(False)


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
            raise TypeError(f"expected dict or {schema.__name__}, got {type(input_spec).__name__}")
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


def get_current_gpu_name() -> str:
    """Resolve the current CUDA device's DB gpu_name key. Raises when CUDA is
    unavailable — callers targeting a non-current or remote GPU must supply the
    gpu_name explicitly instead of relying on this."""
    return _resolve_gpu_name(None)
