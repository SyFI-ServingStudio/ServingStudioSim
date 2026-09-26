"""Batch profiling scheduler owned by L1b.

All runner execution funnels through ``run_profile_batch``. This module
validates specs, groups work by ``(KernelKind, backend, GPU count)``, and hands
chunks to ``GpuPool``; execution backends run workers, and ``Table`` owns SQL.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints

from profiling.db.args import DType, KernelArgs
from profiling.db.kind import KernelKind
from profiling.db.registry import find_kernel_profiler_spec, resolve_spec_backend
from profiling.db.table import ProfileRow, Table
from profiling.gpu_catalog import resolve_gpu_spec
from profiling.instrument import span
from profiling.runners.metrics import Metrics

if TYPE_CHECKING:
    from profiling.db.registry import KernelProfilerSpec
    from profiling.exec.pool import ChunkResult, GpuChunk, GpuPool

GpuCountFn = Callable[[dict[str, Any]], int]

logger = logging.getLogger(__name__)

_MAX_REPORTED_FAILURE_REASONS = 3
_MAX_FAILURE_REASON_CHARS = 240
_MISSING_FAILURE_REASON = "runner returned no metrics and no error"


@dataclass(frozen=True)
class ProfileProvenance:
    """GPU provenance recorded by the profiling execution layer.

    Owned by ``run_profile_batch``: the worker reports the physical GPU it ran on
    (``observed_gpu_name``), which is provenance only — it never becomes a DB cache
    key. ``requested_gpu_name`` is the DB cache key the batch was told to write
    under (None = the backend fell back to the observed name). ``gpu_count`` is the
    largest reserved chunk across the executed specs.
    """

    source: str
    requested_gpu_name: str | None
    observed_gpu_name: str | None = None
    gpu_count: int | None = None


@dataclass(frozen=True)
class ProfileBatchOutcome:
    """Typed result of one profiling batch: results + per-invocation provenance.

    ``execute_profile_batch`` returns this so public facade and CLI share one
    core call without an ambient side channel. The legacy ``run_profile_batch``
    wrapper returns only ``results`` for compatibility.
    """

    results: list[Metrics | None]
    provenance: ProfileProvenance


@dataclass(frozen=True)
class _PreparedProfileSpec:
    """Validated unit of work shared by scheduling, execution, and DB insert.

    ``kernel_args`` is the typed DB identity. ``chunk_spec`` is the normalized
    dict sent to ``GpuChunk.run``; it includes ``backend`` only for execution
    routing and subprocess env/module selection.
    """

    input_index: int
    backend: str
    gpu_count: int
    kernel_args: KernelArgs
    chunk_spec: dict[str, Any]


@dataclass(frozen=True)
class _ChunkAssignment:
    """One reserved chunk plus the specs assigned to that chunk."""

    chunk: GpuChunk
    assigned_specs: list[_PreparedProfileSpec]


def run_profile_batch(
    kernel_kind: KernelKind,
    specs: list[dict[str, Any]],
    *,
    pool: GpuPool | None = None,
    gpu_count_fn: GpuCountFn | None = None,
    db_path: Path | None = None,
    gpu_name: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> list[Metrics | None]:
    """Backward-compatible thin wrapper over ``execute_profile_batch``.

    ``gpu_name`` is the requested DB cache key (``None`` uses the validated
    worker-observed name). Callers that need GPU provenance should call
    ``execute_profile_batch`` directly and read the typed ``ProfileBatchOutcome``;
    this wrapper keeps the historical ``list[Metrics | None]`` return and, when a
    ``provenance`` dict is still passed, fills it for older callers.
    """

    outcome = execute_profile_batch(
        kernel_kind,
        specs,
        pool=pool,
        gpu_count_fn=gpu_count_fn,
        db_path=db_path,
        gpu_name=gpu_name,
    )
    if provenance is not None:
        batch_provenance = outcome.provenance
        provenance["source"] = batch_provenance.source
        provenance["requested_gpu_name"] = batch_provenance.requested_gpu_name
        provenance["observed_gpu_name"] = batch_provenance.observed_gpu_name
        provenance["gpu_count"] = batch_provenance.gpu_count
    return outcome.results


def execute_profile_batch(
    kernel_kind: KernelKind,
    specs: list[dict[str, Any]],
    *,
    pool: GpuPool | None = None,
    gpu_count_fn: GpuCountFn | None = None,
    db_path: Path | None = None,
    gpu_name: str | None = None,
) -> ProfileBatchOutcome:
    """Internal profiling funnel: validate specs, run chunks, then persist.

    This is the single measured-row camera. Every successful row is inserted under the
    requested cache key only after GPU identity validation; a failed mismatch raises
    before ``Table.insert`` so no row is left under the requested key. Runners
    still return only ``Metrics``; physical GPU observations come from the execution
    layer (``ChunkResult.observed_gpu_name``). A successful execution without that
    observation is invalid; cached-only provenance exists only in the facade when
    no worker runs. ``gpu_name`` is the requested DB cache key; ``None`` uses the
    validated worker-observed name.
    """

    prepared_specs = _prepare_specs(kernel_kind, specs, gpu_count_fn)

    if pool is None:
        from profiling.exec import get_default_pool

        selected_pool = get_default_pool()
    else:
        selected_pool = pool

    # Same backend shares one profiler spec/env; same GPU count shares one
    # acquired chunk shape.
    classified_by_backend_gpu_count: dict[tuple[str, int], list[_PreparedProfileSpec]] = (
        defaultdict(list)
    )
    for prepared_spec in prepared_specs:
        classified_by_backend_gpu_count[(prepared_spec.backend, prepared_spec.gpu_count)].append(
            prepared_spec
        )

    results: list[Metrics | None] = [None] * len(specs)
    observed_names: list[str] = []
    pending_groups: list[
        tuple[KernelProfilerSpec, list[tuple[_PreparedProfileSpec, ChunkResult]]]
    ] = []
    attempts_by_backend: Counter[str] = Counter()
    failures_by_backend: dict[str, Counter[str]] = defaultdict(Counter)
    largest_gpu_count = 0

    # Run every group first, collecting rows plus every successful worker observation.
    # Identity validation happens once after all chunks finish, before any insert.
    for (backend, gpu_count), classified_specs in classified_by_backend_gpu_count.items():
        profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)
        with span("pool.acquire", kind=kernel_kind, backend=backend):
            reserved_chunks = list(
                selected_pool.acquire_chunks(
                    gpu_count,
                    max_concurrent=len(classified_specs),
                )
            )
        if not reserved_chunks:
            raise RuntimeError(f"pool returned no chunks for {backend} with {gpu_count} GPU(s)")

        largest_gpu_count = max(largest_gpu_count, gpu_count)
        attempts_by_backend[backend] += len(classified_specs)
        successful_profiles: list[tuple[_PreparedProfileSpec, ChunkResult]] = []
        for prepared_spec, chunk_result in _run_specs_across_chunks(
            kernel_kind,
            classified_specs,
            reserved_chunks,
        ):
            metrics = chunk_result.metrics
            results[prepared_spec.input_index] = metrics
            if metrics is None:
                failures_by_backend[backend][_normalize_failure_reason(chunk_result.error)] += 1
                continue
            observed = chunk_result.observed_gpu_name
            if not observed:
                raise RuntimeError(
                    "successful profiling execution did not report observed_gpu_name"
                )
            if observed not in observed_names:
                observed_names.append(observed)
            successful_profiles.append((prepared_spec, chunk_result))
        if successful_profiles:
            pending_groups.append((profiler_spec, successful_profiles))

    for backend, failures in failures_by_backend.items():
        _log_batch_failures(kernel_kind, backend, attempts_by_backend[backend], failures)

    if pending_groups:
        effective_cache_key, observed_after_validation = _validated_gpu_identity(
            gpu_name, observed_names
        )
        if db_path is not None:
            for profiler_spec, successful_profiles in pending_groups:
                profile_rows = [
                    ProfileRow(
                        args=prepared_spec.kernel_args,
                        metrics=chunk_result.metrics,
                        gpu_name=effective_cache_key,
                        backend=prepared_spec.backend,
                        cuda_version=chunk_result.cuda_version,
                        backend_version=chunk_result.backend_version,
                    )
                    for prepared_spec, chunk_result in successful_profiles
                    if chunk_result.metrics is not None
                ]
                with span("db.insert", kind=kernel_kind, rows=len(profile_rows)):
                    Table(profiler_spec, db_path).insert(profile_rows)
    else:
        effective_cache_key, observed_after_validation = gpu_name, None

    if observed_after_validation is None:
        return ProfileBatchOutcome(
            results=results,
            provenance=ProfileProvenance(
                source="cache_key",
                requested_gpu_name=effective_cache_key,
            ),
        )
    return ProfileBatchOutcome(
        results=results,
        provenance=ProfileProvenance(
            source="measurement",
            requested_gpu_name=effective_cache_key,
            observed_gpu_name=observed_after_validation,
            gpu_count=largest_gpu_count,
        ),
    )


def _log_batch_failures(
    kernel_kind: KernelKind,
    backend: str,
    attempted: int,
    failures: Counter[str],
) -> None:
    """Surface runner errors that would otherwise become anonymous cache misses."""
    failed = sum(failures.values())
    total_failure = failed == attempted
    level = logging.ERROR if total_failure else logging.WARNING
    reasons = [
        # Truncating is right for a partial failure -- a handful of OOM specs
        # inside a large batch should not bury the log. It is wrong when every
        # spec failed: that backend is now a hard stop for whatever asked for
        # it, and the 240-char cut lands in the middle of the traceback,
        # leaving "File ..." and nothing about what actually raised.
        f"{count}x {reason if total_failure else _display_failure_reason(reason)}"
        for reason, count in failures.most_common(_MAX_REPORTED_FAILURE_REASONS)
    ]
    omitted = len(failures) - len(reasons)
    if omitted:
        reasons.append(f"{omitted} additional reason(s)")
    logger.log(
        level,
        "%s:%s profiled %d/%d specs; %d failed and were not inserted (%s)",
        kernel_kind,
        backend,
        attempted - failed,
        attempted,
        failed,
        "; ".join(reasons),
    )


def _display_failure_reason(reason: str) -> str:
    """Keep one grouped reason readable without flooding operator logs."""
    if len(reason) <= _MAX_FAILURE_REASON_CHARS:
        return reason
    return f"{reason[: _MAX_FAILURE_REASON_CHARS - 3]}..."


def _normalize_failure_reason(reason: str | None) -> str:
    """Canonicalize worker text before grouping equivalent failures."""
    return " ".join((reason or "").split()) or _MISSING_FAILURE_REASON


def _validated_gpu_identity(
    requested_gpu_name: str | None,
    observed_names: list[str],
) -> tuple[str, str]:
    """Validate GPU identity before any ``Table.insert``.

    Returns ``(effective_cache_key, observed_name)``. Every observed name must
    canonicalize via ``gpu/spec.json`` and all must share the requested key's
    canonical SKU. Missing observations are rejected while collecting successful
    results, before this helper and before any insert.
    """
    if not observed_names:
        raise RuntimeError("measured profiling batch has no observed GPU identity")

    observed_resolutions = [resolve_gpu_spec(name) for name in observed_names]
    if any(resolution is None for resolution in observed_resolutions):
        unmatched = [
            name
            for name, resolution in zip(observed_names, observed_resolutions, strict=True)
            if resolution is None
        ]
        raise ValueError(
            "worker-observed GPU(s) not in gpu/spec.json: "
            + ", ".join(repr(name) for name in unmatched)
        )
    observed_canonical_skus = {resolution.canonical_name for resolution in observed_resolutions}
    if len(observed_canonical_skus) > 1:
        raise ValueError(
            "workers observed different physical GPU SKUs: "
            + ", ".join(sorted(observed_canonical_skus))
        )
    single_observed_name = observed_names[0]
    if requested_gpu_name is not None:
        requested_resolution = resolve_gpu_spec(requested_gpu_name)
        if requested_resolution is None:
            raise ValueError(f"requested cache key {requested_gpu_name!r} is not in gpu/spec.json")
        if requested_resolution.canonical_name not in observed_canonical_skus:
            raise ValueError(
                f"GPU identity mismatch: requested cache key {requested_gpu_name!r} "
                f"and observed physical GPU {single_observed_name!r} do not resolve to "
                "the same canonical SKU in gpu/spec.json"
            )
        return requested_gpu_name, single_observed_name
    # With no requested DB identity, use the agreed physical name consistently for
    # both provenance and every inserted row.
    return single_observed_name, single_observed_name


def _run_specs_across_chunks(
    kernel_kind: KernelKind,
    classified_specs: list[_PreparedProfileSpec],
    reserved_chunks: list[GpuChunk],
) -> list[tuple[_PreparedProfileSpec, ChunkResult]]:
    """Run one classified set across chunks and preserve input identities."""

    chunk_assignments = _assign_specs_to_chunks_round_robin(classified_specs, reserved_chunks)
    if len(chunk_assignments) == 1:
        return _run_chunk_assignment(kernel_kind, chunk_assignments[0])

    # chunk.run is usually a subprocess or remote submission, so threads only
    # coordinate blocking I/O while each chunk owns its isolated GPU reservation.
    with ThreadPoolExecutor(max_workers=len(chunk_assignments)) as executor:
        futures = [
            executor.submit(_run_chunk_assignment, kernel_kind, chunk_assignment)
            for chunk_assignment in chunk_assignments
        ]
        completed_results: list[tuple[_PreparedProfileSpec, ChunkResult]] = []
        for future in futures:
            completed_results.extend(future.result())
        return completed_results


def _assign_specs_to_chunks_round_robin(
    classified_specs: list[_PreparedProfileSpec],
    reserved_chunks: list[GpuChunk],
) -> list[_ChunkAssignment]:
    """Split specs round-robin, matching the old compute batch JIT behavior."""

    specs_by_chunk_index: list[list[_PreparedProfileSpec]] = []
    for _reserved_chunk in reserved_chunks:
        specs_by_chunk_index.append([])

    for position, prepared_spec in enumerate(classified_specs):
        chunk_index = position % len(reserved_chunks)
        specs_by_chunk_index[chunk_index].append(prepared_spec)

    chunk_assignments = []
    for chunk_index, reserved_chunk in enumerate(reserved_chunks):
        assigned_specs = specs_by_chunk_index[chunk_index]
        if assigned_specs:
            chunk_assignments.append(
                _ChunkAssignment(
                    chunk=reserved_chunk,
                    assigned_specs=assigned_specs,
                )
            )
    return chunk_assignments


def _run_chunk_assignment(
    kernel_kind: KernelKind,
    chunk_assignment: _ChunkAssignment,
) -> list[tuple[_PreparedProfileSpec, ChunkResult]]:
    chunk_specs = [prepared_spec.chunk_spec for prepared_spec in chunk_assignment.assigned_specs]
    # One span per chunk. A kernel group ends when its slowest chunk ends, so
    # the spread between these is the synchronisation loss -- the part that does
    # not shrink by adding cards.
    with span("chunk.run", kind=kernel_kind, specs=len(chunk_specs)):
        chunk_results = chunk_assignment.chunk.run(kernel_kind, chunk_specs)
    if len(chunk_results) != len(chunk_assignment.assigned_specs):
        raise RuntimeError(
            "execution backend returned "
            f"{len(chunk_results)} result(s) for "
            f"{len(chunk_assignment.assigned_specs)} payload(s)"
        )
    return list(zip(chunk_assignment.assigned_specs, chunk_results, strict=True))


@cache
def _schema_field_types(args_schema: type[KernelArgs]) -> dict[str, Any]:
    """Resolved field types of an args dataclass, in field order.

    ``get_type_hints`` re-evaluates every string annotation on each call, and a
    build coerces thousands of specs against a few dozen schemas; resolved once
    per schema, it drops from about half of the profile.db query time to nothing.
    """

    type_hints = get_type_hints(args_schema)
    return {field.name: type_hints[field.name] for field in fields(args_schema)}


def coerce_args(args_schema: type[KernelArgs], spec: dict[str, Any]) -> KernelArgs:
    """Validate a public spec dict against a ``KernelArgs`` dataclass.

    This is the single Python-side spec normalizer used by facade entry points,
    batch scheduling, and local workers. Callers strip routing-only ``backend``
    first; every remaining extra key should fail before a runner is invoked.
    """

    field_types = _schema_field_types(args_schema)
    values = {}
    for name, field_type in field_types.items():
        if name not in spec:
            raise ValueError(f"missing required spec field {name!r}")
        values[name] = _coerce_value(field_type, spec[name])
    extra = set(spec) - field_types.keys()
    if extra:
        raise ValueError(f"unexpected spec fields for {args_schema.__name__}: {sorted(extra)}")
    return args_schema(**values)


def args_to_spec(args: KernelArgs) -> dict[str, Any]:
    """Convert typed args back to the JSON-friendly runner spec shape."""

    values = {}
    for field in fields(args):
        value = getattr(args, field.name)
        values[field.name] = value.value if hasattr(value, "value") else value
    return values


def _prepare_specs(
    kernel_kind: KernelKind,
    input_specs: list[dict[str, Any]],
    gpu_count_fn: GpuCountFn | None,
) -> list[_PreparedProfileSpec]:
    prepared_specs: list[_PreparedProfileSpec] = []
    for input_index, raw_spec in enumerate(input_specs):
        backend = resolve_spec_backend(kernel_kind, raw_spec)
        profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)
        kernel_args = coerce_args(profiler_spec.args_schema, _args_only(raw_spec))
        chunk_spec = args_to_spec(kernel_args)
        # Backend is not a runner kwarg. It stays in the chunk spec only so
        # execution backends can select the registered module/env before calling
        # the runner with schema fields.
        chunk_spec["backend"] = backend

        count_fn = gpu_count_fn or profiler_spec.gpu_count_fn or _one_gpu
        gpu_count = int(count_fn(chunk_spec))
        if gpu_count < 1:
            raise ValueError(f"gpu_count_fn must return >= 1, got {gpu_count}")
        prepared_specs.append(
            _PreparedProfileSpec(
                input_index=input_index,
                backend=backend,
                gpu_count=gpu_count,
                kernel_args=kernel_args,
                chunk_spec=chunk_spec,
            )
        )
    return prepared_specs


def _args_only(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if key != "backend"}


def _one_gpu(_: dict[str, Any]) -> int:
    return 1


def _coerce_value(annotation: Any, value: Any) -> Any:
    if annotation is DType:
        return DType.from_value(value)
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        return str(value)
    origin = get_origin(annotation)
    if origin in (tuple, list):
        # Distribution-sensitive args (e.g. grouped_gemm per_group_batches) arrive
        # from the Rust facade / DB as a JSON list. Coerce to the declared origin
        # (tuple keeps the frozen KernelArgs hashable) and coerce each element.
        elem_types = [arg for arg in get_args(annotation) if arg is not Ellipsis]
        elem_type = elem_types[0] if elem_types else None
        items = [_coerce_value(elem_type, item) for item in value] if elem_type else list(value)
        return tuple(items) if origin is tuple else items
    return value
