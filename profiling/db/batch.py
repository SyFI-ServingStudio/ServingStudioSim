"""Batch profiling scheduler owned by L1b.

All runner execution funnels through ``run_profile_batch``. This module
validates specs, groups work by ``(KernelKind, backend, GPU count)``, and hands
chunks to ``GpuPool``; execution backends run workers, and ``Table`` owns SQL.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_type_hints

from profiling.db.args import DType, KernelArgs
from profiling.db.kind import KernelKind
from profiling.db.registry import find_kernel_profiler_spec, resolve_spec_backend
from profiling.db.table import ProfileRow, Table
from profiling.runners.metrics import Metrics

if TYPE_CHECKING:
    from profiling.exec.pool import ChunkResult, GpuChunk, GpuPool

GpuCountFn = Callable[[dict[str, Any]], int]


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
) -> list[Metrics | None]:
    """Run profiling specs through the registry-selected runner.

    Runners are executed through the selected ``GpuPool``. The default is
    ``LocalGpuPool``; callers may inject ``RemoteGpuPool`` or another backend.
    """

    prepared_specs = _prepare_specs(kernel_kind, specs, gpu_count_fn)

    if pool is None:
        from profiling.exec import get_default_pool

        selected_pool = get_default_pool()
    else:
        selected_pool = pool

    # Same backend shares one profiler spec/env; same GPU count shares one
    # acquired chunk shape.
    classified_by_backend_gpu_count: dict[
        tuple[str, int], list[_PreparedProfileSpec]
    ] = defaultdict(list)
    for prepared_spec in prepared_specs:
        classified_by_backend_gpu_count[
            (prepared_spec.backend, prepared_spec.gpu_count)
        ].append(prepared_spec)

    results: list[Metrics | None] = [None] * len(specs)

    for (backend, gpu_count), classified_specs in classified_by_backend_gpu_count.items():
        profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)
        reserved_chunks = list(
            selected_pool.acquire_chunks(
                gpu_count,
                max_concurrent=len(classified_specs),
            )
        )
        if not reserved_chunks:
            raise RuntimeError(
                f"pool returned no chunks for {backend} with {gpu_count} GPU(s)"
            )

        profile_rows: list[ProfileRow] = []
        for prepared_spec, chunk_result in _run_specs_across_chunks(
            kernel_kind,
            classified_specs,
            reserved_chunks,
        ):
            metrics = chunk_result.metrics
            results[prepared_spec.input_index] = metrics
            if metrics is not None and db_path is not None:
                if chunk_result.gpu_name is None:
                    raise RuntimeError("execution backend did not report gpu_name for saved row")
                profile_rows.append(
                    ProfileRow(
                        args=prepared_spec.kernel_args,
                        metrics=metrics,
                        gpu_name=chunk_result.gpu_name,
                        backend=backend,
                    )
                )
        if db_path is not None and profile_rows:
            Table(profiler_spec, db_path).insert(profile_rows)

    return results


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
    chunk_specs = [
        prepared_spec.chunk_spec
        for prepared_spec in chunk_assignment.assigned_specs
    ]
    chunk_results = chunk_assignment.chunk.run(kernel_kind, chunk_specs)
    if len(chunk_results) != len(chunk_assignment.assigned_specs):
        raise RuntimeError(
            "execution backend returned "
            f"{len(chunk_results)} result(s) for "
            f"{len(chunk_assignment.assigned_specs)} payload(s)"
        )
    return list(zip(chunk_assignment.assigned_specs, chunk_results, strict=True))


def coerce_args(args_schema: type[KernelArgs], spec: dict[str, Any]) -> KernelArgs:
    """Validate a public spec dict against a ``KernelArgs`` dataclass.

    This is the single Python-side spec normalizer used by facade entry points,
    batch scheduling, and local workers. Callers strip routing-only ``backend``
    first; every remaining extra key should fail before a runner is invoked.
    """

    type_hints = get_type_hints(args_schema)
    values = {}
    for field in fields(args_schema):
        if field.name not in spec:
            raise ValueError(f"missing required spec field {field.name!r}")
        values[field.name] = _coerce_value(type_hints[field.name], spec[field.name])
    extra = set(spec) - {field.name for field in fields(args_schema)}
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
    return value
