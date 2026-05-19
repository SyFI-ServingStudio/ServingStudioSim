"""Public re-exports for the L1b profile database package.

Agent note:
- Keep this file as a barrel only. Do not add registry mutation, DB access,
  runner dispatch, or schema construction here.
- Internal L1 modules should import concrete modules directly
  (``profiling.db.args``, ``profiling.db.registry``, etc.) to avoid circular
  imports.
- Adding a new kernel normally should not require touching this file unless the
  new symbol is part of the stable public DB package surface.
"""

from profiling.db.args import DType, KernelArgs, SingleGemmArgs
from profiling.db.batch import args_to_spec, coerce_args, run_profile_batch
from profiling.db.kind import KernelKind
from profiling.db.metadata import (
    DbMetadata,
    ProfilerVersion,
    get_db_metadata,
    get_profiler_versions,
)
from profiling.db.migrate import SCHEMA_HASH, SCHEMA_VERSION, migrate
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    find_args_schema,
    find_kernel_profiler_spec,
    find_table,
    iter_kernel_profiler_specs,
    known_backends,
    load_runner,
    resolve_spec_backend,
)
from profiling.db.table import MissingEntry, ProfileRow, Table, TableMetadata

__all__ = [
    "BatchOutlierPolicy",
    "DbMetadata",
    "DType",
    "KernelArgs",
    "KernelProfilerSpec",
    "KernelKind",
    "MetricFamily",
    "MissingEntry",
    "ProfilerVersion",
    "ProfileRow",
    "RunnerRef",
    "SCHEMA_HASH",
    "SCHEMA_VERSION",
    "SingleGemmArgs",
    "Table",
    "TableMetadata",
    "args_to_spec",
    "coerce_args",
    "find_args_schema",
    "find_kernel_profiler_spec",
    "find_table",
    "get_db_metadata",
    "get_profiler_versions",
    "iter_kernel_profiler_specs",
    "known_backends",
    "load_runner",
    "migrate",
    "resolve_spec_backend",
    "run_profile_batch",
]
