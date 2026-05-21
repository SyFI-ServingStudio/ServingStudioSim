"""Public L1b facade.

All Python-side callers enter here. The facade validates specs, routes through
the registry/table layer, and applies JIT policy.

Agent note:
- Do not add hand-written ``get_<kind>_times`` or ``count_missing_<kind>``
  wrappers here. Add the args schema, runner, and ``KernelProfilerSpec`` row;
  public perf API symbols are generated from registry metadata by
  ``profiling.facade``.
- Keep this file limited to process-wide facade state and controls: DB path,
  JIT toggle, metadata helpers, remote submission, and generated exports.
"""

from __future__ import annotations

from pathlib import Path

from profiling.db.metadata import (
    DbMetadata,
    ProfilerVersion,
)
from profiling.db.metadata import (
    get_db_metadata as _get_db_metadata,
)
from profiling.db.metadata import (
    get_profiler_versions as _get_profiler_versions,
)
from profiling.facade import build_kind_facades
from profiling.facade import get_current_gpu_name as _get_current_gpu_name

DB_PATH = Path(__file__).resolve().parent / "profile.db"

_jit_enabled = False


def enable_jit_profiling() -> None:
    global _jit_enabled
    _jit_enabled = True


def disable_jit_profiling() -> None:
    global _jit_enabled
    _jit_enabled = False


def submit_remote(profile_request):
    raise NotImplementedError("RemoteGpuPool submission has not landed yet")


def get_current_gpu_name() -> str:
    return _get_current_gpu_name()


def get_db_metadata() -> DbMetadata:
    return _get_db_metadata(DB_PATH)


def get_profiler_versions(
    used_op_families: list[str] | None = None,
) -> list[ProfilerVersion]:
    return _get_profiler_versions(DB_PATH, used_op_families)


_GENERATED_FACADES = build_kind_facades(
    db_path=lambda: DB_PATH,
    jit_enabled=lambda: _jit_enabled,
)
globals().update(_GENERATED_FACADES)
