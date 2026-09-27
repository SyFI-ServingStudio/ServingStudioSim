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

import logging
import os
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
from profiling.plan import WorkCollector

DB_PATH = Path(os.environ.get("VIBESIM_PROFILE_DB", Path(__file__).resolve().parent / "profile.db"))

_jit_enabled = False


def enable_jit_profiling() -> None:
    global _jit_enabled
    _jit_enabled = True


def disable_jit_profiling() -> None:
    global _jit_enabled
    _jit_enabled = False


def begin_collect() -> None:
    """Start recording cache misses instead of measuring them.

    Paired with ``issue_collected``. The caller then walks the build cascade in
    the bridge's dry-run mode, which calls only ``count_missing_{kind}``; each
    call records what is absent rather than profiling it.

    Resets rather than refusing when a collector is already open. A build that
    raised part-way through its collect walk leaves one behind, and failing the
    *next* build for that would punish the wrong run.
    """

    from profiling.plan import set_collector

    set_collector(WorkCollector())


def issue_collected(expected_specs: int | None = None) -> None:
    """Measure everything ``begin_collect`` recorded, one whole unit per GPU.

    ``expected_specs`` is the count the caller's own walk arrived at. The count
    and the work list travel on separate channels -- the caller's dry-run report
    holds counts for display, this collector holds the specs -- so the collector
    could be silently inactive (installed on another thread, say) while the
    counts still look right.

    The two are not required to be equal: the count is summed per cost-tree
    node, and two nodes can miss the same shape, which the collector folds into
    one. So recording fewer than were counted is normal. Recording *nothing*
    when something was counted is not, and that is the failure this catches.

    Raises when a unit could not run, for the same reason.
    """

    from profiling.exec import get_default_pool
    from profiling.exec.local import LocalGpuPool, find_idle_gpus
    from profiling.gpu_policy import require_gpu
    from profiling.plan import issue, set_collector

    collector = set_collector(None)
    if collector is None:
        raise RuntimeError("issue_collected without begin_collect")

    recorded = len(collector)
    if expected_specs:
        if recorded == 0:
            raise RuntimeError(
                f"the collect walk counted {expected_specs} missing spec(s) but "
                "recorded none for measurement; the counting and recording paths "
                "disagree, so a cache build here would measure nothing"
            )
        if recorded > expected_specs:
            raise RuntimeError(
                f"recorded {recorded} spec(s) to measure but the collect walk "
                f"counted only {expected_specs} missing; recording more than was "
                "counted cannot happen and means the two walks saw different work"
            )
        logging.getLogger(__name__).info(
            "collected %d distinct spec(s) from %d node-level miss(es)", recorded, expected_specs
        )

    if not collector:
        # A fill with nothing missing must not need a GPU to say so.
        logging.getLogger(__name__).info("nothing to issue: profile.db already covers the run")
        return
    require_gpu(f"measuring {recorded} missing profile.db spec(s)")

    pool = get_default_pool()
    gpus = pool.gpus if isinstance(pool, LocalGpuPool) and pool.gpus else find_idle_gpus()
    if not gpus:
        raise RuntimeError("no GPUs available to issue the collected profiling work")

    report = issue(collector, db_path=DB_PATH, gpu_name=None, gpus=list(gpus))
    logging.getLogger(__name__).info(
        "issued %d unit(s) as %d piece(s), %d spec(s) across %d GPU(s)%s",
        report.units,
        report.pieces,
        report.specs,
        len(gpus),
        f"; split {', '.join(report.split_units)}" if report.split_units else "",
    )
    if report.failures:
        raise RuntimeError(
            f"{len(report.failures)} profiling unit(s) failed: {'; '.join(report.failures[:5])}"
        )


def submit_remote(profile_request):
    raise NotImplementedError("RemoteGpuPool submission has not landed yet")


def measure_kernel(
    kernel_kind,
    spec,
    *,
    backend=None,
    gpu_name=None,
    output_dir,
    duration_s: float = 10.0,
    telemetry_hz: float = 20.0,
    clear_l2: bool = True,
    telemetry: bool = True,
):
    """Cache-free trend+telemetry diagnostic for one CUPTI kernel spec.

    Unlike ``get_<kind>_times``, this does not read or write ``profile.db``; it
    runs a sustained per-launch capture and writes CSV / summary / plots into
    ``output_dir``. See ``profiling.measure``.
    """

    from profiling.measure import measure_kernel as _measure_kernel

    return _measure_kernel(
        kernel_kind,
        spec,
        backend=backend,
        gpu_name=gpu_name,
        output_dir=output_dir,
        duration_s=duration_s,
        telemetry_hz=telemetry_hz,
        clear_l2=clear_l2,
        telemetry=telemetry,
    )


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
