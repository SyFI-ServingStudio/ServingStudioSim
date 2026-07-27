"""Analyzer plot renderer entry: `python analyzer/python render <log_dir>`.

Reads the Rust-emitted payload JSONs in `<log_dir>/payloads/` and writes PNGs to
`<log_dir>/plots/`. Pure rendering — never touches parquet.
"""

from __future__ import annotations

import multiprocessing
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from alignment_e2e import series_plot as alignment_e2e_plot
from alignment_iteration import series_plot as alignment_iteration_plot
from alignment_workload import series_plot as alignment_workload_plot
from backend import kernel_input_distribution_plot
from batch import kernel_throughput_plot, scatter_plot
from breakdown import kernel_time_share_plot
from concurrency import request_state_plot
from concurrency import series_plot as concurrency_plot
from conservation import workload_plot
from kv import kv_occupancy_plot
from optimality import optimality_plot
from request import slo_plot
from sweep import grid_plot as sweep_plot
from throughput import segment_plot
from utilization import util_plot

# Keys match the analyzer subjects (registry.rs). The SLO subjects share one
# renderer, bound to their respective payload files.
RENDERERS = {
    "slo-general": partial(slo_plot.render, payload_name="slo_general_cdf.json"),
    "slo-detailed": partial(slo_plot.render, payload_name="slo_detailed_cdf.json"),
    "slo-goodput": partial(slo_plot.render, payload_name="slo_goodput_cdf.json"),
    "throughput": segment_plot.render,
    "utilization": util_plot.render,
    "batch": scatter_plot.render,
    "kernel-throughput": kernel_throughput_plot.render,
    "kernel-input-distribution": kernel_input_distribution_plot.render,
    "kernel-time-share": kernel_time_share_plot.render,
    "optimality": optimality_plot.render,
    "concurrency": concurrency_plot.render,
    "request-state": request_state_plot.render,
    "workload-conservation": workload_plot.render,
    "kv-occupancy": kv_occupancy_plot.render,
    "alignment-iteration": alignment_iteration_plot.render,
    "alignment-workload": alignment_workload_plot.render,
    "alignment-e2e": alignment_e2e_plot.render,
    "sweep": sweep_plot.render,
}
ALIGNMENT_SUBJECTS = ("alignment-iteration", "alignment-workload", "alignment-e2e")
SWEEP_SUBJECTS = ("sweep",)
RUN_SUBJECTS = tuple(name for name in RENDERERS if name not in ALIGNMENT_SUBJECTS + SWEEP_SUBJECTS)


def _invoke(job: Callable[[], Path]) -> Path:
    """Run one render job (worker entry; must be top-level to be picklable)."""
    return job()


def _run_jobs(jobs: list[Callable[[], Path]]) -> list[Path]:
    """Render every figure in parallel, one process per worker. Fork context (not
    spawn) so workers inherit `sys.path[0]` — the `analyzer/python` dir, needed to
    import `common.*` — and the already-`Agg`-configured matplotlib. matplotlib is
    thread-hostile, so processes (not threads). Cap the pool: alignment can emit
    dozens of figures, while hundreds of forked matplotlib workers only multiply
    memory pressure and make rendering slower."""
    if not jobs:
        return []
    with ProcessPoolExecutor(
        max_workers=min(len(jobs), 8),
        mp_context=multiprocessing.get_context("fork"),
    ) as ex:
        return list(ex.map(_invoke, jobs))


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "render":
        print("usage: python analyzer/python render <log_dir> [subject ...]", file=sys.stderr)
        return 2
    log_dir = Path(argv[1])
    if not log_dir.is_dir():
        print(f"not a directory: {log_dir}", file=sys.stderr)
        return 2
    # Mirror Rust's source Scope gate for the default-is-all behavior. The
    # registry stays flat; the manifest identifies which artifact envelope this
    # directory carries, so a normal run never probes alignment-only payloads
    # and an alignment bundle never probes normal-run payloads.
    if (log_dir / "alignment_manifest.json").is_file():
        default_subjects = ALIGNMENT_SUBJECTS
    elif (log_dir / "sweep_manifest.json").is_file():
        default_subjects = SWEEP_SUBJECTS
    else:
        default_subjects = RUN_SUBJECTS
    subjects = argv[2:] or default_subjects
    jobs: list[Callable[[], Path]] = []
    for subject in subjects:
        renderer = RENDERERS.get(subject)
        if renderer is None:
            print(f"[render] unknown subject {subject!r}; known: {', '.join(RENDERERS)}")
            continue
        jobs.extend(renderer(log_dir))
    paths = _run_jobs(jobs)
    for path in paths:
        print(f"rendered {path}")
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
