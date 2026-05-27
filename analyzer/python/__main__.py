"""Analyzer plot renderer entry: `python analyzer/python render <log_dir>`.

Reads the Rust-emitted payload JSONs in `<log_dir>/payloads/` and writes PNGs to
`<log_dir>/plots/`. Pure rendering — never touches parquet.
"""

from __future__ import annotations

import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable

from request import slo_plot
from throughput import segment_plot

# Keys match the analyzer subjects (registry.rs). Both SLO subjects share one
# renderer, bound to their respective payload file.
RENDERERS = {
    "slo-general": partial(slo_plot.render, payload_name="slo_general_cdf.json"),
    "slo-detailed": partial(slo_plot.render, payload_name="slo_detailed_cdf.json"),
    "throughput": segment_plot.render,
}


def _invoke(job: Callable[[], Path]) -> Path:
    """Run one render job (worker entry; must be top-level to be picklable)."""
    return job()


def _run_jobs(jobs: list[Callable[[], Path]]) -> list[Path]:
    """Render every figure in parallel, one process per worker. Fork context (not
    spawn) so workers inherit `sys.path[0]` — the `analyzer/python` dir, needed to
    import `common.*` — and the already-`Agg`-configured matplotlib. matplotlib is
    thread-hostile, so processes (not threads). max_workers defaults to cpu_count
    and naturally uses fewer when there are fewer figures."""
    if not jobs:
        return []
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork")) as ex:
        return list(ex.map(_invoke, jobs))


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "render":
        print("usage: python analyzer/python render <log_dir> [subject ...]", file=sys.stderr)
        return 2
    log_dir = Path(argv[1])
    if not log_dir.is_dir():
        print(f"not a directory: {log_dir}", file=sys.stderr)
        return 2
    subjects = argv[2:] or list(RENDERERS)
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
