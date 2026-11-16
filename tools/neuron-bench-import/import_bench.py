"""Seed profile.db with the Neuron rows of GLM-5.3-Flash on Trainium2 (LNC2).

Three sources, in order (a later row of the same key replaces an earlier one):

1. Collectives (``all_gather`` / ``reduce_scatter`` / ``all_reduce``, backend ``neuron_cc``): the
   trn2-flashnext branch's imported in-graph rows, kept verbatim in
   ``data/trn2_flashnext_collectives.jsonl`` (``families/collectives.py``).
2. The glm5-trn2 engine's own op benchmarks: every ``*.jsonl`` under ``--bench-dir``
   (``logs/bench/{kda,dsa,moe,mhc_*}.jsonl``), schema ``{kind, impl, phase, tokens, batch, context,
   tp, ep, dtype, device_ms, latency_ms, pipelined_ms, env}``; a row is imported when it has a
   positive ``device_ms`` and no ``error`` (``families/glm53.py``).
3. ``--fill-grid``: every grid cell the arch can ask for (``grid.py``, the Rust Neuron grids) at the
   layouts of ``--attn-tp`` that no measured row covers gets a DERIVED row: the analytic Trainium2
   estimate of the region (``analytic.py``) times the measured/analytic ratio interpolated from the
   same series' measured cells (else the kind's median ratio, else 1). Measured rows are never
   overwritten by derived ones.

Provenance: ``backend_version`` is ``imported:<file>:<line>`` (bench), ``imported:trn2-flashnext``
(collectives) or ``derived:<method>`` (fill). ``--report`` writes the per-kind measured / derived
counts as JSON (``doc/pr/trn2-glm53-flash/l1_coverage.json``), which the config-search report
reads to label predictions that rest on derived rows.

The tool is re-runnable: run it again when the bench JSONLs grow; measured rows replace derived
ones of the same key.

    uv run --no-sync python tools/neuron-bench-import/import_bench.py \\
        --bench-dir /home/ec2-user/glm5-trn2/logs/bench --fill-grid \\
        --report doc/pr/trn2-glm53-flash/l1_coverage.json [--db profiling/profile.db] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

GPU_NAME = "AWS Trainium2 LNC2"


@dataclass(frozen=True)
class ImportRow:
    kind: str
    backend: str
    args: dict
    time_ms: float
    source: str  # "imported:<file>:<line>", "imported:trn2-flashnext" or "derived:<method>"
    flops: float = 0.0
    bytes: float = 0.0
    comm: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.kind, self.backend, json.dumps(self.args, sort_keys=True))

    @property
    def measured(self) -> bool:
        return self.source.startswith("imported:")


def jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    """(1-based line, row) for every parseable line of a JSONL file; missing file -> nothing."""
    if not path.exists():
        return
    with path.open() as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def device_ms(row: dict) -> float | None:
    """The row's device time when it is a clean measurement, else None."""
    if row.get("error"):
        return None
    value = row.get("device_ms")
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def latest(rows: Iterable[ImportRow]) -> list[ImportRow]:
    """Keep the last row per (kind, backend, args): later lines re-measure earlier ones."""
    out: dict = {}
    for row in rows:
        out[row.key] = row
    return list(out.values())


def insert(rows: list[ImportRow], db: Path, dry_run: bool) -> Counter:
    import profiling.kernels  # noqa: F401  registers every kind
    from profiling.db.registry import find_kernel_profiler_spec
    from profiling.db.table import ProfileRow, Table
    from profiling.runners.metrics import CommMetrics, ComputeMetrics

    counts: Counter = Counter()
    by_table: dict[tuple[str, str], list] = defaultdict(list)
    for row in rows:
        spec = find_kernel_profiler_spec(row.kind, row.backend)
        args = spec.args_schema(**row.args)
        seconds = row.time_ms / 1e3
        if row.comm:
            bw = row.bytes / seconds / 1e9 if row.bytes else 0.0
            metrics = CommMetrics(time_ms=row.time_ms, algbw_gbps=bw, busbw_gbps=bw)
        else:
            metrics = ComputeMetrics(
                time_ms=row.time_ms,
                tflops=row.flops / seconds / 1e12,
                memory_bandwidth_gbps=row.bytes / seconds / 1e9,
            )
        by_table[(row.kind, row.backend)].append(
            ProfileRow(
                args=args,
                metrics=metrics,
                gpu_name=GPU_NAME,
                backend=row.backend,
                backend_version=row.source,
                verified=row.measured,
            )
        )
        counts[(row.kind, row.backend, "measured" if row.measured else "derived")] += 1
    if not dry_run:
        for (kind, backend), profile_rows in by_table.items():
            Table(find_kernel_profiler_spec(kind, backend), db).insert(profile_rows)
    return counts


def main(argv: list[str] | None = None) -> None:
    from families import collectives, glm53
    from grid import fill_grid

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=ROOT / "profiling" / "profile.db")
    ap.add_argument("--fill-grid", action="store_true")
    ap.add_argument("--attn-tp", default="1,2,4,8,16", help="layouts the grid fill covers")
    ap.add_argument("--world", type=int, default=32)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rows: list[ImportRow] = []
    rows.extend(collectives.rows(HERE / "data" / "trn2_flashnext_collectives.jsonl"))
    rows.extend(glm53.rows(a.bench_dir))
    rows = latest(rows)
    for (kind, impl), n in sorted(glm53.SKIPPED.items(), key=str):
        print(f"[skip] {n} bench rows of {kind} impl {impl}: no registered backend")
    if a.fill_grid:
        tps = [int(x) for x in a.attn_tp.split(",")]
        rows = rows + fill_grid(rows, tps, a.world)
    counts = insert(rows, a.db, a.dry_run)
    summary: dict = defaultdict(lambda: {"measured": 0, "derived": 0})
    for (kind, backend, origin), n in sorted(counts.items()):
        summary[f"{kind}:{backend}"][origin] += n
    for key, c in sorted(summary.items()):
        print(f"{key:40s} measured {c['measured']:5d}  derived {c['derived']:5d}")
    total_m = sum(c["measured"] for c in summary.values())
    total_d = sum(c["derived"] for c in summary.values())
    print(
        f"{'(dry run) ' if a.dry_run else ''}{total_m} measured + {total_d} derived rows -> {a.db}"
    )
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        sources = sorted({r.source.split(":")[1] for r in rows if r.source.startswith("imported:")})
        a.report.write_text(
            json.dumps(
                {
                    "gpu_name": GPU_NAME,
                    "bench_dir": str(a.bench_dir),
                    "sources": sources,
                    "kinds": dict(sorted(summary.items())),
                },
                indent=1,
            )
            + "\n"
        )


__all__ = ["ImportRow", "jsonl", "device_ms", "latest", "insert", "replace"]

if __name__ == "__main__":
    main()
