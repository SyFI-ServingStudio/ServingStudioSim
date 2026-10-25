"""Seed profile.db with Neuron rows from the flashnext-neuron bench logs.

The flashnext-neuron project's lane benchmarks (`logs/bench/<family>.jsonl`)
time the same op regions, at the same inputs and with the same device-time
contract (`runtime.device_bench`), as this checkout's Neuron profilers
(`profiling/runners/neuron/flashnext_ops.py`). This tool maps every bench row
whose shape lands on a registered Neuron table onto that table's args and
inserts it, so those rows need no second measurement. A row is imported only
when it has a positive `device_ms` and no error; a family module decides which
rows map and how (`families/<family>.py`, each exposing
`rows(bench_dir) -> Iterable[ImportRow]`).

Every imported row records its source file and line in `backend_version`
(`imported:<file>:<line>`), so `profiling query` can tell it from a live row.

    uv run --no-sync python tools/neuron-bench-import/import_bench.py \
        --bench-dir /home/ec2-user/ServingStudio/framework/flashnext-neuron/logs/bench \
        [--db profiling/profile.db] [--family gdn ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GPU_NAME = "AWS Trainium2 LNC2"
FAMILIES = ("gdn", "qsa", "moe", "hc", "ple", "head", "collectives", "launch")


@dataclass(frozen=True)
class ImportRow:
    kind: str
    backend: str
    args: dict
    time_ms: float
    source: str  # "<file>:<line>"
    flops: float = 0.0
    bytes: float = 0.0
    # Comm rows: per-rank bytes moved per call, for busbw (0 leaves it 0).
    comm: bool = False


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
    """Keep the last row per (kind, backend, args): later bench lines re-measure earlier ones."""
    out: dict = {}
    for row in rows:
        out[(row.kind, row.backend, json.dumps(row.args, sort_keys=True))] = row
    return list(out.values())


def insert(rows: list[ImportRow], db: Path, dry_run: bool) -> Counter:
    import profiling.kernels  # noqa: F401  registers every kind
    from profiling.db.registry import find_kernel_profiler_spec
    from profiling.db.table import ProfileRow, Table
    from profiling.runners.metrics import CommMetrics, ComputeMetrics

    counts: Counter = Counter()
    by_table: dict[tuple[str, str], list] = {}
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
        by_table.setdefault((row.kind, row.backend), []).append(
            ProfileRow(
                args=args,
                metrics=metrics,
                gpu_name=GPU_NAME,
                backend=row.backend,
                backend_version=f"imported:{row.source}",
                verified=True,
            )
        )
        counts[(row.kind, row.backend)] += 1
    if not dry_run:
        for (kind, backend), profile_rows in by_table.items():
            Table(find_kernel_profiler_spec(kind, backend), db).insert(profile_rows)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=ROOT / "profiling" / "profile.db")
    ap.add_argument("--family", action="append", choices=FAMILIES)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows: list[ImportRow] = []
    for family in a.family or FAMILIES:
        try:
            module = importlib.import_module(f"families.{family}")
        except ModuleNotFoundError as exc:
            if exc.name == f"families.{family}":
                print(f"[skip] no importer for family {family}")
                continue
            raise
        rows.extend(module.rows(a.bench_dir))
    rows = latest(rows)
    counts = insert(rows, a.db, a.dry_run)
    for (kind, backend), n in sorted(counts.items()):
        print(f"{kind:28s} {backend:24s} {n:5d}")
    print(f"{'(dry run) ' if a.dry_run else ''}{sum(counts.values())} rows -> {a.db}")


if __name__ == "__main__":
    main()
