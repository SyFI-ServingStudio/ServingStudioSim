"""Launch-overhead rows: logs/probe_overhead.json (scripts/probe_overhead.py; WALL clock).

region_launch_overhead is the one kind whose time is wall clock: the pipelined wall per step of
num_regions back-to-back compiled regions. trivial_add at tokens 1 is one trivial region
(num_regions 1); chain_k_regions at tokens 1, k 8 is eight separately compiled 2560 x 2560
linears launched back to back (num_regions 8; it includes their ~0.3 ms of compute, unlike the
live probe's trivial regions). The file sits next to the bench dir (`<bench-dir>/..`).
"""

from __future__ import annotations

import json
from pathlib import Path

from import_bench import ImportRow

from families.costs import cost


def map_case(c: dict) -> int | None:
    """num_regions for one probe case, or None."""
    if c.get("tokens") != 1:
        return None
    if c.get("case") == "trivial_add":
        return 1
    if c.get("case") == "chain_k_regions" and c.get("k") == 8:
        return 8
    return None


def rows(bench_dir: Path):
    path = bench_dir.parent / "probe_overhead.json"
    if not path.exists():
        return
    for i, c in enumerate(json.loads(path.read_text())):
        n = map_case(c)
        ms = c.get("pipelined_ms")
        if n is None or not ms or ms <= 0:
            continue
        args = dict(num_regions=n, dtype="bf16")
        table = "region_launch_overhead"
        yield ImportRow(
            table, "neuron_torch", args, float(ms), f"{path.name}:{i}", *cost(table, args)
        )
