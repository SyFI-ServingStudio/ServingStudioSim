"""In-graph collective rows: logs/bench/collectives_ingraph.jsonl (scripts/probe_collectives.py)
and collectives_ingraph_v2.jsonl (flashnext/neuron/sim_l1/collectives.py: barrier before every
call, median of 3 repeats). Time = `per_cc_device_ms`, the device-time slope of 33 vs 1 chained
collectives; rows with an error or a non-positive slope (launch-skew noise) are dropped. v2 is
read after v1, so a clean v2 row replaces the v1 row of the same key.

Fabric from the cores (4 logical cores per chip): ranks per chip 4 -> neuron_c4 (one chip with 4
ranks, or every core of 2-8 chips), 2 -> neuron_c2, 1 -> neuron_c1 (one rank per chip).
`bytes` on the row is the per-rank traffic used for busbw (nccl-tests factors).
"""

from __future__ import annotations

from pathlib import Path

from import_bench import ImportRow, jsonl

OPS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
FILES = ("collectives_ingraph.jsonl", "collectives_ingraph_v2.jsonl")
CORES_PER_CHIP = 4


def fabric(cores) -> str | None:
    per: dict[int, int] = {}
    for c in cores:
        per[c // CORES_PER_CHIP] = per.get(c // CORES_PER_CHIP, 0) + 1
    counts = set(per.values())
    if len(counts) != 1:
        return None
    return {1: "neuron_c1", 2: "neuron_c2", 4: "neuron_c4"}.get(counts.pop())


def bus_bytes(op: str, world: int, nbytes: int) -> float:
    n = world
    factor = {
        "all_reduce": 2.0 * (n - 1) / n,
        "all_gather": float(n - 1),
        "reduce_scatter": (n - 1) / n,
        "all_to_all": (n - 1) / n,
    }[op]
    return factor * nbytes


def map_row(r: dict) -> tuple[str, dict, float] | None:
    """(table, args, ms) for one clean row, else None."""
    if r.get("error") or r.get("op") not in OPS:
        return None
    ms = r.get("per_cc_device_ms")
    if ms is None or not float(ms) > 0:
        return None
    cores = r.get("cores") or []
    fab = fabric(cores)
    if fab is None or len(cores) != r.get("world") or len(cores) < 2:
        return None
    args = dict(num_gpus=len(cores), message_size_bytes=int(r["bytes"]), dtype="bf16", fabric=fab)
    return r["op"], args, float(ms)


def rows(bench_dir: Path):
    for name in FILES:
        for line, r in jsonl(bench_dir / name):
            mapped = map_row(r)
            if mapped is None:
                continue
            table, args, ms = mapped
            nb = bus_bytes(table, args["num_gpus"], args["message_size_bytes"])
            yield ImportRow(table, "neuron_cc", args, ms, f"{name}:{line}", bytes=nb, comm=True)
