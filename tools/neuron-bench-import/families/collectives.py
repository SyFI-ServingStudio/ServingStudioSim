"""In-graph collective rows on Trainium2 LNC2 cores, from the trn2-flashnext branch.

``data/trn2_flashnext_collectives.jsonl`` is the branch's profile.db ``neuron_cc`` rows verbatim
(``flashnext/neuron/sim_l1/collectives.py`` v2 sweep: device-time slope of 33 vs 1 chained
collectives in one region, barrier before every call, median of 3 repeats). Groups are contiguous
logical cores: fabric ``neuron_c4`` = every core of each chip (4 to 32 ranks), ``neuron_c2`` = two
cores of one chip, ``neuron_c1`` = one core per chip. The GLM engine uses the same runtime and
the same in-graph functional collectives, so the rows transfer as they are.
"""

from __future__ import annotations

from pathlib import Path

from import_bench import ImportRow, jsonl

OPS = ("all_gather", "reduce_scatter")  # all_reduce rows exist but the engine uses none


def bus_bytes(op: str, n: int, nbytes: int) -> float:
    factor = {
        "all_reduce": 2.0 * (n - 1) / n,
        "all_gather": float(n - 1),
        "reduce_scatter": (n - 1) / n,
    }[op]
    return factor * nbytes


def rows(path: Path):
    for line, r in jsonl(path):
        if r.get("op") not in OPS or not float(r.get("time_ms") or 0) > 0:
            continue
        args = dict(
            num_gpus=int(r["num_gpus"]),
            message_size_bytes=int(r["message_size_bytes"]),
            dtype=r.get("dtype", "bf16"),
            fabric=r["fabric"],
        )
        nb = bus_bytes(r["op"], args["num_gpus"], args["message_size_bytes"])
        yield ImportRow(
            r["op"], "neuron_cc", args, float(r["time_ms"]), "imported:trn2-flashnext",
            bytes=nb, comm=True,
        )
