"""Hyper-connection rows: logs/bench/hc.v2.jsonl (device time; hc.jsonl is wall-clock only).

HC weights are replicated on every rank, so a row's per-rank cost depends on its token count
only; decode (tokens == batch) and prefill rows of the same token count are the same shape.
"""

from __future__ import annotations

from pathlib import Path

from import_bench import ImportRow, device_ms, jsonl

from families.costs import cost

HIDDEN, HC, LOWRANK = 2560, 4, 320
MIX = dict(hidden_size=HIDDEN, hc_mult=HC, lowrank=LOWRANK, dtype="bf16")
KINDS = {
    # bench kind -> (table, args builder)
    "hc_grouped_norm": (
        "grouped_rms_norm",
        lambda t: dict(num_tokens=t, hidden_size=HIDDEN, num_groups=HC, hidden_dtype="bf16"),
    ),
    "hc_mix": ("hc_mix", lambda t: dict(num_tokens=t, **MIX)),
    "hc_mix_final": ("hc_mix_final", lambda t: dict(num_tokens=t, **MIX)),
    "hc_combine": (
        "hc_combine",
        lambda t: dict(num_tokens=t, hidden_size=HIDDEN, hc_mult=HC, hidden_dtype="bf16"),
    ),
    "hc_combine_mix": ("hc_combine_mix", lambda t: dict(num_tokens=t, **MIX)),
}


def rows(bench_dir: Path):
    name = "hc.v2.jsonl"
    for line, r in jsonl(bench_dir / name):
        ms = device_ms(r)
        if ms is None or r.get("kind") not in KINDS or r.get("tp", 1) != 1:
            continue
        table, args = KINDS[r["kind"]]
        a = args(int(r["tokens"]))
        yield ImportRow(table, f"neuron_{r['impl']}", a, ms, f"{name}:{line}", *cost(table, a))
