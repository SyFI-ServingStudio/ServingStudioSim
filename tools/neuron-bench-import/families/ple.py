"""PLE rows: logs/bench/ple.v2.jsonl (device time). ple.jsonl is wall clock only: never imported.

bench_ple.py times decode rows as [B, 1] (B sequences x 1 token) and prefill rows as [1, T]. The
Sim's num_tokens kinds take one layout per token count, the same rule as the
`flashnext.neuron.sim_l1.ple` builders: num_tokens <= 256 is a decode step (bench phase
"decode"), larger num_tokens one prefill chunk (phase "prefill"). ple_short_conv and
ple_ngram_ids carry the request layout in their args, so both phases map.
"""

from __future__ import annotations

from pathlib import Path

from import_bench import ImportRow, device_ms, jsonl

from families.costs import cost

H, HC, HEADS, HEAD_DIM = 2560, 4, 16, 160
TABLE_ROWS = 320001536
DECODE_MAX = 256
CHANNELS, CONV_K, DILATION = HC * H, 4, 3


def layout_ok(phase: str, tokens: int) -> bool:
    """The bench row whose layout the builders use for this token count."""
    return phase == ("decode" if tokens <= DECODE_MAX else "prefill")


def map_row(r: dict) -> tuple[str, dict] | None:
    kind, phase = r.get("kind"), r.get("phase")
    if r.get("tokens") is None or phase not in ("decode", "prefill") or r.get("impl") != "torch":
        return None
    T, B = int(r["tokens"]), int(r.get("batch") or 1)
    if kind == "ple_ngram_ids":
        reqs = B if phase == "decode" else 1
        return "ple_ngram_ids", dict(
            num_tokens=T, num_reqs=reqs, ngram_context_len=2, heads_per_ngram=8
        )
    if kind == "ple_short_conv":
        batch, q = (B, 1) if phase == "decode" else (1, T)
        return "ple_short_conv", dict(
            batch_size=batch, query_len=q, channels=CHANNELS, kernel_size=CONV_K,
            dilation=DILATION, mode=phase, dtype="bf16",
        )  # fmt: skip
    if not layout_ok(phase, T):
        return None
    if kind == "ple_project":
        return "ple_project", dict(
            num_tokens=T, embed_dim=HEADS * HEAD_DIM, hidden_size=H, hc_mult=HC, dtype="bf16"
        )
    if kind == "ple_gate":
        return "ple_gate", dict(num_tokens=T, hidden_size=H, hc_mult=HC, dtype="bf16")
    if kind == "ple_residual":
        return "ple_residual", dict(num_tokens=T, hidden_size=H, hc_mult=HC, dtype="bf16")
    if kind == "ple_post":
        return "ple_post", dict(
            num_tokens=T, embed_dim=HEADS * HEAD_DIM, hidden_size=H, hc_mult=HC,
            conv_kernel_size=CONV_K, dilation=DILATION, dtype="bf16",
        )  # fmt: skip
    tp = int(r.get("tp") or 1)
    rows = int(r.get("table_rows") or TABLE_ROWS // tp)
    if kind == "ple_gather":
        return "embedding_row_gather", dict(
            num_tokens=T, num_heads=HEADS, head_dim=HEAD_DIM, table_rows=rows,
            residency="device", weight_dtype="bf16",
        )  # fmt: skip
    if kind == "ple_block":
        return "ple_block", dict(
            num_tokens=T, num_heads=HEADS, head_dim=HEAD_DIM, table_rows=rows,
            hidden_size=H, hc_mult=HC, dtype="bf16",
        )  # fmt: skip
    return None


def rows(bench_dir: Path):
    name = "ple.v2.jsonl"
    for line, r in jsonl(bench_dir / name):
        ms = device_ms(r)
        mapped = map_row(r) if ms is not None else None
        if mapped is None:
            continue
        table, args = mapped
        yield ImportRow(table, "neuron_torch", args, ms, f"{name}:{line}", *cost(table, args))
