"""Embedding / LM head / argmax rows: logs/bench/head.v2.jsonl (device time; head.jsonl is wall).

Vocab parallel: the bench row's tp gives the rank's vocab rows 248320 / tp. Head ops take [N, ...]
rows, so decode and prefill rows of one token count are the same shape (the later line wins).
"""

from __future__ import annotations

from pathlib import Path

from import_bench import ImportRow, device_ms, jsonl

from families.costs import cost

V, H = 248320, 2560


def map_row(r: dict) -> tuple[str, str, dict] | None:
    kind, impl, T = r.get("kind"), r.get("impl"), r.get("tokens")
    tp = int(r.get("tp") or 1)
    if T is None or V % tp:
        return None
    T, vl = int(T), V // tp
    if kind == "embed" and impl == "torch":
        args = dict(
            num_tokens=T,
            num_heads=1,
            head_dim=H,
            table_rows=vl,
            residency="device",
            weight_dtype="bf16",
        )
        return "embedding_row_gather", "neuron_torch", args
    if kind == "lm_head" and impl == "torch":
        return "gemm_fp32_output", "neuron_torch", dict(m=T, n=vl, k=H, input_dtype="bf16")
    if kind == "argmax_local" and impl in ("torch", "nki"):
        return "vocab_argmax", f"neuron_{impl}", dict(num_tokens=T, vocab_size=vl, dtype="bf16")
    if kind == "lm_head_argmax" and impl in ("torch", "nki"):
        args = dict(num_tokens=T, hidden_size=H, vocab_size=vl, dtype="bf16")
        return "lm_head_argmax", f"neuron_{impl}", args
    return None


def rows(bench_dir: Path):
    name = "head.v2.jsonl"
    for line, r in jsonl(bench_dir / name):
        ms = device_ms(r)
        mapped = map_row(r) if ms is not None else None
        if mapped is None:
            continue
        table, backend, args = mapped
        yield ImportRow(table, backend, args, ms, f"{name}:{line}", *cost(table, args))
