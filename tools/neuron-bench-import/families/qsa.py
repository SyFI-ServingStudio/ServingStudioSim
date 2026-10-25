"""QSA rows: logs/bench/qsa.jsonl (bench_qsa.py; device_ms = runtime.device_bench).

Per-rank geometry by the row's tp (L1_CONTRACT.md): num_q_heads = 24 / tp, num_kv_heads = 2 at tp
1, else 1. Every QSA kind takes `mode` = the row's phase and `num_rows` = its tokens (prefill: the
chunk; decode: the requests); a context kind takes `context_len` = the row's context (prefill:
prior + tokens; decode: the KV length bucket).

- `qsa_mixer2_{prefill,decode}` rows with `select_impl == "nki"` -> `qsa_mixer2` (two regions),
  but only when the row's device trace told the two NEFFs apart (`device_neffs == regions`). The
  lite runtime's trace leaves every NEFF's `model_name` unset, so `runtime.device_bench` lumped
  front and back into one "NEFF" (`device_neffs == 1`) and its `device_ms` is the median over
  both regions' executions, not their per-call sum. Every mixer2 row of the 2026-10-25 bench is
  such a row, so none is imported; the profiler sums the regions per call itself. Single-region
  `qsa_mixer_*` rows have no kind.
- Row-only kinds (project, indexer_project, cache_write, indexer_compress, output) carry no
  context arg; the profiler builds them at the bench's context for them (prefill prior 0, decode
  4096), so only those rows map.
- Rows with an error or without a positive device_ms (bench skips) are dropped.
"""

from __future__ import annotations

from pathlib import Path

from import_bench import ImportRow, device_ms, jsonl

SOURCE = "qsa.jsonl"
VALID_TP = (1, 2, 4, 8)
ROW_ONLY_DECODE_CONTEXT = 4096

# bench kind -> (table, impls the table registers, has context_len)
KINDS = {
    "qsa_mixer2_prefill": ("qsa_mixer2", ("torch_sparse", "nki"), True),
    "qsa_mixer2_decode": ("qsa_mixer2", ("torch_sparse", "nki"), True),
    "qsa_project": ("qsa_project", ("torch",), False),
    "indexer_project": ("qsa_indexer_project", ("torch",), False),
    "qsa_cache_write": ("qsa_cache_write", ("torch",), False),
    "indexer_compress": ("qsa_indexer_compress", ("torch",), False),
    "indexer_scores": ("qsa_indexer_scores", ("torch", "torch_bf16"), True),
    "indexer_select": ("qsa_indexer_select", ("torch", "nki"), True),
    "qsa_attention": ("qsa_attention", ("torch_dense", "torch_sparse", "nki"), True),
    "qsa_output": ("qsa_output", ("torch",), False),
}


def geometry(tp: int) -> dict:
    """The common QSA args of a rank at TP size tp."""
    if tp not in VALID_TP:
        raise ValueError(f"tp {tp} does not split 24 query heads")
    return dict(
        hidden_size=2560,
        num_q_heads=24 // tp,
        num_kv_heads=2 if tp == 1 else 1,
        head_dim=256,
        index_heads=4,
        index_head_dim=128,
        index_topk_tokens=2048,
        compress_ratio=4,
        dtype="bf16",
    )


# Analytic per-rank cost. flashnext/neuron/sim_l1/qsa.py sets Region.flops/bytes from the
# same formulas.
SELECTED_GROUPS = 513  # top 512 groups + the open group
SELECTED_SLOTS = SELECTED_GROUPS * 4  # token slots per query row


def _context(a: dict) -> int:
    if "context_len" in a:
        return a["context_len"]
    return a["num_rows"] if a["mode"] == "prefill" else ROW_ONLY_DECODE_CONTEXT


def attended(a: dict) -> int:
    """Tokens all query rows attend to: each row's causal context, capped at the selection."""
    n, C = a["num_rows"], _context(a)
    cap = a["index_topk_tokens"] + a["compress_ratio"]
    if a["mode"] == "decode":
        return n * min(C, cap)
    prior = C - n
    full = max(0, min(n, cap - prior))  # rows below the cap see prior + i + 1 tokens
    return full * (prior + 1) + full * (full - 1) // 2 + (n - full) * cap


def cost(table: str, a: dict) -> tuple[float, float]:
    """(flops, bytes) of one call of `table` at args `a`: minimum logical work and HBM traffic
    (bf16 = 2 B; fp32 scores = 4 B). The two-region mixer is the sum of its ops."""
    n, H, hq, hkv, D = (
        a[k] for k in ("num_rows", "hidden_size", "num_q_heads", "num_kv_heads", "head_dim")
    )
    hi, di, r = a["index_heads"], a["index_head_dim"], a["compress_ratio"]
    C = _context(a)
    reqs = 1 if a["mode"] == "prefill" else n
    groups = C // r  # compressed keys per request context

    def gemm(m, k, out):
        return 2.0 * m * k * out, 2.0 * (m * k + k * out + m * out)

    def project():
        return gemm(n, H, hq * 2 * D + 2 * hkv * D)

    def indexer_project():
        return gemm(n, H, (hi + 1) * di)

    def cache_write():
        return 0.0, 2.0 * 2 * n * (2 * hkv * D + di)

    def compress():
        g = n // r + 1 if a["mode"] == "prefill" else n
        return g * di * (r + 10.0), 2.0 * g * di * (r + 1)

    def scores():
        return 2.0 * n * hi * di * groups, 2.0 * (
            reqs * groups * di + n * hi * di
        ) + 4.0 * n * groups

    def select():
        return float(n * groups), 4.0 * n * groups + n * (8.0 * SELECTED_GROUPS + SELECTED_SLOTS)

    def attention():
        s = attended(a)
        return 4.0 * hq * D * s, 2.0 * 2 * hkv * D * s + 2.0 * 2 * n * hq * D

    def output():
        return gemm(n, hq * D, H)

    ops = dict(
        qsa_project=project,
        qsa_indexer_project=indexer_project,
        qsa_cache_write=cache_write,
        qsa_indexer_compress=compress,
        qsa_indexer_scores=scores,
        qsa_indexer_select=select,
        qsa_attention=attention,
        qsa_output=output,
    )
    if table == "qsa_mixer2":
        parts = [f() for f in ops.values()]
        return sum(p[0] for p in parts), sum(p[1] for p in parts)
    return ops[table]()


def shape_args(r: dict) -> tuple[str, dict] | None:
    """(table, mode + shape args) of one bench row, or None when the row maps to no Neuron table."""
    entry = KINDS.get(r.get("kind"))
    if entry is None or r.get("impl") not in entry[1]:
        return None
    table, _, has_context = entry
    if table == "qsa_mixer2" and (
        r.get("select_impl") != "nki" or r.get("device_neffs") != r.get("regions", 2)
    ):
        return None
    mode, rows_, context = r.get("phase"), int(r["tokens"]), int(r["context"])
    if mode not in ("prefill", "decode"):
        return None
    args = dict(mode=mode, num_rows=rows_)
    if has_context:
        args["context_len"] = context
    elif context != (rows_ if mode == "prefill" else ROW_ONLY_DECODE_CONTEXT):
        return None
    return table, args


def rows(bench_dir: Path):
    for line, r in jsonl(bench_dir / SOURCE):
        ms = device_ms(r)
        if ms is None or r.get("tp", 1) not in VALID_TP:
            continue
        mapped = shape_args(r)
        if mapped is None:
            continue
        table, args = mapped
        args = args | geometry(int(r.get("tp", 1)))
        flops, nbytes = cost(table, args)
        yield ImportRow(table, f"neuron_{r['impl']}", args, ms, f"{SOURCE}:{line}", flops, nbytes)
