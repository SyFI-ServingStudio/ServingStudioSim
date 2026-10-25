"""Analytic flop and byte estimates for the Neuron MoE / HC / PLE / head / launch rows.

The analyzer's headroom rungs read a row's tflops and memory_bandwidth_gbps, so a row with
neither is treated as free. These are per-call, per-rank estimates from the row's args alone
(bf16 = 2 bytes, fp32 = 4): useful math only, weights read once, activations read and written
once. `python families/costs.py --db <profile.db>` applies them to every Neuron row of these
kinds (imported or live), keeping time_ms.
"""

from __future__ import annotations

E_TOTAL, TOP_K = 512, 10


def _touched(local_experts: int, assignments: float, num_experts: int = E_TOTAL) -> float:
    """Expected distinct local experts hit by `assignments` uniform routed slots."""
    return local_experts * (1.0 - (1.0 - 1.0 / num_experts) ** assignments)


def _router(T, H, E):
    return 2.0 * T * H * E, 2.0 * (T * H + E * H) + 4.0 * T * E + 8.0 * T * TOP_K


def _routed(T, H, E, k, EL, inter):
    flops = 2.0 * T * k * 3 * H * inter
    weights = _touched(EL, T * k, E) * 3 * H * inter * 2.0
    return flops, weights + 2.0 * 2 * T * H + 8.0 * T * k


def _shared(T, H, inter):
    return 2.0 * T * 3 * H * inter, 2.0 * (3 * H * inter + H) + 2.0 * 2 * T * H


def _mix(T, H, hc, R, inject=True):
    D = hc * H
    flops = 2.0 * T * D * (2 * R + (hc if inject else 0)) + 8.0 * T * D
    return flops, 2.0 * (T * D + T * H + 2 * R * D + (hc * D if inject else 0) + D)


def _combine(T, H, hc):
    D = hc * H
    return 3.0 * T * D, 2.0 * T * (2 * D + H + hc)


def _project(T, Ed, H, hc):
    C = hc * H
    return 2.0 * T * Ed * (C + H), 2.0 * (T * Ed + Ed * (C + H) + T * (C + H))


def _gate(T, H, hc):
    C = hc * H
    return 12.0 * T * C, 2.0 * T * (C + H + C + 2 * C) + 2.0 * 3 * C


def _conv(B, q, C, K, dil):
    T, L = B * q, (K - 1) * dil
    return T * C * (2.0 * K + 6), 2.0 * (2 * T * C + 2 * B * C * L + C * K)


def _residual(T, H, hc):
    C = hc * H
    return 2.0 * T * C, 2.0 * 4 * T * C


def _ngram(T, B):
    # 16 hash heads x 5 limbs x 3 digits of int32 Horner steps plus the limb products
    return 1000.0 * T, 4.0 * (T * 17 + 2 * B)


def _gather(T, heads, dim):
    return 0.0, 2.0 * 2 * T * heads * dim + 4.0 * T * heads


def _post(T, Ed, H, hc, K, dil):
    parts = [
        _project(T, Ed, H, hc),
        _gate(T, H, hc),
        _conv(T, 1, hc * H, K, dil),
        _residual(T, H, hc),
    ]
    return sum(p[0] for p in parts), sum(p[1] for p in parts)


def cost(kind: str, a: dict) -> tuple[float, float]:
    """(flops, bytes) of one call of `kind` at args `a`; (0, 0) for a kind it does not know."""
    g = a.get
    if kind == "moe_router":
        return _router(a["num_tokens"], a["hidden_size"], a["num_experts"])
    if kind == "moe_routed_experts":
        return _routed(
            a["num_tokens"], a["hidden_size"], a["num_experts"], a["top_k"], a["local_experts"],
            a["intermediate_size"],
        )  # fmt: skip
    if kind == "moe_shared_expert":
        return _shared(a["num_tokens"], a["hidden_size"], a["intermediate_size"])
    if kind == "moe_block":
        T, H = a["num_tokens"], a["hidden_size"]
        parts = [
            _router(T, H, a["num_experts"]),
            _routed(T, H, a["num_experts"], a["top_k"], a["local_experts"], a["intermediate_size"]),
            _shared(T, H, a["shared_intermediate_size"]),
        ]
        return sum(p[0] for p in parts), sum(p[1] for p in parts) - 2 * 2.0 * T * H
    if kind in ("hc_mix", "hc_mix_final"):
        return _mix(a["num_tokens"], a["hidden_size"], a["hc_mult"], a["lowrank"], kind == "hc_mix")
    if kind == "hc_combine_mix":
        c = _combine(a["num_tokens"], a["hidden_size"], a["hc_mult"])
        m = _mix(a["num_tokens"], a["hidden_size"], a["hc_mult"], a["lowrank"])
        return c[0] + m[0], c[1] + m[1]
    if kind == "hc_combine":
        return _combine(a["num_tokens"], a["hidden_size"], a["hc_mult"])
    if kind == "grouped_rms_norm":
        D = a["hidden_size"] * a["num_groups"]
        return 4.0 * a["num_tokens"] * D, 2.0 * (2 * a["num_tokens"] * D + D)
    if kind == "ple_project":
        return _project(a["num_tokens"], a["embed_dim"], a["hidden_size"], a["hc_mult"])
    if kind == "ple_gate":
        return _gate(a["num_tokens"], a["hidden_size"], a["hc_mult"])
    if kind == "ple_residual":
        return _residual(a["num_tokens"], a["hidden_size"], a["hc_mult"])
    if kind == "ple_short_conv":
        return _conv(
            a["batch_size"], a["query_len"], a["channels"], a["kernel_size"], a["dilation"]
        )
    if kind == "ple_ngram_ids":
        return _ngram(a["num_tokens"], a["num_reqs"])
    if kind == "ple_post":
        return _post(
            a["num_tokens"], a["embed_dim"], a["hidden_size"], a["hc_mult"], a["conv_kernel_size"],
            a["dilation"],
        )  # fmt: skip
    if kind == "ple_block":
        T = a["num_tokens"]
        parts = [
            _ngram(T, T),
            _gather(T, a["num_heads"], a["head_dim"]),
            _post(T, a["num_heads"] * a["head_dim"], a["hidden_size"], a["hc_mult"], 4, 3),
        ]
        return sum(p[0] for p in parts), sum(p[1] for p in parts)
    if kind == "embedding_row_gather":
        return _gather(a["num_tokens"], a["num_heads"], a["head_dim"])
    if kind == "gemm_fp32_output":
        m, n, k = a["m"], a["n"], a["k"]
        return 2.0 * m * n * k, 2.0 * (m * k + n * k) + 4.0 * m * n
    if kind == "single_gemm":
        m, n, k = a["m"], a["n"], a["k"]
        return 2.0 * m * n * k, 2.0 * (m * k + n * k + m * n)
    if kind == "vocab_argmax":
        return float(a["num_tokens"] * a["vocab_size"]), 4.0 * a["num_tokens"] * a[
            "vocab_size"
        ] + 8.0 * a["num_tokens"]
    if kind == "lm_head_argmax":
        T, H, V = a["num_tokens"], a["hidden_size"], a["vocab_size"]
        return 2.0 * T * H * V + T * V, 2.0 * (T * H + V * H) + 8.0 * T
    if kind == "region_launch_overhead":
        n = a["num_regions"]
        return 2560.0 * n, 2.0 * 2 * 2560 * n
    if g("num_tokens") is None:
        return 0.0, 0.0
    return 0.0, 0.0


def fix_rows(db, gpu_name: str = "AWS Trainium2 LNC2", kinds=None) -> dict:
    """Recompute tflops / memory_bandwidth_gbps of every Neuron row of the known compute kinds from
    `cost` and the row's own time_ms (imported or live; the time is untouched)."""
    import sqlite3
    from dataclasses import fields

    from profiling.db.args import KernelArgs  # noqa: F401  (profiling on sys.path)
    from profiling.db.registry import find_kernel_profiler_spec

    kinds = kinds or KINDS
    con = sqlite3.connect(db)
    counts = {}
    for kind in kinds:
        try:
            backends = [
                r[0]
                for r in con.execute(
                    f'select distinct backend from "{kind}" where gpu_name=?', (gpu_name,)
                )
            ]
        except sqlite3.OperationalError:
            continue
        n = 0
        for backend in backends:
            if not backend.startswith("neuron_"):
                continue
            names = [f.name for f in fields(find_kernel_profiler_spec(kind, backend).args_schema)]
            rows = con.execute(
                f'select id, time_ms, {",".join(names)} from "{kind}" '
                "where gpu_name=? and backend=?",
                (gpu_name, backend),
            ).fetchall()
            for row in rows:
                rid, ms, vals = row[0], row[1], row[2:]
                flops, nbytes = cost(kind, dict(zip(names, vals)))
                s = ms / 1e3
                con.execute(
                    f'update "{kind}" set tflops=?, memory_bandwidth_gbps=? where id=?',
                    (flops / s / 1e12, nbytes / s / 1e9, rid),
                )
                n += 1
        counts[kind] = n
    con.commit()
    return counts


KINDS = (
    "moe_router", "moe_routed_experts", "moe_shared_expert", "moe_block", "hc_mix", "hc_mix_final",
    "hc_combine_mix", "hc_combine", "grouped_rms_norm", "ple_project", "ple_gate", "ple_residual",
    "ple_short_conv", "ple_ngram_ids", "ple_post", "ple_block", "embedding_row_gather",
    "gemm_fp32_output",
    "single_gemm", "vocab_argmax", "lm_head_argmax", "region_launch_overhead",
)  # fmt: skip


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    ap = argparse.ArgumentParser(
        description="Fill tflops / GB/s of the Neuron rows from analytic estimates."
    )
    ap.add_argument("--db", required=True)
    for kind, n in fix_rows(ap.parse_args().db).items():
        print(f"{kind:24s} {n:5d}")
