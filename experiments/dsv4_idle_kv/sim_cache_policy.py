#!/usr/bin/env python3
"""Session-level KV policy sim for DeepSeek-V4-Flash-shaped cache.

This is *not* the Rust L4 engine. It answers the same scheduling questions as
last week: keep vs always-handoff vs JIT, and migrate-on-idle across GPUs,
using V4 MLA KV footprint (584 bytes/token × 60 layers) and real-text token_ids.

Prefill cost is a linear model in evicted tokens (re-prefill). Decode cost is
linear in output tokens. Numbers are internally consistent A/B, not claimed
as vLLM wall times until the real-GPU cell is run.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# vLLM DeepseekV4 fp8_ds_mla: 448B NoPE + 128B RoPE + 8B scale = 584B / token
KV_BYTES_PER_TOKEN = 584
N_LAYERS = 60  # treat 584B as per-layer MLA page; matches scarce-KV cells
PREFILL_US_PER_TOKEN = 8.0
DECODE_US_PER_TOKEN = 40.0
HBM_BYTES = int(os.environ.get("HBM_GB", "141")) * 1024**3


def kv_bytes(n_tokens: int) -> int:
    return n_tokens * KV_BYTES_PER_TOKEN * N_LAYERS


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def hbm_budget(mem_util: float, n_gpus: int) -> list[int]:
    cap = int(HBM_BYTES * mem_util)
    return [cap] * n_gpus


def run_idle(rows: list[dict], policy: str, mem_util: float, n_gpus: int, jit_ms: int, migrate: bool) -> dict:
    by_sess: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sess[row["session_id"]].append(row)
    for sid in by_sess:
        by_sess[sid].sort(key=lambda r: r["round"])

    gpu_used = [0] * n_gpus
    caps = hbm_budget(mem_util, n_gpus)
    resident: dict[str, bool] = {}
    place: dict[str, int] = {}
    ttfts: list[float] = []
    e2es: list[float] = []
    migrations = 0
    re_prefills = 0
    keep_hits = 0

    for sid, rounds in by_sess.items():
        gpu = int(rounds[0].get("gpu_hint", 0)) % n_gpus
        place[sid] = gpu
        nbytes = kv_bytes(len(rounds[0]["token_ids"]))
        if gpu_used[gpu] + nbytes <= caps[gpu]:
            resident[sid] = True
            gpu_used[gpu] += nbytes
        else:
            resident[sid] = False

    def maybe_migrate(sid: str, wait_ms: int) -> None:
        nonlocal migrations
        if not migrate or wait_ms < jit_ms:
            return
        src = place[sid]
        dst = min(range(n_gpus), key=lambda g: gpu_used[g])
        if dst == src:
            return
        nbytes = kv_bytes(len(by_sess[sid][0]["token_ids"]))
        if resident[sid]:
            gpu_used[src] -= nbytes
        place[sid] = dst
        if resident[sid]:
            gpu_used[dst] += nbytes
        migrations += 1

    for round_i in range(max(len(v) for v in by_sess.values())):
        for sid, rounds in by_sess.items():
            if round_i >= len(rounds):
                continue
            row = rounds[round_i]
            ntok = len(row["token_ids"])
            wait = int(row["wait_ms"])
            gpu = place[sid]
            nbytes = kv_bytes(ntok)

            evict = False
            if policy == "always_handoff":
                evict = True
            elif policy == "jit":
                evict = wait >= jit_ms
            if evict and resident[sid]:
                gpu_used[gpu] = max(0, gpu_used[gpu] - nbytes)
                resident[sid] = False

            maybe_migrate(sid, wait)

            gpu = place[sid]
            cap = caps[gpu]
            nbytes = kv_bytes(ntok)
            # pressure eviction of non-active residents
            if gpu_used[gpu] + (0 if resident[sid] else nbytes) > cap:
                resident[sid] = False

            if resident[sid]:
                prefill_tok = 0
                keep_hits += 1
            else:
                prefill_tok = ntok
                re_prefills += 1
                gpu_used[gpu] += nbytes
                resident[sid] = True

            ttft_ms = prefill_tok * PREFILL_US_PER_TOKEN / 1000.0
            e2e_ms = ttft_ms + row["output_tokens"] * DECODE_US_PER_TOKEN / 1000.0
            ttfts.append(ttft_ms)
            e2es.append(e2e_ms)

    def pct(xs: list[float]) -> float:
        return statistics.median(xs) if xs else 0.0

    return {
        "policy": policy,
        "migrate": migrate,
        "mem_util": mem_util,
        "n_gpus": n_gpus,
        "sessions": len(by_sess),
        "ttft_p50_ms": round(pct(ttfts), 2),
        "e2e_p50_ms": round(pct(e2es), 2),
        "re_prefills": re_prefills,
        "keep_hits": keep_hits,
        "migrations": migrations,
        "hit_rate": round(keep_hits / max(1, keep_hits + re_prefills), 3),
    }


def fork_colocation(rows: list[dict], policy: str) -> dict:
    stems: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        stems[row["stem_id"]].append(row["session_id"])
    n_gpus = 2
    load = [0, 0]
    coloc = 0
    for stem, sids in stems.items():
        sids = sorted(set(sids))
        if policy == "stem_binpack":
            gpu = 0 if load[0] <= load[1] else 1
            for _ in sids:
                load[gpu] += 1
            coloc += 1
        else:
            # scatter: round-robin forks
            for i, _ in enumerate(sids):
                load[i % n_gpus] += 1
    return {
        "policy": policy,
        "stems": len(stems),
        "stems_colocated": coloc if policy == "stem_binpack" else 0,
        "gpu_load": load,
    }


def md_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="all", choices=["all", "idle_kv", "wait_sweep", "membound", "fork"])
    ap.add_argument("--jit-ms", type=int, default=2500)
    args = ap.parse_args()
    wl = HERE / "workloads"
    if not (wl / "idle_24x8.jsonl").exists():
        raise SystemExit("run gen_realtext_workload.py first")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    chunks: list[str] = ["# DSV4 idle-KV policy sim (session-level, V4 KV bytes)\n"]

    if args.cell in ("all", "idle_kv"):
        rows = load_rows(wl / "idle_24x8.jsonl")
        tab = [
            run_idle(rows, p, 0.35, 1, args.jit_ms, False)
            for p in ("keep", "always_handoff", "jit")
        ]
        chunks.append("## idle_kv (24×8, mem 0.35, 1 GPU)\n\n" + md_table(tab))

    if args.cell in ("all", "wait_sweep"):
        tab = []
        for wait in (1000, 4000, 8000, 30000):
            # rebuild waits
            raw = load_rows(wl / "idle_24x8.jsonl")
            for r in raw:
                if r["wait_ms"] >= 1000:
                    r["wait_ms"] = wait
            for migrate in (False, True):
                rec = run_idle(raw, "jit", 0.75, 2, args.jit_ms, migrate)
                rec["wait_h_ms"] = wait
                tab.append(rec)
        chunks.append("## wait_sweep (skew via gpu_hint, 2 GPU)\n\n" + md_table(tab))

    if args.cell in ("all", "membound"):
        rows = load_rows(wl / "idle_64x8.jsonl")
        tab = [
            run_idle(rows, "keep", 0.25, 2, args.jit_ms, False),
            run_idle(rows, "jit", 0.25, 2, args.jit_ms, True),
        ]
        chunks.append("## membound (64×8, mem 0.25, 2 GPU)\n\n" + md_table(tab))

    if args.cell in ("all", "fork"):
        rows = load_rows(wl / "fork_s8_f4.jsonl")
        tab = [fork_colocation(rows, p) for p in ("scatter_stems", "stem_binpack")]
        chunks.append("## fork colocation\n\n" + md_table(tab))

    report = "\n".join(chunks)
    (out_dir / "POLICY_SIM.md").write_text(report)
    print(report)
    print(f"wrote {out_dir / 'POLICY_SIM.md'}")


if __name__ == "__main__":
    main()
