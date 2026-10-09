"""Rounds/s, TTFT percentiles and HBM prefix hit over an arrival window, from a run's request_slo.parquet.

The handoff's tables use this rule: rounds that arrive in [start, end) and complete. A real benchmark that writes
the same columns (arrival_time_ms, ttft_ms, completed, declared_prefix_tokens, prefix_cache_hit_tokens) is scored
the same way.

    uv run python doc/handoff/glm53_flash_pp5/window_metrics.py \
        logs/glm53_flash_pp5/runs_16min/c2656/raw/request_slo.parquet --gpus 5        # 16-min: [5, 15) min
    uv run python doc/handoff/glm53_flash_pp5/window_metrics.py \
        logs/glm53_flash_pp5/runs_8h/c2656/raw/request_slo.parquet --gpus 5 --start-min 60 --end-min 480

prefix_cache_hit_tokens counts tokens restored from any tier, so "prefix hit" is the HBM hit only in the HBM-only
preset. With DRAM and SSD on it is about 100%; take the per-tier split from the run's prefix_tiers_w0.json.
"""

import argparse

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("request_slo")
    ap.add_argument("--gpus", type=int, default=5)
    ap.add_argument("--start-min", type=float, default=5.0)
    ap.add_argument("--end-min", type=float, default=15.0)
    a = ap.parse_args()

    cols = ["arrival_time_ms", "ttft_ms", "completed", "declared_prefix_tokens", "prefix_cache_hit_tokens"]
    t = pq.read_table(a.request_slo, columns=cols).to_pydict()
    arr = np.asarray(t["arrival_time_ms"], dtype=float)
    done = np.asarray(t["completed"], dtype=bool)
    lo, hi = a.start_min * 60_000, a.end_min * 60_000
    m = done & (arr >= lo) & (arr < hi)
    ttft = np.asarray([x for x, k in zip(t["ttft_ms"], m) if k], dtype=float) / 1000
    declared = sum(x for x, k in zip(t["declared_prefix_tokens"], m) if k)
    hit = sum(x for x, k in zip(t["prefix_cache_hit_tokens"], m) if k)

    rate = m.sum() / ((hi - lo) / 1000)
    p50, p90, p99 = np.percentile(ttft, [50, 90, 99])
    print(f"rounds {m.sum()}  rounds/s {rate:.1f} ({rate / a.gpus:.1f} per GPU)")
    print(f"TTFT p50 / p90 / p99 (s) {p50:.2f} / {p90:.2f} / {p99:.2f}")
    print(f"prefix hit {hit / declared:.1%} of declared prefix tokens" if declared else "no declared prefix tokens")


if __name__ == "__main__":
    main()
