"""Benchmark metrics for one 20-minute run, the same for a real run and a simulator run.

Rounds count when their first token comes inside the window [0, 20 min), with t = 0 at the first
request. Rounds still waiting at the end are not in the TTFT percentiles: the real client is
stopped at 20 minutes and never logs them. New tokens are the trace's `input_len` of the counted
rounds: the prompt tokens the workload adds, independent of what any cache holds. Computed tokens
are the prompt tokens not taken from the prefix cache, and the prefix hit share is the cached
tokens over the trace's `prefix_len`. `rounds_ttft_over_60s` counts the counted rounds that
waited more than 60 s. The real server must report its cached tokens (vLLM:
`--enable-prompt-tokens-details`).

usage:
  metrics.py <requests.jsonl>      req-frontend's per-request log (run under `timeout 20m`)
  metrics.py <sim run dir>         a simulator run (raw/request_slo.parquet)
  [--minutes 20]
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--minutes", type=float, default=20.0)
a = ap.parse_args()
W = a.minutes * 60e3


def real(path):
    arr, first, prompt, new, prefix, cached = [], [], [], [], [], []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        o = r.get("outcome", r)
        if o.get("first_token_id_ms") is None or o.get("post_timestamp") is None or str(o.get("status")).upper() != "SUCCESS":
            continue
        usage = o.get("server_usage") or {}
        arr.append(o["submit_timestamp"] * 1e3)
        first.append(o["post_timestamp"] * 1e3 + o["first_token_id_ms"])
        prompt.append(usage.get("prompt_tokens") or 0)
        cached.append(usage.get("cached_prompt_tokens") or 0)
        new.append(r["source"]["data"]["input_len"])
        prefix.append(r["source"]["data"].get("prefix_len") or 0)
    t0 = min(r.get("outcome", r)["submit_timestamp"] for r in map(json.loads, path.read_text().splitlines())) * 1e3
    return (np.array(arr) - t0, np.array(first) - t0, np.array(prompt), np.array(new), np.array(prefix),
            np.array(cached))


def sim(run):
    import pyarrow.parquet as pq
    t = pq.read_table(run / "raw/request_slo.parquet",
                      columns=["arrival_time_ms", "completed", "ttft_ms", "declared_prefix_tokens", "fresh_prompt_tokens",
                               "prefix_cache_hit_tokens"])
    c = t.column("completed").to_numpy(zero_copy_only=False)
    arr = t.column("arrival_time_ms").to_numpy()[c]
    ttft = t.column("ttft_ms").to_numpy(zero_copy_only=False)[c].astype(float)
    new = t.column("fresh_prompt_tokens").to_numpy()[c]
    prefix = t.column("declared_prefix_tokens").to_numpy()[c]
    cached = np.nan_to_num(t.column("prefix_cache_hit_tokens").to_numpy(zero_copy_only=False)[c].astype(float))
    return arr, arr + ttft, prefix + new, new, prefix, cached


arr, first, prompt, new, prefix, cached = real(a.run) if a.run.is_file() else sim(a.run)
m = first < W
ttft = (first - arr)[m] / 1e3
out = {"rounds": int(m.sum()), "rounds_per_s": m.sum() / (W / 1e3), "new_tokens_per_s": float(new[m].sum() / (W / 1e3)),
       "prompt_tokens_per_s": float(prompt[m].sum() / (W / 1e3)),
       "computed_tokens_per_s": float((prompt[m] - cached[m]).sum() / (W / 1e3)),
       "prefix_hit_share": float(cached[m].sum() / max(prefix[m].sum(), 1)),
       "ttft_s": {**{p: float(np.percentile(ttft, q)) for p, q in (("p50", 50), ("p90", 90), ("p99", 99))},
                  "max": float(ttft.max())},
       "rounds_ttft_over_60s": int((ttft > 60).sum())}
print(json.dumps(out))
