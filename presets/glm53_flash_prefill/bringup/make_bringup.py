"""Write the bring-up traces (req-frontend CSVs) beside this file.

For each prompt length L = 1,024 * 2^k, k = 0..7, one output token, no prefix (`text-generation-independent`):
  one_<L>.csv   5 requests of L tokens, 10 s apart: each runs alone on idle GPUs (4_iteration.md).
  uni_<L>.csv   max(128, 2^24 / L) requests of L tokens, all at t = 0: the saturated run (4_iteration.md).
Lengths are named 1k ... 128k. `arrival_time` is milliseconds from the start of replay.

Prefix bring-up (`text-generation-session-execution-v2`, 5_prefix_cache.md):
  prefix_sessions.csv   two-round sessions, one at a time. Round 0 is a prompt of P tokens. Round 1, 10 s after
                        round 0's token, sends it back with that token and A - 1 new tokens: P tokens hit the cache
                        and A are computed (the output token never ran through the model). For each (P, A) in
                        {8k, 32k, 128k, 512k} x {1k, 8k}, 5 sessions; a session starts every 30 s.

Tier bring-up (`text-generation-session-execution-v2`, 7_offloading.md):
  tier_sessions.csv     two-round sessions, one every 30 s, each with P tokens in round 0 and 1k in round 1 (P hit, as
                        in prefix_sessions.csv), round 1 sent 20 s after round 0's token. 6 s after the session starts,
                        five one-round fillers of 256k tokens arrive; with a KV pool of 1,234,944 tokens they push the
                        session out of HBM, so round 1 reads its P tokens back from DRAM or SSD. P in {32k, 128k, 512k},
                        5 sessions each.

Scheduling check (`text-generation-independent`, 6_scheduler.md):
  sched_check.csv       28 requests. At t = 0, longest first: 1 x 128k, 1 x 32k, 8 x 4k, 16 x 1k. Then each alone on
                        an idle pipeline: an 8k prompt at t = 5 s and a 1k prompt at t = 7 s. The scheduler's
                        microbatches, chunk by chunk, follow from its rules alone.
  force_check.csv       a 32k prompt at t = 0, then 1k prompts at 200 per second from t = 0 to t = 80 s: more than
                        the pipeline computes, so a shortest-first order keeps the 32k prompt waiting.

usage: make_bringup.py
"""
import csv
from pathlib import Path

here = Path(__file__).parent
for k in range(8):
    length, name = 1024 << k, f"{1 << k}k"
    plans = {"one": [(i, 10_000 * i) for i in range(5)],
             "uni": [(i, 0) for i in range(max(128, (1 << 24) // length))]}
    for kind, rows in plans.items():
        with open(here / f"{kind}_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "input_len", "output_len", "arrival_time"])
            w.writerows([f"{kind}_{name}_{i}", length, 1, at] for i, at in rows)

PREFIXES = [8 << 10, 32 << 10, 128 << 10, 512 << 10]
APPENDS = [1 << 10, 8 << 10]
with open(here / "prefix_sessions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["request_id", "session_id", "round_idx", "arrival_time_ms", "prefix_len", "input_len", "output_len",
                "tool_wait_after_ms"])
    session = 0
    for prefix in PREFIXES:
        for append in APPENDS:
            for _ in range(5):
                at = f"{30_000 * session:.6f}"
                w.writerow([f"session_{session}_round_000000", session, 0, at, 0, prefix, 1, "10000.000000"])
                w.writerow([f"session_{session}_round_000001", session, 1, at, prefix + 1, append - 1, 1,
                            "0.000000"])
                session += 1

with open(here / "tier_sessions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["request_id", "session_id", "round_idx", "arrival_time_ms", "prefix_len", "input_len", "output_len",
                "tool_wait_after_ms"])
    session, slot = 0, 0
    for prefix in [32 << 10, 128 << 10, 512 << 10]:
        for _ in range(5):
            at = 30_000 * slot
            w.writerow([f"session_{session}_round_000000", session, 0, f"{at:.6f}", 0, prefix, 1, "20000.000000"])
            w.writerow([f"session_{session}_round_000001", session, 1, f"{at:.6f}", prefix + 1, (1 << 10) - 1, 1,
                        "0.000000"])
            for i in range(5):
                w.writerow([f"session_{session + 1 + i}_round_000000", session + 1 + i, 0, f"{at + 6_000:.6f}", 0,
                            256 << 10, 1, "0.000000"])
            session, slot = session + 6, slot + 1

with open(here / "sched_check.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "input_len", "output_len", "arrival_time"])
    rows = [(128 << 10, 0), (32 << 10, 0)] + [(4 << 10, 0)] * 8 + [(1 << 10, 0)] * 16 + [(8 << 10, 5000), (1 << 10, 7000)]
    w.writerows([f"sched_{i}", length, 1, at] for i, (length, at) in enumerate(rows))
with open(here / "force_check.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "input_len", "output_len", "arrival_time"])
    w.writerow(["force_long", 32 << 10, 1, 0])
    w.writerows([f"force_{i}", 1 << 10, 1, f"{i * 5:.1f}"] for i in range(16_000))
