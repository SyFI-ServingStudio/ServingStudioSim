"""Runs on coriander (stdlib, python3.9): every SGLang scheduler iteration of one arm, compacted.

    python3 extract_engine_iters.py <arm_dir> | gzip > iters.csv.gz

One row per sglang_metrics record: rank, timestamp (Unix s, taken just after the
forward is launched), forward_mode, running_batch_size, waiting_queue_size,
kv_tokens_used, prefill_tokens, decode_tokens, and the prefill chunk pairs
summed (sum of first / second element of each [a, b] pair).
"""
import glob, json, os, re, sys

arm = sys.argv[1]
w = sys.stdout.write
w("rank,ts,mode,rbs,waiting,kv_used,prefill_tokens,decode_tokens,n_pairs,pair_a,pair_b\n")
for path in sorted(glob.glob(os.path.join(arm, "sglang_metrics", "sglang_metrics_rank_*.jsonl"))):
    rank = int(re.search(r"rank_(\d+)_", path).group(1))
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            pairs = d.get("prefill_chunk_pairs") or []
            a = sum(p[0] for p in pairs if isinstance(p, list) and len(p) == 2)
            b = sum(p[1] for p in pairs if isinstance(p, list) and len(p) == 2)
            w("%d,%.6f,%s,%d,%d,%d,%d,%d,%d,%d,%d\n" % (
                rank, d["timestamp"], d.get("forward_mode", "")[:1], d.get("running_batch_size", 0),
                d.get("waiting_queue_size", 0), d.get("kv_tokens_used", 0),
                d.get("prefill_tokens", 0) or 0, d.get("decode_tokens", 0) or 0, len(pairs), a, b))
