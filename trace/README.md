# Trace generators and samples

VibeSim and TraceLab's `vibesim` frontend consume the same independent-request
CSV schema:

```text
id,input_len,output_len,arrival_time
```

`arrival_time` is measured in milliseconds. Generate a deterministic fixed-shape
capacity workload with:

```bash
uv run python trace/generate_fixed_shape.py "$TMPDIR/fixed.csv" \
  --requests 1024 --input-len 256 --output-len 256 --interarrival-ms 0
```

Zero interarrival time is an intentional burst. Use a positive spacing when the
experiment needs a controlled offered rate rather than immediate saturation.

## Session-wise traces

Multi-round coding-agent traces carry a wider schema, one row per round:

```text
request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,tool_wait_after_ms
```

`trace/tracelab_preserving.csv` is the canonical one (4,281 sessions / 357,161
rounds, materialized from the public syfi dataset with prefixes preserved). It is
a TraceLab output, deterministic for a given seed, so it is **not tracked** — see
`alignment/load_generator/tracelab/artifacts/trace_facts/csv_export/` to
regenerate rather than expecting it in a fresh clone.
