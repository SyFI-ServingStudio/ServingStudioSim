# Trace generators and samples

VibeSim and req-frontend's `independent` frontend consume the same
independent-request CSV schema:

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
rounds, materialized from the public syfi dataset under the `monotonic` context
policy — the file name predates that policy's rename from `prefix-preserving`);
`trace/tracelab_reported.csv` is its counterpart under `trace-reported`.

Both are ~23 MB deterministic `tracegen` outputs, so `trace/tracelab_*` is
ignored rather than expected in a fresh clone. The names keep the `tracelab_`
prefix because that is the corpus they were derived from, not the tool that
wrote them.

Regenerating takes two steps and two repositories, neither of them vendored
here — the source DuckDB is a 99 MB artifact that was never a clone-and-go
dependency. Export the raw session rounds with
[TraceLab](https://github.com/uw-syfi/TraceLab)'s
`artifacts/trace_facts/csv_export/convert.py`, then materialize them with
`tracegen --policy {trace-reported,monotonic}` from
`alignment/load_generator/req-frontend`. The
`.manifest.json` beside each one **is** tracked: it pins the source SHA-256, the
policy that was applied, and the totals (sessions, rounds, prompt/prefix/output
tokens, planned prefix hit rate) a run needs to confirm it is replaying the trace
it thinks it is.
