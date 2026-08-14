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

The `.manifest.json` beside each one **is** tracked. It pins the source SHA-256,
the policy, the arrival synthesis, and the totals (sessions, rounds,
prompt/prefix/output tokens, planned prefix hit rate) a run needs to confirm it
is replaying the trace it thinks it is.

### Regenerating

Three steps, and the corpus is not vendored here — it is a 99 MB release asset
that was never a clone-and-go dependency. Every flag below is a default except
`--policy`, so the manifests are enough to reproduce these files exactly.

```bash
# 1. The pinned corpus. v0.0.1 is what these manifests describe; v0.0.2 is a
#    larger, later release that produces a different (valid) trace.
curl -L --fail -o "$TMPDIR/syfi_coding_trace.duckdb" \
  https://github.com/uw-syfi/TraceLab/releases/download/v0.0.1/syfi_coding_trace.duckdb

# 2. Export the raw session rounds. Deterministic: no seed, no synthesis.
#    From a TraceLab clone (https://github.com/uw-syfi/TraceLab).
uv run python artifacts/trace_facts/csv_export/convert.py \
  --db "$TMPDIR/syfi_coding_trace.duckdb" -o "$TMPDIR/raw_rounds.csv"
#    -> session-rounds-v2, 4,281 sessions / 357,161 rounds
#    -> sha256 f09b79435bcba218a0aeac26784d0b95cb2270e4d16907bbcbcdb649f5029c5c

# 3. Materialize, from alignment/load_generator/req-frontend.
cargo run --release --bin tracegen -- coding-session \
  --source "$TMPDIR/raw_rounds.csv" --policy monotonic \
  --out <repo>/trace/tracelab_preserving.csv
cargo run --release --bin tracegen -- coding-session \
  --source "$TMPDIR/raw_rounds.csv" --policy trace-reported \
  --out <repo>/trace/tracelab_reported.csv
```

Step 3 also invents the arrival timeline — the corpus has no session arrival
times — at a default `poisson`, 1 session/s, seed 0. The manifest records all
three, so the timeline is reproducible rather than merely plausible.
