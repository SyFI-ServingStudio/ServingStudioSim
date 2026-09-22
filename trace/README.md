# Trace generators and samples

ServingStudio Sim and req-frontend's `independent` frontend consume the same
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

## `diverse_100.csv` — the default alignment workload

A fixed-shape trace answers "what is capacity at this shape"; alignment asks
whether the model tracks measurement *across* shapes, and a workload that only
visits one corner cannot tell a shape-independent error from a shape-dependent
one. This is the spread-out counterpart: 100 requests, 100 distinct shapes.

| | |
|---|---|
| `input_len` | 256 – 131,070, log-uniform, sum 2,061,718 |
| `output_len` | 1 – 8,192, log-uniform, sum 135,244 |
| `input_len + output_len` | at most 131,071, so **`max_model_len: 131072` is enough** |
| arrivals | Poisson, mean 1 req/s, last at 114.9 s |

Both axes are sampled in log space, so every decade is populated rather than the
top one swamping the rest: 23 / 35 / 23 / 19 requests in the 256-1k, 1k-8k,
8k-32k and 32k-128k prefill bands.

Two properties are deliberate and worth not "tidying up":

- **The lengths are not round numbers.** 20,233 and 45,282 rather than 16,384 and
  32,768. Powers of two coincide with tile, page and CUDA-graph capture
  boundaries, so a trace built from them measures the aligned case and hides the
  padding behaviour of everything in between. Only the four pinned corners are
  round, and only because they define the range.
- **The four corners are pinned** — `(256, 1)`, `(256, 8192)`, `(131070, 1)`,
  `(122879, 8192)` — so the file spans the stated range instead of merely
  sampling near it. The two large ones sit just under the context limit, which is
  also why the longest prefill is not paired with the longest generation: the sum
  is capped so the whole trace fits a 128k context.

Generated once and committed as data; there is no generator script, so the table
above is the file's only description. Regenerate it by hand if the ranges need to
move, and update these numbers with it.

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

### `session_execution_v2_placed_smoke.csv` — the placement column

`session_execution_v2_example.csv` plus one column:

```text
...,tool_wait_after_ms,target_worker
```

Same column the independent format already carries (`slime_rollout*.csv`), read
only when the preset lists `input_file_tags: [placement]`, and obeyed only under
`placement: trace-directed`. It is what makes a placement SEQUENCE reproducible:
a load policy can be re-run, but it cannot be replayed.

**Synthetic, and derived rather than measured.** Sessions alternate between two
engines by parity, and session 1 switches engine at round 60 so the file
contains a placement no affinity rule would produce. Three conversations / 211
rounds is enough to exercise the path and not enough to measure anything —
`presets/session_placed_prefix_sweep.yaml` uses it that way.

The real thing is a measured multi-worker replay that recorded the engine per
round. When that lands it goes in the same column, on the full `tracelab_*`
trace, and this file stays as the smoke fixture.

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
