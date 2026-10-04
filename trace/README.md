# Trace generators and samples

ServingStudio Sim and req-frontend's `independent` frontend consume the same
independent-request CSV schema:

```text
id,input_len,output_len,arrival_time
```

`arrival_time` is measured in milliseconds.

### Optional `prefix_len`: a pinned prefix hit

ServingStudio Sim also accepts one optional column, `prefix_len`, in any position:

```text
id,input_len,output_len,arrival_time,prefix_len
r0,4096,1,0,32768
r1,4096,1,0,0
```

`prefix_len` is the number of tokens before this request's prompt that are
already resident in the prefix cache when it arrives. `input_len` stays the
fresh tokens to compute, as in the session schema's `prefix_len,input_len`
pair, so the request's context after prefill is `prefix_len + input_len`.

The hit is forced. It does not depend on cache state, eviction, the prefix
cache mode, sessions, or which worker or partition the request lands on. The
prefix KV is reserved with the request (it counts toward capacity), is released
when the request completes, and is never retained as a shared cache entry.
`request_slo` records `declared_prefix_tokens = prefix_cache_hit_tokens =
prefix_len` and `prefill_processed = input_len`. A missing column, a blank cell,
or `0` loads exactly as the four-column file does. A row that also declares a
`session_id` through the `session` tag must use `prefix_kv` instead.

req-frontend does not declare `prefix_len` yet: its replay client rejects a file
that carries the column (`header does not match ... unexpected: ["prefix_len"]`).
Keep pinned-prefix traces to the simulator until the client models the hit.

Generate a deterministic fixed-shape
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

### Prefill-only replay

`session_to_prefill_only.py` turns either session trace into an independent
trace for prefill studies. Every round becomes one request: its fresh tokens
are `input_len`, its planned prefix is a pinned `prefix_len` hit, and
`output_len` is 1. Sessions, tool waits, and the session timeline are dropped.

```bash
uv run python trace/session_to_prefill_only.py trace/tracelab_preserving.csv \
  "$TMPDIR/prefill_5k.csv" --requests 5000 --seed 0
```

`--requests` takes a seeded uniform sample of rounds in random order, and
arrivals are Poisson at 1 request/s, so a preset's `workload.request_rate`
sweeps the offered load. With the same seed, the two policies sample the same
rounds at the same arrival times and differ only in the prefix/fresh split.
Contexts reach 999,888 tokens, so runs need `max_model_len: 1048576`. The
pinned prefix is reserved only while its request runs, so this replay measures
prefill compute, not prefix-cache capacity.
