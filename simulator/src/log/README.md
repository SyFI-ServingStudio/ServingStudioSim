# Log — streaming parquet output

Cross-layer logging: turn the sim's per-request and per-iteration events into
parquet streams plus JSON sidecars (`run_meta.json` and per-worker
`cost_manifest/*.json`) under `<log_dir>/raw/`. The design
constraint is that **logging must not slow the sim**: the sim thread only fills
row buffers and hands full chunks to a background thread that does the heavy
parquet encode + ZSTD compression off the critical path.

This is the practical, code-matching reference; the code is the ground truth.
This README specifies the *tables* — the schemas, the two-tier (per-request
vs. cost_log) split, the replay flow. The **threading model below is the
implementation's own**: the table layout follows
`ref/moesim-rs`, but the sim-thread/writer-thread split is this code's decision,
stated here and in the `session.rs` / `cost_logger.rs` headers.

## The threading model (both sessions share it)

```
sim thread                         bounded sync_channel (CHANNEL_CAP=64)        writer thread
  push Entry into Vec  ──fill──▶    [chunk] [chunk] ...  ──backpressure──▶   *_to_record_batch
  at STREAM_FLUSH_ROWS (8192):                                              ArrowWriter::write
  send the chunk, swap a fresh Vec                                          (dict / RLE / stats / ZSTD-L3)
```

The channel is **bounded**, so a slow writer applies backpressure instead of
growing memory without limit (sized so a dense `request_state` snapshot — tens of
chunks in one tick — queues without stalling the sim). `flush_all` (also `Drop`)
sends the buffer tails, closes the channel, and joins the writer, propagating its
first error. Files are created **lazily** on the first non-empty write, so a
stream that never produces a row leaves no file behind.

## Directory map

Speculative iterations append nullable `groups.speculative_geometry` JSON with
`draft_tokens`, `max_model_len`, prefill `(prefix, query)` pairs and decode
`(final_context, query)` pairs. Ordinary rows leave it null. These are raw request
shapes, not kernel FLOPs or bytes; the independent model accountant owns those
formulas. Geometry is encoded on the writer thread.

`request_slo.speculative_progress` independently persists admission/completion
observations: verify width, completed prefill chunks and decode rounds, resident
KV sum, emitted tokens, and an optional pending prefill/decode. This bounded
per-request record survives optional timing logs being disabled. Pending work
lets conservation account exactly for a sim-end stop before completion. Rejected
candidates remain part of executed work; only committed output advances KV.

```
schemas.rs        Arrow schemas for the streams + ALL_STREAMS. The cost_log
                  envelope (universal columns) vs. the full cost_log schema
                  (envelope + per-group input_section + per-slot breakdown lists).
rows.rs           Per-table row types (RequestStateEntry, RequestSloEntry,
                  CostLogEntry, GroupInputLog, FinalPhase) + their
                  `*_to_record_batch` Arrow conversions + CostLogChunk.
parquet_writer.rs StreamingParquetWriter: append RecordBatches to one file, lazy
                  create on first row, ZSTD-L3. The shared low-level writer.
session.rs        LoggerSession — the per-REQUEST streams (request_state +
                  request_slo). Opened by the L7 run loop.
cost_logger.rs    CostLogger — the per-ITERATION cost_log stream, standalone so it
                  doesn't perturb the per-request streams. Owned by each L5 worker.
kv_sampler.rs     KvSampler — per-worker KV occupancy sampling and parquet writer.
prefix_cache_logger.rs
                  PrefixCacheLogger — exact per-worker retained-prefix mutation
                  replay, ordered independently of sampled occupancy.
network_logger.rs NetworkLogger — the shared GpuCluster transfer stream writer.
run_meta.rs       write_run_meta — the run_meta.json GPU-facts sidecar (plain
                  serde_json, not parquet / not threaded).
```

## The streams & files (under `<log_dir>/raw/`)

| File | Writer | Opened by | Content |
|---|---|---|---|
| `request_slo.parquet` | `LoggerSession` | L7 `run_sim` | one terminal row per completed request, or one sim-end partial row per incomplete arrived request; includes the request's nullable TTFT/TPOT/E2E SLOs, declared prefix tokens, and nullable admission-time cache-hit tokens |
| `request_state.parquet` | `LoggerSession` | L7 `run_sim` | periodic dense snapshot over the admitted set |
| `cost_log/worker_<pool_tag>_<worker_id>.parquet` | `CostLogger` | each L5 worker | one row per iteration: envelope + per-group `input_section` + the CostTree per-slot breakdown |
| `cost_manifest/worker_<pool_tag>_<worker_id>.json` | `CostLogger` | each L5 worker | the matching `CostManifest` (slots + flat aggregation nodes) written once at open |
| `kv_snapshot/worker_<pool_tag>_<worker_id>.parquet` | `KvSampler` | each KV-owning L5 worker | throttled per-partition active/retained-prefix/projected/promised KV series |
| `prefix_cache_event/worker_<pool_tag>_<worker_id>.parquet` | `PrefixCacheLogger` | each prefix-capable KV-owning L5 worker | every session-cache ownership transition, with exact entry/cache before and after token counts |
| `gpu_cluster.parquet` | `NetworkLogger` | shared `GpuCluster` for PD/AFD | one resolved cross-worker transfer with both endpoints and timing window |
| `run_meta.json` | `run_meta` | L7 (pre-loop) | GPU registry, worker grouping, KV capacities, comm groups, and stage vocabulary |

`network_event` remains a reserved schema with no writer; production transfer
logging uses the richer `gpu_cluster` stream.

For prefix-aware text requests, `request_slo.declared_prefix_tokens` is the
immutable request requirement and `prefix_cache_hit_tokens` is the worker-local
observation copied at successful admission. A null hit means the request never
reached prefix resolution; zero is a resolved cache miss. Consumers derive miss
tokens as `declared - hit` and the hit rate from those two counts rather than
depending on another redundant column. `fresh_prompt_tokens` separately records
the immutable new suffix, while `prefill_processed` records work actually done.
Once a request has produced its first output token, conservation therefore has
the exact request-level invariant
`prefix_cache_hit_tokens + prefill_processed = fresh_prompt_tokens + declared_prefix_tokens`.

The nullable `declared_ttft_slo_ms`, `declared_tpot_slo_ms`, and
`declared_e2e_slo_ms` columns preserve the trace's per-request obligations.
Null means that metric was not bounded for that request; the three columns are
independent and scheduling priority is not an SLO column.

`kv_snapshot.active_kv` is total committed attention KV and therefore already
includes retained prefix-cache entries. `retained_prefix_kv` exposes that component
without creating another pool: it is captured from the same raw submit that set the
throttle window's `active_kv` peak. Consequently every new-schema row satisfies
`retained_prefix_kv <= active_kv`, and `active_kv - retained_prefix_kv` is the
non-prefix committed occupancy represented by that sample.

`prefix_cache_event` is the non-sampled operation replay. A session request that
successfully reserves KV emits `activate`: `hit` destructively transfers the
whole retained entry to the active request, while `miss` records a resolved cold
lookup. Request completion emits `retain`; PD prefill emits it only after the
decode pull acknowledgement releases held source KV. Cache removals emit
`evict`, with one of `active-kv-pressure`, `replacement-policy`,
`retention-capacity`, or `same-session-replacement`. A completed session that
cannot retain any KV still emits `retain/no-cache-capacity` with a zero-sized
transition.

`sequence` is strictly increasing within one worker file and totally orders
events that share `time_ms`. Replaying rows in sequence per partition must obey:

```text
cache_used_after
  = cache_used_before - entry_tokens_before + entry_tokens_after
next.cache_used_before = previous.cache_used_after
```

The worker constructs these rows through `PrefixCacheEventKind`, whose typed
variants pair each operation with only its legal reason family. `operation` and
`reason` become strings only at the parquet boundary.

The row's `request_id` identifies the operation trigger and `session_id`
identifies the mutated entry. `requested_tokens` and `hit_tokens` explain an
`activate`; for a partial declaration the entire count-only entry is removed but
only the declared portion is a hit. This stream does not mirror active decode
growth and does not maintain a second cache ledger: `PrefixCache` returns the
mutation receipts that are written verbatim. `kv_snapshot` remains the compact
occupancy view; `prefix_cache_event` is the exact diagnostic replay.

## cost_log & the manifest (INV-5)

The `cost_log` row carries the per-slot breakdown as **position-keyed parallel
lists** — `slot_time_ms` (`List<f32>`), `slot_coverage` (`List<u8>` of
`CoverageFlags` bits), and the captured `slot_input` column — never slot *names*. Names
live once in the per-worker `cost_manifest/worker_<pool_tag>_<worker_id>.json`
sidecar (which also carries the flattened aggregation nodes), so a consumer
reproduces `total_time_ms` from a row's per-slot times by re-running
`CostTree::aggregate`, and labels the list positions from the matching manifest.
The row carries `pool_tag` because `worker_id` is per-pool; consumers key
manifests by `(pool_tag, worker_id)`. See [../timing/COST_TREE.md](../timing/COST_TREE.md)
for the manifest shape. The worker fills the per-slot buffers during its CostTree
eval pass and hands them to `CostLogger::record` (slot-aligned, one input list per
row).

The captured `slot_input` values are handed over as inline enums and **serialized to
JSON strings on the writer thread** (`*_to_record_batch`, off the sim's critical
path), with a `"null"` fallback if serialization fails — never panicking the run.

## Up / down

- **Above (writers):** the L7 `sim` loop (`LoggerSession` + `run_meta`), each
  KV-owning L5 worker (`CostLogger` + `KvSampler`), and the shared `GpuCluster`
  (`NetworkLogger` for PD/AFD transfers).
- **Below (used):** Arrow / parquet (`arrow_array`, `arrow_schema`, `parquet`
  with ZSTD).
- **Consumer (downstream):** the standalone `analyzer` crate reads these parquet
  streams + JSON sidecars back (see `analyzer/README.md`).
