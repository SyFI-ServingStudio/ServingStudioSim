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
network_logger.rs NetworkLogger — the shared GpuCluster transfer stream writer.
run_meta.rs       write_run_meta — the run_meta.json GPU-facts sidecar (plain
                  serde_json, not parquet / not threaded).
```

## The streams & files (under `<log_dir>/raw/`)

| File | Writer | Opened by | Content |
|---|---|---|---|
| `request_slo.parquet` | `LoggerSession` | L7 `run_sim` | one terminal row per completed request, or one sim-end partial row per incomplete arrived request |
| `request_state.parquet` | `LoggerSession` | L7 `run_sim` | periodic dense snapshot over the admitted set |
| `cost_log/worker_<pool_tag>_<worker_id>.parquet` | `CostLogger` | each L5 worker | one row per iteration: envelope + per-group `input_section` + the CostTree per-slot breakdown |
| `cost_manifest/worker_<pool_tag>_<worker_id>.json` | `CostLogger` | each L5 worker | the matching `CostManifest` (slots + flat aggregation nodes) written once at open |
| `kv_snapshot/worker_<pool_tag>_<worker_id>.parquet` | `KvSampler` | each KV-owning L5 worker | throttled per-partition occupancy/projected-peak series |
| `gpu_cluster.parquet` | `NetworkLogger` | shared `GpuCluster` for PD/AFD | one resolved cross-worker transfer with both endpoints and timing window |
| `run_meta.json` | `run_meta` | L7 (pre-loop) | GPU registry, worker grouping, KV capacities, comm groups, and stage vocabulary |

`network_event` remains a reserved schema with no writer; production transfer
logging uses the richer `gpu_cluster` stream.

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
