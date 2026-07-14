//! Arrow schema definitions for the five VibeSim parquet streams.
//!
//! Modeled on `ref/moesim-rs/src/logging/schemas.rs` — each stream gets a
//! `pub fn xxx_schema() -> Arc<Schema>` returning a real
//! `arrow_schema::Schema`, ready for direct use by `StreamingParquetWriter`,
//! `LoggerSession`, and `CostLogger`.
//!
//! Field name / type / nullability follow the local logging contract. `cost_log`
//! is written as one file per `(pool_tag, worker_id)` stream; variable-length
//! per-group input and per-slot CostTree data live inside list columns.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema};

/// Legacy ref-style scalar envelope helper. The current writer uses
/// [`cost_log_schema`] below; do not infer the live parquet columns from this
/// compatibility surface.
pub fn cost_log_envelope_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("pool_tag", DataType::Utf8, false),
        Field::new("worker_id", DataType::UInt16, false),
        Field::new("worker_kind", DataType::Utf8, false),
        // Iteration index (one forward-pass cycle); shared by every batch a worker
        // runs in that iteration. `batch_id` distinguishes those batches.
        Field::new("iter_id", DataType::UInt64, false),
        Field::new("batch_id", DataType::UInt64, false),
        Field::new("call_seq", DataType::UInt32, false),
        Field::new("layer", DataType::Int16, false),
        Field::new("group_id", DataType::Int16, false),
        Field::new("wall_start_ms", DataType::Float64, false),
        Field::new("wall_end_ms", DataType::Float64, false),
        Field::new("total_time_ms", DataType::Float64, false),
        Field::new("energy_j", DataType::Float64, false),
    ]))
}

/// The per-group `input_section` struct (see `simulator/src/log/README.md`): one struct per
/// `ArchGroupInput` the worker fed to the model_arch this iteration. Prefill is
/// kept at full per-request fidelity (the two parallel `prefill_*_lens` lists);
/// decode is aggregated to two scalars (`decode_request_count` / `decode_kv_total`)
/// — the per-decode-request KV-length list is the size driver (re-logged every
/// step) and is intentionally dropped. `prefill_request_count` is the list length,
/// so it is not a separate field.
pub(crate) fn group_input_fields() -> Fields {
    let u32_item = || Arc::new(Field::new("item", DataType::UInt32, false));
    Fields::from(vec![
        Field::new("batch_tokens", DataType::UInt32, false),
        Field::new("prefill_tokens", DataType::UInt32, false),
        Field::new("decode_request_count", DataType::UInt32, false),
        Field::new("decode_kv_total", DataType::UInt32, false),
        Field::new("prefill_prefix_lens", DataType::List(u32_item()), false),
        Field::new("prefill_append_lens", DataType::List(u32_item()), false),
    ])
}

/// `cost_log` (CostTree per-iter form) — the envelope scalars, the per-iteration
/// `input_section` (`groups`: one struct per HP group, see [`group_input_fields`]),
/// then the compiled CostTree's per-slot breakdown as two parallel lists:
/// `slot_time_ms` (`List<f32>`) and `slot_coverage` (`List<u8>`, the
/// `CoverageFlags` bits). Slot *names* are NOT a column (INV-5 — names live at
/// compile time only); they live once in the matching
/// `cost_manifest/worker_<pool_tag>_<worker_id>.json` sidecar and label the
/// list positions.
pub fn cost_log_schema() -> Arc<Schema> {
    let time_item = Arc::new(Field::new("item", DataType::Float32, false));
    let cov_item = Arc::new(Field::new("item", DataType::UInt8, false));
    let group_item = Arc::new(Field::new(
        "item",
        DataType::Struct(group_input_fields()),
        false,
    ));
    Arc::new(Schema::new(vec![
        // Pool tag + worker id are the stable manifest key. WorkerId is per-pool
        // (PD has prefill worker 0 and decode worker 0), so worker_id alone is
        // not globally unique inside one run.
        Field::new("pool_tag", DataType::Utf8, false),
        Field::new("worker_id", DataType::UInt16, false),
        // `iter_id` is the per-worker iteration index (one forward-pass cycle);
        // `batch_id` is the batch *within* that iteration. They coincide today
        // (one batch per iteration) but diverge under AFD/TBO, where a worker runs
        // several batches in one iteration — so the row key is (pool_tag,
        // worker_id, iter_id, batch_id), not iter_id alone.
        Field::new("iter_id", DataType::UInt64, false),
        Field::new("batch_id", DataType::UInt64, false),
        // `wall_end_ms` is intentionally NOT a column: it is exactly
        // `wall_start_ms + total_time_ms`, so a consumer derives it. Dropping the
        // redundant f64 (which barely compressed) was the single largest size win.
        Field::new("wall_start_ms", DataType::Float64, false),
        Field::new("total_time_ms", DataType::Float64, false),
        Field::new("energy_j", DataType::Float64, false),
        Field::new("groups", DataType::List(group_item), false),
        Field::new("slot_time_ms", DataType::List(time_item), false),
        Field::new("slot_coverage", DataType::List(cov_item), false),
        // Per-slot input JSON (one string per leaf, slot-aligned to slot_time_ms).
        // Appended last: append-only, so old readers are unaffected.
        Field::new(
            "slot_input",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),
            false,
        ),
        // `section` / `layer` make a row's building-block group explicit (vs the
        // iter-wise form, where one Scale-folded row covers the whole iteration).
        // `section` names the cost group — `iter` for the fused unified path, or
        // `attn` / `prologue` / `pre_attn` / `post_attn` / `epilogue` for the AFD
        // layer-wise path; it selects which sub-manifest in the `cost_manifest/`
        // sidecar interprets this row's `slot_*` lists. `layer` is the layer index
        // (`-1` when the row is whole-iteration / Scale-folded). Appended last:
        // append-only, so old readers are unaffected.
        Field::new("section", DataType::Utf8, false),
        Field::new("layer", DataType::Int16, false),
        // Per-slot achieved FLOPs / bytes (slot-aligned to `slot_time_ms`; `0`
        // when the profile row carried no tflops/bandwidth rate — the same
        // "not measured" sentinel `KernelMetrics::flops`/`bytes` returns). Lets a
        // consumer derive per-op achieved TFLOP/s (`flops / time`) and GB/s
        // (`bytes / time`). Appended last: append-only, so old readers are
        // unaffected.
        Field::new(
            "slot_flops",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
        Field::new(
            "slot_bytes",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
        // Per-slot selected backend (slot-aligned to `slot_time_ms`): the
        // position-local index into the leaf's candidate list (`backends` in the
        // matching `cost_manifest` sidecar), or `255` when the leaf was not
        // executed this iteration. Lets a consumer see which of a multi-backend
        // position's candidates best-of-N picked. Appended last: append-only, so
        // old readers are unaffected.
        Field::new(
            "slot_backend",
            DataType::List(Arc::new(Field::new("item", DataType::UInt8, false))),
            false,
        ),
    ]))
}

/// `kv_snapshot` (§4) — the per-worker KV-pool occupancy time series: one row per
/// `(pool_tag, worker_id, group_id)` per throttled sample (emitted by
/// [`KvSampler`](crate::log::kv_sampler::KvSampler), which owns the sampling
/// policy). Only raw token counts live here. The static per-pool `capacity` moves
/// to `run_meta.json` (`kv_pools`), and the occupancy percentages are the
/// analyzer's join-and-divide, not stored columns — single source of truth, the
/// same reasoning that drops `wall_end_ms` from `cost_log`. `pool_tag` is part of
/// the key because `worker_id` restarts at 0 per pool (PD's prefill#0 vs
/// decode#0), exactly as in `cost_log`.
pub fn kv_snapshot_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("pool_tag", DataType::Utf8, false),
        Field::new("worker_id", DataType::UInt16, false),
        Field::new("group_id", DataType::UInt16, false),
        Field::new("time_ms", DataType::Float64, false),
        // `active_kv`: peak committed KV over the throttle window (a running max,
        // so decimated sampling never hides an occupancy spike) — the "current
        // size". `projected_peak`: the max KV the currently-admitted set will reach
        // as it drains (`Batch::projected_peak_kv`) — the "future estimate".
        // `promised_kv`: admitted-but-not-yet-realized tokens.
        Field::new("active_kv", DataType::UInt64, false),
        Field::new("projected_peak", DataType::UInt64, false),
        Field::new("promised_kv", DataType::UInt64, false),
    ]))
}

/// `network_event` (§5) — one row per cross-worker pull/push (disagg only).
pub fn network_event_schema() -> Arc<Schema> {
    let request_ids_item = Arc::new(Field::new("item", DataType::UInt32, false));
    Arc::new(Schema::new(vec![
        Field::new("src_worker_id", DataType::UInt16, false),
        Field::new("src_kind", DataType::Utf8, false),
        Field::new("dst_worker_id", DataType::UInt16, false),
        Field::new("dst_kind", DataType::Utf8, false),
        Field::new("request_count", DataType::UInt32, false),
        Field::new("request_ids", DataType::List(request_ids_item), false),
        Field::new("bytes", DataType::UInt64, false),
        Field::new("pull_request_ms", DataType::Float64, false),
        Field::new("pull_ready_ms", DataType::Float64, false),
        Field::new("net_first_start_ms", DataType::Float64, false),
        Field::new("net_last_start_ms", DataType::Float64, false),
        Field::new("pull_complete_ms", DataType::Float64, false),
        Field::new("bandwidth_gbps", DataType::Float64, false),
        Field::new("flow_id_first", DataType::UInt64, false),
        Field::new("flow_count", DataType::UInt32, false),
    ]))
}

/// `gpu_cluster` — one row per cross-worker transfer submitted to the shared
/// [`GpuCluster`](crate::worker::gpu_cluster::GpuCluster) (the run's single
/// transfer oracle), written by the [`NetworkLogger`](crate::log::NetworkLogger).
/// Unlike the ref-shaped [`network_event_schema`] (reserved for future full
/// disagg-pull semantics), this is the *actual* `submit_transfer` surface: the
/// on-wire window (`net_start_ms` / `net_end_ms`, both incl. queueing on the same
/// sim clock as `cost_log.wall_start_ms`), **both** endpoints' worker identity
/// (`src_*` = sender = `send_gid`'s comm-group owner; `dst_*` = receiver =
/// `recv_gid`'s owner), the two comm-group ids, the byte count, and a stable
/// `kind` + free-form `tag` each deployment fills to label the event. All columns
/// are always populated (no ref-style null padding). The `(dst_pool_tag,
/// dst_worker_id)` pair is the same key `cost_log` uses, so `analyze trace`
/// overlays a transfer onto the receiving worker's row.
pub fn gpu_cluster_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("net_start_ms", DataType::Float64, false),
        Field::new("net_end_ms", DataType::Float64, false),
        Field::new("src_pool_tag", DataType::Utf8, false),
        Field::new("src_worker_id", DataType::UInt16, false),
        Field::new("dst_pool_tag", DataType::Utf8, false),
        Field::new("dst_worker_id", DataType::UInt16, false),
        Field::new("send_gid", DataType::UInt16, false),
        Field::new("recv_gid", DataType::UInt16, false),
        Field::new("bytes", DataType::UInt64, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("tag", DataType::Utf8, false),
        // GPU count of the send / recv comm group (`CommGroup::count`, always >= 1).
        // Per-link share is `bytes / count`; lets a consumer derive per-link
        // effective bandwidth. The full gid→gpu-set mapping lives in
        // `run_meta.json`'s `comm_groups`. Appended last: append-only.
        Field::new("send_count", DataType::UInt16, false),
        Field::new("recv_count", DataType::UInt16, false),
        // When the *sender's* link is freed = end of its transmission slice. For a
        // gather (attn/ffn pull) this is `net_start + transfer_time` (latency-
        // stripped, < `net_end`), so the sender overlaps the tail of the collective;
        // for a coupled `submit_transfer` the sender is held to `net_end` (== this).
        // Lets `analyze trace` draw the send occupancy separately from the full
        // on-wire window. Appended last: append-only.
        Field::new("send_end_ms", DataType::Float64, false),
    ]))
}

/// `request_state` (§7.1) — one **aggregate** row per snapshot tick (not per
/// request). The analyzer's only consumer (`throughput/segment.rs`) reduces the
/// per-request rows to `Σ completed_input_len` / `Σ completed_output_len` per
/// `logging_time` and diffs consecutive ticks, so the sum is computed at the
/// source: each row carries the cumulative prefill/decode token totals over the
/// admitted set at that tick, plus the admitted/completed request counts. This
/// collapses the table from `O(admitted × ticks)` (tens of millions of rows on a
/// saturated run, where prefill admits the whole trace while decode lags) to one
/// row per tick. Per-request session columns that fed the analyzer's session-E2E
/// rollup move to `request_slo` (terminal-per-request).
pub fn request_state_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("logging_time", DataType::Float64, false),
        Field::new("prefill_tokens_cum", DataType::UInt64, false),
        Field::new("decode_tokens_cum", DataType::UInt64, false),
        Field::new("n_admitted", DataType::UInt64, false),
        Field::new("n_completed", DataType::UInt64, false),
    ]))
}

/// `request_slo` — one row per request with full per-token timing.
/// Column list / types / nullability match `simulator/src/log/README.md`.
/// `output_token_times` is `List<f32>` (absolute sim-time ms); it is empty
/// unless `io.log_output_token_times` is on (the scalars below are always present).
/// The convenience scalars (`ttft_ms`, `tpot_*_ms`, `last_token_time_ms`) are
/// nullable: `null` when there are too few output tokens to define them (e.g. a
/// sim-end-flush of a request still in prefill). `finish_decode_time_ms` is the
/// absolute sim-time ms of the final decoded token; E2E = `finish_decode_time_ms
/// - arrival_time_ms` (kept as a scalar so E2E survives with the array off).
pub fn request_slo_schema() -> Arc<Schema> {
    let token_times_item = Arc::new(Field::new("item", DataType::Float32, false));
    Arc::new(Schema::new(vec![
        Field::new("request_id", DataType::UInt32, false),
        Field::new("logging_time", DataType::Float64, false),
        Field::new("completed", DataType::Boolean, false),
        Field::new("arrival_time_ms", DataType::Float64, false),
        Field::new(
            "output_token_times",
            DataType::List(token_times_item),
            false,
        ),
        Field::new("num_output_tokens", DataType::UInt32, false),
        Field::new("ttft_ms", DataType::Float32, true),
        Field::new("tpot_mean_ms", DataType::Float32, true),
        Field::new("tpot_p50_ms", DataType::Float32, true),
        Field::new("tpot_p99_ms", DataType::Float32, true),
        Field::new("tpot_max_ms", DataType::Float32, true),
        Field::new("finish_decode_time_ms", DataType::Float32, true),
        // Terminal prefill length (`= RequestRecord::prefill_processed`): tokens
        // prefilled for this request by the time the row was written (full prompt
        // for a completed request, partial for a sim-end-flush still in prefill).
        // Appended last (append-only, old readers unaffected). Pairs with
        // `num_output_tokens` to give per-request (p, d) for the
        // `workload_conservation` analyzer's expected-work closed forms.
        Field::new("prefill_processed", DataType::UInt32, false),
    ]))
}

/// All stream names — useful for `LoggerSession::open` registration loops
/// and analyzer file-discovery.
pub const ALL_STREAMS: &[&str] = &[
    "cost_log",
    "gpu_cluster",
    "kv_snapshot",
    "network_event",
    "request_state",
    "request_slo",
];
