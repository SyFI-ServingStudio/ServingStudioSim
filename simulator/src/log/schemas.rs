//! Arrow schema definitions for the five MLSim parquet streams.
//!
//! Modeled on `ref/moesim-rs/src/logging/schemas.rs` — each stream gets a
//! `pub fn xxx_schema() -> Arc<Schema>` returning a real
//! `arrow_schema::Schema`, ready for direct use by Phase 3's
//! `StreamingParquetWriter` + `SimLogger`.
//!
//! Field name / type / nullability follow `docs/logging.md` §3–§7. The
//! `cost_log` table is *partitioned by `worker_kind`*, so this module exposes
//! only the worker_kind-agnostic envelope. Per-worker_kind `input_section` +
//! per-model_arch `output_section` schemas land alongside L5 / L4 work in
//! Phase 2+.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};

/// `cost_log` envelope (§3.1) — universal columns shared by every worker_kind
/// partition. Per-worker_kind extensions append `input_section` /
/// `output_section` struct columns onto this base.
pub fn cost_log_envelope_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
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

/// `cost_log` (CostTree per-iter form) — the envelope scalars plus the compiled
/// CostTree's per-slot breakdown as two parallel lists: `slot_time_ms`
/// (`List<f32>`) and `slot_coverage` (`List<u8>`, the `CoverageFlags` bits).
/// Slot *names* are NOT a column (INV-5 — names live at compile time only); they
/// live once in the `cost_manifest.json` sidecar and label the list positions.
pub fn cost_log_schema() -> Arc<Schema> {
    let time_item = Arc::new(Field::new("item", DataType::Float32, false));
    let cov_item = Arc::new(Field::new("item", DataType::UInt8, false));
    Arc::new(Schema::new(vec![
        Field::new("worker_id", DataType::UInt16, false),
        // `iter_id` is the per-worker iteration index (one forward-pass cycle);
        // `batch_id` is the batch *within* that iteration. They coincide today
        // (one batch per iteration) but diverge under AFD/TBO, where a worker runs
        // several batches in one iteration — so the row key is (worker_id, iter_id,
        // batch_id), not iter_id alone.
        Field::new("iter_id", DataType::UInt64, false),
        Field::new("batch_id", DataType::UInt64, false),
        Field::new("wall_start_ms", DataType::Float64, false),
        Field::new("wall_end_ms", DataType::Float64, false),
        Field::new("total_time_ms", DataType::Float64, false),
        Field::new("energy_j", DataType::Float64, false),
        Field::new("slot_time_ms", DataType::List(time_item), false),
        Field::new("slot_coverage", DataType::List(cov_item), false),
    ]))
}

/// `kv_snapshot` (§4) — sampled KV pool state per worker × group.
pub fn kv_snapshot_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("worker_id", DataType::UInt16, false),
        Field::new("group_id", DataType::UInt8, false),
        Field::new("time_ms", DataType::Float64, false),
        Field::new("active_kv", DataType::UInt64, false),
        Field::new("projected_peak", DataType::UInt64, false),
        Field::new("promised_kv", DataType::UInt64, false),
        Field::new("suspended_kv", DataType::UInt64, false),
        Field::new("capacity", DataType::UInt64, false),
        Field::new("active_pct", DataType::Float32, false),
        Field::new("peak_pct", DataType::Float32, false),
        Field::new("promised_pct", DataType::Float32, false),
        Field::new("suspended_pct", DataType::Float32, false),
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

/// `request_state` (§7.1) — periodic snapshot per request × sampling tick.
/// Column list / order / types / nullability match `docs/logging.md §7.1`
/// (and the legacy `ref/moesim-rs/src/logging/schemas.rs::request_final_state_schema`,
/// which the new doc takes verbatim).
pub fn request_state_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("request_id", DataType::UInt32, false),
        Field::new("logging_time", DataType::Float64, false),
        Field::new("arrival_time_ms", DataType::Float64, false),
        Field::new("first_token_time_ms", DataType::Float64, true),
        Field::new("completion_time_ms", DataType::Float64, true),
        Field::new("completed", DataType::Boolean, false),
        Field::new("input_len", DataType::UInt32, false),
        Field::new("output_len", DataType::UInt32, false),
        Field::new("completed_input_len", DataType::UInt32, false),
        Field::new("completed_output_len", DataType::UInt32, false),
        Field::new("final_phase", DataType::LargeUtf8, false),
        Field::new("session_id", DataType::UInt32, false),
        Field::new("round_idx", DataType::UInt32, false),
        Field::new("total_rounds", DataType::UInt32, false),
        Field::new("tool_wait_after_ms", DataType::Float64, false),
        Field::new("session_arrival_time_ms", DataType::Float64, false),
        Field::new("preserved_prefix_kv", DataType::UInt32, false),
    ]))
}

/// `request_slo` (§7.2) — one row per request with full per-token timing.
/// Column list / types / nullability match `docs/logging.md §7.2`.
/// `output_token_times` is `List<f32>` (absolute sim-time ms). The five
/// convenience scalars (`ttft_ms`, `tpot_*_ms`) are nullable: `null` when
/// `num_output_tokens == 0` (e.g. sim-end-flush of a request still in prefill).
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
    ]))
}

/// All stream names — useful for `LoggerSession::open` registration loops
/// and analyzer file-discovery.
pub const ALL_STREAMS: &[&str] = &[
    "cost_log",
    "kv_snapshot",
    "network_event",
    "request_state",
    "request_slo",
];
