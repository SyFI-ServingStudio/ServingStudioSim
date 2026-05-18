//! Cross-layer logging — Arrow schemas (Phase 0) + parquet writers + the
//! `SimLogger` row buffer / flush / Drop lifecycle (Phase 3).
//!
//! Phase 0 ships only the schemas (`schemas` submodule). The runtime pieces
//! (`StreamingParquetWriter`, `SimLogger` with `STREAM_FLUSH_ROWS` row
//! buffers + `maybe_flush_X` per stream + Drop-time final flush) follow the
//! shape of `ref/moesim-rs/src/logging/{parquet_writer.rs,mod.rs}` and land
//! in Phase 3 alongside the first L5 worker that writes to them.

pub mod schemas;

pub use schemas::{
    cost_log_envelope_schema, kv_snapshot_schema, network_event_schema, request_slo_schema,
    request_state_schema, ALL_STREAMS,
};
