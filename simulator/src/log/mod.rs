//! Cross-layer logging — Arrow schemas (`schemas`), the streaming parquet writer
//! (`parquet_writer`), per-table row types + Arrow conversions (`rows`), and the
//! `LoggerSession` row-buffer / flush / Drop lifecycle (`session`).
//!
//! Shape follows `ref/moesim-rs/src/logging/{parquet_writer.rs,mod.rs}`: each
//! stream buffers rows and flushes a `RecordBatch` once it reaches
//! `STREAM_FLUSH_ROWS`; `flush_all` (also on `Drop`) force-flushes and closes.
//! This batch wires only the two per-request tables (`request_state` /
//! `request_slo`); the worker-internal streams (`cost_log` / `kv_snapshot` /
//! `network_event`) land with L5 logging.

pub mod cost_logger;
pub mod parquet_writer;
pub mod rows;
pub mod run_meta;
pub mod schemas;
pub mod session;

pub use cost_logger::CostLogger;
pub use parquet_writer::StreamingParquetWriter;
pub use rows::{CostLogEntry, GroupInputLog, RequestSloEntry, RequestStateEntry};
pub use run_meta::write_run_meta;
pub use schemas::{
    cost_log_envelope_schema, cost_log_schema, kv_snapshot_schema, network_event_schema,
    request_slo_schema, request_state_schema, ALL_STREAMS,
};
pub use session::LoggerSession;
