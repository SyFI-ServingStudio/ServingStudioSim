//! `LoggerSession` — owns the per-table parquet writers + their row buffers,
//! flushing a `RecordBatch` once a buffer reaches `STREAM_FLUSH_ROWS` and
//! force-flushing + closing on `flush_all` (also called on `Drop`). Shape
//! follows `ref/moesim-rs/src/logging/mod.rs`.

use std::path::Path;

use anyhow::Result;

use crate::log::parquet_writer::StreamingParquetWriter;
use crate::log::rows::{
    slo_to_record_batch, state_to_record_batch, RequestSloEntry, RequestStateEntry,
};
use crate::log::schemas::{request_slo_schema, request_state_schema};

/// Buffered rows per stream before a `RecordBatch` flush (matches ref).
const STREAM_FLUSH_ROWS: usize = 8_192;

/// Owns the per-request parquet writers + their row buffers. Created once per
/// run (`open`), fed one row at a time, force-flushed + closed by `flush_all`
/// (also on `Drop`).
pub struct LoggerSession {
    state_writer: StreamingParquetWriter,
    slo_writer: StreamingParquetWriter,
    state_buf: Vec<RequestStateEntry>,
    slo_buf: Vec<RequestSloEntry>,
    closed: bool,
}

impl LoggerSession {
    /// Open writers under `<log_dir>/raw/{request_state,request_slo}.parquet`.
    /// Files are created lazily on first row (so an empty stream writes nothing).
    pub fn open(log_dir: &Path) -> Result<Self> {
        let raw = log_dir.join("raw");
        Ok(Self {
            state_writer: StreamingParquetWriter::new(
                raw.join("request_state.parquet"),
                request_state_schema(),
            ),
            slo_writer: StreamingParquetWriter::new(
                raw.join("request_slo.parquet"),
                request_slo_schema(),
            ),
            state_buf: Vec::new(),
            slo_buf: Vec::new(),
            closed: false,
        })
    }

    pub fn record_request_state(&mut self, entry: RequestStateEntry) -> Result<()> {
        self.state_buf.push(entry);
        self.maybe_flush_state(false)
    }

    pub fn record_request_slo(&mut self, entry: RequestSloEntry) -> Result<()> {
        self.slo_buf.push(entry);
        self.maybe_flush_slo(false)
    }

    fn maybe_flush_state(&mut self, force: bool) -> Result<()> {
        if self.state_buf.is_empty() || (!force && self.state_buf.len() < STREAM_FLUSH_ROWS) {
            return Ok(());
        }
        let batch = state_to_record_batch(&self.state_buf)?;
        self.state_writer.write(&batch)?;
        self.state_buf.clear();
        Ok(())
    }

    fn maybe_flush_slo(&mut self, force: bool) -> Result<()> {
        if self.slo_buf.is_empty() || (!force && self.slo_buf.len() < STREAM_FLUSH_ROWS) {
            return Ok(());
        }
        let batch = slo_to_record_batch(&self.slo_buf)?;
        self.slo_writer.write(&batch)?;
        self.slo_buf.clear();
        Ok(())
    }

    /// Force-flush both buffers and close the writers. Idempotent.
    pub fn flush_all(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.maybe_flush_state(true)?;
        self.maybe_flush_slo(true)?;
        self.state_writer.close()?;
        self.slo_writer.close()?;
        self.closed = true;
        Ok(())
    }
}

impl Drop for LoggerSession {
    fn drop(&mut self) {
        let _ = self.flush_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::rows::RequestSloEntry;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    fn slo_entry(id: u32, times: Vec<f32>) -> RequestSloEntry {
        RequestSloEntry {
            request_id: id,
            logging_time_ms: 100.0,
            completed: true,
            arrival_time_ms: 0.0,
            output_token_times_ms: times,
            ttft_ms: Some(1.0),
            tpot_mean_ms: Some(2.0),
            tpot_p50_ms: Some(2.0),
            tpot_p99_ms: Some(3.0),
            tpot_max_ms: Some(3.0),
        }
    }

    #[test]
    fn logger_session_writes_both_parquets() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut log = LoggerSession::open(dir.path()).unwrap();
            log.record_request_slo(slo_entry(0, vec![1.0, 2.0])).unwrap();
            log.flush_all().unwrap();
        }
        let slo_path = dir.path().join("raw/request_slo.parquet");
        assert!(slo_path.exists());
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&slo_path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let rows: usize = reader.map(|b| b.unwrap().num_rows()).sum();
        assert_eq!(rows, 1);
        // request_state had no rows → no file written.
        assert!(!dir.path().join("raw/request_state.parquet").exists());
    }
}
