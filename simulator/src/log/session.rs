//! `LoggerSession` — buffers rows on the sim thread and offloads the heavy
//! parquet encode + ZSTD compression to a dedicated background writer thread.
//! The sim thread only fills `Vec<…Entry>` row buffers and hands full chunks
//! over a bounded channel; the writer thread owns the parquet writers and does
//! `…_to_record_batch` + `ArrowWriter::write` (dictionary-intern, RLE, column
//! stats, ZSTD-L3) off the critical path. The channel is bounded so a slow
//! writer applies backpressure instead of growing memory without limit.
//! Shape follows `ref/moesim-rs/src/logging/mod.rs`; the threading is ours.

use std::path::Path;
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::thread::JoinHandle;

use anyhow::{anyhow, Result};

use crate::log::parquet_writer::StreamingParquetWriter;
use crate::log::rows::{
    slo_to_record_batch, state_to_record_batch, RequestSloEntry, RequestStateEntry,
};
use crate::log::schemas::{request_slo_schema, request_state_schema};

/// Buffered rows per stream before a chunk is handed to the writer thread.
const STREAM_FLUSH_ROWS: usize = 8_192;

/// In-flight chunks the channel holds before the sim thread blocks on `send`
/// (backpressure). `request_state` is now one aggregate row per snapshot tick
/// (not a per-request dump), so the burst per tick is tiny; the cap mainly
/// smooths `request_slo` completion bursts. At `STREAM_FLUSH_ROWS` rows/chunk
/// this bounds buffered memory to ~tens of MB.
const CHANNEL_CAP: usize = 64;

/// A full row chunk handed to the writer thread (ownership transferred).
enum LogMsg {
    State(Vec<RequestStateEntry>),
    Slo(Vec<RequestSloEntry>),
}

/// Sim-thread handle: owns the row buffers + the channel to the writer thread.
/// Fed one row at a time; `flush_all` (also `Drop`) sends the tails, closes the
/// channel, and joins the writer (propagating its first error).
pub struct LoggerSession {
    tx: Option<SyncSender<LogMsg>>,
    handle: Option<JoinHandle<Result<()>>>,
    state_buf: Vec<RequestStateEntry>,
    slo_buf: Vec<RequestSloEntry>,
    closed: bool,
}

impl LoggerSession {
    /// Open writers under `<log_dir>/raw/{request_state,request_slo}.parquet`
    /// and spawn the background writer thread. Files are created lazily on the
    /// first row (an empty stream writes nothing). `log_output_token_times` controls
    /// whether the per-token `output_token_times` array column is materialized
    /// (the derived SLO scalars are written either way). `log_stage_transitions`
    /// likewise gates the per-request stage-transition list columns.
    pub fn open(
        log_dir: &Path,
        log_output_token_times: bool,
        log_stage_transitions: bool,
    ) -> Result<Self> {
        let raw = log_dir.join("raw");
        let mut state_writer =
            StreamingParquetWriter::new(raw.join("request_state.parquet"), request_state_schema());
        let mut slo_writer =
            StreamingParquetWriter::new(raw.join("request_slo.parquet"), request_slo_schema());

        let (tx, rx) = sync_channel::<LogMsg>(CHANNEL_CAP);
        let handle = std::thread::Builder::new()
            .name("vibesim-logger".to_string())
            .spawn(move || -> Result<()> {
                // Encode + compress + write each chunk off the sim thread. The
                // loop ends when every `tx` is dropped (channel disconnected).
                for msg in rx {
                    match msg {
                        LogMsg::State(buf) => state_writer.write(&state_to_record_batch(&buf)?)?,
                        LogMsg::Slo(buf) => slo_writer.write(&slo_to_record_batch(
                            &buf,
                            log_output_token_times,
                            log_stage_transitions,
                        )?)?,
                    };
                }
                state_writer.close()?;
                slo_writer.close()?;
                Ok(())
            })?;

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            state_buf: Vec::with_capacity(STREAM_FLUSH_ROWS),
            slo_buf: Vec::with_capacity(STREAM_FLUSH_ROWS),
            closed: false,
        })
    }

    pub fn record_request_state(&mut self, entry: RequestStateEntry) -> Result<()> {
        self.state_buf.push(entry);
        if self.state_buf.len() >= STREAM_FLUSH_ROWS {
            self.send_state()?;
        }
        Ok(())
    }

    pub fn record_request_slo(&mut self, entry: RequestSloEntry) -> Result<()> {
        self.slo_buf.push(entry);
        if self.slo_buf.len() >= STREAM_FLUSH_ROWS {
            self.send_slo()?;
        }
        Ok(())
    }

    fn send_state(&mut self) -> Result<()> {
        if self.state_buf.is_empty() {
            return Ok(());
        }
        let buf = std::mem::replace(&mut self.state_buf, Vec::with_capacity(STREAM_FLUSH_ROWS));
        self.send(LogMsg::State(buf))
    }

    fn send_slo(&mut self) -> Result<()> {
        if self.slo_buf.is_empty() {
            return Ok(());
        }
        let buf = std::mem::replace(&mut self.slo_buf, Vec::with_capacity(STREAM_FLUSH_ROWS));
        self.send(LogMsg::Slo(buf))
    }

    /// Hand a chunk to the writer thread. A send error means the writer died;
    /// surface its real error by joining rather than the generic disconnect.
    /// `try_send` first so a full channel (writer behind) warns about
    /// backpressure before falling back to the blocking `send` that stalls the
    /// sim thread until the writer drains a slot.
    fn send(&mut self, msg: LogMsg) -> Result<()> {
        let tx = self.tx.as_ref().expect("tx present until flush_all");
        let msg = match tx.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(msg)) => {
                tracing::warn!(
                    "logger channel full ({CHANNEL_CAP} chunks in flight): sim thread \
                     blocking on writer backpressure"
                );
                msg
            }
            Err(TrySendError::Disconnected(_)) => return Err(self.writer_died()),
        };
        match self.tx.as_ref().expect("tx present").send(msg) {
            Ok(()) => Ok(()),
            Err(_) => Err(self.writer_died()),
        }
    }

    /// The writer thread dropped its `rx`: join to surface its real error
    /// instead of the generic "disconnected".
    fn writer_died(&mut self) -> anyhow::Error {
        self.join_writer()
            .err()
            .unwrap_or_else(|| anyhow!("logger writer thread disconnected"))
    }

    /// Drop the sender (ends the writer's `recv` loop) and join, returning the
    /// writer thread's result. Safe to call once; later calls are no-ops.
    fn join_writer(&mut self) -> Result<()> {
        drop(self.tx.take());
        match self.handle.take() {
            Some(h) => h
                .join()
                .map_err(|_| anyhow!("logger writer thread panicked"))?,
            None => Ok(()),
        }
    }

    /// Flush both buffer tails, close the channel, and join the writer thread,
    /// propagating its first error. Idempotent.
    pub fn flush_all(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        self.send_state()?;
        self.send_slo()?;
        self.join_writer()
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
        let num = times.len() as u32;
        let finish = times.last().copied();
        RequestSloEntry {
            request_id: id,
            logging_time_ms: 100.0,
            completed: true,
            arrival_time_ms: 0.0,
            declared_ttft_slo_ms: None,
            declared_tpot_slo_ms: None,
            declared_e2e_slo_ms: None,
            output_token_times_ms: times,
            ttft_ms: Some(1.0),
            num_output_tokens: num,
            tpot_mean_ms: None,
            finish_decode_time_ms: finish,
            prefill_processed: 0,
            stage_times_ms: Vec::new(),
            stage_codes: Vec::new(),
            stage_pool_ids: Vec::new(),
            stage_worker_ids: Vec::new(),
            declared_prefix_tokens: 0,
            prefix_cache_hit_tokens: Some(0),
            fresh_prompt_tokens: 0,
            retraction_count: 0,
            reprocessed_prefill_output_tokens_before: Vec::new(),
            reprocessed_prefill_prefix_hit_tokens: Vec::new(),
            reprocessed_prefill_processed_tokens: Vec::new(),
            reprocessed_prefill_completed: Vec::new(),
            speculative_progress: None,
        }
    }

    #[test]
    fn logger_session_writes_both_parquets() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut log = LoggerSession::open(dir.path(), true, false).unwrap();
            log.record_request_slo(slo_entry(0, vec![1.0, 2.0]))
                .unwrap();
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
