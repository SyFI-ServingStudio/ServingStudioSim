//! `CostLogger` — an isolated per-iteration `cost_log` writer owned by a worker.
//!
//! Mirrors [`LoggerSession`](crate::log::session::LoggerSession)'s threading
//! (sim thread buffers rows; a background thread encodes + ZSTD-compresses +
//! writes the parquet), but stands alone so cost logging doesn't perturb the
//! per-request streams or the run loop. The worker constructs one when a log dir
//! is available, writes the `cost_manifest.json` sidecar once (slots + aggregation
//! structure, for reproducing the total), then pushes a `CostLogEntry` per iteration.

use std::path::Path;
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread::JoinHandle;

use anyhow::{anyhow, Result};

use crate::log::parquet_writer::StreamingParquetWriter;
use crate::log::rows::{cost_to_record_batch, CostLogChunk, CostLogEntry};
use crate::log::schemas::cost_log_schema;
use crate::timing::{CostManifest, SlotInput};

const STREAM_FLUSH_ROWS: usize = 8_192;
const CHANNEL_CAP: usize = 64;

/// Sim-thread handle: buffers `CostLogEntry` rows and offloads encode/write to a
/// background thread. `flush_all` (also `Drop`) sends the tail and joins.
pub struct CostLogger {
    tx: Option<SyncSender<CostLogChunk>>,
    handle: Option<JoinHandle<Result<()>>>,
    buf: CostLogChunk,
    slot_inputs_per_row: usize,
    closed: bool,
}

impl CostLogger {
    /// Open `<log_dir>/raw/cost_log.parquet` + spawn the writer thread, and write
    /// the [`CostManifest`] to `<log_dir>/raw/cost_manifest.json`. The manifest
    /// carries the ordered slots (positions map to the parquet's `slot_*` lists)
    /// *and* the flattened aggregation nodes, so a consumer can reproduce
    /// `total_time_ms` from a row's per-slot breakdown.
    pub fn open(log_dir: &Path, manifest: &CostManifest) -> Result<Self> {
        let raw = log_dir.join("raw");
        std::fs::create_dir_all(&raw)?;
        std::fs::write(
            raw.join("cost_manifest.json"),
            serde_json::to_vec_pretty(manifest)?,
        )?;

        let mut writer =
            StreamingParquetWriter::new(raw.join("cost_log.parquet"), cost_log_schema());
        let (tx, rx) = sync_channel::<CostLogChunk>(CHANNEL_CAP);
        let handle = std::thread::Builder::new()
            .name("mlsim-cost-logger".to_string())
            .spawn(move || -> Result<()> {
                for chunk in rx {
                    writer.write(&cost_to_record_batch(&chunk)?)?;
                }
                writer.close()?;
                Ok(())
            })?;

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            buf: CostLogChunk::with_capacity(STREAM_FLUSH_ROWS, 0),
            slot_inputs_per_row: 0,
            closed: false,
        })
    }

    pub fn record(
        &mut self,
        mut entry: CostLogEntry,
        slot_inputs: &mut Vec<SlotInput>,
    ) -> Result<()> {
        let slot_input_len = slot_inputs.len();
        if slot_input_len > 0 {
            if self.slot_inputs_per_row == 0 {
                self.slot_inputs_per_row = slot_input_len;
                self.buf
                    .slot_inputs
                    .reserve_exact(STREAM_FLUSH_ROWS * slot_input_len);
            } else {
                debug_assert_eq!(
                    slot_input_len, self.slot_inputs_per_row,
                    "slot_input capture must stay slot-aligned"
                );
            }
        }
        entry.slot_input_len = slot_input_len;
        self.buf.slot_inputs.append(slot_inputs);
        self.buf.entries.push(entry);
        if self.buf.len() >= STREAM_FLUSH_ROWS {
            self.send()?;
        }
        Ok(())
    }

    fn send(&mut self) -> Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let slot_input_capacity = STREAM_FLUSH_ROWS * self.slot_inputs_per_row;
        let chunk = std::mem::replace(
            &mut self.buf,
            CostLogChunk::with_capacity(STREAM_FLUSH_ROWS, slot_input_capacity),
        );
        match self
            .tx
            .as_ref()
            .expect("tx present until flush")
            .send(chunk)
        {
            Ok(()) => Ok(()),
            Err(_) => Err(self
                .join_writer()
                .err()
                .unwrap_or_else(|| anyhow!("cost-logger writer thread disconnected"))),
        }
    }

    fn join_writer(&mut self) -> Result<()> {
        drop(self.tx.take());
        match self.handle.take() {
            Some(h) => h
                .join()
                .map_err(|_| anyhow!("cost-logger writer thread panicked"))?,
            None => Ok(()),
        }
    }

    /// Flush the buffer tail, close the channel, and join the writer. Idempotent.
    pub fn flush_all(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        self.send()?;
        self.join_writer()
    }
}

impl Drop for CostLogger {
    fn drop(&mut self) {
        let _ = self.flush_all();
    }
}
