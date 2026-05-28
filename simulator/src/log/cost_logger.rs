//! `CostLogger` — an isolated per-iteration `cost_log` writer owned by a worker.
//!
//! Mirrors [`LoggerSession`](crate::log::session::LoggerSession)'s threading
//! (sim thread buffers rows; a background thread encodes + ZSTD-compresses +
//! writes the parquet), but stands alone so cost logging doesn't perturb the
//! per-request streams or the run loop. The worker constructs one when a log dir
//! is available, writes its matching `cost_manifest/worker_<pool>_<id>.json`
//! sidecar once (slots + aggregation structure, for reproducing the total), then
//! pushes a `CostLogEntry` per iteration.

use std::path::Path;
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::thread::JoinHandle;

use anyhow::{anyhow, Result};

use crate::common::WorkerId;
use crate::log::parquet_writer::StreamingParquetWriter;
use crate::log::rows::{cost_to_record_batch, CostLogChunk, CostLogEntry, GroupInputLog};
use crate::log::schemas::cost_log_schema;
use crate::timing::{CostManifest, LeafMetrics, SlotInput};

const STREAM_FLUSH_ROWS: usize = 8_192;
const CHANNEL_CAP: usize = 64;

pub(crate) fn cost_artifact_stem(pool_tag: &str, worker_id: WorkerId) -> String {
    assert!(
        !pool_tag.contains('/') && !pool_tag.contains('\\'),
        "pool_tag must be path-segment safe"
    );
    format!("worker_{}_{}", pool_tag, worker_id.0)
}

/// Sim-thread handle: buffers `CostLogEntry` rows and offloads encode/write to a
/// background thread. `flush_all` (also `Drop`) sends the tail and joins.
pub struct CostLogger {
    tx: Option<SyncSender<CostLogChunk>>,
    handle: Option<JoinHandle<Result<()>>>,
    buf: CostLogChunk,
    pool_tag: &'static str,
    /// Per-row flat-buffer sizes, learned from the first recorded row (all rows
    /// of one worker share the same compiled CostTree, so these are constant).
    /// Used to size a freshly-rotated chunk's buffers in [`Self::send`].
    groups_per_row: usize,
    slots_per_row: usize,
    slot_inputs_per_row: usize,
    closed: bool,
}

impl CostLogger {
    /// Open `<log_dir>/raw/cost_log/worker_<pool_tag>_<id>.parquet` (per-worker
    /// file under a shared `cost_log/` directory) + spawn the writer thread, and
    /// write the matching [`CostManifest`] to
    /// `<log_dir>/raw/cost_manifest/worker_<pool_tag>_<id>.json`.
    /// Per-worker files avoid a race: all workers ran `File::create` on one
    /// shared `cost_log.parquet` and trampled each other's writes — only the
    /// last worker's footer ever survived. The `pool_tag` disambiguates across
    /// pools whose `WorkerId`s both restart at 0 (PD has prefill worker 0 *and*
    /// decode worker 0). The manifest carries the ordered slots (positions map
    /// to the parquet's `slot_*` lists) *and* the flattened aggregation nodes,
    /// so a consumer can reproduce `total_time_ms` from a row's per-slot
    /// breakdown for that specific `(pool_tag, worker_id)` stream.
    pub fn open(
        log_dir: &Path,
        pool_tag: &'static str,
        worker_id: WorkerId,
        manifest: &CostManifest,
    ) -> Result<Self> {
        let raw = log_dir.join("raw");
        let cost_dir = raw.join("cost_log");
        let manifest_dir = raw.join("cost_manifest");
        std::fs::create_dir_all(&cost_dir)?;
        std::fs::create_dir_all(&manifest_dir)?;

        let stem = cost_artifact_stem(pool_tag, worker_id);
        std::fs::write(
            manifest_dir.join(format!("{stem}.json")),
            serde_json::to_vec_pretty(manifest)?,
        )?;

        let path = cost_dir.join(format!("{stem}.parquet"));
        let mut writer = StreamingParquetWriter::new(path, cost_log_schema());
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
            buf: CostLogChunk::with_capacity(pool_tag, STREAM_FLUSH_ROWS, 0, 0, 0),
            pool_tag,
            groups_per_row: 0,
            slots_per_row: 0,
            slot_inputs_per_row: 0,
            closed: false,
        })
    }

    /// Buffer one cost-log row. The variable-length fields are appended into the
    /// chunk's flat buffers (reused capacity) rather than owned per-row `Vec`s:
    /// `slots` (the per-slot `LeafMetrics`) is split into the `time`/`coverage`
    /// columns here; `groups` and `slot_inputs` are moved over (drained, so the
    /// worker keeps their capacity). The first row fixes the per-row sizes used
    /// to pre-size a rotated chunk in [`Self::send`].
    pub fn record(
        &mut self,
        mut entry: CostLogEntry,
        slots: &[LeafMetrics],
        groups: &mut Vec<GroupInputLog>,
        slot_inputs: &mut Vec<SlotInput>,
    ) -> Result<()> {
        let (group_len, slot_len, slot_input_len) = (groups.len(), slots.len(), slot_inputs.len());
        if self.slots_per_row == 0 && slot_len > 0 {
            self.groups_per_row = group_len;
            self.slots_per_row = slot_len;
            self.slot_inputs_per_row = slot_input_len;
            self.buf
                .group_logs
                .reserve(STREAM_FLUSH_ROWS * group_len.max(1));
            self.buf.slot_times.reserve(STREAM_FLUSH_ROWS * slot_len);
            self.buf.slot_covs.reserve(STREAM_FLUSH_ROWS * slot_len);
            self.buf
                .slot_inputs
                .reserve(STREAM_FLUSH_ROWS * slot_input_len);
        }
        entry.group_len = group_len;
        entry.slot_len = slot_len;
        entry.slot_input_len = slot_input_len;
        self.buf
            .slot_times
            .extend(slots.iter().map(|l| l.m.time_ms));
        self.buf
            .slot_covs
            .extend(slots.iter().map(|l| l.coverage.bits()));
        self.buf.group_logs.append(groups);
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
        let chunk = std::mem::replace(
            &mut self.buf,
            CostLogChunk::with_capacity(
                self.pool_tag,
                STREAM_FLUSH_ROWS,
                STREAM_FLUSH_ROWS * self.groups_per_row.max(1),
                STREAM_FLUSH_ROWS * self.slots_per_row,
                STREAM_FLUSH_ROWS * self.slot_inputs_per_row,
            ),
        );
        let tx = self.tx.as_ref().expect("tx present until flush");
        // `try_send` first so a full channel (writer behind) warns about
        // backpressure before falling back to the blocking `send` that stalls the
        // sim thread until the writer drains a slot. Mirrors `LoggerSession::send`.
        let chunk = match tx.try_send(chunk) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(chunk)) => {
                tracing::warn!(
                    "cost-log channel full ({CHANNEL_CAP} chunks in flight): sim thread \
                     blocking on cost-logger backpressure"
                );
                chunk
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(self
                    .join_writer()
                    .err()
                    .unwrap_or_else(|| anyhow!("cost-logger writer thread disconnected")))
            }
        };
        match self.tx.as_ref().expect("tx present").send(chunk) {
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;

    use arrow_array::StringArray;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    use crate::log::CostLogEntry;
    use crate::timing::{CoverageFlags, FlatCostNode, LeafDesc, Metrics4};

    #[test]
    fn writes_pool_worker_scoped_cost_artifacts() {
        let dir = tempdir().unwrap();
        let manifest = CostManifest {
            slots: vec![LeafDesc {
                name: "m.test".to_owned(),
                kind: "unit".to_owned(),
                config: "shape=1".to_owned(),
            }],
            nodes: vec![FlatCostNode::Leaf(0)],
            node_labels: vec![None],
        };

        let mut logger =
            CostLogger::open(dir.path(), "decode", WorkerId(7), &manifest).unwrap();
        let entry = CostLogEntry {
            worker_id: 7,
            iter_id: 3,
            batch_id: 0,
            wall_start_ms: 10.0,
            total_time_ms: 1.25,
            energy_j: 0.0,
            group_len: 0,
            slot_len: 0,
            slot_input_len: 0,
        };
        let slots = vec![LeafMetrics {
            m: Metrics4 {
                time_ms: 1.25,
                flops: 0.0,
                bytes: 0.0,
                energy_j: 0.0,
            },
            coverage: CoverageFlags::EMPTY,
        }];
        let mut groups = Vec::new();
        let mut slot_inputs = Vec::new();
        logger
            .record(entry, &slots, &mut groups, &mut slot_inputs)
            .unwrap();
        logger.flush_all().unwrap();

        let manifest_path = dir
            .path()
            .join("raw/cost_manifest/worker_decode_7.json");
        assert!(manifest_path.exists());
        let parquet_path = dir.path().join("raw/cost_log/worker_decode_7.parquet");
        let mut reader =
            ParquetRecordBatchReaderBuilder::try_new(File::open(&parquet_path).unwrap())
                .unwrap()
                .build()
                .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        let pool = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(pool.value(0), "decode");
    }
}
