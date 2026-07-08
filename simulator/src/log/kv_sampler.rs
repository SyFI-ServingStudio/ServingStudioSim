//! `KvSampler` — the per-worker `kv_snapshot` stream writer **and** the KV-pool
//! occupancy sampling policy.
//!
//! The write half mirrors [`NetworkLogger`](crate::log::network_logger::NetworkLogger)
//! (sim thread buffers scalar rows; a background thread encodes + ZSTD-compresses +
//! writes the parquet) and the per-worker file layout of
//! [`CostLogger`](crate::log::cost_logger::CostLogger) (`worker_<pool_tag>_<id>`).
//! What it adds is the one thing no other logger has: a **sampling policy**. Every
//! other stream records exactly what it is handed, but a KV pool changes every
//! iteration, so logging each iteration's occupancy verbatim would be as chatty as
//! `cost_log`. Instead the worker `submit`s its per-group occupancy every
//! iteration (a dumb producer, holding no sampling state) and the sampler decides
//! what becomes a row:
//!   - a **running max** of `active_kv` over each throttle window, so decimated
//!     sampling never hides the true occupancy peak (the "current size");
//!   - a **stride throttle** — one row per `stride` submits — plus a baseline row
//!     on the first submit (a t≈0 anchor) and a tail row at [`flush_all`] (the
//!     endpoint), so the series always has both ends;
//!   - `projected_peak` / `promised_kv` are sampled at emit time — they are already
//!     forward-looking, so a running max would buy nothing.
//!
//! The worker opens one when a log dir is available and calls
//! [`submit`](KvSampler::submit) per group per iteration; the whole policy lives
//! here, not in the worker.

use std::path::Path;
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::thread::JoinHandle;

use anyhow::{anyhow, Result};

use crate::common::{Time, WorkerId};
use crate::log::cost_logger::cost_artifact_stem;
use crate::log::parquet_writer::StreamingParquetWriter;
use crate::log::rows::{kv_to_record_batch, KvSnapshotEntry};
use crate::log::schemas::kv_snapshot_schema;

const STREAM_FLUSH_ROWS: usize = 8_192;
const CHANNEL_CAP: usize = 64;

/// The three occupancy figures a worker hands the sampler each iteration, all in
/// tokens. `active_kv` is the committed KV *now*; `projected_peak` is the max the
/// currently-admitted set will reach as it drains; `promised_kv` is admitted-but-
/// not-yet-realized. See [`KvSampler::submit`].
#[derive(Clone, Copy, Debug)]
pub struct KvSubmit {
    pub active_kv: u64,
    pub projected_peak: u64,
    pub promised_kv: u64,
}

/// Per-group sampling accumulator. `window_count` is the stride counter (submits
/// since the last emitted row); `peak_active` is the running max of `active_kv`
/// over that window; the `last_*` fields hold the most recent submit so
/// [`KvSampler::flush_all`] can emit a tail row at the true endpoint.
#[derive(Clone, Copy, Default)]
struct Slot {
    window_count: u32,
    peak_active: u64,
    emitted_any: bool,
    /// Un-emitted window data is buffered (a tail row is owed at flush).
    pending: bool,
    last_time_ms: f64,
    last_projected_peak: u64,
    last_promised: u64,
}

/// Sim-thread handle: runs the sampling policy, buffers the emitted
/// [`KvSnapshotEntry`] rows, and offloads encode/write to a background thread.
/// `flush_all` (also `Drop`) emits each group's tail row, sends the buffer tail,
/// and joins. One per worker (like `CostLogger`), so one
/// `raw/kv_snapshot/worker_<pool_tag>_<id>.parquet` per worker.
pub struct KvSampler {
    tx: Option<SyncSender<Vec<KvSnapshotEntry>>>,
    handle: Option<JoinHandle<Result<()>>>,
    buf: Vec<KvSnapshotEntry>,
    /// Per-group accumulators, indexed by `group_id` (length = `num_groups`).
    slots: Vec<Slot>,
    /// Emit one row every `stride` submits (>= 1; a value of 1 logs every submit).
    stride: u32,
    worker_id: WorkerId,
    closed: bool,
}

impl KvSampler {
    /// Convenience for worker construction: `None` when there is no log dir or the
    /// writer fails to open (logged, never fatal — a logging problem must not abort
    /// the sim), mirroring how `CostBuffers` degrades its `CostLogger`.
    pub fn open_opt(
        log_dir: Option<&Path>,
        pool_tag: &'static str,
        worker_id: WorkerId,
        num_groups: usize,
        stride: u32,
    ) -> Option<Self> {
        let dir = log_dir?;
        match Self::open(dir, pool_tag, worker_id, num_groups, stride) {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!("kv_snapshot disabled: failed to open writer: {e:#}");
                None
            }
        }
    }

    /// Open `<log_dir>/raw/kv_snapshot/worker_<pool_tag>_<id>.parquet` + spawn the
    /// writer thread. Per-worker file (a shared `kv_snapshot/` dir) for the same
    /// reason `cost_log/` is per-worker — every worker writing one shared file
    /// trampled each other's footer. `pool_tag` is captured into the writer thread
    /// (it is the whole stream's tag, so `kv_to_record_batch` takes it once rather
    /// than per row).
    pub fn open(
        log_dir: &Path,
        pool_tag: &'static str,
        worker_id: WorkerId,
        num_groups: usize,
        stride: u32,
    ) -> Result<Self> {
        let kv_dir = log_dir.join("raw").join("kv_snapshot");
        std::fs::create_dir_all(&kv_dir)?;
        let path = kv_dir.join(format!("{}.parquet", cost_artifact_stem(pool_tag, worker_id)));
        let mut writer = StreamingParquetWriter::new(path, kv_snapshot_schema());
        let (tx, rx) = sync_channel::<Vec<KvSnapshotEntry>>(CHANNEL_CAP);
        let handle = std::thread::Builder::new()
            .name("vibesim-kv-sampler".to_string())
            .spawn(move || -> Result<()> {
                for chunk in rx {
                    writer.write(&kv_to_record_batch(pool_tag, &chunk)?)?;
                }
                writer.close()?;
                Ok(())
            })?;

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            buf: Vec::with_capacity(STREAM_FLUSH_ROWS),
            slots: vec![Slot::default(); num_groups.max(1)],
            stride: stride.max(1),
            worker_id,
            closed: false,
        })
    }

    /// Feed one group's occupancy for this iteration. Infallible to the worker: the
    /// sampling policy runs here, and a row is emitted only on the baseline / every
    /// `stride`th submit (its `active_kv` is the window running max). A downstream
    /// writer failure is logged once and disables further logging rather than
    /// propagating into the sim FSM.
    pub fn submit(&mut self, group_id: u16, s: KvSubmit, now: Time) {
        if self.closed {
            return;
        }
        let stride = self.stride;
        let slot = &mut self.slots[group_id as usize];
        slot.peak_active = slot.peak_active.max(s.active_kv);
        slot.last_time_ms = now.as_ms();
        slot.last_projected_peak = s.projected_peak;
        slot.last_promised = s.promised_kv;
        slot.window_count += 1;
        // Baseline anchors the series at the first submit; afterwards emit once the
        // window fills `stride` submits.
        let emit = !slot.emitted_any || slot.window_count >= stride;
        if !emit {
            slot.pending = true;
            return;
        }
        let row = KvSnapshotEntry {
            worker_id: self.worker_id.0,
            group_id,
            time_ms: slot.last_time_ms,
            active_kv: slot.peak_active,
            projected_peak: slot.last_projected_peak,
            promised_kv: slot.last_promised,
        };
        slot.emitted_any = true;
        slot.window_count = 0;
        slot.peak_active = 0;
        slot.pending = false;
        self.push_row(row);
    }

    /// Buffer one emitted row; flush the chunk once it reaches `STREAM_FLUSH_ROWS`.
    /// A flush failure warns once and latches `closed` so the sim never blocks on a
    /// dead writer.
    fn push_row(&mut self, row: KvSnapshotEntry) {
        self.buf.push(row);
        if self.buf.len() >= STREAM_FLUSH_ROWS {
            if let Err(e) = self.send() {
                tracing::warn!("kv_snapshot log record failed: {e:#}");
                self.closed = true;
            }
        }
    }

    fn send(&mut self) -> Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let chunk = std::mem::replace(&mut self.buf, Vec::with_capacity(STREAM_FLUSH_ROWS));
        let tx = self.tx.as_ref().expect("tx present until flush");
        // `try_send` first so a full channel (writer behind) warns about
        // backpressure before the blocking `send`. Mirrors `NetworkLogger::send`.
        let chunk = match tx.try_send(chunk) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(chunk)) => {
                tracing::warn!(
                    "kv-sampler channel full ({CHANNEL_CAP} chunks in flight): sim thread \
                     blocking on kv-sampler backpressure"
                );
                chunk
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(self
                    .join_writer()
                    .err()
                    .unwrap_or_else(|| anyhow!("kv-sampler writer thread disconnected")))
            }
        };
        match self.tx.as_ref().expect("tx present").send(chunk) {
            Ok(()) => Ok(()),
            Err(_) => Err(self
                .join_writer()
                .err()
                .unwrap_or_else(|| anyhow!("kv-sampler writer thread disconnected"))),
        }
    }

    fn join_writer(&mut self) -> Result<()> {
        drop(self.tx.take());
        match self.handle.take() {
            Some(h) => h
                .join()
                .map_err(|_| anyhow!("kv-sampler writer thread panicked"))?,
            None => Ok(()),
        }
    }

    /// Emit each group's owed tail row (so every series ends at its last activity),
    /// flush the buffer tail, close the channel, and join the writer. Idempotent.
    pub fn flush_all(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        // Tail rows: any group whose last window never reached `stride` still owes
        // one row at its final sampled state.
        let tails: Vec<KvSnapshotEntry> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.pending)
            .map(|(gid, slot)| KvSnapshotEntry {
                worker_id: self.worker_id.0,
                group_id: gid as u16,
                time_ms: slot.last_time_ms,
                active_kv: slot.peak_active,
                projected_peak: slot.last_projected_peak,
                promised_kv: slot.last_promised,
            })
            .collect();
        self.buf.extend(tails);
        self.send()?;
        self.join_writer()
    }
}

impl Drop for KvSampler {
    fn drop(&mut self) {
        let _ = self.flush_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;

    use arrow_array::{UInt64Array, UInt16Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    fn sub(active: u64) -> KvSubmit {
        KvSubmit { active_kv: active, projected_peak: active, promised_kv: 0 }
    }

    /// baseline + stride throttle + running-max + tail, end to end through the
    /// on-disk parquet. stride=3, one group; the submit sequence is chosen so the
    /// window max (50) lands strictly between two emit points, proving decimation
    /// does not hide the peak.
    #[test]
    fn throttle_running_max_baseline_and_tail() {
        let dir = tempdir().unwrap();
        let mut kv = KvSampler::open(dir.path(), "main", WorkerId(0), 1, 3).unwrap();
        // #1 baseline → row @10; #2/#3 accumulate (peak 50); #4 fills stride → row
        // @50 (running max over {50,20,30}); #5 pending → tail @5.
        for (i, active) in [10u64, 50, 20, 30, 5].iter().enumerate() {
            kv.submit(0, sub(*active), Time::from_ms(i as f64));
        }
        kv.flush_all().unwrap();

        let path = dir.path().join("raw/kv_snapshot/worker_main_0.parquet");
        assert!(path.exists());
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3, "baseline + one stride emit + tail");
        let active = batch
            .column(4)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(active.values(), &[10, 50, 5], "window running max preserved");
        // group_id column (index 2) is all group 0.
        let gid = batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap();
        assert_eq!(gid.value(0), 0);
    }

    /// A group that never fills a stride still gets exactly one (baseline) row, and
    /// the tail does not duplicate it (baseline already cleared `pending`).
    #[test]
    fn single_submit_is_one_baseline_row() {
        let dir = tempdir().unwrap();
        let mut kv = KvSampler::open(dir.path(), "decode", WorkerId(2), 1, 8).unwrap();
        kv.submit(0, sub(7), Time::from_ms(1.0));
        kv.flush_all().unwrap();

        let path = dir.path().join("raw/kv_snapshot/worker_decode_2.parquet");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1, "baseline only; tail must not duplicate it");
    }
}
