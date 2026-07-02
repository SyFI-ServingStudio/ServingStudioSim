//! `NetworkLogger` — the `gpu_cluster` stream writer, owned by the run's single
//! shared [`GpuCluster`](crate::worker::gpu_cluster::GpuCluster).
//!
//! Mirrors [`CostLogger`](crate::log::cost_logger::CostLogger)'s threading (sim
//! thread buffers rows; a background thread encodes + ZSTD-compresses + writes the
//! parquet) so transfer logging never stalls the sim, but is much simpler:
//!   - there is exactly ONE `GpuCluster` per run (a shared `Rc<RefCell<>>`), so
//!     exactly one writer → a single `raw/gpu_cluster.parquet`, with none of the
//!     per-worker-file race `CostLogger` guards against;
//!   - rows are all scalar (no per-slot list columns) and far rarer than cost-log
//!     rows (one per KV / activation handoff, not one per iteration), so a plain
//!     `Vec<GpuClusterEntry>` chunk suffices — no flat-buffer optimization.
//!
//! The cluster constructs one via [`attach_logger`](crate::worker::gpu_cluster::GpuCluster::attach_logger)
//! when a log dir is available, then pushes a [`GpuClusterEntry`] per transfer.

use std::path::Path;
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::thread::JoinHandle;

use anyhow::{anyhow, Result};

use crate::log::parquet_writer::StreamingParquetWriter;
use crate::log::rows::{gpu_cluster_to_record_batch, GpuClusterEntry};
use crate::log::schemas::gpu_cluster_schema;

const STREAM_FLUSH_ROWS: usize = 8_192;
const CHANNEL_CAP: usize = 64;

/// Sim-thread handle: buffers `GpuClusterEntry` rows and offloads encode/write to
/// a background thread. `flush_all` (also `Drop`) sends the tail and joins.
pub struct NetworkLogger {
    tx: Option<SyncSender<Vec<GpuClusterEntry>>>,
    handle: Option<JoinHandle<Result<()>>>,
    buf: Vec<GpuClusterEntry>,
    closed: bool,
}

impl NetworkLogger {
    /// Open `<log_dir>/raw/gpu_cluster.parquet` + spawn the writer thread. A single
    /// file (not a per-worker directory like `cost_log/`) because the whole run
    /// shares one `GpuCluster`, so there is only ever one writer.
    pub fn open(log_dir: &Path) -> Result<Self> {
        let raw = log_dir.join("raw");
        std::fs::create_dir_all(&raw)?;

        let path = raw.join("gpu_cluster.parquet");
        let mut writer = StreamingParquetWriter::new(path, gpu_cluster_schema());
        let (tx, rx) = sync_channel::<Vec<GpuClusterEntry>>(CHANNEL_CAP);
        let handle = std::thread::Builder::new()
            .name("mlsim-net-logger".to_string())
            .spawn(move || -> Result<()> {
                for chunk in rx {
                    writer.write(&gpu_cluster_to_record_batch(&chunk)?)?;
                }
                writer.close()?;
                Ok(())
            })?;

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            buf: Vec::with_capacity(STREAM_FLUSH_ROWS),
            closed: false,
        })
    }

    /// Buffer one transfer row; flush the chunk once it reaches `STREAM_FLUSH_ROWS`.
    pub fn record(&mut self, entry: GpuClusterEntry) -> Result<()> {
        self.buf.push(entry);
        if self.buf.len() >= STREAM_FLUSH_ROWS {
            self.send()?;
        }
        Ok(())
    }

    fn send(&mut self) -> Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let chunk = std::mem::replace(&mut self.buf, Vec::with_capacity(STREAM_FLUSH_ROWS));
        let tx = self.tx.as_ref().expect("tx present until flush");
        // `try_send` first so a full channel (writer behind) warns about
        // backpressure before falling back to the blocking `send`. Mirrors
        // `CostLogger::send`.
        let chunk = match tx.try_send(chunk) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(chunk)) => {
                tracing::warn!(
                    "gpu-cluster-log channel full ({CHANNEL_CAP} chunks in flight): sim thread \
                     blocking on net-logger backpressure"
                );
                chunk
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(self
                    .join_writer()
                    .err()
                    .unwrap_or_else(|| anyhow!("net-logger writer thread disconnected")))
            }
        };
        match self.tx.as_ref().expect("tx present").send(chunk) {
            Ok(()) => Ok(()),
            Err(_) => Err(self
                .join_writer()
                .err()
                .unwrap_or_else(|| anyhow!("net-logger writer thread disconnected"))),
        }
    }

    fn join_writer(&mut self) -> Result<()> {
        drop(self.tx.take());
        match self.handle.take() {
            Some(h) => h
                .join()
                .map_err(|_| anyhow!("net-logger writer thread panicked"))?,
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

impl Drop for NetworkLogger {
    fn drop(&mut self) {
        let _ = self.flush_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;

    use arrow_array::{StringArray, UInt16Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    #[test]
    fn writes_gpu_cluster_parquet_with_both_endpoints() {
        let dir = tempdir().unwrap();
        let mut logger = NetworkLogger::open(dir.path()).unwrap();
        logger
            .record(GpuClusterEntry {
                net_start_ms: 5.0,
                net_end_ms: 9.0,
                src_pool_tag: "prefill",
                src_worker_id: 2,
                dst_pool_tag: "decode",
                dst_worker_id: 7,
                send_gid: 3,
                recv_gid: 11,
                send_count: 8,
                recv_count: 2,
                bytes: 4_000_000,
                kind: "pd_kv_pull",
                tag: "req=42".to_string(),
            })
            .unwrap();
        logger.flush_all().unwrap();

        let path = dir.path().join("raw/gpu_cluster.parquet");
        assert!(path.exists());
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        // Both endpoints' worker identity round-trip (src at cols 2/3, dst at 4/5).
        let src_pool = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(src_pool.value(0), "prefill");
        let dst_pool = batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(dst_pool.value(0), "decode");
        let dst_worker = batch
            .column(5)
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap();
        assert_eq!(dst_worker.value(0), 7);
        // send_count / recv_count round-trip (the two appended UInt16 columns).
        let send_count = batch
            .column_by_name("send_count")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap();
        assert_eq!(send_count.value(0), 8);
        let recv_count = batch
            .column_by_name("recv_count")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap();
        assert_eq!(recv_count.value(0), 2);
    }
}
