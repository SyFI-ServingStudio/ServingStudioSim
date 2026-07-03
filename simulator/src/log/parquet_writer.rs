//! Generic streaming Parquet writer — appends multiple Arrow `RecordBatch`es to
//! one file, opening the file lazily on first non-empty write. Ported from
//! `ref/moesim-rs/src/logging/parquet_writer.rs` (ZSTD-L3 compression).

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties};

fn writer_properties(dictionary_enabled: bool, statistics_enabled: bool) -> Result<WriterProperties> {
    Ok(WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        // Dictionary encoding dedups each cell against a per-column dictionary via a
        // hash + `memcmp` (`Interner::intern`). For a column with a handful of
        // distinct short strings written millions of times (the `gpu_cluster` net log:
        // `src/dst_pool_tag`, `kind`, always-empty `tag`) that per-cell dedup is the
        // dominant encode cost, yet buys almost nothing over PLAIN + ZSTD (the
        // repetition compresses away regardless). Callers that stream such a column at
        // high volume disable it to keep the writer thread off the sim's critical path.
        .set_dictionary_enabled(dictionary_enabled)
        // Per-column min/max statistics compare every cell to the running min/max —
        // for byte-array (string) columns that is a `memcmp` per cell, the dominant
        // remaining encode cost once dictionary is off. Nothing reads the net log's
        // per-column min/max, so high-volume callers disable it as well.
        .set_statistics_enabled(if statistics_enabled {
            EnabledStatistics::Page
        } else {
            EnabledStatistics::None
        })
        .build())
}

/// Appends record batches to one parquet file. The file (and its parent dir)
/// is created on the first non-empty `write`, so a stream that never produces a
/// row leaves no file behind.
pub struct StreamingParquetWriter {
    path: PathBuf,
    schema: Arc<Schema>,
    writer: Option<ArrowWriter<File>>,
    rows_written: usize,
    /// Parquet dictionary encoding — on by default (best on-disk size for the
    /// low-cardinality cost/kv streams). A high-volume, low-cardinality-string
    /// stream (the net log) turns it off via [`Self::with_dictionary_enabled`] to
    /// drop the per-cell interner `memcmp` that otherwise gates the writer thread.
    dictionary_enabled: bool,
    /// Parquet per-column min/max statistics — on by default. The net log turns it
    /// off via [`Self::with_statistics_enabled`] to drop the per-cell `memcmp` its
    /// (unused) string-column min/max tracking costs.
    statistics_enabled: bool,
}

impl StreamingParquetWriter {
    pub fn new(path: PathBuf, schema: Arc<Schema>) -> Self {
        Self {
            path,
            schema,
            writer: None,
            rows_written: 0,
            dictionary_enabled: true,
            statistics_enabled: true,
        }
    }

    /// Opt out of parquet dictionary encoding for this stream (see the field docs).
    /// Must be set before the first `write` opens the file.
    pub fn with_dictionary_enabled(mut self, enabled: bool) -> Self {
        self.dictionary_enabled = enabled;
        self
    }

    /// Opt out of parquet per-column min/max statistics for this stream (see the
    /// field docs). Must be set before the first `write` opens the file.
    pub fn with_statistics_enabled(mut self, enabled: bool) -> Self {
        self.statistics_enabled = enabled;
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn rows_written(&self) -> usize {
        self.rows_written
    }

    fn writer_mut(&mut self) -> Result<&mut ArrowWriter<File>> {
        if self.writer.is_none() {
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = File::create(&self.path)?;
            let writer = ArrowWriter::try_new(
                file,
                self.schema.clone(),
                Some(writer_properties(
                    self.dictionary_enabled,
                    self.statistics_enabled,
                )?),
            )?;
            self.writer = Some(writer);
        }
        Ok(self.writer.as_mut().unwrap())
    }

    pub fn write(&mut self, batch: &RecordBatch) -> Result<usize> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(0);
        }
        self.writer_mut()?.write(batch)?;
        self.rows_written += num_rows;
        Ok(num_rows)
    }

    pub fn close(&mut self) -> Result<usize> {
        if let Some(writer) = self.writer.take() {
            writer.close()?;
        }
        Ok(self.rows_written)
    }
}

#[cfg(test)]
mod tests {
    //! Round-trip diagnostics for the cost_log corruption. Each test writes
    //! multiple batches through `StreamingParquetWriter` exactly the way
    //! `cost_logger.rs` does, then reads them back with the public parquet
    //! reader — so we exercise the real on-disk encode/decode path, not just
    //! the in-memory `RecordBatch` construction that `rows.rs` tests cover.
    use super::*;
    use crate::log::rows::{cost_to_record_batch, CostLogChunk, CostLogEntry, GroupInputLog};
    use crate::log::schemas::cost_log_schema;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;
    use tempfile::tempdir;

    /// One synthetic cost_log row in the flat layout: every variable-length
    /// field gets pushed into the chunk's flat buffer; the entry carries only
    /// the per-row lengths. Mirrors the worker's `start_iter` path.
    fn push_row(chunk: &mut CostLogChunk, worker_id: u16, iter_id: u64, slot_count: usize) {
        let g = GroupInputLog {
            batch_tokens: 8,
            prefill_tokens: 4,
            decode_request_count: 1,
            decode_kv_total: 10,
            prefill_chunk_pairs: vec![(0, 4)],
        };
        chunk.group_logs.push(g);
        for s in 0..slot_count {
            chunk.slot_times.push(0.1 + s as f32 * 0.01);
            chunk.slot_covs.push((s & 0xFF) as u8);
            chunk.slot_flops.push(s as f32 * 1e9);
            chunk.slot_bytes.push(s as f32 * 1e6);
        }
        chunk.entries.push(CostLogEntry {
            worker_id,
            iter_id,
            batch_id: 0,
            wall_start_ms: iter_id as f64 * 0.1,
            total_time_ms: 0.5,
            energy_j: 0.1,
            section: "iter",
            layer: -1,
            group_len: 1,
            slot_len: slot_count,
            slot_input_len: 0,
        });
    }

    fn count_rows(path: &Path) -> Result<usize> {
        let reader =
            ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?;
        let mut total = 0;
        for b in reader {
            total += b?.num_rows();
        }
        Ok(total)
    }

    #[test]
    fn streaming_cost_log_round_trip_single_batch() {
        // One 8_192-row batch (one row group), no slot_inputs. The simplest
        // shape — if this fails, the writer can't even produce a valid file.
        let dir = tempdir().unwrap();
        let path = dir.path().join("cost_log.parquet");
        let mut writer = StreamingParquetWriter::new(path.clone(), cost_log_schema());
        let mut chunk = CostLogChunk::with_capacity("main", 8192, 8192, 8192 * 4, 0);
        for i in 0..8192u64 {
            push_row(&mut chunk, 0, i, 4);
        }
        let rb = cost_to_record_batch(&chunk).unwrap();
        writer.write(&rb).unwrap();
        writer.close().unwrap();
        assert_eq!(count_rows(&path).unwrap(), 8192);
    }

    #[test]
    fn streaming_cost_log_round_trip_multi_batch() {
        // Two 8_192-row batches sent through the same writer instance — this
        // is the production pattern (`cost_logger.rs` ships one chunk per
        // 8_192 buffered rows). Reproduces the "Page was smaller than
        // expected" corruption when there is something wrong about how the
        // writer threads multiple batches through one row-group boundary.
        let dir = tempdir().unwrap();
        let path = dir.path().join("cost_log.parquet");
        let mut writer = StreamingParquetWriter::new(path.clone(), cost_log_schema());
        for batch_idx in 0..2u64 {
            let mut chunk = CostLogChunk::with_capacity("main", 8192, 8192, 8192 * 4, 0);
            for i in 0..8192u64 {
                push_row(&mut chunk, 0, batch_idx * 8192 + i, 4);
            }
            let rb = cost_to_record_batch(&chunk).unwrap();
            writer.write(&rb).unwrap();
        }
        writer.close().unwrap();
        assert_eq!(count_rows(&path).unwrap(), 16384);
    }
}
