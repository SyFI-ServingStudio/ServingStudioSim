//! Generic streaming Parquet writer — appends multiple Arrow `RecordBatch`es to
//! one file, opening the file lazily on first non-empty write. Ported from
//! `ref/moesim-rs/src/logging/parquet_writer.rs` (ZSTD-L3 compression).

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema};
use parquet::arrow::arrow_writer::{compute_leaves, get_column_writers, ArrowColumnWriter};
use parquet::arrow::{add_encoded_arrow_schema_to_metadata, ArrowSchemaConverter};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::{ColumnPath, SchemaDescriptor};

fn writer_properties(
    dictionary_enabled: bool,
    statistics_enabled: bool,
    no_dictionary_columns: &[ColumnPath],
    no_statistics_columns: &[ColumnPath],
) -> Result<WriterProperties> {
    let mut builder = WriterProperties::builder();
    for column in no_dictionary_columns {
        builder = builder.set_column_dictionary_enabled(column.clone(), false);
    }
    for column in no_statistics_columns {
        builder = builder.set_column_statistics_enabled(column.clone(), EnabledStatistics::None);
    }
    Ok(builder
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

/// Arrow bytes a row group may take in before it is written out. The writer
/// otherwise closes a row group only at 1M rows, which a cost log never reaches,
/// and it buffers each finished page in a `Vec` sized to the page's
/// *uncompressed* bytes: an 8 h PP8 run held every stage's whole cost log at
/// its uncompressed size (~4 GB a stage) until the run ended. The input batches'
/// Arrow size bounds that, where the writer's own `in_progress_size` counts the
/// compressed bytes, ~11x fewer for `slot_input`.
const ROW_GROUP_FLUSH_BYTES: usize = 64 << 20;

/// Appends record batches to one parquet file. The file (and its parent dir)
/// is created on the first non-empty `write`, so a stream that never produces a
/// row leaves no file behind.
pub struct StreamingParquetWriter {
    path: PathBuf,
    schema: Arc<Schema>,
    writer: Option<OpenFile>,
    rows_written: usize,
    /// Arrow bytes written since the last row group closed.
    unflushed_bytes: usize,
    /// Parquet dictionary encoding — on by default (best on-disk size for the
    /// low-cardinality cost/kv streams). A high-volume, low-cardinality-string
    /// stream (the net log) turns it off via [`Self::with_dictionary_enabled`] to
    /// drop the per-cell interner `memcmp` that otherwise gates the writer thread.
    dictionary_enabled: bool,
    /// Parquet per-column min/max statistics — on by default. The net log turns it
    /// off via [`Self::with_statistics_enabled`] to drop the per-cell `memcmp` its
    /// (unused) string-column min/max tracking costs.
    statistics_enabled: bool,
    /// Leaf columns whose dictionary is disabled individually. See
    /// [`Self::with_column_dictionary_disabled`].
    no_dictionary_columns: Vec<ColumnPath>,
    /// Leaf columns whose min/max statistics are disabled individually. See
    /// [`Self::with_column_statistics_disabled`].
    no_statistics_columns: Vec<ColumnPath>,
    /// Threads that encode a batch's columns, read at every `write`. Starts at
    /// one; an owner that sees this writer fall behind raises it (see
    /// [`EncodeThreads`]). The file is the same at any count: each column is
    /// encoded by exactly one thread, in batch order.
    encode_threads: Arc<AtomicUsize>,
}

/// An open file and the column writers of its in-progress row group: the
/// pieces of `parquet::arrow::ArrowWriter`, held apart so the columns of one
/// batch can be encoded on several threads.
struct OpenFile {
    file: SerializedFileWriter<File>,
    /// `None` between row groups.
    columns: Option<Vec<ArrowColumnWriter>>,
    buffered_rows: usize,
}

impl StreamingParquetWriter {
    pub fn new(path: PathBuf, schema: Arc<Schema>) -> Self {
        Self {
            path,
            schema,
            writer: None,
            rows_written: 0,
            unflushed_bytes: 0,
            dictionary_enabled: true,
            statistics_enabled: true,
            no_dictionary_columns: Vec::new(),
            no_statistics_columns: Vec::new(),
            encode_threads: Arc::new(AtomicUsize::new(1)),
        }
    }

    /// Share the encode thread count with the stream's owner, which raises it
    /// when the writer cannot keep up (see the field docs).
    pub fn with_encode_threads(mut self, threads: Arc<AtomicUsize>) -> Self {
        self.encode_threads = threads;
        self
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

    /// Disable dictionary encoding for individual leaf columns, leaving the rest
    /// of the stream's default alone.
    ///
    /// Dictionary is per *cell*: a hash plus a `memcmp` against the interner. It
    /// earns that back only where the column's cardinality is small enough for the
    /// dictionary to stand in for the data. Measured on a real 8h run's cost log
    /// (975,624 rows x 1,271 slots), per column, compressed:
    ///
    /// | column | with dictionary | without |
    /// |---|---|---|
    /// | `slot_time_ms` / `slot_flops` / `slot_bytes` | 105.1 MB | 103.7 MB |
    /// | `slot_input` | 120.8 MB | 119.7 MB |
    /// | `slot_coverage` | 0.6 MB | 198.6 MB |
    /// | `slot_backend` | 0.4 MB | 131.8 MB |
    ///
    /// So near-unique cells pay the interning and then overflow into PLAIN anyway
    /// (0.99x — very slightly *worse* on disk), while a handful of distinct `u8`s
    /// compress 322-346x through the dictionary's RLE. Disabling it blanket-wide
    /// more than doubled this file (247 -> 572 MB); disabling it only where it does
    /// not pay is free.
    ///
    /// Paths are per *leaf*, so a `List<item>` column is three parts — build them
    /// with [`ColumnPath::new`], since `ColumnPath::from("a.list.item")` does not
    /// split on `.` and silently matches nothing.
    /// Must be set before the first `write` opens the file.
    pub fn with_column_dictionary_disabled(mut self, columns: Vec<ColumnPath>) -> Self {
        self.no_dictionary_columns = columns;
        self
    }

    /// Disable min/max statistics for individual leaf columns, leaving the rest of
    /// the stream's default alone.
    ///
    /// Also per cell, and for a list column nothing can ever read the result: a
    /// predicate does not push down to a list element, so the page min/max is
    /// written and never consulted. On the same run's rows, turning it off for the
    /// six `slot_*` list columns cut the parquet encode 41% (227 -> 133 ns/cell)
    /// with the file byte-for-byte the same size.
    ///
    /// Same per-leaf path rule as [`Self::with_column_dictionary_disabled`].
    /// Must be set before the first `write` opens the file.
    pub fn with_column_statistics_disabled(mut self, columns: Vec<ColumnPath>) -> Self {
        self.no_statistics_columns = columns;
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn rows_written(&self) -> usize {
        self.rows_written
    }

    fn open_file(&mut self) -> Result<&mut OpenFile> {
        if self.writer.is_none() {
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut props = writer_properties(
                self.dictionary_enabled,
                self.statistics_enabled,
                &self.no_dictionary_columns,
                &self.no_statistics_columns,
            )?;
            // What `ArrowWriter::try_new` does, so the file is unchanged.
            let parquet_schema = ArrowSchemaConverter::new()
                .with_coerce_types(props.coerce_types())
                .convert(&self.schema)?;
            add_encoded_arrow_schema_to_metadata(&self.schema, &mut props);
            let file = SerializedFileWriter::new(
                File::create(&self.path)?,
                parquet_schema.root_schema_ptr(),
                Arc::new(props),
            )?;
            self.writer = Some(OpenFile {
                file,
                columns: None,
                buffered_rows: 0,
            });
        }
        Ok(self.writer.as_mut().unwrap())
    }

    pub fn write(&mut self, batch: &RecordBatch) -> Result<usize> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(0);
        }
        self.encode(batch)?;
        self.unflushed_bytes += batch.get_array_memory_size();
        if self.unflushed_bytes >= ROW_GROUP_FLUSH_BYTES {
            self.flush_row_group()?;
            self.unflushed_bytes = 0;
        }
        self.rows_written += num_rows;
        Ok(num_rows)
    }

    /// Append `batch` to the in-progress row group, splitting it at the row
    /// group size cap the way `ArrowWriter::write` does.
    fn encode(&mut self, batch: &RecordBatch) -> Result<()> {
        let threads = self.encode_threads.load(Ordering::Relaxed).max(1);
        let schema = self.schema.clone();
        let open = self.open_file()?;
        let max_rows = open.file.properties().max_row_group_size();
        if open.buffered_rows + batch.num_rows() > max_rows {
            let head = max_rows - open.buffered_rows;
            self.encode(&batch.slice(0, head))?;
            return self.encode(&batch.slice(head, batch.num_rows() - head));
        }
        if open.columns.is_none() {
            open.columns = Some(get_column_writers(
                open.file.schema_descr(),
                open.file.properties(),
                &schema,
            )?);
        }
        encode_columns(
            &schema,
            open.file.schema_descr(),
            batch,
            open.columns.as_mut().unwrap(),
            threads,
        )?;
        open.buffered_rows += batch.num_rows();
        if open.buffered_rows >= max_rows {
            self.flush_row_group()?;
        }
        Ok(())
    }

    /// Close the in-progress row group, if any (`ArrowWriter::flush`).
    fn flush_row_group(&mut self) -> Result<()> {
        let Some(open) = self.writer.as_mut() else {
            return Ok(());
        };
        let Some(columns) = open.columns.take() else {
            return Ok(());
        };
        open.buffered_rows = 0;
        let mut row_group = open.file.next_row_group()?;
        for column in columns {
            column.close()?.append_to_row_group(&mut row_group)?;
        }
        row_group.close()?;
        Ok(())
    }

    pub fn close(&mut self) -> Result<usize> {
        self.flush_row_group()?;
        if let Some(open) = self.writer.take() {
            open.file.close()?;
        }
        Ok(self.rows_written)
    }
}

/// One top-level field of a batch and the leaf column writers it feeds.
struct FieldWork<'a> {
    field: &'a Field,
    array: &'a ArrayRef,
    writers: &'a mut [ArrowColumnWriter],
    /// Arrow bytes, the cost estimate used to balance threads.
    weight: usize,
}

impl FieldWork<'_> {
    fn encode(&mut self) -> Result<()> {
        let leaves = compute_leaves(self.field, self.array)?;
        for (writer, leaf) in self.writers.iter_mut().zip(&leaves) {
            writer.write(leaf)?;
        }
        Ok(())
    }
}

/// Encode `batch` into `writers` (one per leaf of `parquet_schema`, in schema
/// order) on up to `threads` threads. Each field stays on one thread, and
/// fields are dealt heaviest first to the least-loaded thread.
fn encode_columns(
    schema: &Schema,
    parquet_schema: &SchemaDescriptor,
    batch: &RecordBatch,
    writers: &mut [ArrowColumnWriter],
    threads: usize,
) -> Result<()> {
    let mut leaves_per_field = vec![0; schema.fields().len()];
    for leaf in 0..parquet_schema.num_columns() {
        leaves_per_field[parquet_schema.get_column_root_idx(leaf)] += 1;
    }
    let mut rest = writers;
    let mut work = Vec::with_capacity(schema.fields().len());
    for ((field, array), leaves) in schema
        .fields()
        .iter()
        .zip(batch.columns())
        .zip(leaves_per_field)
    {
        let (mine, tail) = std::mem::take(&mut rest).split_at_mut(leaves);
        rest = tail;
        work.push(FieldWork {
            field: field.as_ref(),
            array,
            writers: mine,
            weight: array.get_array_memory_size(),
        });
    }
    if threads <= 1 || work.len() <= 1 {
        return work.iter_mut().try_for_each(FieldWork::encode);
    }
    work.sort_by(|a, b| b.weight.cmp(&a.weight));
    let mut bins: Vec<(usize, Vec<FieldWork>)> = (0..threads.min(work.len()))
        .map(|_| (0, Vec::new()))
        .collect();
    for item in work {
        let bin = bins.iter_mut().min_by_key(|(load, _)| *load).unwrap();
        bin.0 += item.weight;
        bin.1.push(item);
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = bins
            .into_iter()
            .map(|(_, mut items)| {
                scope.spawn(move || items.iter_mut().try_for_each(FieldWork::encode))
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|handle| handle.join().expect("parquet column encoder panicked"))
    })
}

/// Encode threads a process's writers may add beyond their first, shared by
/// every stream: a run with many workers must not multiply its threads.
static EXTRA_ENCODE_THREADS: AtomicUsize = AtomicUsize::new(0);

/// The encode thread count of one stream, raised only when the stream's
/// writer falls behind. Each raise draws one thread from a per-process budget
/// and the drop returns them, so streams that keep up stay at one thread and
/// a process never runs more than [`Self::PROCESS_EXTRA`] beyond one per stream.
pub struct EncodeThreads {
    count: Arc<AtomicUsize>,
    max: usize,
}

impl EncodeThreads {
    /// Extra threads one process may hold across all its streams.
    const PROCESS_EXTRA: usize = 8;

    pub fn new(max: usize) -> Self {
        Self {
            count: Arc::new(AtomicUsize::new(1)),
            max: max.max(1),
        }
    }

    pub fn handle(&self) -> Arc<AtomicUsize> {
        self.count.clone()
    }

    pub fn current(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Add one thread if this stream is below its cap and the process has one
    /// to spare. Returns the new count when it grew.
    pub fn grow(&self) -> Option<usize> {
        let current = self.current();
        if current >= self.max {
            return None;
        }
        EXTRA_ENCODE_THREADS
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                (used < Self::PROCESS_EXTRA).then_some(used + 1)
            })
            .ok()?;
        self.count.store(current + 1, Ordering::Relaxed);
        Some(current + 1)
    }
}

impl Drop for EncodeThreads {
    fn drop(&mut self) {
        let extra = self.current() - 1;
        EXTRA_ENCODE_THREADS.fetch_sub(extra, Ordering::Relaxed);
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
    use crate::log::rows::{cost_to_record_batches, CostLogChunk, CostLogEntry, GroupInputLog};
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
            decode_query_rows: 1,
            speculative_geometry: None,
            decode_kv_lens: None,
        };
        chunk.group_logs.push(g);
        for s in 0..slot_count {
            chunk.slot_times.push(0.1 + s as f32 * 0.01);
            chunk.slot_covs.push((s & 0xFF) as u8);
            chunk.slot_flops.push(s as f32 * 1e9);
            chunk.slot_bytes.push(s as f32 * 1e6);
            chunk.slot_backends.push((s & 0xFF) as u8);
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
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?;
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
        for rb in cost_to_record_batches(&chunk).unwrap() {
            writer.write(&rb).unwrap();
        }
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
            for rb in cost_to_record_batches(&chunk).unwrap() {
                writer.write(&rb).unwrap();
            }
        }
        writer.close().unwrap();
        assert_eq!(count_rows(&path).unwrap(), 16384);
    }

    /// Batches wide enough to cross [`ROW_GROUP_FLUSH_BYTES`] once, so the
    /// files hold two row groups.
    fn wide_batches() -> Vec<RecordBatch> {
        (0..3u64)
            .flat_map(|batch_idx| {
                let mut chunk = CostLogChunk::with_capacity("main", 4096, 4096, 4096 * 600, 0);
                for i in 0..4096u64 {
                    push_row(&mut chunk, 0, batch_idx * 4096 + i, 600);
                }
                cost_to_record_batches(&chunk).unwrap()
            })
            .collect()
    }

    fn cost_stream(path: PathBuf, threads: usize) -> StreamingParquetWriter {
        let list = |c: &str| ColumnPath::new(vec![c.to_owned(), "list".into(), "item".into()]);
        StreamingParquetWriter::new(path, cost_log_schema())
            .with_column_dictionary_disabled(vec![list("slot_time_ms"), list("slot_flops")])
            .with_column_statistics_disabled(vec![list("slot_time_ms"), list("slot_backend")])
            .with_encode_threads(Arc::new(AtomicUsize::new(threads)))
    }

    /// Catches a column-parallel encode that changes the file: threads only
    /// move whole columns, so every count must write the bytes `ArrowWriter`
    /// writes for the same batches and row-group boundaries.
    #[test]
    fn encode_threads_write_the_file_arrow_writer_writes() {
        let dir = tempdir().unwrap();
        let batches = wide_batches();
        let mut written = Vec::new();
        for threads in [1, 4] {
            let path = dir.path().join(format!("t{threads}.parquet"));
            let mut writer = cost_stream(path.clone(), threads);
            for batch in &batches {
                writer.write(batch).unwrap();
            }
            writer.close().unwrap();
            written.push(std::fs::read(path).unwrap());
        }

        let list = |c: &str| ColumnPath::new(vec![c.to_owned(), "list".into(), "item".into()]);
        let props = writer_properties(
            true,
            true,
            &[list("slot_time_ms"), list("slot_flops")],
            &[list("slot_time_ms"), list("slot_backend")],
        )
        .unwrap();
        let path = dir.path().join("arrow_writer.parquet");
        let mut reference = parquet::arrow::ArrowWriter::try_new(
            File::create(&path).unwrap(),
            cost_log_schema(),
            Some(props),
        )
        .unwrap();
        let mut unflushed = 0;
        for batch in &batches {
            reference.write(batch).unwrap();
            unflushed += batch.get_array_memory_size();
            if unflushed >= ROW_GROUP_FLUSH_BYTES {
                reference.flush().unwrap();
                unflushed = 0;
            }
        }
        reference.close().unwrap();
        let expected = std::fs::read(&path).unwrap();
        let groups = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
            .unwrap()
            .metadata()
            .num_row_groups();
        assert_eq!(groups, 2);
        assert!(written[0] == expected, "1 thread differs from ArrowWriter");
        assert!(written[1] == expected, "4 threads differ from ArrowWriter");
    }

    /// Catches streams that each grow their encode threads without a shared
    /// cap: the process budget runs out before every stream has grown.
    #[test]
    fn encode_threads_grow_within_the_process_budget() {
        let streams: Vec<_> = (0..EncodeThreads::PROCESS_EXTRA + 2)
            .map(|_| EncodeThreads::new(2))
            .collect();
        let grown = streams.iter().filter(|s| s.grow().is_some()).count();
        assert!(grown <= EncodeThreads::PROCESS_EXTRA, "{grown}");
        assert!(streams.iter().all(|s| s.current() <= 2));
    }
}
