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
use crate::timing::{CostManifestDoc, LeafMetrics, SlotInput};
use parquet::schema::types::ColumnPath;

const STREAM_FLUSH_ROWS: usize = 8_192;
const CHANNEL_CAP: usize = 64;

/// Parquet *leaf* path of one per-slot list column.
fn slot_list_column(column: &str) -> ColumnPath {
    ColumnPath::new(vec![
        column.to_owned(),
        "list".to_owned(),
        "item".to_owned(),
    ])
}

/// Every per-slot list column. One row carries ~1,271 slots on a GLM-5.2-class
/// deployment, so these six hold three orders of magnitude more cells than all
/// the scalar columns combined — which is why their per-cell encoding options,
/// and only theirs, decide whether the writer thread keeps up with the sim.
fn slot_list_columns() -> Vec<ColumnPath> {
    [
        "slot_time_ms",
        "slot_coverage",
        "slot_input",
        "slot_flops",
        "slot_bytes",
        "slot_backend",
    ]
    .iter()
    .map(|column| slot_list_column(column))
    .collect()
}

/// The per-slot columns whose cells are near-unique, so the dictionary's per-cell
/// hash buys nothing (measured 0.99x on disk — see
/// [`StreamingParquetWriter::with_column_dictionary_disabled`]). `slot_coverage`
/// and `slot_backend` are deliberately absent: they hold a handful of distinct
/// `u8`s and the dictionary is worth 322-346x on them.
fn high_cardinality_slot_columns() -> Vec<ColumnPath> {
    ["slot_time_ms", "slot_input", "slot_flops", "slot_bytes"]
        .iter()
        .map(|column| slot_list_column(column))
        .collect()
}

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
    /// of one worker share the same compiled `CostTree`, so these are constant).
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
    /// breakdown for that specific `(pool_tag, worker_id)` stream. The manifest is
    /// a [`CostManifestDoc`] — one or more named sections, indexed by each row's
    /// `section` field (the iter-wise path passes a single `iter` section).
    pub fn open(
        log_dir: &Path,
        pool_tag: &'static str,
        worker_id: WorkerId,
        manifest: &CostManifestDoc,
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
        let mut writer = StreamingParquetWriter::new(path, cost_log_schema())
            .with_column_dictionary_disabled(high_cardinality_slot_columns())
            .with_column_statistics_disabled(slot_list_columns());
        let (tx, rx) = sync_channel::<CostLogChunk>(CHANNEL_CAP);
        let handle = std::thread::Builder::new()
            .name("vibesim-cost-logger".to_string())
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
    /// columns here; `groups` is moved over (drained, so the worker keeps its
    /// capacity); `slot_inputs` is copied by reference (so the same slice can be
    /// replayed from a cached section snapshot). The first row fixes the per-row
    /// sizes used to pre-size a rotated chunk in [`Self::send`].
    pub fn record(
        &mut self,
        mut entry: CostLogEntry,
        slots: &[LeafMetrics],
        groups: &mut Vec<GroupInputLog>,
        slot_inputs: &[SlotInput],
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
            self.buf.slot_flops.reserve(STREAM_FLUSH_ROWS * slot_len);
            self.buf.slot_bytes.reserve(STREAM_FLUSH_ROWS * slot_len);
            self.buf.slot_backends.reserve(STREAM_FLUSH_ROWS * slot_len);
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
        self.buf.slot_flops.extend(slots.iter().map(|l| l.m.flops));
        self.buf.slot_bytes.extend(slots.iter().map(|l| l.m.bytes));
        self.buf
            .slot_backends
            .extend(slots.iter().map(|l| l.backend_index));
        self.buf.group_logs.append(groups);
        self.buf.slot_inputs.extend_from_slice(slot_inputs);
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

    use arrow_array::{Array, ListArray, StringArray, UInt8Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::{Compression, ZstdLevel};
    use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterPropertiesBuilder};
    use parquet::schema::types::ColumnPath;
    use tempfile::tempdir;

    use crate::log::CostLogEntry;
    use crate::timing::{CostManifest, CoverageFlags, FlatCostNode, LeafDesc, Metrics4};

    /// Slots per row on the GLM-5.2 EP8 unified run whose writer thread showed
    /// backpressure (`logs/20260810_1_*`: 975,624 rows x 1,271 slots).
    const BENCH_SLOTS_PER_ROW: usize = 1_271;

    fn bench_chunk(rows: usize) -> CostLogChunk {
        use crate::timing::kernels::SingleGemmKernelInput;
        use crate::timing::AttnPrefillLog;

        let slots = rows * BENCH_SLOTS_PER_ROW;
        let mut chunk = CostLogChunk::with_capacity("main", rows, 0, slots, slots);
        for row in 0..rows {
            chunk.entries.push(CostLogEntry {
                worker_id: 0,
                iter_id: row as u64,
                batch_id: 0,
                wall_start_ms: row as f64 * 0.37,
                total_time_ms: 12.5,
                energy_j: 3.25,
                section: "iter",
                layer: -1,
                group_len: 0,
                slot_len: BENCH_SLOTS_PER_ROW,
                slot_input_len: BENCH_SLOTS_PER_ROW,
            });
            for slot in 0..BENCH_SLOTS_PER_ROW {
                // High-entropy floats (as in a real run: near-unique per cell) and
                // low-cardinality inputs (a handful of distinct batch shapes).
                let seed = (row * BENCH_SLOTS_PER_ROW + slot) as f32;
                chunk.slot_times.push(seed * 1.000_003_7);
                chunk.slot_covs.push((slot % 4) as u8);
                chunk.slot_flops.push(seed * 977.0);
                chunk.slot_bytes.push(seed * 31.5);
                chunk.slot_backends.push((slot % 3) as u8);
                // Half short (`{"m":N}`, 9 B), half list-carrying (~34 B), for a
                // ~21 B/cell mean — the real 8h run's `slot_input` averaged 19.4 B
                // (24.0 GB uncompressed over 1.24e9 cells), and serde_json cost
                // tracks output size, so a uniformly-short mix understates it.
                let matmul_tokens = (row % 64) as u32 + 1;
                if slot % 2 == 0 {
                    chunk
                        .slot_inputs
                        .push(SingleGemmKernelInput { m: matmul_tokens }.into());
                } else {
                    chunk.slot_inputs.push(
                        AttnPrefillLog {
                            prefill_chunk_pairs: vec![(seed as u32, matmul_tokens)],
                        }
                        .into(),
                    );
                }
            }
        }
        chunk
    }

    /// Where the cost-log writer thread's wall time goes, at the row shape that
    /// made the sim thread block on `CHANNEL_CAP`. Reports the Arrow/JSON
    /// conversion separately from the parquet encode, and the encode under each
    /// column-encoding setting, so the fix targets the term that dominates.
    ///
    /// Synthetic rows by default. Point `VIBESIM_COST_LOG_BENCH_PARQUET` at a real
    /// `raw/cost_log/*.parquet` to encode that run's actual cell distribution —
    /// synthetic floats are higher-entropy and synthetic `slot_input` far
    /// lower-cardinality than a real run, which biases both size and ZSTD time.
    #[test]
    #[ignore = "internal microbench; run: cargo test --release --lib cost_log_writer_breakdown -- --ignored --nocapture"]
    fn cost_log_writer_breakdown() {
        let rows = 2_048;
        let mut convert_s = f64::NAN;
        let (batch, cells) = match std::env::var("VIBESIM_COST_LOG_BENCH_PARQUET") {
            Ok(path) => {
                let mut reader =
                    ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
                        .unwrap()
                        .with_batch_size(rows)
                        .build()
                        .unwrap();
                let batch = reader.next().unwrap().unwrap();
                let slots = batch
                    .column_by_name("slot_time_ms")
                    .expect("slot_time_ms column")
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap()
                    .value_offsets()
                    .last()
                    .copied()
                    .unwrap_or(0) as usize;
                println!("(real batch from {path})");
                (batch, slots)
            }
            Err(_) => {
                let chunk = bench_chunk(rows);
                let mut batch = cost_to_record_batch(&chunk).unwrap();
                convert_s = f64::INFINITY;
                for _ in 0..5 {
                    let start = std::time::Instant::now();
                    batch = cost_to_record_batch(&chunk).unwrap();
                    convert_s = convert_s.min(start.elapsed().as_secs_f64());
                }
                // Same chunk with the slot_input column emptied: the difference is
                // the per-slot serde_json term, the rest is Arrow builder append.
                let mut no_input = bench_chunk(rows);
                no_input.slot_inputs.clear();
                for entry in &mut no_input.entries {
                    entry.slot_input_len = 0;
                }
                let mut no_input_s = f64::INFINITY;
                for _ in 0..5 {
                    let start = std::time::Instant::now();
                    let _ = cost_to_record_batch(&no_input).unwrap();
                    no_input_s = no_input_s.min(start.elapsed().as_secs_f64());
                }
                println!(
                    "(convert without slot_input: {:.3} s = {:.1} ns/cell -> serde_json share {:.0}%)",
                    no_input_s,
                    no_input_s / (rows * BENCH_SLOTS_PER_ROW) as f64 * 1e9,
                    (1.0 - no_input_s / convert_s) * 100.0
                );
                let json_bytes: usize = chunk
                    .slot_inputs
                    .iter()
                    .map(|slot| serde_json::to_vec(slot).map_or(0, |v| v.len()))
                    .sum();
                println!(
                    "(synthetic slot_input mean {:.1} B/cell)",
                    json_bytes as f64 / chunk.slot_inputs.len() as f64
                );
                (batch, rows * BENCH_SLOTS_PER_ROW)
            }
        };
        let rows = batch.num_rows();

        // Element paths of the six per-slot list columns — the ones carrying
        // ~1.24e9 cells on the run that backpressured. Built part-by-part on
        // purpose: `ColumnPath::from("a.list.item")` does NOT split on `.` (it
        // makes a one-part path), so a dotted string silently matches nothing and
        // every per-column override is a no-op.
        let slot_input_column = slot_list_column("slot_input");
        let numeric_slot_columns: Vec<ColumnPath> = slot_list_columns()
            .into_iter()
            .filter(|column| *column != slot_input_column)
            .collect();

        let keep = std::env::var("VIBESIM_COST_LOG_BENCH_OUT").ok();
        let dir = tempdir().unwrap();
        let out_dir = keep
            .as_deref()
            .map_or_else(|| dir.path().to_path_buf(), std::path::PathBuf::from);
        std::fs::create_dir_all(&out_dir).unwrap();
        let mut report = vec![("cost_to_record_batch".to_owned(), convert_s, 0u64)];
        let all_slot_columns = |input: bool| -> Vec<ColumnPath> {
            let mut columns = numeric_slot_columns.clone();
            if input {
                columns.push(slot_input_column.clone());
            }
            columns
        };
        let (stats_only, both_off, keep_input_dict) = (
            all_slot_columns(true),
            all_slot_columns(true),
            all_slot_columns(false),
        );
        let slot_input_for_stats = slot_input_column.clone();
        let variants: Vec<(
            &str,
            Box<dyn Fn(WriterPropertiesBuilder) -> WriterPropertiesBuilder>,
        )> = vec![
            ("dict=on  stats=on   (current)", Box::new(|b| b)),
            (
                "dict=off stats=on",
                Box::new(|b| b.set_dictionary_enabled(false)),
            ),
            (
                "dict=on  stats=off",
                Box::new(|b| b.set_statistics_enabled(EnabledStatistics::None)),
            ),
            (
                "dict=off stats=off",
                Box::new(|b| {
                    b.set_dictionary_enabled(false)
                        .set_statistics_enabled(EnabledStatistics::None)
                }),
            ),
            (
                "dict=on  stats=chunk",
                Box::new(|b| b.set_statistics_enabled(EnabledStatistics::Chunk)),
            ),
            (
                "slot cols: stats=off only",
                Box::new(move |mut builder| {
                    for column in &stats_only {
                        builder = builder
                            .set_column_statistics_enabled(column.clone(), EnabledStatistics::None);
                    }
                    builder
                }),
            ),
            (
                "slot cols: dict=off stats=off",
                Box::new(move |mut builder| {
                    for column in &both_off {
                        builder = builder
                            .set_column_dictionary_enabled(column.clone(), false)
                            .set_column_statistics_enabled(column.clone(), EnabledStatistics::None);
                    }
                    builder
                }),
            ),
            (
                "slot cols: stats=off, dict=off except coverage/backend (shipped)",
                Box::new(|mut builder| {
                    for column in high_cardinality_slot_columns() {
                        builder = builder.set_column_dictionary_enabled(column, false);
                    }
                    for column in slot_list_columns() {
                        builder =
                            builder.set_column_statistics_enabled(column, EnabledStatistics::None);
                    }
                    builder
                }),
            ),
            (
                "slot cols: dict=off stats=off, keep slot_input dict",
                Box::new(move |mut builder| {
                    for column in &keep_input_dict {
                        builder = builder
                            .set_column_dictionary_enabled(column.clone(), false)
                            .set_column_statistics_enabled(column.clone(), EnabledStatistics::None);
                    }
                    builder.set_column_statistics_enabled(
                        slot_input_for_stats.clone(),
                        EnabledStatistics::None,
                    )
                }),
            ),
        ];
        // Min of several passes: this box is shared, and single-shot timings
        // drifted ~20% run to run — enough to invert two adjacent variants.
        const PASSES: usize = 5;
        for (label, configure) in variants {
            let path = out_dir.join(format!("variant_{}.parquet", report.len()));
            let mut best = f64::INFINITY;
            let mut size = 0u64;
            for _ in 0..PASSES {
                let properties = configure(
                    WriterProperties::builder()
                        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
                        .set_statistics_enabled(EnabledStatistics::Page),
                )
                .build();
                let file = File::create(&path).unwrap();
                let mut writer =
                    ArrowWriter::try_new(file, cost_log_schema(), Some(properties)).unwrap();
                let start = std::time::Instant::now();
                writer.write(&batch).unwrap();
                writer.close().unwrap();
                best = best.min(start.elapsed().as_secs_f64());
                size = std::fs::metadata(&path).unwrap().len();
            }
            report.push((label.to_owned(), best, size));
        }

        println!("\ncost-log writer breakdown: {rows} rows, {cells} slot cells");
        for (label, seconds, size) in &report {
            if seconds.is_nan() {
                continue;
            }
            let per_cell_ns = seconds / cells as f64 * 1e9;
            let size_note = if *size > 0 {
                format!("  file={:.1} MB", *size as f64 / 1e6)
            } else {
                String::new()
            };
            println!("  {label:38}  {seconds:7.3} s  ({per_cell_ns:6.1} ns/cell){size_note}");
        }
    }

    /// The per-column encoding overrides actually reach the writer.
    ///
    /// This is a silent-failure guard, not a style check. The overrides are keyed by
    /// parquet *leaf* path, and `ColumnPath::from("slot_input.list.item")` does not
    /// split on `.` — it builds a one-part path that matches no column, so an
    /// override written that way is accepted, applied to nothing, and produces a
    /// file byte-for-byte identical to having set nothing at all. Asserting on the
    /// resulting column metadata is the only way that mistake shows up.
    #[test]
    fn slot_columns_drop_the_encodings_nothing_reads() {
        use crate::timing::kernels::SingleGemmKernelInput;

        let dir = tempdir().unwrap();
        let manifest = CostManifest {
            slots: vec![LeafDesc {
                name: "m.test".to_owned(),
                kind: "unit".to_owned(),
                kernel_config: serde_json::json!({"shape": 1, "backends": ["torch"]}),
            }],
            nodes: vec![FlatCostNode::Leaf(0)],
            node_labels: vec![None],
        };
        let doc = CostManifestDoc::single("iter", manifest);
        let mut logger = CostLogger::open(dir.path(), "main", WorkerId(0), &doc).unwrap();
        for iter_id in 0..64u64 {
            let entry = CostLogEntry {
                worker_id: 0,
                iter_id,
                batch_id: 0,
                wall_start_ms: iter_id as f64,
                total_time_ms: 1.25,
                energy_j: 0.0,
                section: "iter",
                layer: -1,
                group_len: 0,
                slot_len: 0,
                slot_input_len: 0,
            };
            let slots = vec![LeafMetrics {
                m: Metrics4 {
                    time_ms: iter_id as f32 * 1.5,
                    flops: iter_id as f32 * 7.0,
                    bytes: iter_id as f32 * 11.0,
                    energy_j: 0.0,
                },
                coverage: CoverageFlags::EMPTY,
                backend_index: 0,
            }];
            let mut groups = Vec::new();
            let slot_inputs = vec![SingleGemmKernelInput { m: 1 }.into()];
            logger
                .record(entry, &slots, &mut groups, &slot_inputs)
                .unwrap();
        }
        logger.flush_all().unwrap();

        let path = dir.path().join("raw/cost_log/worker_main_0.parquet");
        let metadata = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
            .unwrap()
            .metadata()
            .clone();
        let row_group = metadata.row_group(0);
        let column = |name: &str| {
            (0..row_group.num_columns())
                .map(|index| row_group.column(index))
                .find(|column| column.column_path().string() == name)
                .unwrap_or_else(|| panic!("no column {name:?}"))
        };

        // Near-unique cells: the dictionary's per-cell hash buys nothing (0.99x on a
        // real run), so it is off. Nothing can push a predicate to a list element,
        // so the statistics are off too.
        for name in [
            "slot_time_ms.list.item",
            "slot_flops.list.item",
            "slot_bytes.list.item",
            "slot_input.list.item",
        ] {
            let column = column(name);
            assert!(
                column.dictionary_page_offset().is_none(),
                "{name} should not be dictionary-encoded"
            );
            assert!(
                column.statistics().is_none(),
                "{name} should not carry statistics"
            );
        }
        // A handful of distinct `u8`s: here the dictionary's RLE is worth 322-346x on
        // disk, so it stays on — only the statistics go.
        for name in ["slot_coverage.list.item", "slot_backend.list.item"] {
            let column = column(name);
            assert!(
                column.dictionary_page_offset().is_some(),
                "{name} should keep its dictionary"
            );
            assert!(
                column.statistics().is_none(),
                "{name} should not carry statistics"
            );
        }
        // The overrides are per column, not a stream-wide switch: scalars keep both.
        let iter_id = column("iter_id");
        assert!(
            iter_id.statistics().is_some(),
            "iter_id should keep statistics"
        );
        assert!(
            iter_id.dictionary_page_offset().is_some(),
            "iter_id should keep its dictionary"
        );
    }

    #[test]
    fn writes_pool_worker_scoped_cost_artifacts() {
        let dir = tempdir().unwrap();
        let manifest = CostManifest {
            slots: vec![LeafDesc {
                name: "m.test".to_owned(),
                kind: "unit".to_owned(),
                kernel_config: serde_json::json!({"shape": 1, "backends": ["torch"]}),
            }],
            nodes: vec![FlatCostNode::Leaf(0)],
            node_labels: vec![None],
        };

        let doc = CostManifestDoc::single("iter", manifest);
        let mut logger = CostLogger::open(dir.path(), "decode", WorkerId(7), &doc).unwrap();
        let entry = CostLogEntry {
            worker_id: 7,
            iter_id: 3,
            batch_id: 0,
            wall_start_ms: 10.0,
            total_time_ms: 1.25,
            energy_j: 0.0,
            section: "iter",
            layer: -1,
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
            backend_index: 0,
        }];
        let mut groups = Vec::new();
        let slot_inputs = Vec::new();
        logger
            .record(entry, &slots, &mut groups, &slot_inputs)
            .unwrap();
        logger.flush_all().unwrap();

        let manifest_path = dir.path().join("raw/cost_manifest/worker_decode_7.json");
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
        // slot_backend round-trips the selected index (last column, appended).
        let backend_col = batch
            .column_by_name("slot_backend")
            .expect("slot_backend column")
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let backend_slots = backend_col
            .value(0)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .clone();
        assert_eq!(backend_slots.len(), 1);
        assert_eq!(backend_slots.value(0), 0);
    }
}
