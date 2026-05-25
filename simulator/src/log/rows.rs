//! Per-table row entry types + their Arrow `RecordBatch` conversions. One
//! `*Entry` struct + one `*_to_record_batch` per logged table; the matching
//! Arrow `Schema` lives in [`crate::log::schemas`].

use std::sync::Arc;

use anyhow::{ensure, Result};
use arrow_array::builder::{
    Float32Builder, ListBuilder, StringBuilder, StructBuilder, UInt32Builder, UInt8Builder,
};
use arrow_array::{
    BooleanArray, Float32Array, Float64Array, LargeStringArray, ListArray, RecordBatch,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field};

use crate::log::schemas::{
    cost_log_schema, group_input_fields, request_slo_schema, request_state_schema,
};
use crate::timing::SlotInput;

/// A request's lifecycle phase at snapshot time. A small `#[repr(u8)]` code
/// rather than a `String`: dense `request_state` emits one row per live request
/// per interval (tens of millions of rows on a long run), so a per-row `String`
/// allocation here dominated both the sim thread (alloc) and the writer thread
/// (dictionary intern). The parquet column is still `LargeUtf8` — `as_str()`
/// hands back a `&'static str`, so the on-disk output is byte-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinalPhase {
    Queued,
    Prefill,
    Decode,
    Complete,
}

impl FinalPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            FinalPhase::Queued => "queued",
            FinalPhase::Prefill => "prefill",
            FinalPhase::Decode => "decode",
            FinalPhase::Complete => "complete",
        }
    }
}

/// One `request_state` row — a snapshot of a request at `logging_time_ms`.
/// Single-round runs default the multi-round columns (`session_id = request_id`,
/// `round_idx = 0`, `total_rounds = 1`, `tool_wait_after_ms = 0`,
/// `session_arrival_time_ms = arrival`, `preserved_prefix_kv = 0`).
#[derive(Clone, Debug)]
pub struct RequestStateEntry {
    pub request_id: u32,
    pub logging_time_ms: f64,
    pub arrival_time_ms: f64,
    pub first_token_time_ms: Option<f64>,
    pub completion_time_ms: Option<f64>,
    pub completed: bool,
    pub input_len: u32,
    pub output_len: u32,
    pub completed_input_len: u32,
    pub completed_output_len: u32,
    pub final_phase: FinalPhase,
    pub session_id: u32,
    pub round_idx: u32,
    pub total_rounds: u32,
    pub tool_wait_after_ms: f64,
    pub session_arrival_time_ms: f64,
    pub preserved_prefix_kv: u32,
}

/// One `request_slo` row — terminal (or sim-end-flush) per-token timing for one
/// request. `output_token_times_ms` holds absolute sim-time ms. The TPOT
/// percentile scalars are NOT carried on the entry: they are derived from
/// `output_token_times_ms` on the writer thread in [`slo_to_record_batch`]
/// (see [`tpot_stats_ms`]), keeping the `O(n log n)` sort off the sim hot path.
/// `ttft_ms` stays here — it is a cheap scalar (first-token minus arrival), not a
/// sort, and needs `first_token_time` which only the sim side holds.
#[derive(Clone, Debug)]
pub struct RequestSloEntry {
    pub request_id: u32,
    pub logging_time_ms: f64,
    pub completed: bool,
    pub arrival_time_ms: f64,
    pub output_token_times_ms: Vec<f32>,
    pub ttft_ms: Option<f32>,
}

/// Inter-token gaps (ms) → `(mean, p50, p99, max)`, all `None` when fewer than
/// two output tokens (no gap defined). Runs on the **writer thread** from the
/// already-logged `output_token_times_ms`, so the percentile sort never touches
/// the sim hot path. Numerically equivalent to the old sim-side `tpot_stats`
/// (gaps from consecutive ms timestamps; the f32 vs f64 subtraction difference is
/// negligible for a timing statistic).
fn tpot_stats_ms(times_ms: &[f32]) -> (Option<f32>, Option<f32>, Option<f32>, Option<f32>) {
    if times_ms.len() < 2 {
        return (None, None, None, None);
    }
    let mut gaps: Vec<f32> = times_ms.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = gaps.iter().sum::<f32>() / gaps.len() as f32;
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f32 {
        let idx = ((p * (gaps.len() - 1) as f64).round() as usize).min(gaps.len() - 1);
        gaps[idx]
    };
    (
        Some(mean),
        Some(pct(0.5)),
        Some(pct(0.99)),
        Some(*gaps.last().unwrap()),
    )
}

/// One HP group's input context for a `cost_log` row — the per-iteration
/// `input_section` (`docs/logging.md` §3.2), one per `ArchGroupInput` the worker
/// fed the model_arch. Prefill is kept full (`prefill_chunk_pairs`, moved over
/// un-split — the writer thread splits `(prefix, append)` into the two parallel
/// list columns); decode is aggregated to `decode_request_count` /
/// `decode_kv_total` (the per-decode KV-length list is the size driver and is
/// dropped). `tokens_per_source_rank` (the ffn/EP routing view) is NOT logged —
/// it is empty for the dense-local model and its layer ownership is unsettled.
#[derive(Clone, Debug)]
pub struct GroupInputLog {
    pub batch_tokens: u32,
    pub prefill_tokens: u32,
    pub decode_request_count: u32,
    pub decode_kv_total: u32,
    /// Per prefill request `(prefix_len, append_len)`, moved from the
    /// `ArchGroupInput`; split into two parallel list columns at serialization.
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
}

/// One `cost_log` row — a whole-iteration cost query via the compiled CostTree.
/// `groups` is the per-iteration input context (one entry per HP group);
/// `slot_time_ms` / `slot_coverage` are the per-slot cost breakdown (positions
/// named by the `cost_manifest.json` sidecar); the scalars are the rolled-up
/// totals.
///
/// Not `Clone`/`Debug`: the row is only ever moved into the writer-thread
/// channel, never cloned or formatted on the sim thread. Per-slot inputs are
/// stored in [`CostLogChunk::slot_inputs`] as one flat chunk buffer; this row
/// carries only its slice length so the worker's capture buffer can keep its
/// allocation across iterations.
pub struct CostLogEntry {
    pub worker_id: u16,
    /// Per-worker iteration index (one forward-pass cycle).
    pub iter_id: u64,
    /// Batch index *within* the iteration. 0 today (one batch per iteration);
    /// 0..k under AFD/TBO where a worker runs several batches in one iteration.
    pub batch_id: u64,
    pub wall_start_ms: f64,
    /// `wall_end_ms` is deliberately not stored — it is derivable as
    /// `wall_start_ms + total_time_ms`, so keeping it duplicated a high-entropy
    /// f64 that barely compressed.
    pub total_time_ms: f64,
    pub energy_j: f64,
    pub groups: Vec<GroupInputLog>,
    pub slot_time_ms: Vec<f32>,
    pub slot_coverage: Vec<u8>,
    /// Number of entries in [`CostLogChunk::slot_inputs`] belonging to this row.
    /// Zero only for models that do not expose a compiled CostTree.
    pub slot_input_len: usize,
}

/// A cost-log transfer unit sent from the sim thread to the writer thread.
/// `entries` stores row metadata; `slot_inputs` stores all captured per-slot
/// inputs back-to-back, avoiding one owned `Vec<SlotInput>` allocation per row.
pub struct CostLogChunk {
    pub entries: Vec<CostLogEntry>,
    pub slot_inputs: Vec<SlotInput>,
}

impl CostLogChunk {
    pub fn with_capacity(row_capacity: usize, slot_input_capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(row_capacity),
            slot_inputs: Vec::with_capacity(slot_input_capacity),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Build the nested `groups` `List<Struct>` column (the per-iteration
/// `input_section`): one list per row holding that iteration's groups, each a
/// struct of one [`GroupInputLog`]. The `(prefix, append)` prefill pairs are
/// split into the two parallel `prefill_*_lens` lists *here*, on the writer
/// thread (the worker hands the pairs over un-split). Item/list fields are
/// non-nullable to match [`group_input_fields`] / `cost_log_schema`.
fn build_groups_column(entries: &[CostLogEntry]) -> ListArray {
    let fields = group_input_fields();
    let u32_item = || Arc::new(Field::new("item", DataType::UInt32, false));
    let new_struct_builder = || {
        StructBuilder::new(
            fields.clone(),
            vec![
                Box::new(UInt32Builder::new()),
                Box::new(UInt32Builder::new()),
                Box::new(UInt32Builder::new()),
                Box::new(UInt32Builder::new()),
                Box::new(ListBuilder::new(UInt32Builder::new()).with_field(u32_item())),
                Box::new(ListBuilder::new(UInt32Builder::new()).with_field(u32_item())),
            ],
        )
    };
    let struct_field = Arc::new(Field::new("item", DataType::Struct(fields.clone()), false));
    let mut groups_builder = ListBuilder::new(new_struct_builder()).with_field(struct_field);
    for e in entries {
        let sb = groups_builder.values();
        for g in &e.groups {
            sb.field_builder::<UInt32Builder>(0)
                .unwrap()
                .append_value(g.batch_tokens);
            sb.field_builder::<UInt32Builder>(1)
                .unwrap()
                .append_value(g.prefill_tokens);
            sb.field_builder::<UInt32Builder>(2)
                .unwrap()
                .append_value(g.decode_request_count);
            sb.field_builder::<UInt32Builder>(3)
                .unwrap()
                .append_value(g.decode_kv_total);
            {
                let prefix_b = sb.field_builder::<ListBuilder<UInt32Builder>>(4).unwrap();
                for &(prefix, _) in &g.prefill_chunk_pairs {
                    prefix_b.values().append_value(prefix);
                }
                prefix_b.append(true);
            }
            {
                let append_b = sb.field_builder::<ListBuilder<UInt32Builder>>(5).unwrap();
                for &(_, append) in &g.prefill_chunk_pairs {
                    append_b.values().append_value(append);
                }
                append_b.append(true);
            }
            sb.append(true);
        }
        groups_builder.append(true);
    }
    groups_builder.finish()
}

pub(crate) fn cost_to_record_batch(chunk: &CostLogChunk) -> Result<RecordBatch> {
    let entries = &chunk.entries;
    let worker_id: Vec<u16> = entries.iter().map(|e| e.worker_id).collect();
    let iter_id: Vec<u64> = entries.iter().map(|e| e.iter_id).collect();
    let batch_id: Vec<u64> = entries.iter().map(|e| e.batch_id).collect();
    let wall_start: Vec<f64> = entries.iter().map(|e| e.wall_start_ms).collect();
    let total_time: Vec<f64> = entries.iter().map(|e| e.total_time_ms).collect();
    let energy: Vec<f64> = entries.iter().map(|e| e.energy_j).collect();

    // Per-iteration input_section: one List<Struct> entry per row.
    let groups = build_groups_column(entries);

    // Two parallel List columns, one (non-null, possibly empty) list per row.
    // Non-nullable `item` to match the schema (ListBuilder defaults to nullable).
    let mut time_builder = ListBuilder::new(Float32Builder::new())
        .with_field(Arc::new(Field::new("item", DataType::Float32, false)));
    let mut cov_builder = ListBuilder::new(UInt8Builder::new()).with_field(Arc::new(Field::new(
        "item",
        DataType::UInt8,
        false,
    )));
    // Per-slot input JSON, serialized HERE on the writer thread (the sim thread
    // only handed over the inline `SlotInput` enums). One (possibly empty) list per row.
    let mut input_builder = ListBuilder::new(StringBuilder::new())
        .with_field(Arc::new(Field::new("item", DataType::Utf8, false)));
    let mut slot_input_cursor = 0usize;
    let mut json_buf = Vec::new();
    for e in entries {
        for &t in &e.slot_time_ms {
            time_builder.values().append_value(t);
        }
        time_builder.append(true);
        for &c in &e.slot_coverage {
            cov_builder.values().append_value(c);
        }
        cov_builder.append(true);
        let slot_input_end = slot_input_cursor + e.slot_input_len;
        ensure!(
            slot_input_end <= chunk.slot_inputs.len(),
            "cost_log slot_input slice exceeds chunk buffer"
        );
        for slot in &chunk.slot_inputs[slot_input_cursor..slot_input_end] {
            json_buf.clear();
            let json = match serde_json::to_writer(&mut json_buf, slot) {
                Ok(()) => std::str::from_utf8(&json_buf).unwrap_or("null"),
                Err(_) => "null",
            };
            input_builder.values().append_value(json);
        }
        slot_input_cursor = slot_input_end;
        input_builder.append(true);
    }
    ensure!(
        slot_input_cursor == chunk.slot_inputs.len(),
        "cost_log chunk has unused slot_input values"
    );

    Ok(RecordBatch::try_new(
        cost_log_schema(),
        vec![
            Arc::new(UInt16Array::from(worker_id)),
            Arc::new(UInt64Array::from(iter_id)),
            Arc::new(UInt64Array::from(batch_id)),
            Arc::new(Float64Array::from(wall_start)),
            Arc::new(Float64Array::from(total_time)),
            Arc::new(Float64Array::from(energy)),
            Arc::new(groups),
            Arc::new(time_builder.finish()),
            Arc::new(cov_builder.finish()),
            Arc::new(input_builder.finish()),
        ],
    )?)
}

pub(crate) fn state_to_record_batch(entries: &[RequestStateEntry]) -> Result<RecordBatch> {
    let request_id: Vec<u32> = entries.iter().map(|e| e.request_id).collect();
    let logging_time: Vec<f64> = entries.iter().map(|e| e.logging_time_ms).collect();
    let arrival: Vec<f64> = entries.iter().map(|e| e.arrival_time_ms).collect();
    let first_token: Vec<Option<f64>> = entries.iter().map(|e| e.first_token_time_ms).collect();
    let completion: Vec<Option<f64>> = entries.iter().map(|e| e.completion_time_ms).collect();
    let completed: Vec<bool> = entries.iter().map(|e| e.completed).collect();
    let input_len: Vec<u32> = entries.iter().map(|e| e.input_len).collect();
    let output_len: Vec<u32> = entries.iter().map(|e| e.output_len).collect();
    let completed_input: Vec<u32> = entries.iter().map(|e| e.completed_input_len).collect();
    let completed_output: Vec<u32> = entries.iter().map(|e| e.completed_output_len).collect();
    let final_phase: Vec<&str> = entries.iter().map(|e| e.final_phase.as_str()).collect();
    let session_id: Vec<u32> = entries.iter().map(|e| e.session_id).collect();
    let round_idx: Vec<u32> = entries.iter().map(|e| e.round_idx).collect();
    let total_rounds: Vec<u32> = entries.iter().map(|e| e.total_rounds).collect();
    let tool_wait: Vec<f64> = entries.iter().map(|e| e.tool_wait_after_ms).collect();
    let session_arrival: Vec<f64> = entries.iter().map(|e| e.session_arrival_time_ms).collect();
    let prefix_kv: Vec<u32> = entries.iter().map(|e| e.preserved_prefix_kv).collect();

    Ok(RecordBatch::try_new(
        request_state_schema(),
        vec![
            Arc::new(UInt32Array::from(request_id)),
            Arc::new(Float64Array::from(logging_time)),
            Arc::new(Float64Array::from(arrival)),
            Arc::new(Float64Array::from(first_token)),
            Arc::new(Float64Array::from(completion)),
            Arc::new(BooleanArray::from(completed)),
            Arc::new(UInt32Array::from(input_len)),
            Arc::new(UInt32Array::from(output_len)),
            Arc::new(UInt32Array::from(completed_input)),
            Arc::new(UInt32Array::from(completed_output)),
            Arc::new(LargeStringArray::from(final_phase)),
            Arc::new(UInt32Array::from(session_id)),
            Arc::new(UInt32Array::from(round_idx)),
            Arc::new(UInt32Array::from(total_rounds)),
            Arc::new(Float64Array::from(tool_wait)),
            Arc::new(Float64Array::from(session_arrival)),
            Arc::new(UInt32Array::from(prefix_kv)),
        ],
    )?)
}

pub(crate) fn slo_to_record_batch(entries: &[RequestSloEntry]) -> Result<RecordBatch> {
    let request_id: Vec<u32> = entries.iter().map(|e| e.request_id).collect();
    let logging_time: Vec<f64> = entries.iter().map(|e| e.logging_time_ms).collect();
    let completed: Vec<bool> = entries.iter().map(|e| e.completed).collect();
    let arrival: Vec<f64> = entries.iter().map(|e| e.arrival_time_ms).collect();
    let num_tokens: Vec<u32> = entries
        .iter()
        .map(|e| e.output_token_times_ms.len() as u32)
        .collect();
    let ttft: Vec<Option<f32>> = entries.iter().map(|e| e.ttft_ms).collect();
    // TPOT percentiles derived here on the writer thread from the per-row token
    // times (the sort is off the sim hot path; the times list already crossed
    // the channel, so this is zero extra data movement).
    let mut tpot_mean: Vec<Option<f32>> = Vec::with_capacity(entries.len());
    let mut tpot_p50: Vec<Option<f32>> = Vec::with_capacity(entries.len());
    let mut tpot_p99: Vec<Option<f32>> = Vec::with_capacity(entries.len());
    let mut tpot_max: Vec<Option<f32>> = Vec::with_capacity(entries.len());
    for e in entries {
        let (mean, p50, p99, max) = tpot_stats_ms(&e.output_token_times_ms);
        tpot_mean.push(mean);
        tpot_p50.push(p50);
        tpot_p99.push(p99);
        tpot_max.push(max);
    }

    // List<f32> column: one non-null list per row (may be empty). The item
    // field must match the schema's non-nullable `item` (ListBuilder defaults to
    // nullable items, which fails the RecordBatch schema check otherwise).
    let mut times_builder = ListBuilder::new(Float32Builder::new())
        .with_field(Arc::new(Field::new("item", DataType::Float32, false)));
    for e in entries {
        for &t in &e.output_token_times_ms {
            times_builder.values().append_value(t);
        }
        times_builder.append(true);
    }
    let output_token_times = times_builder.finish();

    Ok(RecordBatch::try_new(
        request_slo_schema(),
        vec![
            Arc::new(UInt32Array::from(request_id)),
            Arc::new(Float64Array::from(logging_time)),
            Arc::new(BooleanArray::from(completed)),
            Arc::new(Float64Array::from(arrival)),
            Arc::new(output_token_times),
            Arc::new(UInt32Array::from(num_tokens)),
            Arc::new(Float32Array::from(ttft)),
            Arc::new(Float32Array::from(tpot_mean)),
            Arc::new(Float32Array::from(tpot_p50)),
            Arc::new(Float32Array::from(tpot_p99)),
            Arc::new(Float32Array::from(tpot_max)),
        ],
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, StructArray};

    fn slo_entry(id: u32, times: Vec<f32>) -> RequestSloEntry {
        RequestSloEntry {
            request_id: id,
            logging_time_ms: 100.0,
            completed: true,
            arrival_time_ms: 0.0,
            output_token_times_ms: times,
            ttft_ms: Some(1.0),
        }
    }

    #[test]
    fn cost_log_parallel_list_columns_round_trip() {
        let entries = vec![
            CostLogEntry {
                worker_id: 0,
                iter_id: 7,
                batch_id: 0,
                wall_start_ms: 1.0,
                total_time_ms: 2.0,
                energy_j: 0.5,
                groups: vec![
                    // group 0: two prefills + a decode aggregate.
                    GroupInputLog {
                        batch_tokens: 20,
                        prefill_tokens: 18,
                        decode_request_count: 2,
                        decode_kv_total: 100,
                        prefill_chunk_pairs: vec![(0, 8), (4, 10)],
                    },
                    // group 1: pure decode (no prefill pairs).
                    GroupInputLog {
                        batch_tokens: 3,
                        prefill_tokens: 0,
                        decode_request_count: 3,
                        decode_kv_total: 60,
                        prefill_chunk_pairs: vec![],
                    },
                ],
                slot_time_ms: vec![1.0, 0.5, 0.5],
                slot_coverage: vec![0, 1, 0],
                slot_input_len: 0,
            },
            CostLogEntry {
                worker_id: 0,
                iter_id: 8,
                batch_id: 0,
                wall_start_ms: 3.0,
                total_time_ms: 1.0,
                energy_j: 0.2,
                groups: vec![GroupInputLog {
                    batch_tokens: 5,
                    prefill_tokens: 5,
                    decode_request_count: 0,
                    decode_kv_total: 0,
                    prefill_chunk_pairs: vec![(0, 5)],
                }],
                slot_time_ms: vec![0.4, 0.6, 0.0],
                slot_coverage: vec![0, 0, 0],
                slot_input_len: 0,
            },
        ];
        let batch = cost_to_record_batch(&CostLogChunk {
            entries,
            slot_inputs: vec![],
        })
        .unwrap();
        assert_eq!(batch.num_rows(), 2);
        // iter_id column (index 1) carries the per-worker iteration counter.
        let it = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(it.value(0), 7);
        assert_eq!(it.value(1), 8);

        // groups column (index 6): List<Struct>. Row 0 has 2 groups, row 1 has 1.
        let groups_col = batch
            .column(6)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let row0 = groups_col.value(0);
        let g0 = row0.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(g0.len(), 2);
        let batch_tokens = g0.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
        assert_eq!(batch_tokens.value(0), 20);
        assert_eq!(batch_tokens.value(1), 3);
        // decode aggregates land in struct fields 2/3.
        let dec_count = g0.column(2).as_any().downcast_ref::<UInt32Array>().unwrap();
        assert_eq!(dec_count.value(0), 2);
        // prefill split: field 4 = prefix lens, field 5 = append lens.
        let prefix_lists = g0.column(4).as_any().downcast_ref::<ListArray>().unwrap();
        let append_lists = g0.column(5).as_any().downcast_ref::<ListArray>().unwrap();
        let prefix0 = prefix_lists.value(0);
        let append0 = append_lists.value(0);
        let prefix0 = prefix0.as_any().downcast_ref::<UInt32Array>().unwrap();
        let append0 = append0.as_any().downcast_ref::<UInt32Array>().unwrap();
        assert_eq!(prefix0.len(), append0.len());
        assert_eq!(prefix0.values(), &[0, 4]);
        assert_eq!(append0.values(), &[8, 10]);
        // group 1 of row 0 is pure decode: empty prefill lists.
        assert_eq!(prefix_lists.value(1).len(), 0);

        let row1 = groups_col.value(1);
        let g1 = row1.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(g1.len(), 1);
    }

    #[test]
    fn slo_list_column_round_trips() {
        let entries = vec![slo_entry(0, vec![1.0, 3.0, 5.0]), slo_entry(1, vec![2.0])];
        let batch = slo_to_record_batch(&entries).unwrap();
        assert_eq!(batch.num_rows(), 2);
        // num_output_tokens column (index 5) reflects the list lengths.
        let n = batch
            .column(5)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(n.value(0), 3);
        assert_eq!(n.value(1), 1);
        // TPOT is now derived on the writer thread from the token times. Row 0
        // gaps = [2,2] → mean=p50=p99=max=2; row 1 has <2 tokens → null.
        let mean = batch
            .column(7)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(mean.value(0), 2.0);
        assert!(mean.is_null(1));
        let max = batch
            .column(10)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(max.value(0), 2.0);
        assert!(max.is_null(1));
    }
}
