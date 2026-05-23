//! Per-table row entry types + their Arrow `RecordBatch` conversions. One
//! `*Entry` struct + one `*_to_record_batch` per logged table; the matching
//! Arrow `Schema` lives in [`crate::log::schemas`].

use std::sync::Arc;

use anyhow::Result;
use arrow_array::builder::{Float32Builder, ListBuilder, UInt8Builder};
use arrow_array::{
    BooleanArray, Float32Array, Float64Array, LargeStringArray, RecordBatch, UInt16Array,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field};

use crate::log::schemas::{cost_log_schema, request_slo_schema, request_state_schema};

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
/// request. `output_token_times_ms` holds absolute sim-time ms; the TPOT/TTFT
/// scalars are `None` when there are no output tokens yet.
#[derive(Clone, Debug)]
pub struct RequestSloEntry {
    pub request_id: u32,
    pub logging_time_ms: f64,
    pub completed: bool,
    pub arrival_time_ms: f64,
    pub output_token_times_ms: Vec<f32>,
    pub ttft_ms: Option<f32>,
    pub tpot_mean_ms: Option<f32>,
    pub tpot_p50_ms: Option<f32>,
    pub tpot_p99_ms: Option<f32>,
    pub tpot_max_ms: Option<f32>,
}

/// One `cost_log` row — a whole-iteration cost query via the compiled CostTree.
/// `slot_time_ms` / `slot_coverage` are the per-slot breakdown (positions named
/// by the `cost_manifest.json` sidecar); the scalars are the rolled-up totals.
#[derive(Clone, Debug)]
pub struct CostLogEntry {
    pub worker_id: u16,
    pub batch_id: u64,
    pub wall_start_ms: f64,
    pub wall_end_ms: f64,
    pub total_time_ms: f64,
    pub energy_j: f64,
    pub slot_time_ms: Vec<f32>,
    pub slot_coverage: Vec<u8>,
}

pub(crate) fn cost_to_record_batch(entries: &[CostLogEntry]) -> Result<RecordBatch> {
    let worker_id: Vec<u16> = entries.iter().map(|e| e.worker_id).collect();
    let batch_id: Vec<u64> = entries.iter().map(|e| e.batch_id).collect();
    let wall_start: Vec<f64> = entries.iter().map(|e| e.wall_start_ms).collect();
    let wall_end: Vec<f64> = entries.iter().map(|e| e.wall_end_ms).collect();
    let total_time: Vec<f64> = entries.iter().map(|e| e.total_time_ms).collect();
    let energy: Vec<f64> = entries.iter().map(|e| e.energy_j).collect();

    // Two parallel List columns, one (non-null, possibly empty) list per row.
    // Non-nullable `item` to match the schema (ListBuilder defaults to nullable).
    let mut time_builder = ListBuilder::new(Float32Builder::new())
        .with_field(Arc::new(Field::new("item", DataType::Float32, false)));
    let mut cov_builder = ListBuilder::new(UInt8Builder::new())
        .with_field(Arc::new(Field::new("item", DataType::UInt8, false)));
    for e in entries {
        for &t in &e.slot_time_ms {
            time_builder.values().append_value(t);
        }
        time_builder.append(true);
        for &c in &e.slot_coverage {
            cov_builder.values().append_value(c);
        }
        cov_builder.append(true);
    }

    Ok(RecordBatch::try_new(
        cost_log_schema(),
        vec![
            Arc::new(UInt16Array::from(worker_id)),
            Arc::new(UInt64Array::from(batch_id)),
            Arc::new(Float64Array::from(wall_start)),
            Arc::new(Float64Array::from(wall_end)),
            Arc::new(Float64Array::from(total_time)),
            Arc::new(Float64Array::from(energy)),
            Arc::new(time_builder.finish()),
            Arc::new(cov_builder.finish()),
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
    let tpot_mean: Vec<Option<f32>> = entries.iter().map(|e| e.tpot_mean_ms).collect();
    let tpot_p50: Vec<Option<f32>> = entries.iter().map(|e| e.tpot_p50_ms).collect();
    let tpot_p99: Vec<Option<f32>> = entries.iter().map(|e| e.tpot_p99_ms).collect();
    let tpot_max: Vec<Option<f32>> = entries.iter().map(|e| e.tpot_max_ms).collect();

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
    fn cost_log_parallel_list_columns_round_trip() {
        let entries = vec![
            CostLogEntry {
                worker_id: 0,
                batch_id: 7,
                wall_start_ms: 1.0,
                wall_end_ms: 3.0,
                total_time_ms: 2.0,
                energy_j: 0.5,
                slot_time_ms: vec![1.0, 0.5, 0.5],
                slot_coverage: vec![0, 1, 0],
            },
            CostLogEntry {
                worker_id: 0,
                batch_id: 8,
                wall_start_ms: 3.0,
                wall_end_ms: 4.0,
                total_time_ms: 1.0,
                energy_j: 0.2,
                slot_time_ms: vec![0.4, 0.6, 0.0],
                slot_coverage: vec![0, 0, 0],
            },
        ];
        let batch = cost_to_record_batch(&entries).unwrap();
        assert_eq!(batch.num_rows(), 2);
        // batch_id column (index 1) carries the iter counter.
        let b = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(b.value(0), 7);
        assert_eq!(b.value(1), 8);
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
    }
}
