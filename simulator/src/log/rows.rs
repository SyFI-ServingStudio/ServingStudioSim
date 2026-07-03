//! Per-table row entry types + their Arrow `RecordBatch` conversions. One
//! `*Entry` struct + one `*_to_record_batch` per logged table; the matching
//! Arrow `Schema` lives in [`crate::log::schemas`].

use std::sync::Arc;

use anyhow::{ensure, Result};
use arrow_array::builder::{
    Float32Builder, ListBuilder, StringBuilder, StructBuilder, UInt32Builder, UInt8Builder,
};
use arrow_array::{
    BooleanArray, Float32Array, Float64Array, Int16Array, ListArray, RecordBatch, StringArray,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field};

use crate::log::schemas::{
    cost_log_schema, gpu_cluster_schema, group_input_fields, kv_snapshot_schema,
    request_slo_schema, request_state_schema,
};
use crate::timing::SlotInput;

/// One `request_state` row — an **aggregate** over the admitted set at one
/// snapshot tick (`logging_time_ms`), not a per-request row. `*_tokens_cum` are
/// the cumulative prefill/decode token totals across all admitted requests; the
/// analyzer diffs them between consecutive ticks for per-segment throughput. The
/// counts are diagnostics (admitted = touched-by-a-worker, completed = finished).
#[derive(Clone, Copy, Debug)]
pub struct RequestStateEntry {
    pub logging_time_ms: f64,
    pub prefill_tokens_cum: u64,
    pub decode_tokens_cum: u64,
    pub n_admitted: u64,
    pub n_completed: u64,
}

/// One `request_slo` row — terminal (or sim-end-flush) per-token timing for one
/// request. The full `output_token_times_ms` (absolute sim-time ms) is always
/// carried to the writer thread, which derives the scalar columns from it
/// (`num_output_tokens`, the `tpot_*` percentiles, `finish_decode_time_ms`) in
/// [`slo_to_record_batch`]. The *array column itself* is persisted only when
/// `io.log_output_token_times` is on (it is the dominant size + encode cost); the
/// derived scalars are always written, so dropping the array keeps E2E / TPOT.
/// `ttft_ms` is computed sim-side here (needs `first_token_time`).
#[derive(Clone, Debug)]
pub struct RequestSloEntry {
    pub request_id: u32,
    pub logging_time_ms: f64,
    pub completed: bool,
    pub arrival_time_ms: f64,
    /// Per-token output timestamps (ms). Only populated when
    /// `io.log_output_token_times` is on; feeds `slo-detailed` ITL. The scalars
    /// below (`slo-general`) are computed sim-side and do NOT depend on it.
    pub output_token_times_ms: Vec<f32>,
    pub ttft_ms: Option<f32>,
    /// Output tokens emitted (`= tokens_emitted`), independent of the array.
    pub num_output_tokens: u32,
    /// Mean inter-token gap (ms) = `(last - first) / (tokens - 1)`; `None` for
    /// fewer than two output tokens. Computed sim-side from scalars.
    pub tpot_mean_ms: Option<f32>,
    /// Absolute sim-time (ms) of the final output token (= `last_token_time`),
    /// so E2E = `finish - arrival` survives `log_output_token_times` off.
    pub finish_decode_time_ms: Option<f32>,
    /// Terminal prefill length (`= RequestRecord::prefill_processed`): the `p` in
    /// the `workload_conservation` expected-work closed forms (`num_output_tokens`
    /// is the matching `d`). Full prompt for a completed request, partial for a
    /// sim-end-flush still in prefill.
    pub prefill_processed: u32,
    // Multi-round / session columns are deliberately NOT carried here today.
    // The current request lifecycle is single-round, so `session_id`,
    // `round_idx`, `total_rounds`,
    // `tool_wait_after_ms`, `session_arrival_time_ms`, `preserved_prefix_kv`,
    // and the terminal `final_phase` were either hardcoded constants
    // (`round_idx = 0`, `total_rounds = 1`, ...) or degenerate duplicates of
    // existing columns (`session_id == request_id`,
    // `session_arrival_time_ms == arrival_time_ms`). The analyzer's session-E2E
    // subject treated each request as its own session — i.e. it computed
    // request E2E under a different column name. So adding them as defaults
    // here would be dead schema. When real multi-turn lifecycle lands (the
    // `common/request.rs` "L7 lifecycle" comment), extend `RequestSloEntry`
    // with the actually-populated fields and update the analyzer to read
    // session-grouping columns from `request_slo` (not the per-tick
    // `request_state`, which is now an aggregate).
}

/// Inter-token gaps (ms) → `(mean, p50, p99, max)`, all `None` when fewer than
/// two output tokens (no gap defined). Selecting a percentile from unsorted data
/// is `O(n)` (quickselect, `select_nth_unstable_by`) — a full `O(n log n)` sort
/// is wasteful when only two order statistics are needed. `mean`/`max` are
/// single passes. Each `select_nth` partially reorders `gaps`, but it operates
/// on any order, so two successive selections are both correct (and still `O(n)`
/// each). `f32::total_cmp` gives a total order without the `partial_cmp` unwrap.
fn tpot_stats_ms(times_ms: &[f32]) -> (Option<f32>, Option<f32>, Option<f32>, Option<f32>) {
    if times_ms.len() < 2 {
        return (None, None, None, None);
    }
    let mut gaps: Vec<f32> = times_ms.windows(2).map(|w| w[1] - w[0]).collect();
    let n = gaps.len();
    let mean = gaps.iter().sum::<f32>() / n as f32;
    let max = gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let kth = |gaps: &mut [f32], p: f64| -> f32 {
        let idx = ((p * (n - 1) as f64).round() as usize).min(n - 1);
        *gaps.select_nth_unstable_by(idx, f32::total_cmp).1
    };
    let p50 = kth(&mut gaps, 0.5);
    let p99 = kth(&mut gaps, 0.99);
    (Some(mean), Some(p50), Some(p99), Some(max))
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
/// named by the matching per-worker `cost_manifest/` sidecar); the scalars are
/// the rolled-up totals.
///
/// Not `Clone`/`Debug`: the row is only ever moved into the writer-thread
/// channel, never cloned or formatted on the sim thread. The row owns **no**
/// `Vec`s — its variable-length data (per-iteration `groups`, the per-slot
/// `time`/`coverage`/`input` breakdowns) lives in the parallel flat buffers on
/// [`CostLogChunk`]; the row carries only the slice lengths. This keeps the
/// per-iteration record allocation-free (the worker appends into the chunk's
/// reused buffers instead of `collect`ing three fresh `Vec`s every forward pass
/// that would then be freed cross-thread on the writer).
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
    /// Building-block group this row costs: `iter` for the fused unified path, or
    /// `attn` / `prologue` / `pre_attn` / `post_attn` / `epilogue` for the AFD
    /// layer-wise path. Selects which sub-manifest in the `cost_manifest/` sidecar
    /// interprets this row's `slot_*` lists. A `&'static str` (section names are
    /// literals), so the row stays `Copy`-cheap to move to the writer thread.
    pub section: &'static str,
    /// Layer index this row costs; `-1` when the row is whole-iteration (the
    /// iter-wise Scale-folded form, where one row covers all layers).
    pub layer: i16,
    /// Number of entries in [`CostLogChunk::group_logs`] belonging to this row.
    pub group_len: usize,
    /// Number of entries in [`CostLogChunk::slot_times`] / [`CostLogChunk::slot_covs`]
    /// belonging to this row (the per-slot cost breakdown length).
    pub slot_len: usize,
    /// Number of entries in [`CostLogChunk::slot_inputs`] belonging to this row.
    /// Zero only for models that do not expose a compiled CostTree.
    pub slot_input_len: usize,
}

/// A cost-log transfer unit sent from the sim thread to the writer thread.
/// `entries` stores row metadata; every variable-length field is stored
/// back-to-back in a parallel flat buffer (`group_logs` / `slot_times` /
/// `slot_covs` / `slot_inputs`) so the sim thread appends into reused capacity
/// instead of allocating an owned `Vec` per row. The writer walks all four with
/// cursors keyed by each row's `*_len`.
pub struct CostLogChunk {
    /// Stable pool tag for every row in this chunk. A worker owns one CostLogger,
    /// so the whole chunk belongs to one `(pool_tag, worker_id)` stream.
    pub pool_tag: &'static str,
    pub entries: Vec<CostLogEntry>,
    pub group_logs: Vec<GroupInputLog>,
    pub slot_times: Vec<f32>,
    pub slot_covs: Vec<u8>,
    /// Per-slot achieved FLOPs / bytes, parallel to `slot_times` (same cursor,
    /// same `slot_len`). `0` marks a slot whose profile row had no throughput
    /// rate. Surfaced as the `slot_flops` / `slot_bytes` cost-log columns.
    pub slot_flops: Vec<f32>,
    pub slot_bytes: Vec<f32>,
    pub slot_inputs: Vec<SlotInput>,
}

impl CostLogChunk {
    pub fn with_capacity(
        pool_tag: &'static str,
        row_capacity: usize,
        groups_capacity: usize,
        slot_capacity: usize,
        slot_input_capacity: usize,
    ) -> Self {
        Self {
            pool_tag,
            entries: Vec::with_capacity(row_capacity),
            group_logs: Vec::with_capacity(groups_capacity),
            slot_times: Vec::with_capacity(slot_capacity),
            slot_covs: Vec::with_capacity(slot_capacity),
            slot_flops: Vec::with_capacity(slot_capacity),
            slot_bytes: Vec::with_capacity(slot_capacity),
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
fn build_groups_column(entries: &[CostLogEntry], group_logs: &[GroupInputLog]) -> ListArray {
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
    let mut group_cursor = 0usize;
    for e in entries {
        let sb = groups_builder.values();
        let group_end = group_cursor + e.group_len;
        for g in &group_logs[group_cursor..group_end] {
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
        group_cursor = group_end;
        groups_builder.append(true);
    }
    groups_builder.finish()
}

pub(crate) fn cost_to_record_batch(chunk: &CostLogChunk) -> Result<RecordBatch> {
    let entries = &chunk.entries;
    let pool_tag: Vec<&str> = entries.iter().map(|_| chunk.pool_tag).collect();
    let worker_id: Vec<u16> = entries.iter().map(|e| e.worker_id).collect();
    let iter_id: Vec<u64> = entries.iter().map(|e| e.iter_id).collect();
    let batch_id: Vec<u64> = entries.iter().map(|e| e.batch_id).collect();
    let wall_start: Vec<f64> = entries.iter().map(|e| e.wall_start_ms).collect();
    let total_time: Vec<f64> = entries.iter().map(|e| e.total_time_ms).collect();
    let energy: Vec<f64> = entries.iter().map(|e| e.energy_j).collect();
    let section: Vec<&str> = entries.iter().map(|e| e.section).collect();
    let layer: Vec<i16> = entries.iter().map(|e| e.layer).collect();

    // Per-iteration input_section: one List<Struct> entry per row.
    let groups = build_groups_column(entries, &chunk.group_logs);

    // Two parallel List columns, one (non-null, possibly empty) list per row.
    // Non-nullable `item` to match the schema (ListBuilder defaults to nullable).
    let mut time_builder = ListBuilder::new(Float32Builder::new())
        .with_field(Arc::new(Field::new("item", DataType::Float32, false)));
    let mut cov_builder = ListBuilder::new(UInt8Builder::new()).with_field(Arc::new(Field::new(
        "item",
        DataType::UInt8,
        false,
    )));
    // Achieved FLOPs / bytes, slot-aligned to `time_builder` (same cursor).
    let mut flops_builder = ListBuilder::new(Float32Builder::new())
        .with_field(Arc::new(Field::new("item", DataType::Float32, false)));
    let mut bytes_builder = ListBuilder::new(Float32Builder::new())
        .with_field(Arc::new(Field::new("item", DataType::Float32, false)));
    // Per-slot input JSON, serialized HERE on the writer thread (the sim thread
    // only handed over the inline `SlotInput` enums). One (possibly empty) list per row.
    let mut input_builder = ListBuilder::new(StringBuilder::new())
        .with_field(Arc::new(Field::new("item", DataType::Utf8, false)));
    let mut slot_input_cursor = 0usize;
    let mut slot_cursor = 0usize;
    let mut json_buf = Vec::new();
    for e in entries {
        let slot_end = slot_cursor + e.slot_len;
        ensure!(
            slot_end <= chunk.slot_times.len()
                && slot_end <= chunk.slot_covs.len()
                && slot_end <= chunk.slot_flops.len()
                && slot_end <= chunk.slot_bytes.len(),
            "cost_log slot slice exceeds chunk buffer"
        );
        for &t in &chunk.slot_times[slot_cursor..slot_end] {
            time_builder.values().append_value(t);
        }
        time_builder.append(true);
        for &c in &chunk.slot_covs[slot_cursor..slot_end] {
            cov_builder.values().append_value(c);
        }
        cov_builder.append(true);
        for &f in &chunk.slot_flops[slot_cursor..slot_end] {
            flops_builder.values().append_value(f);
        }
        flops_builder.append(true);
        for &b in &chunk.slot_bytes[slot_cursor..slot_end] {
            bytes_builder.values().append_value(b);
        }
        bytes_builder.append(true);
        slot_cursor = slot_end;
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
    ensure!(
        slot_cursor == chunk.slot_times.len()
            && slot_cursor == chunk.slot_covs.len()
            && slot_cursor == chunk.slot_flops.len()
            && slot_cursor == chunk.slot_bytes.len(),
        "cost_log chunk has unused slot time/coverage/flops/bytes values"
    );

    Ok(RecordBatch::try_new(
        cost_log_schema(),
        vec![
            Arc::new(StringArray::from(pool_tag)),
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
            Arc::new(StringArray::from(section)),
            Arc::new(Int16Array::from(layer)),
            Arc::new(flops_builder.finish()),
            Arc::new(bytes_builder.finish()),
        ],
    )?)
}

pub(crate) fn state_to_record_batch(entries: &[RequestStateEntry]) -> Result<RecordBatch> {
    let logging_time: Vec<f64> = entries.iter().map(|e| e.logging_time_ms).collect();
    let prefill_cum: Vec<u64> = entries.iter().map(|e| e.prefill_tokens_cum).collect();
    let decode_cum: Vec<u64> = entries.iter().map(|e| e.decode_tokens_cum).collect();
    let n_admitted: Vec<u64> = entries.iter().map(|e| e.n_admitted).collect();
    let n_completed: Vec<u64> = entries.iter().map(|e| e.n_completed).collect();

    Ok(RecordBatch::try_new(
        request_state_schema(),
        vec![
            Arc::new(Float64Array::from(logging_time)),
            Arc::new(UInt64Array::from(prefill_cum)),
            Arc::new(UInt64Array::from(decode_cum)),
            Arc::new(UInt64Array::from(n_admitted)),
            Arc::new(UInt64Array::from(n_completed)),
        ],
    )?)
}

/// One `kv_snapshot` row — a KV-pool occupancy sample for one group of one worker
/// at `time_ms`. All fields are token counts; `pool_tag` is carried once per chunk
/// (the whole chunk is one worker's stream), not per row. The pool's static
/// `capacity` lives in `run_meta.json` (`kv_pools`) and the occupancy percentages
/// are the analyzer's job, so neither is a column here. Emitted by
/// [`KvSampler`](crate::log::kv_sampler::KvSampler), which owns the throttle +
/// running-max policy deciding which samples become rows.
#[derive(Clone, Copy, Debug)]
pub struct KvSnapshotEntry {
    pub worker_id: u16,
    pub group_id: u16,
    pub time_ms: f64,
    /// Peak committed KV over the throttle window (the "current size").
    pub active_kv: u64,
    /// Peak KV the currently-admitted set will reach as it drains
    /// (`Batch::projected_peak_kv`) — the "future estimate".
    pub projected_peak: u64,
    /// Admitted-but-not-yet-realized tokens (the `promised` ledger).
    pub promised_kv: u64,
}

/// `pool_tag` is the whole chunk's stream tag (one worker owns one `KvSampler`),
/// so it is passed once here rather than duplicated into every [`KvSnapshotEntry`]
/// — mirrors how `cost_to_record_batch` takes `CostLogChunk::pool_tag`.
pub(crate) fn kv_to_record_batch(pool_tag: &str, entries: &[KvSnapshotEntry]) -> Result<RecordBatch> {
    let pool: Vec<&str> = entries.iter().map(|_| pool_tag).collect();
    let worker_id: Vec<u16> = entries.iter().map(|e| e.worker_id).collect();
    let group_id: Vec<u16> = entries.iter().map(|e| e.group_id).collect();
    let time_ms: Vec<f64> = entries.iter().map(|e| e.time_ms).collect();
    let active: Vec<u64> = entries.iter().map(|e| e.active_kv).collect();
    let peak: Vec<u64> = entries.iter().map(|e| e.projected_peak).collect();
    let promised: Vec<u64> = entries.iter().map(|e| e.promised_kv).collect();

    Ok(RecordBatch::try_new(
        kv_snapshot_schema(),
        vec![
            Arc::new(StringArray::from(pool)),
            Arc::new(UInt16Array::from(worker_id)),
            Arc::new(UInt16Array::from(group_id)),
            Arc::new(Float64Array::from(time_ms)),
            Arc::new(UInt64Array::from(active)),
            Arc::new(UInt64Array::from(peak)),
            Arc::new(UInt64Array::from(promised)),
        ],
    )?)
}

/// Derive the scalar columns from each row's `output_token_times_ms` (num,
/// `tpot_*` percentiles, `finish_decode_time_ms`) on the writer thread, then
/// transpose into a `RecordBatch`. The per-token array *column* is only
/// materialized when `log_output_token_times` is on — otherwise it is written as empty
/// lists (the scalars, which is all the analyzer needs for E2E / TPOT, are
/// always present). The array still crosses the channel either way; this only
/// skips its (dominant) parquet encode + the on-disk bytes.
pub(crate) fn slo_to_record_batch(
    entries: &[RequestSloEntry],
    log_output_token_times: bool,
) -> Result<RecordBatch> {
    let request_id: Vec<u32> = entries.iter().map(|e| e.request_id).collect();
    let logging_time: Vec<f64> = entries.iter().map(|e| e.logging_time_ms).collect();
    let completed: Vec<bool> = entries.iter().map(|e| e.completed).collect();
    let arrival: Vec<f64> = entries.iter().map(|e| e.arrival_time_ms).collect();
    // `slo-general` scalars are sim-computed (independent of the per-token array,
    // which is empty when `log_output_token_times` is off).
    let num_tokens: Vec<u32> = entries.iter().map(|e| e.num_output_tokens).collect();
    let prefill_processed: Vec<u32> = entries.iter().map(|e| e.prefill_processed).collect();
    let ttft: Vec<Option<f32>> = entries.iter().map(|e| e.ttft_ms).collect();
    let finish_decode: Vec<Option<f32>> = entries.iter().map(|e| e.finish_decode_time_ms).collect();
    let tpot_mean: Vec<Option<f32>> = entries.iter().map(|e| e.tpot_mean_ms).collect();
    // Per-request TPOT percentiles need the per-token series, so they are derived
    // from the array and are `None` when it was not logged (unused by the current
    // analyzer subjects, which read only `tpot_mean_ms`).
    let mut tpot_p50: Vec<Option<f32>> = Vec::with_capacity(entries.len());
    let mut tpot_p99: Vec<Option<f32>> = Vec::with_capacity(entries.len());
    let mut tpot_max: Vec<Option<f32>> = Vec::with_capacity(entries.len());
    for e in entries {
        let (_, p50, p99, max) = tpot_stats_ms(&e.output_token_times_ms);
        tpot_p50.push(p50);
        tpot_p99.push(p99);
        tpot_max.push(max);
    }

    // List<f32> column: one non-null list per row (may be empty). The item
    // field must match the schema's non-nullable `item` (ListBuilder defaults to
    // nullable items, which fails the RecordBatch schema check otherwise). When
    // `log_output_token_times` is off we still emit one (empty) list per row so the
    // column stays non-null and row-aligned.
    let mut times_builder = ListBuilder::new(Float32Builder::new())
        .with_field(Arc::new(Field::new("item", DataType::Float32, false)));
    for e in entries {
        if log_output_token_times {
            for &t in &e.output_token_times_ms {
                times_builder.values().append_value(t);
            }
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
            Arc::new(Float32Array::from(finish_decode)),
            Arc::new(UInt32Array::from(prefill_processed)),
        ],
    )?)
}

/// One `gpu_cluster` row — a single cross-worker transfer submitted to the shared
/// [`GpuCluster`](crate::worker::gpu_cluster::GpuCluster). Both endpoints'
/// identity is resolved from the two comm groups: `src_*` is the `send_gid`
/// group's owner, `dst_*` the `recv_gid` group's owner. Pool tags are `&'static
/// str` (worker pool literals); `tag` is the free-form per-deployment label
/// (owned, since callers may build it from request ids). One row is small and
/// transfers are far rarer than cost-log rows, so — unlike [`CostLogChunk`] —
/// there is no flat-buffer optimization; the [`NetworkLogger`](crate::log::NetworkLogger)
/// just buffers a `Vec<GpuClusterEntry>`.
#[derive(Clone, Debug)]
pub struct GpuClusterEntry {
    pub net_start_ms: f64,
    pub net_end_ms: f64,
    pub src_pool_tag: &'static str,
    pub src_worker_id: u16,
    pub dst_pool_tag: &'static str,
    pub dst_worker_id: u16,
    pub send_gid: u16,
    pub recv_gid: u16,
    /// GPU counts of the send / recv comm groups (`CommGroup::count`). The
    /// per-link share is `bytes / count`, so a consumer derives per-link
    /// effective bandwidth without joining the `run_meta` comm-group table.
    /// Always >= 1 (a transfer with a zero-count endpoint is never logged).
    pub send_count: u16,
    pub recv_count: u16,
    pub bytes: u64,
    /// Stable event category (`pd_kv_pull` / `afd_ffn_pull` / `afd_attn_pull`).
    pub kind: &'static str,
    /// Free-form per-deployment identifier (e.g. a request-id list); may be empty.
    pub tag: String,
    /// When the *sender's* link frees = end of its transmission slice. For a gather
    /// this is `net_start + transfer_time` (latency-stripped, ≤ `net_end`); for a
    /// coupled `submit_transfer` the sender is held to `net_end` (so == `net_end`).
    pub send_end_ms: f64,
}

pub(crate) fn gpu_cluster_to_record_batch(entries: &[GpuClusterEntry]) -> Result<RecordBatch> {
    let net_start: Vec<f64> = entries.iter().map(|e| e.net_start_ms).collect();
    let net_end: Vec<f64> = entries.iter().map(|e| e.net_end_ms).collect();
    let src_pool: Vec<&str> = entries.iter().map(|e| e.src_pool_tag).collect();
    let src_worker: Vec<u16> = entries.iter().map(|e| e.src_worker_id).collect();
    let dst_pool: Vec<&str> = entries.iter().map(|e| e.dst_pool_tag).collect();
    let dst_worker: Vec<u16> = entries.iter().map(|e| e.dst_worker_id).collect();
    let send_gid: Vec<u16> = entries.iter().map(|e| e.send_gid).collect();
    let recv_gid: Vec<u16> = entries.iter().map(|e| e.recv_gid).collect();
    let send_count: Vec<u16> = entries.iter().map(|e| e.send_count).collect();
    let recv_count: Vec<u16> = entries.iter().map(|e| e.recv_count).collect();
    let bytes: Vec<u64> = entries.iter().map(|e| e.bytes).collect();
    let kind: Vec<&str> = entries.iter().map(|e| e.kind).collect();
    let tag: Vec<&str> = entries.iter().map(|e| e.tag.as_str()).collect();
    let send_end: Vec<f64> = entries.iter().map(|e| e.send_end_ms).collect();

    Ok(RecordBatch::try_new(
        gpu_cluster_schema(),
        vec![
            Arc::new(Float64Array::from(net_start)),
            Arc::new(Float64Array::from(net_end)),
            Arc::new(StringArray::from(src_pool)),
            Arc::new(UInt16Array::from(src_worker)),
            Arc::new(StringArray::from(dst_pool)),
            Arc::new(UInt16Array::from(dst_worker)),
            Arc::new(UInt16Array::from(send_gid)),
            Arc::new(UInt16Array::from(recv_gid)),
            Arc::new(UInt64Array::from(bytes)),
            Arc::new(StringArray::from(kind)),
            Arc::new(StringArray::from(tag)),
            Arc::new(UInt16Array::from(send_count)),
            Arc::new(UInt16Array::from(recv_count)),
            Arc::new(Float64Array::from(send_end)),
        ],
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, StructArray};

    fn slo_entry(id: u32, times: Vec<f32>) -> RequestSloEntry {
        // Mirror `sim::run::slo_entry`: the general scalars are computed from
        // first/last/count, not the array, so they survive the array being off.
        let num = times.len() as u32;
        let finish = times.last().copied();
        let tpot_mean = if num > 1 {
            Some((times[num as usize - 1] - times[0]) / (num - 1) as f32)
        } else {
            None
        };
        RequestSloEntry {
            request_id: id,
            logging_time_ms: 100.0,
            completed: true,
            arrival_time_ms: 0.0,
            output_token_times_ms: times,
            ttft_ms: Some(1.0),
            num_output_tokens: num,
            tpot_mean_ms: tpot_mean,
            finish_decode_time_ms: finish,
            prefill_processed: 0,
        }
    }

    #[test]
    fn cost_log_parallel_list_columns_round_trip() {
        // Flat layout: row metadata in `entries` (carrying the per-row lengths),
        // variable-length data back-to-back in the chunk's parallel buffers.
        let entries = vec![
            CostLogEntry {
                worker_id: 0,
                iter_id: 7,
                batch_id: 0,
                wall_start_ms: 1.0,
                total_time_ms: 2.0,
                energy_j: 0.5,
                section: "iter",
                layer: -1,
                group_len: 2,
                slot_len: 3,
                slot_input_len: 0,
            },
            CostLogEntry {
                worker_id: 0,
                iter_id: 8,
                batch_id: 0,
                wall_start_ms: 3.0,
                total_time_ms: 1.0,
                energy_j: 0.2,
                section: "iter",
                layer: -1,
                group_len: 1,
                slot_len: 3,
                slot_input_len: 0,
            },
        ];
        let group_logs = vec![
            // row 0, group 0: two prefills + a decode aggregate.
            GroupInputLog {
                batch_tokens: 20,
                prefill_tokens: 18,
                decode_request_count: 2,
                decode_kv_total: 100,
                prefill_chunk_pairs: vec![(0, 8), (4, 10)],
            },
            // row 0, group 1: pure decode (no prefill pairs).
            GroupInputLog {
                batch_tokens: 3,
                prefill_tokens: 0,
                decode_request_count: 3,
                decode_kv_total: 60,
                prefill_chunk_pairs: vec![],
            },
            // row 1, group 0.
            GroupInputLog {
                batch_tokens: 5,
                prefill_tokens: 5,
                decode_request_count: 0,
                decode_kv_total: 0,
                prefill_chunk_pairs: vec![(0, 5)],
            },
        ];
        let batch = cost_to_record_batch(&CostLogChunk {
            pool_tag: "decode",
            entries,
            group_logs,
            slot_times: vec![1.0, 0.5, 0.5, 0.4, 0.6, 0.0],
            slot_covs: vec![0, 1, 0, 0, 0, 0],
            slot_flops: vec![10.0, 20.0, 30.0, 40.0, 50.0, 0.0],
            slot_bytes: vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0],
            slot_inputs: vec![],
        })
        .unwrap();
        assert_eq!(batch.num_rows(), 2);
        // slot_flops / slot_bytes round-trip as the last two List<f32> columns,
        // slot-aligned to slot_time_ms (both rows have slot_len 3).
        let flops_col = batch
            .column_by_name("slot_flops")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let flops0 = flops_col.value(0);
        let flops0 = flops0.as_any().downcast_ref::<Float32Array>().unwrap();
        assert_eq!(flops0.values(), &[10.0, 20.0, 30.0]);
        let bytes_col = batch
            .column_by_name("slot_bytes")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let bytes1 = bytes_col.value(1);
        let bytes1 = bytes1.as_any().downcast_ref::<Float32Array>().unwrap();
        assert_eq!(bytes1.values(), &[4.0, 5.0, 0.0]);
        let pool = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(pool.value(0), "decode");
        assert_eq!(pool.value(1), "decode");

        // iter_id column (index 2) carries the per-worker iteration counter.
        let it = batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(it.value(0), 7);
        assert_eq!(it.value(1), 8);

        // groups column (index 7): List<Struct>. Row 0 has 2 groups, row 1 has 1.
        let groups_col = batch
            .column(7)
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
        let batch = slo_to_record_batch(&entries, true).unwrap();
        assert_eq!(batch.num_rows(), 2);
        // num_output_tokens column (index 5) reflects the list lengths.
        let n = batch
            .column(5)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(n.value(0), 3);
        assert_eq!(n.value(1), 1);
        // TPOT is derived on the writer thread from the token times. Row 0
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
        // finish_decode_time_ms (index 11) = last token time; null when no tokens.
        let finish = batch
            .column(11)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(finish.value(0), 5.0);
        assert_eq!(finish.value(1), 2.0);
        // List column carries the full per-token times when logging is on.
        let times = batch
            .column(4)
            .as_any()
            .downcast_ref::<arrow_array::ListArray>()
            .unwrap();
        assert_eq!(times.value(0).len(), 3);
    }

    #[test]
    fn slo_list_column_omitted_when_off() {
        // log_output_token_times = false → the array column is empty, but the derived
        // scalars (num, tpot, finish_decode) are still computed from the array.
        let entries = vec![slo_entry(0, vec![1.0, 3.0, 5.0])];
        let batch = slo_to_record_batch(&entries, false).unwrap();
        let times = batch
            .column(4)
            .as_any()
            .downcast_ref::<arrow_array::ListArray>()
            .unwrap();
        assert_eq!(times.value(0).len(), 0, "array column must be empty");
        let n = batch
            .column(5)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(n.value(0), 3, "num_output_tokens still from the array");
        let finish = batch
            .column(11)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(finish.value(0), 5.0, "E2E scalar survives array-off");
    }
}
