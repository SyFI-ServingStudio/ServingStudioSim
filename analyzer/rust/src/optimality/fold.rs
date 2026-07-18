//! Fold algorithm — the ladder's compute core, fed by [`super::prepare`].
//!
//! R0/R1 are exact SQL sums over every row ([`read_exact_worker_totals`]). R2..R5 fold
//! a stride-sampled set of rows ([`accumulate_fold`]): the mean-mode fold is linear,
//! so each rung is `Σ_slot α·value` with the precomputed per-slot weight `α`, one
//! dot product per row. Each leaf's per-rung value comes from [`leaf_optimal_ms`], a
//! roofline over the leaf's grid-peak (R3) then the GPU-spec peak (R5). The sampled
//! sums land in each worker's accumulators and, per slot, in `(location, worker)`
//! cells for the kernel-level attribution.

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use arrow_array::{Array, Float32Array, ListArray, RecordBatch, StringArray};
use datafusion::prelude::SessionContext;

use crate::session::{col, collect, value_f64};

use super::prepare::SectionFoldPlan;
use super::{MAX_STRIDE, TARGET_SAMPLED_ITERS};

/// Per-worker exact totals + sampled fold accumulators.
pub(super) struct WorkerFoldAccumulator {
    pub(super) pool_tag: String,
    pub(super) worker_id: u16,
    pub(super) gpu_count: f64,
    /// Exact `Σ total_time_ms` and wall span (ms) over ALL rows.
    pub(super) busy_ms: f64,
    pub(super) span_ms: f64,
    /// Sampled `Σ total_time_ms` and the sampled mean-fold sums for R2..R5 (ms).
    pub(super) sampled_busy_ms: f64,
    pub(super) sampled_rungs_ms: [f64; 4],
}

/// Exact per-worker `Σ total_time_ms` and wall span from ALL rows (cheap SQL).
pub(super) async fn read_exact_worker_totals(
    ctx: &SessionContext,
    gpu_count_by_worker: &HashMap<(String, u16), f64>,
) -> Result<Vec<WorkerFoldAccumulator>> {
    let record_batches = collect(
        ctx,
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, \
                SUM(total_time_ms) AS busy, \
                MIN(wall_start_ms) AS w0, \
                MAX(wall_start_ms + total_time_ms) AS w1 \
         FROM cost_log GROUP BY pool_tag, worker_id",
    )
    .await?;
    let mut worker_accumulators = Vec::new();
    for batch in &record_batches {
        let pool_tags = string_column(batch, "pool_tag")?;
        let worker_ids = col(batch, "worker_id")?;
        let busy_times = col(batch, "busy")?;
        let wall_starts = col(batch, "w0")?;
        let wall_ends = col(batch, "w1")?;
        for row in 0..batch.num_rows() {
            let pool_tag = pool_tags.value(row).to_string();
            let worker_id = value_f64(worker_ids, row)? as u16;
            let gpu_count = gpu_count_by_worker
                .get(&(pool_tag.clone(), worker_id))
                .copied()
                .unwrap_or(1.0);
            let busy_ms = value_f64(busy_times, row)?.max(0.0);
            let span_ms = (value_f64(wall_ends, row)? - value_f64(wall_starts, row)?).max(0.0);
            worker_accumulators.push(WorkerFoldAccumulator {
                pool_tag,
                worker_id,
                gpu_count,
                busy_ms,
                span_ms,
                sampled_busy_ms: 0.0,
                sampled_rungs_ms: [0.0; 4],
            });
        }
    }
    Ok(worker_accumulators)
}

/// Pick the iteration stride: `num_iters / TARGET`, clamped `[1, MAX_STRIDE]`.
pub(super) async fn choose_stride(ctx: &SessionContext) -> Result<u64> {
    let record_batches = collect(
        ctx,
        "SELECT CAST(COALESCE(MAX(iter_id), 0) AS BIGINT) AS mx FROM cost_log",
    )
    .await?;
    let mut num_iters = 1u64;
    if let Some(first_batch) = record_batches.first() {
        if first_batch.num_rows() > 0 {
            let max_iter_id = value_f64(col(first_batch, "mx")?, 0)?;
            if max_iter_id.is_finite() {
                num_iters = max_iter_id as u64 + 1;
            }
        }
    }
    Ok((num_iters / TARGET_SAMPLED_ITERS).clamp(1, MAX_STRIDE))
}

/// Fold the stride-sampled rows: per row `Σ_slot α·value` for R2..R5 into its
/// worker, and the same per-slot contributions into `(location, worker)`.
pub(super) async fn accumulate_fold(
    ctx: &SessionContext,
    stride: u64,
    hardware_bandwidth_gbps: f64,
    section_fold_plan_by_key: &HashMap<(String, u16, String), SectionFoldPlan>,
    worker_index_by_key: &HashMap<(String, u16), usize>,
    workers: &mut [WorkerFoldAccumulator],
    sampled_rungs_by_location_worker: &mut HashMap<(u32, usize), [f64; 4]>,
) -> Result<u64> {
    let sql = format!(
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, \
                CAST(section AS VARCHAR) AS section, total_time_ms, \
                slot_time_ms, slot_flops, slot_bytes \
         FROM cost_log WHERE iter_id % {stride} = 0"
    );
    let record_batches = collect(ctx, &sql).await?;
    let mut sampled_rows = 0u64;
    for batch in &record_batches {
        let pool_tags = string_column(batch, "pool_tag")?;
        let worker_ids = col(batch, "worker_id")?;
        let sections = string_column(batch, "section")?;
        let total_times = col(batch, "total_time_ms")?;
        let (offsets, times) = list_f32_column(batch, "slot_time_ms")?;
        let (_, flops) = list_f32_column(batch, "slot_flops")?;
        let (_, bytes) = list_f32_column(batch, "slot_bytes")?;
        // Cache the resolved (meta, worker index) across the run of rows sharing
        // one (pool, worker, section) — cost_log is worker/iter ordered.
        let mut cached_plan: Option<((String, u16, String), (&SectionFoldPlan, usize))> = None;
        for row in 0..batch.num_rows() {
            let pool_tag = pool_tags.value(row);
            let worker_id = value_f64(worker_ids, row)? as u16;
            let section = sections.value(row);
            let resolved_plan = match &cached_plan {
                Some((cached_key, cached_value))
                    if cached_key.0 == pool_tag
                        && cached_key.1 == worker_id
                        && cached_key.2 == section =>
                {
                    Some(*cached_value)
                }
                _ => {
                    let section_key = (pool_tag.to_string(), worker_id, section.to_string());
                    match (
                        section_fold_plan_by_key.get(&section_key),
                        worker_index_by_key.get(&(pool_tag.to_string(), worker_id)),
                    ) {
                        (Some(fold_plan), Some(&worker_index)) => {
                            cached_plan = Some((section_key, (fold_plan, worker_index)));
                            Some((fold_plan, worker_index))
                        }
                        _ => None,
                    }
                }
            };
            let Some((fold_plan, worker_index)) = resolved_plan else {
                continue;
            };
            sampled_rows += 1;
            workers[worker_index].sampled_busy_ms += value_f64(total_times, row)?.max(0.0);

            let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
            let mut row_rungs_ms = [0.0f64; 4];
            for (slot, value_index) in (start..end).enumerate() {
                let Some(&mean_fold_weight) = fold_plan.mean_fold_weight_by_slot.get(slot) else {
                    break;
                };
                if mean_fold_weight == 0.0 {
                    continue;
                }
                let observed_time_ms = times.value(value_index) as f64;
                let leaf_flops = flops.value(value_index) as f64;
                let leaf_bytes = bytes.value(value_index) as f64;
                let is_communication = fold_plan.is_communication_by_slot[slot];
                let per_config_best_ms = leaf_optimal_ms(
                    observed_time_ms,
                    leaf_flops,
                    leaf_bytes,
                    is_communication,
                    fold_plan.grid_peak_tflops_by_slot[slot],
                    fold_plan.grid_peak_gbps_by_slot[slot],
                );
                let hardware_limit_ms = if is_communication {
                    0.0
                } else {
                    // Drop the memory term for a non-physical byte count (grouped_gemm),
                    // leaving the compute roofline — its real hardware gap survives.
                    let effective_hardware_bandwidth_gbps =
                        if fold_plan.has_physical_memory_bytes_by_slot[slot] {
                            hardware_bandwidth_gbps
                        } else {
                            0.0
                        };
                    leaf_optimal_ms(
                        per_config_best_ms,
                        leaf_flops,
                        leaf_bytes,
                        false,
                        fold_plan.hardware_peak_tflops_by_slot[slot],
                        effective_hardware_bandwidth_gbps,
                    )
                };
                // R2 real, R3 per-config-best, R4 drop-comm, R5 hardware.
                let leaf_rungs_ms = [
                    observed_time_ms,
                    per_config_best_ms,
                    if is_communication {
                        0.0
                    } else {
                        per_config_best_ms
                    },
                    if is_communication {
                        0.0
                    } else {
                        hardware_limit_ms
                    },
                ];
                let weighted_rungs_ms = leaf_rungs_ms.map(|value| mean_fold_weight * value);
                for rung_index in 0..4 {
                    row_rungs_ms[rung_index] += weighted_rungs_ms[rung_index];
                }
                let location_worker_rungs = sampled_rungs_by_location_worker
                    .entry((fold_plan.location_id_by_slot[slot], worker_index))
                    .or_insert([0.0; 4]);
                for rung_index in 0..4 {
                    location_worker_rungs[rung_index] += weighted_rungs_ms[rung_index];
                }
            }
            for rung_index in 0..4 {
                workers[worker_index].sampled_rungs_ms[rung_index] += row_rungs_ms[rung_index];
            }
        }
    }
    Ok(sampled_rows)
}

/// One leaf's optimal time (ms) = `work / peak_rate`, roofline over compute and
/// bandwidth, clamped to `real` (a peak can't make a leaf slower than observed).
/// A comm leaf uses only the bandwidth term. No measurable work/peak → `real`
/// (that leaf contributes no headroom at this rung).
fn leaf_optimal_ms(
    observed_upper_bound_ms: f64,
    work_flops: f64,
    traffic_bytes: f64,
    is_communication: bool,
    ceiling_tflops: f64,
    ceiling_gbps: f64,
) -> f64 {
    let mut lower_bound_ms = 0.0f64;
    let mut has_rate_ceiling = false;
    if !is_communication && ceiling_tflops > 0.0 && work_flops > 0.0 {
        lower_bound_ms = lower_bound_ms.max(work_flops / ceiling_tflops / 1e9);
        has_rate_ceiling = true;
    }
    if ceiling_gbps > 0.0 && traffic_bytes > 0.0 {
        lower_bound_ms = lower_bound_ms.max(traffic_bytes / ceiling_gbps / 1e6);
        has_rate_ceiling = true;
    }
    if !has_rate_ceiling {
        return observed_upper_bound_ms;
    }
    lower_bound_ms.min(observed_upper_bound_ms).max(0.0)
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    col(batch, name)?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("`{name}` is not a Utf8 array"))
}

/// Downcast a cost_log `List<f32>` column to `(list_offsets, flat_values)`.
fn list_f32_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<(&'a [i32], &'a Float32Array)> {
    let list = col(batch, name)?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("`{name}` is not a List array"))?;
    let vals = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| anyhow!("`{name}` is not List<f32>"))?;
    Ok((list.value_offsets(), vals))
}
