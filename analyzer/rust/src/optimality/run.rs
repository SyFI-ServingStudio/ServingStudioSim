//! Orchestration — `run_optimality` wires preparation → fold → tiers → kernel and
//! assembles the top-level `meta` / `report` / `payload` JSON. Also owns the output
//! contract shared by the degrade paths (`definitions`, `unavailable*`). The caveat
//! bookkeeping (gpu-count degrade, missing roster, no spec, empty peaks) stays here
//! because it is orchestration glue, not part of any one stage.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context, Result};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_cost_manifests, read_run_meta, read_worker_gpu_counts, SCHEMA_VERSION};
use crate::kernel_query::repo_root;
use crate::session::{register_cost_log, require_columns, COST_LOG_TABLE};

use super::ladder::NecessaryWorkPolicy;
use super::spec::{self, GpuSpec};
use super::{
    bucket_keys, ms_to_s, ratio, rung_keys, BUCKET_KEYS, FLOORS_TARGET_SAMPLED_ITERS,
    KERNEL_RUNG_KEYS, RUNG_KEYS, UNLOCKED_ITERATION_REPLICATION_FACTOR,
};
use super::{floors, fold, grid_peaks, kernel, levels, location, prepare};

/// cost_log columns this subject depends on (drift guard).
pub(super) const COST_COLS: &[&str] = &[
    "pool_tag",
    "worker_id",
    "section",
    "iter_id",
    "wall_start_ms",
    "total_time_ms",
    "slot_time_ms",
    "slot_flops",
    "slot_bytes",
];

pub async fn run_optimality(
    ctx: &SessionContext,
    log_dir: &Path,
    lock_batch_size: bool,
) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((
            unavailable(log_dir, reason, lock_batch_size),
            unavailable_payload(log_dir, reason, lock_batch_size),
        ));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLS).await?;
    let manifests_by_worker = match read_cost_manifests(log_dir) {
        Ok(manifests_by_worker) => manifests_by_worker,
        Err(error) => {
            let reason =
                format!("cost_manifest/ unreadable ({error:#}); needed to fold the ladder");
            return Ok((
                unavailable(log_dir, &reason, lock_batch_size),
                unavailable_payload(log_dir, &reason, lock_batch_size),
            ));
        }
    };

    let mut caveats: Vec<String> = Vec::new();

    // G_worker from run_meta — READ, never inferred. Absent → single-GPU degrade.
    let worker_gpu_counts = read_worker_gpu_counts(log_dir);
    let has_worker_gpu_counts = worker_gpu_counts.is_some();
    if !has_worker_gpu_counts {
        caveats.push(
            "run_meta worker GPU counts unavailable; treating every worker as 1 GPU \
             (worker/pool/cluster GPU·s are not physical)"
                .to_string(),
        );
    }
    let gpu_count_by_worker: HashMap<(String, u16), f64> = worker_gpu_counts
        .unwrap_or_default()
        .into_iter()
        .map(|(pool_tag, worker_id, gpu_count)| ((pool_tag, worker_id), gpu_count.max(1) as f64))
        .collect();

    // Hardware rate ceilings (R5) + grid-peak ceilings (R3).
    let (_num_gpus, gpu_name) = read_run_meta(log_dir).unwrap_or((1, String::new()));
    let repository_root = repo_root().ok();
    let matched_gpu_spec = repository_root
        .as_deref()
        .and_then(|root| spec::load_gpu_spec(root, &gpu_name));
    let (gpu_spec_matched, gpu_spec): (Option<String>, GpuSpec) = match matched_gpu_spec {
        Some((matched_name, spec)) => (Some(matched_name), spec),
        None => {
            caveats.push(format!(
                "no gpu/spec.json entry for gpu_name {gpu_name:?}; R3 defaults to its \
                 bandwidth basis and R5 hardware limit collapses onto R4 — add an alias"
            ));
            (None, GpuSpec::default())
        }
    };
    let hardware_bandwidth_gbps = gpu_spec.mem_bandwidth_gbps;

    let grid_peak_catalog = if lock_batch_size {
        grid_peaks::GridPeakCatalog::batch_locked()
    } else {
        grid_peaks::load_or_generate(log_dir, &manifests_by_worker)
    };
    if !lock_batch_size && grid_peak_catalog.is_empty() {
        caveats.push(format!(
            "grid-peaks sidecar {}: batching headroom not computed (R3 = R2)",
            grid_peak_catalog.source
        ));
    }

    // Intern locations + precompute per-section metadata.
    let (kernel_locations, section_fold_plan_by_key) =
        prepare::build_section_fold_plans(&manifests_by_worker, &grid_peak_catalog, &gpu_spec);
    let kernel_locations_by_worker = prepare::kernel_locations_by_worker(&manifests_by_worker);

    // Exact per-worker busy + span (all rows).
    let mut workers = fold::read_exact_worker_totals(ctx, &gpu_count_by_worker).await?;
    if workers.is_empty() {
        let reason = "cost_log has no (pool_tag, worker_id) rows";
        return Ok((
            unavailable(log_dir, reason, lock_batch_size),
            unavailable_payload(log_dir, reason, lock_batch_size),
        ));
    }
    // A cost_log worker with no run_meta GPU count falls back to G=1 (see
    // `read_exact_worker_totals`), which understates its GPU·s. On a v4 roster this never
    // happens; on a pre-v4 log a non-KV worker (null `pool_tag`) is missing from the
    // tagged roster — surface it rather than silently under-counting.
    if has_worker_gpu_counts {
        let missing_worker_count = workers
            .iter()
            .filter(|worker| {
                !gpu_count_by_worker.contains_key(&(worker.pool_tag.clone(), worker.worker_id))
            })
            .count();
        if missing_worker_count > 0 {
            caveats.push(format!(
                "{missing_worker_count} worker(s) in cost_log are absent from run_meta's tagged roster \
                 (pre-v4 run_meta with a null non-KV pool_tag?); their GPU·s use G=1 and are \
                 understated — re-simulate to v4 for exact GPU counts"
            ));
        }
    }
    let worker_index_by_key: HashMap<(String, u16), usize> = workers
        .iter()
        .enumerate()
        .map(|(worker_index, worker)| ((worker.pool_tag.clone(), worker.worker_id), worker_index))
        .collect();

    // Sampled fold for R2..R5 + per-(location, worker) raw contributions.
    let sample_stride = fold::choose_stride(ctx).await?;
    let mut sampled_rungs_by_location_worker: HashMap<(u32, usize), [f64; 4]> = HashMap::new();
    let sampled_row_count = fold::accumulate_fold(
        ctx,
        sample_stride,
        hardware_bandwidth_gbps,
        lock_batch_size,
        &section_fold_plan_by_key,
        &worker_index_by_key,
        &mut workers,
        &mut sampled_rungs_by_location_worker,
    )
    .await?;

    // Assemble rungs (GPU·ms) per worker → pool → cluster; then per-kernel.
    let tier_aggregates = levels::assemble_tiers(&workers);

    // Locked composition labels each distinct observed iteration shape and adds
    // its roofline with occurrence weight. Unlocked composition uses one saturated
    // large-batch label per worker and may recompute rooflines after work rollup.
    let run_label_result = if lock_batch_size {
        floors::compute_batch_locked_run_labels(ctx, log_dir, FLOORS_TARGET_SAMPLED_ITERS).await
    } else {
        floors::compute_saturated_run_labels(ctx, log_dir, UNLOCKED_ITERATION_REPLICATION_FACTOR)
            .await
    };
    let run_labels = match run_label_result {
        Ok(computed_labels) => Some(computed_labels),
        Err(error) => {
            caveats.push(format!(
                "labeler necessary-work floors unavailable ({error:#}); \
                 waterfall shows the plain hardware-optimal floor"
            ));
            None
        }
    };
    let necessary_work_floors = run_labels.as_ref().map(|labels| &labels.floors);
    let has_necessary_work_floors = necessary_work_floors.is_some_and(|floors| !floors.is_empty());
    if let Some(labels) = run_labels.as_ref() {
        for (level_key, error) in &labels.errors {
            caveats.push(format!(
                "necessary-work label unavailable for {level_key:?} ({error})"
            ));
        }
    }
    let cluster_floor = necessary_work_floors
        .and_then(|by_level| by_level.get("cluster"))
        .copied();
    let cluster_necessary_ratio = cluster_floor.map(|floor| {
        let green_s = ms_to_s(tier_aggregates.cluster_rungs.hardware_limit);
        ratio(
            floor.fused.clamp(0.0, green_s),
            ms_to_s(tier_aggregates.cluster_rungs.real),
        )
    });
    let kernel_rungs_by_location = kernel::aggregate_kernel_rungs(
        &sampled_rungs_by_location_worker,
        &tier_aggregates.worker_anchor_factors,
        kernel_locations.len(),
    );
    let mut worker_kernel_ladders = kernel::worker_kernel_ladders(
        &kernel_locations,
        &workers,
        &tier_aggregates.worker_rungs_in_fold_order,
        &sampled_rungs_by_location_worker,
        &tier_aggregates.worker_anchor_factors,
    );
    let mut composed_location_mapping_ids = BTreeSet::new();
    let mut attributed_worker_count = 0usize;
    if let Some(labels) = run_labels.as_ref() {
        let catalog = repository_root
            .as_deref()
            .context("repo root unavailable for semantic-location maps")
            .and_then(|repository_root| location::LocationCatalog::load(repository_root, log_dir));
        match catalog {
            Ok(catalog) => {
                for ladder in &mut worker_kernel_ladders {
                    let (pool_tag, worker_id) = ladder
                        .worker_ref()
                        .context("worker ladder unexpectedly has aggregate scope")?;
                    let pool_tag = pool_tag.to_string();
                    let worker_key = (pool_tag.clone(), worker_id);
                    let Some(composition) = labels.workers.get(&worker_key) else {
                        continue;
                    };
                    let attribution_result = (|| -> Result<_> {
                        let locations =
                            kernel_locations_by_worker
                                .get(&worker_key)
                                .with_context(|| {
                                    format!("manifest locations missing for {pool_tag}/{worker_id}")
                                })?;
                        let gpu_count = workers
                            .iter()
                            .find(|worker| {
                                worker.pool_tag == pool_tag && worker.worker_id == worker_id
                            })
                            .map(|worker| worker.gpu_count)
                            .unwrap_or(1.0);
                        catalog.attribute_composed_ladder(
                            &pool_tag,
                            locations,
                            ladder,
                            composition,
                            gpu_spec,
                            gpu_count,
                            if lock_batch_size {
                                NecessaryWorkPolicy::BatchLocked
                            } else {
                                NecessaryWorkPolicy::Saturated {
                                    replication_factor: UNLOCKED_ITERATION_REPLICATION_FACTOR,
                                }
                            },
                        )
                    })();
                    match attribution_result {
                        Ok(attribution) => {
                            attributed_worker_count += 1;
                            composed_location_mapping_ids.insert(attribution.mapping_id);
                        }
                        Err(error) => caveats.push(format!(
                            "per-location necessary work unavailable for {pool_tag}/{worker_id} ({error:#})"
                        )),
                    }
                }
            }
            Err(error) => caveats.push(format!(
                "per-location necessary work catalog unavailable ({error:#})"
            )),
        }
    }
    let composed_necessary_work_available =
        attributed_worker_count == worker_kernel_ladders.len() && !worker_kernel_ladders.is_empty();
    let any_composed_necessary_work = attributed_worker_count > 0;
    let aggregate_kernel_ladders = kernel::aggregate_kernel_ladders(&worker_kernel_ladders)?;
    let worker_kernel_ladder_values = worker_kernel_ladders
        .iter()
        .map(|ladder| ladder.to_json())
        .collect::<Result<Vec<_>>>()?;
    let aggregate_kernel_ladder_values = aggregate_kernel_ladders
        .iter()
        .map(|ladder| ladder.to_json())
        .collect::<Result<Vec<_>>>()?;

    let cluster_optimality_ratio = ratio(
        tier_aggregates.cluster_rungs.hardware_limit,
        tier_aggregates.cluster_rungs.real,
    );

    let level_entries = levels::levels_json(&tier_aggregates, necessary_work_floors);
    let kernel_entries = kernel::kernel_levels_json(&kernel_locations, &kernel_rungs_by_location);

    let analysis_meta = json!({
        "log_dir": log_dir.display().to_string(),
        "gpu_name": gpu_name,
        "batch_size_locked": lock_batch_size,
        "gpu_spec_matched": gpu_spec_matched,
        "peaks_source": grid_peak_catalog.source,
        "sample_stride": sample_stride,
        "sampled_rows": sampled_row_count,
        "gpu_counts_available": has_worker_gpu_counts,
        "num_workers": workers.len(),
        "num_locations": kernel_locations.len(),
        "composed_necessary_work_available": composed_necessary_work_available,
        "composed_necessary_work_workers": attributed_worker_count,
        "composed_necessary_work_basis": if lock_batch_size { "fixed_iteration_composition" } else { "saturated_worker_workload" },
        "composed_necessary_work_replication_factor": if lock_batch_size { 1 } else { UNLOCKED_ITERATION_REPLICATION_FACTOR },
        "composed_necessary_work_unique_shapes": run_labels.as_ref().and_then(|labels| labels.batch_locked_unique_shapes),
        "composed_necessary_work_iterations": run_labels.as_ref().and_then(|labels| labels.batch_locked_iterations),
        "composed_necessary_work_sampled_iterations": run_labels.as_ref().and_then(|labels| labels.batch_locked_sampled_iterations),
        "composed_necessary_work_sample_stride": run_labels.as_ref().and_then(|labels| labels.batch_locked_sample_stride),
        "composed_necessary_work_affine_bases": run_labels.as_ref().and_then(|labels| labels.batch_locked_affine_bases),
        "composed_necessary_work_direct_fallback_bases": run_labels.as_ref().and_then(|labels| labels.batch_locked_direct_fallback_bases),
        "composed_location_mapping_ids": composed_location_mapping_ids,
        "caveats": caveats,
    });

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "available": true,
        "meta": analysis_meta,
        "optimality_ratio": cluster_optimality_ratio,
        "necessary_ratio": cluster_necessary_ratio,
        "unit": "gpu_seconds",
        "cluster": levels::rung_report_json(&tier_aggregates.cluster_rungs, cluster_floor),
        "pools": tier_aggregates
            .pool_rungs_by_tag
            .iter()
            .map(|(pool_tag, pool_rungs)| {
                json!({
                    "pool": pool_tag,
                    "rungs": levels::rung_report_json(
                        pool_rungs,
                        necessary_work_floors
                            .and_then(|by_level| by_level.get(pool_tag))
                            .copied(),
                    ),
                })
            })
            .collect::<Vec<_>>(),
        "worst_batching_kernels": kernel::worst_batching_json(
            &kernel_locations,
            &kernel_rungs_by_location,
        ),
        "definitions": definitions(lock_batch_size),
    });

    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "available": true,
        "meta": analysis_meta,
        "unit": "gpu_seconds",
        "optimality_ratio": cluster_optimality_ratio,
        "necessary_ratio": cluster_necessary_ratio,
        "bucket_keys": bucket_keys(has_necessary_work_floors),
        "rung_keys": rung_keys(has_necessary_work_floors),
        "kernel_rung_keys": if any_composed_necessary_work {
            KERNEL_RUNG_KEYS.iter().copied().chain(["necessary_limit"]).collect::<Vec<_>>()
        } else {
            KERNEL_RUNG_KEYS.to_vec()
        },
        "levels": level_entries,
        "kernels": kernel_entries,
        "worker_kernel_ladders": worker_kernel_ladder_values,
        "aggregate_kernel_ladders": aggregate_kernel_ladder_values,
        "definitions": definitions(lock_batch_size),
    });

    Ok((report, payload))
}

fn definitions(lock_batch_size: bool) -> Value {
    let per_config_best = if lock_batch_size {
        "batch size locked: identical to balanced (R3=R2); batching gap is zero"
    } else {
        "large-batch grid ceiling for the leaf's fixed config: peak TFLOP/s when any fitted point crosses the GPU ridge, otherwise peak GB/s; gap = batching"
    };
    let necessary_work = if lock_batch_size {
        "model/work labels each distinct fixed-batch iteration shape once; occurrence-weighted per-iteration rooflines add through worker, pool, and cluster without rebatching"
    } else {
        "model/work labeler bounds below R5: segmented necessary work sums per-location rooflines; scope-fused necessary work adds FLOPs/bytes first and applies one roofline at that scope"
    };
    let buckets = "idle, imbalance, batching, communication, hardware_gap, then R5 split into excess_over_necessary, fusion, hardware_necessary when labeler bounds are available; sum to Real";
    json!({
        "scope": "per-worker CostTree manifest re-folded with per-rung leaf substitutions; \
                  unit GPU-seconds = wall time × the worker's run_meta gpu_ids count",
        "ladder": {
            "real": "span × G — GPU·s actually held (includes idle)",
            "busy": "Σ total_time_ms × G — no scheduler idle; gap vs real = idle",
            "balanced": "real slot times, Max folded to mean — perfect load balance; gap = imbalance",
            "per_config_best": per_config_best,
            "ignore_network": "per_config_best with comm leaves dropped; gap = communication",
            "hardware_limit": "active compute or memory work unit / matching gpu-spec peak; unlocked reuses the R3 grid regime, locked uses each current operating point; gap = profiled↔hardware",
            "necessary_work": necessary_work,
        },
        "buckets": buckets,
        "sampling": "R0/R1 exact over all rows; R2..R5 folded over 1-in-sample_stride iterations \
                     and anchored to the exact R1 by their sampled ratio",
        "kernel_level": "per manifest location (name); Max siblings + DP replicas pool; \
                         single-leaf so no idle/imbalance — only batching/communication/hw",
        "kernel_ladders": "the analyzer emits complete worker/pool/cluster ladders: R0/R1 reuse \
                           the attributable R2 kernel baseline plus aggregate idle/imbalance; \
                           R2..R6 carry per-location values; R7 is aggregate-only; unlocked parents \
                           recompute R6/R7 from additive FLOPs/bytes, while locked parents add values \
                           already evaluated at fixed-iteration boundaries; the UI never reconstructs them",
    })
}

fn unavailable(log_dir: &Path, reason: &str, lock_batch_size: bool) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string(),
                 "batch_size_locked": lock_batch_size},
        "available": false,
        "reason": reason,
        "definitions": definitions(lock_batch_size),
    })
}

fn unavailable_payload(log_dir: &Path, reason: &str, lock_batch_size: bool) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string(), "available": false, "reason": reason,
                 "batch_size_locked": lock_batch_size},
        "bucket_keys": BUCKET_KEYS,
        "rung_keys": RUNG_KEYS,
        "kernel_rung_keys": KERNEL_RUNG_KEYS,
        "levels": [],
        "kernels": [],
        "worker_kernel_ladders": [],
        "aggregate_kernel_ladders": [],
    })
}
