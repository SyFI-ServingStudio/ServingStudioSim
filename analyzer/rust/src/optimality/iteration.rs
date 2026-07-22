//! On-demand exact optimality details for one `(worker, iter_id)`. The waterfall
//! and kernel ladder have separate transport contracts, while both reuse this
//! module's all-row fold so their R0..R5 values are defined identically.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{bail, Context, Result};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_cost_manifests, read_run_meta, read_worker_gpu_counts, SCHEMA_VERSION};
use crate::session::{build_session, register_cost_log, require_columns, COST_LOG_TABLE};

use super::ladder::{KernelLadder, NecessaryWorkPolicy};
use super::run::COST_COLS;
use super::spec::{self, GpuSpec};
use super::{
    floors, fold, grid_peaks, kernel, levels, location, prepare,
    UNLOCKED_ITERATION_REPLICATION_FACTOR,
};

fn necessary_work_replication_factor(lock_batch_size: bool) -> u32 {
    if lock_batch_size {
        1
    } else {
        UNLOCKED_ITERATION_REPLICATION_FACTOR
    }
}

struct ExactIterationFold {
    ladder: KernelLadder,
    worker_rungs_gpu_ms: levels::BaseRungs,
    gpu_name: String,
    gpu_spec_matched: Option<String>,
    peaks_source: String,
    gpu_count: f64,
    folded_rows: u64,
    kernel_locations: Vec<prepare::KernelLocation>,
    gpu_spec: GpuSpec,
}

/// Shared typed computation projected by the two exact-iteration resources.
/// Keeping the transport contracts separate no longer duplicates fold/label rules.
struct ExactIterationAnalysis {
    exact: ExactIterationFold,
    label: Option<floors::IterationLabel>,
    label_caveats: Vec<String>,
    replication_factor: u32,
}

/// Build the exact stacked-kernel ladder for one selected worker iteration.
/// Locked mode adds mapped R6 necessary work when strict attribution succeeds.
/// R0 equals R1 because an iteration has no scheduler holding-span boundary;
/// imbalance remains the exact R1-R2 aggregate chunk.
pub(crate) async fn iteration_kernel_ladder(
    repo_root: &Path,
    log_dir: &Path,
    pool_tag: &str,
    worker_id: u16,
    iter_id: u64,
    lock_batch_size: bool,
) -> Result<Value> {
    let mut analysis = analyze_exact_iteration(
        repo_root,
        log_dir,
        pool_tag,
        worker_id,
        iter_id,
        lock_batch_size,
    )
    .await?;
    let mut caveats = analysis.label_caveats.clone();
    let attribution = match analysis.label.as_ref() {
        Some(label) => {
            match location::LocationCatalog::load(repo_root, log_dir).and_then(|catalog| {
                catalog.attribute_ladder(
                    pool_tag,
                    &analysis.exact.kernel_locations,
                    &mut analysis.exact.ladder,
                    label,
                    analysis.exact.gpu_spec,
                    analysis.exact.gpu_count,
                    if lock_batch_size {
                        NecessaryWorkPolicy::BatchLocked
                    } else {
                        NecessaryWorkPolicy::Saturated {
                            replication_factor: analysis.replication_factor,
                        }
                    },
                )
            }) {
                Ok(attribution) => Some(attribution),
                Err(error) => {
                    caveats.push(format!(
                        "per-location necessary work unavailable ({error:#})"
                    ));
                    None
                }
            }
        }
        None => None,
    };
    let ladder_value = analysis.exact.ladder.to_json()?;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "unit": "gpu_seconds",
        "worker": {"pool_tag": pool_tag, "worker_id": worker_id},
        "iter_id": iter_id,
        "rungs": ladder_value["rungs"],
        "special_chunks": ladder_value["special_chunks"],
        "kernels": ladder_value["kernels"],
        "meta": {
            "gpu_name": analysis.exact.gpu_name,
            "gpu_spec_matched": analysis.exact.gpu_spec_matched,
            "peaks_source": analysis.exact.peaks_source,
            "gpu_count": analysis.exact.gpu_count,
            "folded_rows": analysis.exact.folded_rows,
            "batch_size_locked": lock_batch_size,
            "necessary_work_available": attribution.is_some(),
            "necessary_work_replication_factor": analysis.replication_factor,
            "necessary_work_mode": if lock_batch_size { "batch_locked" } else { "replicated_large_batch" },
            "location_mapping_id": attribution.as_ref().map(|value| value.mapping_id.as_str()),
            "caveats": caveats,
        },
    }))
}

/// Build the full telescoping waterfall for one selected worker iteration.
/// The scope-fused floor belongs only to this aggregate view; the sibling ladder
/// may expose the segmented floor because that one is location-attributable.
pub(crate) async fn iteration_waterfall(
    repo_root: &Path,
    log_dir: &Path,
    pool_tag: &str,
    worker_id: u16,
    iter_id: u64,
    lock_batch_size: bool,
) -> Result<Value> {
    let analysis = analyze_exact_iteration(
        repo_root,
        log_dir,
        pool_tag,
        worker_id,
        iter_id,
        lock_batch_size,
    )
    .await?;
    let iteration_floor = analysis.label.as_ref().map(|label| label.floors);
    let level_key = format!("{pool_tag}/{worker_id}/{iter_id}");
    let level_label = format!("{pool_tag}/{worker_id} / iter {iter_id}");
    let level = levels::exact_iteration_level_json(
        &level_key,
        &level_label,
        &analysis.exact.worker_rungs_gpu_ms,
        iteration_floor,
    );

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "unit": "gpu_seconds",
        "worker": {"pool_tag": pool_tag, "worker_id": worker_id},
        "iter_id": iter_id,
        "level": level,
        "meta": {
            "gpu_name": analysis.exact.gpu_name,
            "gpu_spec_matched": analysis.exact.gpu_spec_matched,
            "peaks_source": analysis.exact.peaks_source,
            "gpu_count": analysis.exact.gpu_count,
            "folded_rows": analysis.exact.folded_rows,
            "batch_size_locked": lock_batch_size,
            "necessary_work_available": iteration_floor.is_some(),
            "necessary_work_replication_factor": analysis.replication_factor,
            "necessary_work_mode": if lock_batch_size { "batch_locked" } else { "replicated_large_batch" },
            "caveats": analysis.label_caveats,
        },
    }))
}

async fn analyze_exact_iteration(
    repo_root: &Path,
    log_dir: &Path,
    pool_tag: &str,
    worker_id: u16,
    iter_id: u64,
    lock_batch_size: bool,
) -> Result<ExactIterationAnalysis> {
    let context = build_session();
    if !register_cost_log(&context, log_dir).await? {
        bail!("cost_log/ dir not found");
    }
    require_columns(&context, COST_LOG_TABLE, COST_COLS).await?;
    let exact = fold_exact_iteration(
        &context,
        repo_root,
        log_dir,
        pool_tag,
        worker_id,
        iter_id,
        lock_batch_size,
    )
    .await?;
    let replication_factor = necessary_work_replication_factor(lock_batch_size);
    let (label, label_caveats) = match floors::compute_iteration_label(
        &context,
        log_dir,
        pool_tag,
        worker_id,
        iter_id,
        replication_factor,
    )
    .await
    {
        Ok(label) => (Some(label), Vec::new()),
        Err(error) => (
            None,
            vec![format!(
                "exact iteration necessary work unavailable ({error:#})"
            )],
        ),
    };
    Ok(ExactIterationAnalysis {
        exact,
        label,
        label_caveats,
        replication_factor,
    })
}

async fn fold_exact_iteration(
    context: &SessionContext,
    repo_root: &Path,
    log_dir: &Path,
    pool_tag: &str,
    worker_id: u16,
    iter_id: u64,
    lock_batch_size: bool,
) -> Result<ExactIterationFold> {
    // The UI chooses the same explicit variant for the aggregate payload and
    // this exact fold. Do not infer mode from a mutable report file: both
    // variants coexist for every launcher-produced run.
    let mut manifests_by_worker = read_cost_manifests(log_dir)?;
    let worker_key = (pool_tag.to_string(), worker_id);
    let manifest = manifests_by_worker
        .remove(&worker_key)
        .with_context(|| format!("cost manifest missing for worker {pool_tag}/{worker_id}"))?;
    let manifests_by_worker = BTreeMap::from([(worker_key.clone(), manifest)]);

    let worker_gpu_counts = read_worker_gpu_counts(log_dir).unwrap_or_default();
    let gpu_count = worker_gpu_counts
        .into_iter()
        .find_map(|(candidate_pool, candidate_worker, count)| {
            (candidate_pool == pool_tag && candidate_worker == worker_id).then_some(count)
        })
        .unwrap_or(1)
        .max(1) as f64;

    let (_num_gpus, gpu_name) = read_run_meta(log_dir).unwrap_or((1, String::new()));
    let (gpu_spec_matched, gpu_spec) = spec::load_gpu_spec(repo_root, &gpu_name)
        .map(|(name, spec)| (Some(name), spec))
        .unwrap_or((None, GpuSpec::default()));
    let hardware_bandwidth_gbps = gpu_spec.mem_bandwidth_gbps;
    let grid_peak_catalog = if lock_batch_size {
        grid_peaks::GridPeakCatalog::batch_locked()
    } else {
        grid_peaks::load_cached(log_dir)
    };
    let peaks_source = grid_peak_catalog.source.clone();
    let (kernel_locations, section_fold_plan_by_key) =
        prepare::build_section_fold_plans(&manifests_by_worker, &grid_peak_catalog, &gpu_spec);

    let Some(worker) =
        fold::read_exact_iteration_total(context, pool_tag, worker_id, iter_id, gpu_count).await?
    else {
        bail!("iteration {iter_id} has no rows for worker {pool_tag}/{worker_id}");
    };
    let mut workers = vec![worker];
    let worker_index_by_key = HashMap::from([(worker_key, 0usize)]);
    let mut rungs_by_location_worker = HashMap::new();
    let folded_rows = fold::accumulate_iteration_fold(
        context,
        pool_tag,
        worker_id,
        iter_id,
        hardware_bandwidth_gbps,
        lock_batch_size,
        &section_fold_plan_by_key,
        &worker_index_by_key,
        &mut workers,
        &mut rungs_by_location_worker,
    )
    .await?;
    if folded_rows == 0 {
        bail!("iteration {iter_id} rows did not match a manifest section");
    }

    let tier_aggregates = levels::assemble_tiers(&workers);
    let ladder = kernel::worker_kernel_ladders(
        &kernel_locations,
        &workers,
        &tier_aggregates.worker_rungs_in_fold_order,
        &rungs_by_location_worker,
        &tier_aggregates.worker_anchor_factors,
    )
    .into_iter()
    .next()
    .context("iteration fold produced no worker ladder")?;

    Ok(ExactIterationFold {
        ladder,
        worker_rungs_gpu_ms: tier_aggregates.worker_rungs_in_fold_order[0],
        gpu_name,
        gpu_spec_matched,
        peaks_source,
        gpu_count,
        folded_rows,
        kernel_locations,
        gpu_spec,
    })
}
