//! On-demand exact kernel ladder for one `(worker, iter_id)`. This is the
//! high-cardinality detail counterpart of the run payload's sampled per-worker
//! ladders: it reuses the same preparation, leaf rate ceilings, and attribution code,
//! but folds every row belonging to the selected iteration.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::io::{read_cost_manifests, read_run_meta, read_worker_gpu_counts, SCHEMA_VERSION};
use crate::session::{build_session, register_cost_log, require_columns, COST_LOG_TABLE};

use super::run::COST_COLS;
use super::spec::{self, GpuSpec};
use super::{fold, grid_peaks, kernel, levels, prepare};

/// Build the exact R0..R5 stacked-kernel ladder for one selected worker
/// iteration. R0 equals R1 because an iteration has no scheduler holding-span
/// boundary; imbalance remains the exact R1-R2 aggregate chunk.
pub(crate) async fn iteration_kernel_ladder(
    repo_root: &Path,
    log_dir: &Path,
    pool_tag: &str,
    worker_id: u16,
    iter_id: u64,
    lock_batch_size: bool,
) -> Result<Value> {
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

    let ctx = build_session();
    if !register_cost_log(&ctx, log_dir).await? {
        bail!("cost_log/ dir not found");
    }
    require_columns(&ctx, COST_LOG_TABLE, COST_COLS).await?;
    let Some(worker) =
        fold::read_exact_iteration_total(&ctx, pool_tag, worker_id, iter_id, gpu_count).await?
    else {
        bail!("iteration {iter_id} has no rows for worker {pool_tag}/{worker_id}");
    };
    let mut workers = vec![worker];
    let worker_index_by_key = HashMap::from([(worker_key, 0usize)]);
    let mut rungs_by_location_worker = HashMap::new();
    let folded_rows = fold::accumulate_iteration_fold(
        &ctx,
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
    let ladder = kernel::worker_kernel_ladders_json(
        &kernel_locations,
        &workers,
        &tier_aggregates.worker_rungs_in_fold_order,
        &rungs_by_location_worker,
        &tier_aggregates.worker_anchor_factors,
    )
    .into_iter()
    .next()
    .context("iteration fold produced no worker ladder")?;

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "unit": "gpu_seconds",
        "worker": {"pool_tag": pool_tag, "worker_id": worker_id},
        "iter_id": iter_id,
        "rungs": ladder["rungs"],
        "special_chunks": ladder["special_chunks"],
        "kernels": ladder["kernels"],
        "meta": {
            "gpu_name": gpu_name,
            "gpu_spec_matched": gpu_spec_matched,
            "peaks_source": peaks_source,
            "gpu_count": gpu_count,
            "folded_rows": folded_rows,
            "batch_size_locked": lock_batch_size,
        },
    }))
}
