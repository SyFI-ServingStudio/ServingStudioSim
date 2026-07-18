//! Orchestration — `run_optimality` wires preparation → fold → tiers → kernel and
//! assembles the top-level `meta` / `report` / `payload` JSON. Also owns the output
//! contract shared by the degrade paths (`definitions`, `unavailable*`). The caveat
//! bookkeeping (gpu-count degrade, missing roster, no spec, empty peaks) stays here
//! because it is orchestration glue, not part of any one stage.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_cost_manifests, read_run_meta, read_worker_gpu_counts, SCHEMA_VERSION};
use crate::kernel_query::repo_root;
use crate::session::{register_cost_log, require_columns, COST_LOG_TABLE};

use super::spec::{self, GpuSpec};
use super::{fold, grid_peaks, kernel, levels, prepare};
use super::{ratio, BUCKET_KEYS, KERNEL_RUNG_KEYS, R0, R5, RUNG_KEYS};

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

pub async fn run_optimality(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLS).await?;
    let manifests_by_worker = match read_cost_manifests(log_dir) {
        Ok(manifests_by_worker) => manifests_by_worker,
        Err(error) => {
            let reason =
                format!("cost_manifest/ unreadable ({error:#}); needed to fold the ladder");
            return Ok((
                unavailable(log_dir, &reason),
                unavailable_payload(log_dir, &reason),
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

    // Hardware roofline (R5) + grid-peak ceilings (R3).
    let (_num_gpus, gpu_name) = read_run_meta(log_dir).unwrap_or((1, String::new()));
    let repository_root = repo_root().ok();
    let matched_gpu_spec = repository_root
        .as_deref()
        .and_then(|root| spec::load_gpu_spec(root, &gpu_name));
    let (gpu_spec_matched, gpu_spec): (Option<String>, GpuSpec) = match matched_gpu_spec {
        Some((matched_name, spec)) => (Some(matched_name), spec),
        None => {
            caveats.push(format!(
                "no gpu/spec.json entry for gpu_name {gpu_name:?}; R5 hardware limit \
                 collapses onto R4 (no hardware-gap bucket) — add an alias"
            ));
            (None, GpuSpec::default())
        }
    };
    let hardware_bandwidth_gbps = gpu_spec.mem_bandwidth_gbps;

    let grid_peak_catalog = grid_peaks::load_or_generate(log_dir, &manifests_by_worker);
    if grid_peak_catalog.is_empty() {
        caveats.push(format!(
            "grid-peaks sidecar {}: batching headroom not computed (R3 = R2)",
            grid_peak_catalog.source
        ));
    }

    // Intern locations + precompute per-section metadata.
    let (kernel_locations, section_fold_plan_by_key) =
        prepare::build_section_fold_plans(&manifests_by_worker, &grid_peak_catalog, &gpu_spec);

    // Exact per-worker busy + span (all rows).
    let mut workers = fold::read_exact_worker_totals(ctx, &gpu_count_by_worker).await?;
    if workers.is_empty() {
        let reason = "cost_log has no (pool_tag, worker_id) rows";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
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
        &section_fold_plan_by_key,
        &worker_index_by_key,
        &mut workers,
        &mut sampled_rungs_by_location_worker,
    )
    .await?;

    // Assemble rungs (GPU·ms) per worker → pool → cluster; then per-kernel.
    let tier_aggregates = levels::assemble_tiers(&workers);
    let kernel_rungs_by_location = kernel::aggregate_kernel_rungs(
        &sampled_rungs_by_location_worker,
        &tier_aggregates.worker_anchor_factors,
        kernel_locations.len(),
    );
    let worker_kernel_ladders = kernel::worker_kernel_ladders_json(
        &kernel_locations,
        &workers,
        &tier_aggregates.worker_rungs_in_fold_order,
        &sampled_rungs_by_location_worker,
        &tier_aggregates.worker_anchor_factors,
    );

    let cluster_optimality_ratio = ratio(
        tier_aggregates.cluster_rungs[R5],
        tier_aggregates.cluster_rungs[R0],
    );

    let level_entries = levels::levels_json(&tier_aggregates);
    let kernel_entries = kernel::kernel_levels_json(&kernel_locations, &kernel_rungs_by_location);

    let analysis_meta = json!({
        "log_dir": log_dir.display().to_string(),
        "gpu_name": gpu_name,
        "gpu_spec_matched": gpu_spec_matched,
        "peaks_source": grid_peak_catalog.source,
        "sample_stride": sample_stride,
        "sampled_rows": sampled_row_count,
        "gpu_counts_available": has_worker_gpu_counts,
        "num_workers": workers.len(),
        "num_locations": kernel_locations.len(),
        "caveats": caveats,
    });

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "available": true,
        "meta": analysis_meta,
        "optimality_ratio": cluster_optimality_ratio,
        "unit": "gpu_seconds",
        "cluster": levels::rung_report_json(&tier_aggregates.cluster_rungs),
        "pools": tier_aggregates
            .pool_rungs_by_tag
            .iter()
            .map(|(pool_tag, pool_rungs)| {
                json!({
                    "pool": pool_tag,
                    "rungs": levels::rung_report_json(pool_rungs),
                })
            })
            .collect::<Vec<_>>(),
        "worst_batching_kernels": kernel::worst_batching_json(
            &kernel_locations,
            &kernel_rungs_by_location,
        ),
        "definitions": definitions(),
    });

    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "available": true,
        "meta": analysis_meta,
        "unit": "gpu_seconds",
        "optimality_ratio": cluster_optimality_ratio,
        "bucket_keys": BUCKET_KEYS,
        "rung_keys": RUNG_KEYS,
        "kernel_rung_keys": KERNEL_RUNG_KEYS,
        "levels": level_entries,
        "kernels": kernel_entries,
        "worker_kernel_ladders": worker_kernel_ladders,
        "definitions": definitions(),
    });

    Ok((report, payload))
}

fn definitions() -> Value {
    json!({
        "scope": "per-worker CostTree manifest re-folded with per-rung leaf substitutions; \
                  unit GPU-seconds = wall time × the worker's run_meta gpu_ids count",
        "ladder": {
            "real": "span × G — GPU·s actually held (includes idle)",
            "busy": "Σ total_time_ms × G — no scheduler idle; gap vs real = idle",
            "balanced": "real slot times, Max folded to mean — perfect load balance; gap = imbalance",
            "per_config_best": "work / grid-peak rate for the leaf's fixed config over its batch axis; gap = batching",
            "ignore_network": "per_config_best with comm leaves dropped; gap = communication",
            "hardware_limit": "work / gpu-spec dense peak (roofline); gap = profiled↔hardware; this rung = irreducible",
        },
        "buckets": "idle, imbalance, batching, communication, hardware_gap, hardware_optimal — \
                    telescoping differences of the rungs; sum to Real",
        "sampling": "R0/R1 exact over all rows; R2..R5 folded over 1-in-sample_stride iterations \
                     and anchored to the exact R1 by their sampled ratio",
        "kernel_level": "per manifest location (name); Max siblings + DP replicas pool; \
                         single-leaf so no idle/imbalance — only batching/communication/hw",
        "worker_kernel_ladder": "R0/R1 reuse the attributable R2 kernel baseline and append \
                                 aggregate idle/imbalance chunks; R2..R5 carry per-kernel values",
    })
}

fn unavailable(log_dir: &Path, reason: &str) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string()},
        "available": false,
        "reason": reason,
        "definitions": definitions(),
    })
}

fn unavailable_payload(log_dir: &Path, reason: &str) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string(), "available": false, "reason": reason},
        "bucket_keys": BUCKET_KEYS,
        "rung_keys": RUNG_KEYS,
        "kernel_rung_keys": KERNEL_RUNG_KEYS,
        "levels": [],
        "kernels": [],
        "worker_kernel_ladders": [],
    })
}
