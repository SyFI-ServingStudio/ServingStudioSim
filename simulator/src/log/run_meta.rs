//! `run_meta.json` — run-level GPU facts sidecar (L7).
//!
//! The sim's own record of the GPUs a run modeled (id / name / pool / owning
//! worker), so the analyzer can normalize throughput per-GPU and label plots with
//! the GPU name. Distinct from the launcher-written `params.json` (run inputs)
//! and the per-worker `cost_manifest/` sidecars (CostTree structure). Written
//! once after the flow is built; mirrors `cost_logger`'s manifest write
//! (serde_json → fs).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde_json::json;

use crate::orchestrator::GpuInventory;

const SCHEMA_VERSION: u32 = 1;

/// Write `<log_dir>/raw/run_meta.json` from the run's [`GpuInventory`]. Emits the
/// flat per-GPU list plus a derived worker→gpu grouping and the total count.
pub fn write_run_meta(log_dir: &Path, inventory: &GpuInventory) -> Result<()> {
    let raw = log_dir.join("raw");
    std::fs::create_dir_all(&raw)?;

    // Derived (pool, worker)→gpu inverse of the flat list (downstream
    // convenience; the flat `gpus` already carries both ids per gpu). WorkerId is
    // per-pool, so the pair is the unique run-level worker key.
    let mut by_worker: BTreeMap<(u16, u16), Vec<u16>> = BTreeMap::new();
    for g in &inventory.gpus {
        by_worker.entry((g.pool, g.worker_id)).or_default().push(g.id);
    }
    let workers: Vec<_> = by_worker
        .into_iter()
        .map(|((pool, worker_id), gpu_ids)| {
            json!({ "worker_id": worker_id, "pool": pool, "gpu_ids": gpu_ids })
        })
        .collect();

    let meta = json!({
        "schema_version": SCHEMA_VERSION,
        "num_gpus": inventory.num_gpus(),
        "gpus": inventory.gpus,
        "workers": workers,
    });
    std::fs::write(raw.join("run_meta.json"), serde_json::to_vec_pretty(&meta)?)?;
    Ok(())
}
