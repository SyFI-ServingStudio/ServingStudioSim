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

use crate::worker::GpuCluster;

// v2 adds the `comm_groups` array (gid → base/count/gpu_ids). Append-only; the
// analyzer reads by field name and never asserts the version, so v1 readers are
// unaffected.
const SCHEMA_VERSION: u32 = 2;

/// Write `<log_dir>/raw/run_meta.json` from the run's [`GpuCluster`]'s GPU
/// registry. Emits the flat per-GPU list, a derived worker→gpu grouping, the
/// total count, and the comm-group registry (gid → base/count/gpu_ids, via
/// [`GpuCluster::comm_groups`]). (The cluster's runtime state — `streams` /
/// `cost` — is `#[serde(skip)]` and never reaches disk.)
pub fn write_run_meta(log_dir: &Path, cluster: &GpuCluster) -> Result<()> {
    let raw = log_dir.join("raw");
    std::fs::create_dir_all(&raw)?;

    // Derived (pool, worker)→gpu inverse of the flat list (downstream
    // convenience; the flat `gpus` already carries both ids per gpu). WorkerId is
    // per-pool, so the pair is the unique run-level worker key.
    let mut by_worker: BTreeMap<(u16, u16), Vec<u16>> = BTreeMap::new();
    for g in &cluster.gpus {
        by_worker.entry((g.pool, g.worker_id)).or_default().push(g.id);
    }
    let workers: Vec<_> = by_worker
        .into_iter()
        .map(|((pool, worker_id), gpu_ids)| {
            json!({ "worker_id": worker_id, "pool": pool, "gpu_ids": gpu_ids })
        })
        .collect();

    // Comm-group registry: the canonical gid → gpu-set mapping. `gpu_cluster`
    // rows carry `send_gid`/`recv_gid` (plus a denormalized `send_count`/
    // `recv_count`); this table is what an analysis holding only a gid uses to
    // recover the group's size and exactly which GPUs it spans (the contiguous
    // block `[base, base+count)`).
    let comm_groups: Vec<_> = cluster
        .comm_groups()
        .map(|(gid, base, count, owner_pool, owner_worker_id)| {
            json!({
                "gid": gid,
                "base": base,
                "count": count,
                "gpu_ids": (base..base + count).collect::<Vec<u16>>(),
                "owner_pool": owner_pool,
                "owner_worker_id": owner_worker_id,
            })
        })
        .collect();

    let meta = json!({
        "schema_version": SCHEMA_VERSION,
        "num_gpus": cluster.num_gpus(),
        "gpus": cluster.gpus,
        "workers": workers,
        "comm_groups": comm_groups,
    });
    std::fs::write(raw.join("run_meta.json"), serde_json::to_vec_pretty(&meta)?)?;
    Ok(())
}
