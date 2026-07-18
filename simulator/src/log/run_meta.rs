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

// v4 makes `workers[].pool_tag` authoritative for every worker (stamped on each GPU
// at `GpuCluster::allocate`), so non-KV workers (e.g. AFD ffn) no longer carry a null
// tag that consumers had to reverse-recover through `comm_groups`. v3 added per-worker
// `kv_pools` (`group_id` → KV token `capacity_tokens`); v2 added the `comm_groups`
// array. Append-only; the analyzer reads by field name and never asserts the version,
// so older readers are unaffected and older logs (null ffn tag) still parse via the
// analyzer's legacy `comm_groups` fallback.
const SCHEMA_VERSION: u32 = 4;

/// Write `<log_dir>/raw/run_meta.json` from the run's [`GpuCluster`]'s GPU
/// registry. Emits the flat per-GPU list, a derived worker→gpu grouping (each
/// worker also carrying its `kv_pools` = per-group KV token capacity, via
/// [`GpuCluster::kv_capacities`]), the total count, and the comm-group registry
/// (gid → base/count/gpu_ids, via [`GpuCluster::comm_groups`]). (The cluster's
/// runtime state — `cost` / `logger` / `kv_caps` — is `#[serde(skip)]` and never
/// reaches disk via the derived `Serialize`; only the reads above export it.)
pub fn write_run_meta(log_dir: &Path, cluster: &GpuCluster) -> Result<()> {
    let raw = log_dir.join("raw");
    std::fs::create_dir_all(&raw)?;

    // Derived (pool, worker)→(gpu ids, pool_tag) inverse of the flat list
    // (downstream convenience; the flat `gpus` already carries all three per gpu).
    // WorkerId is per-pool, so the pair is the unique run-level worker key. Every
    // GPU of a worker shares its `pool_tag` (stamped at `allocate`), so the first
    // one seen is authoritative — this is what makes each worker's tag independent
    // of whether it registered a KV pool or a comm group.
    let mut by_worker: BTreeMap<(u16, u16), (Vec<u16>, &str)> = BTreeMap::new();
    for g in &cluster.gpus {
        let entry = by_worker
            .entry((g.pool, g.worker_id))
            .or_insert_with(|| (Vec::new(), g.pool_tag.as_str()));
        entry.0.push(g.id);
    }

    // Per-worker KV-pool capacities (static token counts), grouped by the same
    // (pool, worker_id) key as `workers`. This is where the `kv_snapshot` stream's
    // former `capacity` column lives now — recorded once here, not per occupancy
    // row. Empty for workers with no KV pool (e.g. AFD ffn); the worker's tag no
    // longer rides along here since `by_worker` now carries it for every worker.
    let mut kv_by_worker: BTreeMap<(u16, u16), Vec<(u16, u64)>> = BTreeMap::new();
    for (_pool_tag, pool, worker_id, group_id, capacity) in cluster.kv_capacities() {
        kv_by_worker
            .entry((pool, worker_id))
            .or_default()
            .push((group_id, capacity));
    }

    let workers: Vec<_> = by_worker
        .into_iter()
        .map(|((pool, worker_id), (gpu_ids, pool_tag))| {
            let kv_pools: Vec<_> = kv_by_worker
                .get(&(pool, worker_id))
                .into_iter()
                .flatten()
                .map(|(group_id, capacity)| {
                    json!({ "group_id": group_id, "capacity_tokens": capacity })
                })
                .collect();
            // `pool_tag` is authoritative for every worker (stamped per GPU at
            // `allocate`), so non-KV workers keep their role tag — no `comm_groups`
            // reverse-recovery needed downstream.
            json!({
                "worker_id": worker_id,
                "pool": pool,
                "pool_tag": pool_tag,
                "gpu_ids": gpu_ids,
                "kv_pools": kv_pools,
            })
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
