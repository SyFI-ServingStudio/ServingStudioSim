//! Artifact path helpers + JSON writer. The run dir uses the ref layout:
//! `raw/` (sim parquet) · `reports/` (numbers JSON) · `payloads/` (plot JSON) ·
//! `plots/` (Python PNG). The analyzer writes directly into `reports/` /
//! `payloads/`, so no post-hoc file shuffling is needed.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;

/// Version of the report/payload JSON contract shared by every metric. Bump on
/// any breaking change to the envelope shape (`meta`/`metrics`/`series`/…) so the
/// Python renderer can refuse a payload it doesn't understand.
pub const SCHEMA_VERSION: u32 = 1;

const RAW_DIR: &str = "raw";
const REPORTS_DIR: &str = "reports";
const PAYLOADS_DIR: &str = "payloads";
const PLOTS_DIR: &str = "plots";

/// Find an input artifact, searching the run root then `raw/`/`reports/`/… (the
/// sim writes parquet under `raw/`; older runs may have them at the root).
pub fn resolve_artifact_path(log_dir: &Path, name: &str) -> PathBuf {
    for sub in ["", RAW_DIR, REPORTS_DIR, PAYLOADS_DIR, PLOTS_DIR] {
        let p = if sub.is_empty() {
            log_dir.join(name)
        } else {
            log_dir.join(sub).join(name)
        };
        if p.exists() {
            return p;
        }
    }
    log_dir.join(RAW_DIR).join(name)
}

/// The run's deployment name (e.g. `"unified"`), read as a bare string from
/// `params.json`. This is the analyzer's only deployment input — it stays a
/// string (no `simulator` dep), mirroring how we read parquet by column name.
/// `None` if the file is absent/unparseable; the applicability gate then keeps
/// only `Applies::All` subjects, which is the safe default for an unknown run.
pub fn read_deployment(log_dir: &Path) -> Option<String> {
    let path = resolve_artifact_path(log_dir, "params.json");
    let text = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    json.get("deployment")?.as_str().map(str::to_owned)
}

/// The run's GPU facts from `run_meta.json` (sim-written, L7): `(num_gpus,
/// gpu_name)`. Like [`read_deployment`], read as bare JSON (no `simulator` dep).
/// `None` if absent/unparseable; the throughput subject then treats the run as
/// single-GPU (the honest default for a run that predates the sidecar).
pub fn read_run_meta(log_dir: &Path) -> Option<(usize, String)> {
    let path = resolve_artifact_path(log_dir, "run_meta.json");
    let text = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let num_gpus = json.get("num_gpus")?.as_u64()? as usize;
    let gpu_name = json
        .get("gpus")
        .and_then(|g| g.get(0))
        .and_then(|g| g.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .to_owned();
    Some((num_gpus, gpu_name))
}

/// Read + parse per-worker cost manifests under `raw/cost_manifest/`.
///
/// Manifest filenames are keyed exactly like cost-log parquet filenames:
/// `worker_<pool_tag>_<worker_id>.json`. `worker_id` is only unique within a
/// pool, so trace consumers must use the `(pool_tag, worker_id)` pair from each
/// cost_log row to select the right CostTree.
pub fn read_cost_manifests(
    log_dir: &Path,
) -> Result<BTreeMap<(String, u16), crate::trace::manifest::ManifestDoc>> {
    use anyhow::{bail, Context};
    let dir = resolve_artifact_path(log_dir, "cost_manifest");
    if !dir.is_dir() {
        bail!("cost_manifest/ dir not found under {}", log_dir.display());
    }

    let mut manifests = BTreeMap::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .with_context(|| format!("invalid manifest filename {}", path.display()))?;
        let rest = stem
            .strip_prefix("worker_")
            .with_context(|| {
                format!(
                    "manifest filename must start with worker_: {}",
                    path.display()
                )
            })?;
        let (pool_tag, worker_id) = rest
            .rsplit_once('_')
            .with_context(|| {
                format!(
                    "manifest filename must end with _<worker_id>: {}",
                    path.display()
                )
            })?;
        let worker_id: u16 = worker_id
            .parse()
            .with_context(|| format!("parse worker id from {}", path.display()))?;
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read cost manifest at {}", path.display()))?;
        let manifest =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        let old = manifests.insert((pool_tag.to_owned(), worker_id), manifest);
        if old.is_some() {
            bail!("duplicate cost manifest key ({pool_tag}, {worker_id})");
        }
    }
    if manifests.is_empty() {
        bail!(
            "cost_manifest/ contains no worker_*.json files under {}",
            log_dir.display()
        );
    }
    Ok(manifests)
}

/// Output path for a Perfetto trace: `<log_dir>/traces/<prefix>.pftrace.gz`.
pub fn trace_path(log_dir: &Path, prefix: &str) -> PathBuf {
    log_dir.join("traces").join(format!("{prefix}.pftrace.gz"))
}

/// The run's `worker_id → pool` map from `run_meta.json`'s `workers[]` (sim-written,
/// L7). Read as bare JSON (no `simulator` dep), like [`read_run_meta`]. `None` if
/// absent/unparseable; the utilization subject then treats every `cost_log` worker
/// as belonging to a single pool 0 (the honest default for a run pre-dating the
/// sidecar, where the deployment is a single DP pool anyway).
pub fn read_worker_pools(log_dir: &Path) -> Option<Vec<(u64, u64)>> {
    let path = resolve_artifact_path(log_dir, "run_meta.json");
    let text = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let workers = json.get("workers")?.as_array()?;
    let pairs: Vec<(u64, u64)> = workers
        .iter()
        .filter_map(|w| Some((w.get("worker_id")?.as_u64()?, w.get("pool")?.as_u64()?)))
        .collect();
    (!pairs.is_empty()).then_some(pairs)
}

/// Per-worker × group KV-pool token capacities from `run_meta.json`'s `workers[]`
/// (v3+), as `(pool_tag, worker_id, group_id, capacity_tokens)`. This is the static
/// denominator the `kv-occupancy` subject divides the `kv_snapshot` series by; the
/// `pool_tag` matches the snapshot rows' key exactly (worker_id alone collides
/// across pools). Read as bare JSON (no `simulator` dep), like [`read_worker_pools`].
/// `None` when the sidecar is absent/unparseable or carries no KV worker (a run
/// with KV logging off, or a deployment with no KV pool) — the subject then reports
/// raw token occupancy without a capacity reference.
pub fn read_kv_capacities(log_dir: &Path) -> Option<Vec<(String, u64, u64, u64)>> {
    let path = resolve_artifact_path(log_dir, "run_meta.json");
    let text = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let workers = json.get("workers")?.as_array()?;
    let mut out = Vec::new();
    for w in workers {
        // `pool_tag` is null for non-KV workers (AFD ffn); their `kv_pools` is empty
        // too, so this loop simply skips them.
        let Some(tag) = w.get("pool_tag").and_then(|t| t.as_str()) else {
            continue;
        };
        let worker_id = w.get("worker_id").and_then(|v| v.as_u64()).unwrap_or(0);
        let Some(pools) = w.get("kv_pools").and_then(|p| p.as_array()) else {
            continue;
        };
        for p in pools {
            let group_id = p.get("group_id").and_then(|v| v.as_u64()).unwrap_or(0);
            let capacity = p.get("capacity_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            out.push((tag.to_owned(), worker_id, group_id, capacity));
        }
    }
    (!out.is_empty()).then_some(out)
}

pub fn report_path(log_dir: &Path, name: &str) -> PathBuf {
    log_dir.join(REPORTS_DIR).join(name)
}

pub fn payload_path(log_dir: &Path, name: &str) -> PathBuf {
    log_dir.join(PAYLOADS_DIR).join(name)
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    println!("wrote {}", path.display());
    Ok(())
}
