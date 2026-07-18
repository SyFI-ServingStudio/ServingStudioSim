//! Artifact path helpers + JSON writer. The run dir uses the ref layout:
//! `raw/` (sim parquet) · `reports/` (numbers JSON) · `payloads/` (plot JSON) ·
//! `plots/` (Python PNG). The analyzer writes directly into `reports/` /
//! `payloads/`, so no post-hoc file shuffling is needed.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

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

/// Deployment-defined request-stage vocabulary from `run_meta.json` (v5+).
/// Stage codes in `request_slo` are intentionally opaque to the analyzer until
/// decoded through this sidecar; keeping that lookup here prevents subjects from
/// hard-coding deployment enums or importing simulator types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageVocabDoc {
    pub deployment: String,
    pub names: Vec<String>,
}

pub fn read_stage_vocab(log_dir: &Path) -> Option<StageVocabDoc> {
    let path = resolve_artifact_path(log_dir, "run_meta.json");
    let text = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    stage_vocab_from_run_meta(&json)
}

fn stage_vocab_from_run_meta(json: &serde_json::Value) -> Option<StageVocabDoc> {
    let vocab = json.get("stage_vocab")?;
    let deployment = vocab.get("deployment")?.as_str()?.to_owned();
    let names = vocab
        .get("names")?
        .as_array()?
        .iter()
        .map(|name| name.as_str().map(str::to_owned))
        .collect::<Option<Vec<_>>>()?;
    (!names.is_empty()).then_some(StageVocabDoc { deployment, names })
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
        let rest = stem.strip_prefix("worker_").with_context(|| {
            format!(
                "manifest filename must start with worker_: {}",
                path.display()
            )
        })?;
        let (pool_tag, worker_id) = rest.rsplit_once('_').with_context(|| {
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

/// The run's `(pool_tag, worker_id) → numeric pool` rows from `run_meta.json`
/// (sim-written, L7). `worker_id` is only unique within a pool, so dropping the
/// tag would alias workers such as AFD `attn/0` and `ffn/0`.
///
/// Every worker carries an authoritative `pool_tag` in `workers[]` (run_meta v4+,
/// stamped per GPU at `GpuCluster::allocate`), so the role tag is read directly —
/// no `comm_groups` reverse-recovery. Read as bare JSON (no `simulator` dep), like
/// [`read_run_meta`]. `None` means the sidecar lacks a usable tagged roster (absent
/// file, or a pre-v4 log whose non-KV workers still have a null tag); the
/// utilization subject then derives its roster from the observed composite keys in
/// `cost_log`.
pub fn read_worker_pools(log_dir: &Path) -> Option<Vec<(String, u64, u64)>> {
    let path = resolve_artifact_path(log_dir, "run_meta.json");
    let text = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    worker_pools_from_run_meta(&json)
}

fn worker_pools_from_run_meta(json: &serde_json::Value) -> Option<Vec<(String, u64, u64)>> {
    let workers = json.get("workers")?.as_array()?;
    let worker_pool_rows: Vec<(String, u64, u64)> = workers
        .iter()
        .filter_map(|worker| {
            let worker_id = worker.get("worker_id")?.as_u64()?;
            let pool = worker.get("pool")?.as_u64()?;
            let pool_tag = worker.get("pool_tag").and_then(Value::as_str)?.to_owned();
            Some((pool_tag, worker_id, pool))
        })
        .collect();
    (!worker_pool_rows.is_empty()).then_some(worker_pool_rows)
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
        // Every worker carries a `pool_tag` (v4+); a non-KV worker (AFD ffn) simply
        // has an empty `kv_pools`, so the inner loop emits nothing for it.
        let Some(tag) = w.get("pool_tag").and_then(|t| t.as_str()) else {
            continue;
        };
        let worker_id = w.get("worker_id").and_then(|v| v.as_u64()).unwrap_or(0);
        let Some(pools) = w.get("kv_pools").and_then(|p| p.as_array()) else {
            continue;
        };
        for p in pools {
            let group_id = p.get("group_id").and_then(|v| v.as_u64()).unwrap_or(0);
            let capacity = p
                .get("capacity_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            out.push((tag.to_owned(), worker_id, group_id, capacity));
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Per-worker physical GPU counts from `run_meta.json`'s `workers[]`, keyed by
/// the `(pool_tag, worker_id)` composite the cost_log rows and cost manifests use
/// — `(pool_tag, worker_id, gpu_ids.len())`.
///
/// `G_worker` MUST be read here, never inferred from tp×dp×ep degrees: whether a
/// "tp4+dp8" pool is one 32-GPU worker or eight 4-GPU workers is a deployment
/// definition, and it is what the `optimality` subject multiplies busy/held wall
/// time by to get GPU·seconds. Every worker carries an authoritative `pool_tag`
/// (run_meta v4+), so the role tag is read directly — no `comm_groups` recovery.
/// `None` when the sidecar is absent/unparseable or carries no worker roster (or a
/// pre-v4 log whose non-KV workers still have a null tag); the subject then degrades
/// to a single aggregate GPU.
pub fn read_worker_gpu_counts(log_dir: &Path) -> Option<Vec<(String, u16, usize)>> {
    let path = resolve_artifact_path(log_dir, "run_meta.json");
    let text = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let workers = json.get("workers")?.as_array()?;
    let rows: Vec<(String, u16, usize)> = workers
        .iter()
        .filter_map(|worker| {
            let worker_id = worker.get("worker_id")?.as_u64()?;
            let pool_tag = worker.get("pool_tag").and_then(Value::as_str)?.to_owned();
            let gpus = worker.get("gpu_ids").and_then(Value::as_array)?.len();
            Some((pool_tag, worker_id as u16, gpus))
        })
        .collect();
    (!rows.is_empty()).then_some(rows)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stage_vocab_parser_preserves_open_category_names() {
        let doc = stage_vocab_from_run_meta(&json!({
            "stage_vocab": {
                "deployment": "future",
                "names": ["pending:prefill", "suspended:preempted", "done:request"]
            }
        }))
        .unwrap();

        assert_eq!(doc.deployment, "future");
        assert_eq!(
            doc.names,
            vec!["pending:prefill", "suspended:preempted", "done:request"]
        );
    }
}
