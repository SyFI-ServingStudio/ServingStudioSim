//! Artifact path helpers + JSON writer. The run dir uses the ref layout:
//! `raw/` (sim parquet) · `reports/` (numbers JSON) · `payloads/` (plot JSON) ·
//! `plots/` (Python PNG). The analyzer writes directly into `reports/` /
//! `payloads/`, so no post-hoc file shuffling is needed.

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
