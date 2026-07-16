//! Descriptor plus the simulator-owned summary and topology resources.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::discovery::{regular_file, timestamp, DiscoveredRun, StageStatus};
use super::{ArtifactNotFound, PROTOCOL_VERSION};

pub(super) fn build_descriptor(run: &DiscoveredRun) -> Result<Value> {
    let params = read_run_json(&run.path, "raw/params.json")?;
    let deployment = params
        .get("deployment")
        .and_then(Value::as_str)
        .filter(|value| matches!(*value, "unified" | "pd" | "afd"))
        .context("raw/params.json has no supported deployment")?;
    let mut descriptor = json!({
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run.run_id,
        "kind": "simulation",
        "display_name": run.display_name,
        "deployment": deployment,
        "lifecycle": run.lifecycle,
        "summary": { "href": "summary" },
        "subjects": {},
        "details": {},
        "traces": {},
    });

    if regular_file(&run.path.join("raw/run_meta.json")) {
        descriptor["topology"] = json!({
            "href": "topology",
            "media_type": "application/json",
            "schema_version": 1,
        });
    }
    if let Some(model_name) = model_name(&params) {
        descriptor["model_name"] = Value::String(model_name);
    }
    if run.lifecycle.analysis == StageStatus::Complete {
        let timing_path = run.path.join("reports/analyzer_timing.json");
        let timing_bytes = read_regular_file(&timing_path)?;
        let generated_at = timing_path
            .metadata()
            .and_then(|metadata| metadata.modified())
            .map(timestamp)
            .context("read analyzer timing modification time")?;
        descriptor["analysis"] = json!({
            "revision": format!("legacy-sha256-{:x}", Sha256::digest(&timing_bytes)),
            "generated_at": generated_at,
            "generator_version": "legacy-analyzer-v1",
        });
    }
    Ok(descriptor)
}

pub(super) fn read_summary(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, "summary.json")
}

pub(super) fn build_topology(run: &DiscoveredRun) -> Result<Value> {
    Ok(json!({
        "schema_version": 1,
        "params": read_run_json(&run.path, "raw/params.json")?,
        "run_meta": read_run_json(&run.path, "raw/run_meta.json")?,
    }))
}

fn read_run_json(run: &Path, relative: &str) -> Result<Value> {
    let path = run.join(relative);
    let bytes = read_regular_file(&path)?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn read_regular_file(path: &Path) -> Result<Vec<u8>> {
    if !regular_file(path) {
        return Err(ArtifactNotFound.into());
    }
    fs::read(path).with_context(|| format!("read {}", path.display()))
}

fn model_name(params: &Value) -> Option<String> {
    params
        .get("pools")?
        .as_object()?
        .values()
        .find_map(|pool| {
            pool.get("groups")?
                .as_array()?
                .iter()
                .find_map(|group| group.get("arch")?.get("model_config")?.as_str())
        })
        .map(str::to_owned)
}
