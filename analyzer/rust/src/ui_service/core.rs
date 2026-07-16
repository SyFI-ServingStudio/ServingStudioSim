//! Core run resources: the descriptor index and simulator summary.

use anyhow::{Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::artifact::{read_bytes, read_run_json};
use super::discovery::{regular_file, timestamp, DiscoveredRun, StageStatus};
use super::model::model_config_path;
use super::PROTOCOL_VERSION;

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
    if let Some(model_config) = model_config_path(&params)? {
        descriptor["model_name"] = Value::String(model_config);
        descriptor["model"] = json!({
            "href": "model",
            "media_type": "application/json",
            "schema_version": 1,
        });
    }
    if run.lifecycle.analysis == StageStatus::Complete {
        let timing_path = run.path.join("reports/analyzer_timing.json");
        let timing_bytes = read_bytes(&timing_path)?;
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
