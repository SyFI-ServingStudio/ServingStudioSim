//! Topology resource projected directly from simulator-owned run artifacts.

use anyhow::Result;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::DiscoveredRun;

pub(super) fn build_topology(run: &DiscoveredRun) -> Result<Value> {
    Ok(json!({
        "schema_version": 1,
        "params": read_run_json(&run.path, "raw/params.json")?,
        "run_meta": read_run_json(&run.path, "raw/run_meta.json")?,
    }))
}
