//! UI publication for the analyzer's worker and pool GPU utilization artifacts.

use anyhow::Result;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun, StageStatus};

const UTILIZATION_REPORT: &str = "reports/utilization_report.json";
const UTILIZATION_PAYLOAD: &str = "payloads/utilization_series.json";

pub(super) fn utilization_descriptor(run: &DiscoveredRun) -> Result<Option<Value>> {
    if run.lifecycle.analysis != StageStatus::Complete
        || !regular_file(&run.path.join(UTILIZATION_REPORT))
        || !regular_file(&run.path.join(UTILIZATION_PAYLOAD))
        || !latest_run_succeeded(run)?
    {
        return Ok(None);
    }
    let report = read_utilization_report(run)?;
    if report.get("available").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    Ok(Some(json!({
        "status": "ready",
        "schema_version": 1,
        "views": ["report", "payload"],
    })))
}

pub(super) fn read_utilization_report(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, UTILIZATION_REPORT)
}

pub(super) fn read_utilization_payload(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, UTILIZATION_PAYLOAD)
}

fn latest_run_succeeded(run: &DiscoveredRun) -> Result<bool> {
    let timing = read_run_json(&run.path, "reports/analyzer_timing.json")?;
    Ok(timing
        .get("subjects")
        .and_then(Value::as_array)
        .is_some_and(|subjects| {
            subjects.iter().any(|entry| {
                entry.get("name").and_then(Value::as_str) == Some("utilization")
                    && entry.get("status").and_then(Value::as_str) == Some("ok")
            })
        }))
}
