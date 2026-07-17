//! UI publication for analyzer-produced workload-conservation artifacts.

use anyhow::Result;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun, StageStatus};

const REPORT: &str = "reports/workload_conservation_report.json";
const PAYLOAD: &str = "payloads/workload_conservation_checks.json";

pub(super) fn workload_conservation_descriptor(run: &DiscoveredRun) -> Result<Option<Value>> {
    if run.lifecycle.analysis != StageStatus::Complete
        || !regular_file(&run.path.join(REPORT))
        || !regular_file(&run.path.join(PAYLOAD))
        || !latest_run_succeeded(run)?
    {
        return Ok(None);
    }
    // An unavailable artifact is still generated evidence: the payload reason
    // lets the typed UI distinguish missing inputs from a subject never run.
    Ok(Some(json!({
        "status": "ready",
        "schema_version": 1,
        "report_href": "subjects/workload-conservation/report",
        "payload_href": "subjects/workload-conservation/payload",
    })))
}

pub(super) fn read_workload_conservation_report(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, REPORT)
}

pub(super) fn read_workload_conservation_payload(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, PAYLOAD)
}

fn latest_run_succeeded(run: &DiscoveredRun) -> Result<bool> {
    let timing = read_run_json(&run.path, "reports/analyzer_timing.json")?;
    Ok(timing
        .get("subjects")
        .and_then(Value::as_array)
        .is_some_and(|subjects| {
            subjects.iter().any(|entry| {
                entry.get("name").and_then(Value::as_str) == Some("workload-conservation")
                    && entry.get("status").and_then(Value::as_str) == Some("ok")
            })
        }))
}
