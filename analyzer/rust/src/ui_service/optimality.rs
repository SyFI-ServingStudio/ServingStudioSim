//! UI publication for the optimality sub-optimality waterfall artifacts.

use anyhow::Result;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun, StageStatus};

const REPORT: &str = "reports/optimality_report.json";
const PAYLOAD: &str = "payloads/optimality_waterfall.json";

pub(super) fn optimality_descriptor(run: &DiscoveredRun) -> Result<Option<Value>> {
    if run.lifecycle.analysis != StageStatus::Complete
        || !regular_file(&run.path.join(REPORT))
        || !regular_file(&run.path.join(PAYLOAD))
        || !latest_run_succeeded(run)?
    {
        return Ok(None);
    }
    let report = read_optimality_report(run)?;
    if report.get("available").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    Ok(Some(json!({
        "status": "ready",
        "schema_version": 1,
        "report_href": "subjects/optimality/report",
        "payload_href": "subjects/optimality/payload",
    })))
}

pub(super) fn read_optimality_report(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, REPORT)
}

pub(super) fn read_optimality_payload(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, PAYLOAD)
}

fn latest_run_succeeded(run: &DiscoveredRun) -> Result<bool> {
    let timing = read_run_json(&run.path, "reports/analyzer_timing.json")?;
    Ok(timing
        .get("subjects")
        .and_then(Value::as_array)
        .is_some_and(|subjects| {
            subjects.iter().any(|entry| {
                entry.get("name").and_then(Value::as_str) == Some("optimality")
                    && entry.get("status").and_then(Value::as_str) == Some("ok")
            })
        }))
}
