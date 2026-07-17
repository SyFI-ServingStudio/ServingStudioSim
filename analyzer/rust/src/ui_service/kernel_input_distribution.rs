//! UI publication for analyzer-produced kernel-input distribution artifacts.

use anyhow::Result;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun, StageStatus};

const REPORT: &str = "reports/kernel_input_distribution_report.json";
const PAYLOAD: &str = "payloads/kernel_input_distribution_scatter.json";

pub(super) fn kernel_input_distribution_descriptor(run: &DiscoveredRun) -> Result<Option<Value>> {
    if run.lifecycle.analysis != StageStatus::Complete
        || !regular_file(&run.path.join(REPORT))
        || !regular_file(&run.path.join(PAYLOAD))
        || !latest_run_succeeded(run)?
    {
        return Ok(None);
    }
    // `available: false` is still a successfully generated analyzer result. Its
    // reason tells the UI why an older run cannot provide distribution points.
    Ok(Some(json!({
        "status": "ready",
        "schema_version": 1,
        "report_href": "subjects/kernel-input-distribution/report",
        "payload_href": "subjects/kernel-input-distribution/payload",
    })))
}

pub(super) fn read_kernel_input_distribution_report(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, REPORT)
}

pub(super) fn read_kernel_input_distribution_payload(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, PAYLOAD)
}

fn latest_run_succeeded(run: &DiscoveredRun) -> Result<bool> {
    let timing = read_run_json(&run.path, "reports/analyzer_timing.json")?;
    Ok(timing
        .get("subjects")
        .and_then(Value::as_array)
        .is_some_and(|subjects| {
            subjects.iter().any(|entry| {
                entry.get("name").and_then(Value::as_str) == Some("kernel-input-distribution")
                    && entry.get("status").and_then(Value::as_str) == Some("ok")
            })
        }))
}
