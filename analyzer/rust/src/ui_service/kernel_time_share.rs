//! UI publication for critical-path kernel-time composition artifacts.

use anyhow::Result;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun, StageStatus};

const REPORT: &str = "reports/kernel_time_share_report.json";
const PAYLOAD: &str = "payloads/kernel_time_share_composition.json";

pub(super) fn kernel_time_share_descriptor(run: &DiscoveredRun) -> Result<Option<Value>> {
    if run.lifecycle.analysis != StageStatus::Complete
        || !regular_file(&run.path.join(REPORT))
        || !regular_file(&run.path.join(PAYLOAD))
        || !latest_run_succeeded(run)?
    {
        return Ok(None);
    }
    let report = read_kernel_time_share_report(run)?;
    if report.get("available").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    Ok(Some(json!({
        "status": "ready",
        "schema_version": 1,
        "report_href": "subjects/kernel-time-share/report",
        "payload_href": "subjects/kernel-time-share/payload",
    })))
}

pub(super) fn read_kernel_time_share_report(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, REPORT)
}

pub(super) fn read_kernel_time_share_payload(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, PAYLOAD)
}

fn latest_run_succeeded(run: &DiscoveredRun) -> Result<bool> {
    let timing = read_run_json(&run.path, "reports/analyzer_timing.json")?;
    Ok(timing
        .get("subjects")
        .and_then(Value::as_array)
        .is_some_and(|subjects| {
            subjects.iter().any(|entry| {
                entry.get("name").and_then(Value::as_str) == Some("kernel-time-share")
                    && entry.get("status").and_then(Value::as_str) == Some("ok")
            })
        }))
}
