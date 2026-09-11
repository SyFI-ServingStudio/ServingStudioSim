//! UI publication for analyzer-produced concurrency artifacts.

use anyhow::Result;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun, StageStatus};

const CONCURRENCY_REPORT: &str = "reports/concurrency_report.json";
const CONCURRENCY_PAYLOAD: &str = "payloads/concurrency_series.json";

pub(super) fn concurrency_descriptor(run: &DiscoveredRun) -> Result<Option<Value>> {
    if run.lifecycle.analysis != StageStatus::Complete
        || !regular_file(&run.path.join(CONCURRENCY_REPORT))
        || !regular_file(&run.path.join(CONCURRENCY_PAYLOAD))
        || !latest_run_succeeded(run, "concurrency")?
    {
        return Ok(None);
    }
    let report = read_concurrency_report(run)?;
    if report.get("available").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    Ok(Some(json!({
        "status": "ready",
        "schema_version": 1,
        "views": ["report", "payload"],
    })))
}

pub(super) fn read_concurrency_report(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, CONCURRENCY_REPORT)
}

pub(super) fn read_concurrency_payload(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, CONCURRENCY_PAYLOAD)
}

fn latest_run_succeeded(run: &DiscoveredRun, subject: &str) -> Result<bool> {
    let timing = read_run_json(&run.path, "reports/analyzer_timing.json")?;
    Ok(timing
        .get("subjects")
        .and_then(Value::as_array)
        .is_some_and(|subjects| {
            subjects.iter().any(|entry| {
                entry.get("name").and_then(Value::as_str) == Some(subject)
                    && entry.get("status").and_then(Value::as_str) == Some("ok")
            })
        }))
}
