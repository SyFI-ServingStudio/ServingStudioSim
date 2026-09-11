//! UI publication for the analyzer's run-throughput artifacts.

use anyhow::Result;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun, StageStatus};

const THROUGHPUT_REPORT: &str = "reports/throughput_report.json";
const THROUGHPUT_PAYLOAD: &str = "payloads/throughput_segments.json";

pub(super) fn throughput_descriptor(run: &DiscoveredRun) -> Result<Option<Value>> {
    if run.lifecycle.analysis != StageStatus::Complete
        || !regular_file(&run.path.join(THROUGHPUT_REPORT))
        || !regular_file(&run.path.join(THROUGHPUT_PAYLOAD))
        || !latest_run_succeeded(run)?
    {
        return Ok(None);
    }
    let report = read_throughput_report(run)?;
    if report.get("available").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    Ok(Some(json!({
        "status": "ready",
        "schema_version": 1,
        "views": ["report", "payload"],
    })))
}

pub(super) fn read_throughput_report(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, THROUGHPUT_REPORT)
}

pub(super) fn read_throughput_payload(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, THROUGHPUT_PAYLOAD)
}

fn latest_run_succeeded(run: &DiscoveredRun) -> Result<bool> {
    let timing = read_run_json(&run.path, "reports/analyzer_timing.json")?;
    Ok(timing
        .get("subjects")
        .and_then(Value::as_array)
        .is_some_and(|subjects| {
            subjects.iter().any(|entry| {
                entry.get("name").and_then(Value::as_str) == Some("throughput")
                    && entry.get("status").and_then(Value::as_str) == Some("ok")
            })
        }))
}
