//! UI publication for the analyzer's pool and worker KV-occupancy artifacts.

use anyhow::Result;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun, StageStatus};

const KV_OCCUPANCY_REPORT: &str = "reports/kv_occupancy_report.json";
const KV_OCCUPANCY_PAYLOAD: &str = "payloads/kv_occupancy_series.json";

pub(super) fn kv_occupancy_descriptor(run: &DiscoveredRun) -> Result<Option<Value>> {
    if run.lifecycle.analysis != StageStatus::Complete
        || !regular_file(&run.path.join(KV_OCCUPANCY_REPORT))
        || !regular_file(&run.path.join(KV_OCCUPANCY_PAYLOAD))
        || !latest_run_succeeded(run)?
    {
        return Ok(None);
    }
    let report = read_kv_occupancy_report(run)?;
    if report.get("available").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    Ok(Some(json!({
        "status": "ready",
        "schema_version": 1,
        "report_href": "subjects/kv-occupancy/report",
        "payload_href": "subjects/kv-occupancy/payload",
    })))
}

pub(super) fn read_kv_occupancy_report(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, KV_OCCUPANCY_REPORT)
}

pub(super) fn read_kv_occupancy_payload(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, KV_OCCUPANCY_PAYLOAD)
}

fn latest_run_succeeded(run: &DiscoveredRun) -> Result<bool> {
    let timing = read_run_json(&run.path, "reports/analyzer_timing.json")?;
    Ok(timing
        .get("subjects")
        .and_then(Value::as_array)
        .is_some_and(|subjects| {
            subjects.iter().any(|entry| {
                entry.get("name").and_then(Value::as_str) == Some("kv-occupancy")
                    && entry.get("status").and_then(Value::as_str) == Some("ok")
            })
        }))
}
