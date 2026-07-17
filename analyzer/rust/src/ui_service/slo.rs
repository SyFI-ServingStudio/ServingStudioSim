//! UI publication for the analyzer's general SLO artifacts.

use anyhow::Result;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun, StageStatus};

const SLO_GENERAL_REPORT: &str = "reports/slo_general_report.json";
const SLO_GENERAL_PAYLOAD: &str = "payloads/slo_general_cdf.json";

pub(super) fn slo_general_descriptor(run: &DiscoveredRun) -> Result<Option<Value>> {
    if run.lifecycle.analysis != StageStatus::Complete
        || !regular_file(&run.path.join(SLO_GENERAL_REPORT))
        || !regular_file(&run.path.join(SLO_GENERAL_PAYLOAD))
        || !latest_run_succeeded(run)?
    {
        return Ok(None);
    }
    let report = read_slo_general_report(run)?;
    if report.get("available").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    Ok(Some(json!({
        "status": "ready",
        "schema_version": 1,
        "report_href": "subjects/slo-general/report",
        "payload_href": "subjects/slo-general/payload",
    })))
}

pub(super) fn read_slo_general_report(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, SLO_GENERAL_REPORT)
}

pub(super) fn read_slo_general_payload(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, SLO_GENERAL_PAYLOAD)
}

fn latest_run_succeeded(run: &DiscoveredRun) -> Result<bool> {
    let timing = read_run_json(&run.path, "reports/analyzer_timing.json")?;
    Ok(timing
        .get("subjects")
        .and_then(Value::as_array)
        .is_some_and(|subjects| {
            subjects.iter().any(|entry| {
                entry.get("name").and_then(Value::as_str) == Some("slo-general")
                    && entry.get("status").and_then(Value::as_str) == Some("ok")
            })
        }))
}
