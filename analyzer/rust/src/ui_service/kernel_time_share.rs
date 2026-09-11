//! UI publication for critical-path kernel-time composition artifacts.
//!
//! The analysis writes one document holding every scope of a run: the overall
//! composition, one entry per pool, and one entry per worker. Serving it whole
//! means a reader looking at one worker also downloads every other worker's
//! segments, and that cost grows with the size of the cluster — the direction in
//! which runs are getting larger.
//!
//! The split follows what a reader actually asks for, not the shape of the file.
//! The cluster read keeps `overall` and `pools` in full, because the panel's
//! entry view *is* a comparison across pools and splitting those would trade one
//! request for many without saving anything: pool count is a property of the
//! topology, not of the cluster's size. What it does not keep is each worker's
//! segments, which is the `workers × positions` term. Workers appear in the
//! cluster read as an index — identity, kernel time, and sampling counts — and
//! each worker's composition has its own address.
//!
//! The worker read carries no `definitions` and no `positions`: both describe
//! the run rather than the worker, so repeating them once per drill-down would
//! rebuild exactly the cost this split removes. They arrive with the cluster
//! read, which is the subject's own address and the only one a descriptor
//! declares — a scope refines it, never replaces it.

use anyhow::Result;
use serde_json::{json, Map, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun, StageStatus};
use super::ArtifactNotFound;

const REPORT: &str = "reports/kernel_time_share_report.json";
const PAYLOAD: &str = "payloads/kernel_time_share_composition.json";

/// Bumped from 1 when `workers` stopped carrying segments and became an index.
const SCHEMA_VERSION: u64 = 2;

/// Everything a worker entry holds apart from its composition. The cluster read
/// keeps these so its sampling totals stay checkable against their parts.
const WORKER_INDEX_FIELDS: [&str; 6] = [
    "pool_tag",
    "worker_id",
    "kernel_time_ms",
    "raw_rows",
    "sampled_rows",
    "sample_stride",
];

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
        "schema_version": SCHEMA_VERSION,
        "views": ["report", "payload"],
    })))
}

pub(super) fn read_kernel_time_share_report(run: &DiscoveredRun) -> Result<Value> {
    read_run_json(&run.path, REPORT)
}

/// The cluster read: the analysis document with each worker's segments withheld.
pub(super) fn read_kernel_time_share_payload(run: &DiscoveredRun) -> Result<Value> {
    let mut payload = read_run_json(&run.path, PAYLOAD)?;
    let Some(document) = payload.as_object_mut() else {
        return Ok(payload);
    };
    document.insert("schema_version".into(), json!(SCHEMA_VERSION));
    if document.get("available").and_then(Value::as_bool) != Some(true) {
        return Ok(payload);
    }
    let index = workers(document)
        .map(worker_index_entry)
        .collect::<Vec<_>>();
    document.insert("workers".into(), Value::Array(index));
    Ok(payload)
}

/// One worker's composition, addressed as
/// `runs/{id}/workers/{pool_tag}/{worker_id}/subjects/kernel-time-share/payload`.
///
/// The sampling fields travel with the worker as well as in the index: a stride
/// is chosen from that worker's own row count, so a reader judging how exact
/// these numbers are must not have to hold the cluster read to find out.
pub(super) fn read_kernel_time_share_worker_payload(
    run: &DiscoveredRun,
    pool_tag: &str,
    worker_id: u16,
) -> Result<Value> {
    let payload = read_run_json(&run.path, PAYLOAD)?;
    let document = payload.as_object().ok_or(ArtifactNotFound)?;
    if document.get("available").and_then(Value::as_bool) != Some(true) {
        return Err(ArtifactNotFound.into());
    }
    let worker = workers(document)
        .find(|worker| {
            worker.get("pool_tag").and_then(Value::as_str) == Some(pool_tag)
                && worker.get("worker_id").and_then(Value::as_u64) == Some(u64::from(worker_id))
        })
        .ok_or(ArtifactNotFound)?;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "scope": {"kind": "worker", "pool_tag": pool_tag, "worker_id": worker_id},
        "kernel_time_ms": field(worker, "kernel_time_ms"),
        "raw_rows": field(worker, "raw_rows"),
        "sampled_rows": field(worker, "sampled_rows"),
        "sample_stride": field(worker, "sample_stride"),
        "segments": field(worker, "segments"),
    }))
}

fn workers(document: &Map<String, Value>) -> impl Iterator<Item = &Value> {
    document
        .get("workers")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
}

/// A worker entry minus its segments. Copying the named fields rather than
/// removing `segments` keeps the index closed: a field the analysis adds later
/// does not silently join the cluster read.
fn worker_index_entry(worker: &Value) -> Value {
    let mut entry = Map::new();
    for name in WORKER_INDEX_FIELDS {
        entry.insert(name.to_owned(), field(worker, name));
    }
    Value::Object(entry)
}

fn field(value: &Value, key: &str) -> Value {
    value.get(key).cloned().unwrap_or(Value::Null)
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
