//! Cross-run sweep collection over already-computed run artifacts.
//!
//! The launcher owns membership and coordinate order in `sweep_manifest.json`.
//! This module deliberately does not discover sibling directories or rescan
//! parquet: it projects stable scalars from each member's existing analyzer
//! reports into one experiment-level report/payload pair.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::io::{payload_path, report_path, write_json, SCHEMA_VERSION};

const MANIFEST_NAME: &str = "sweep_manifest.json";
const REPORT_NAME: &str = "sweep_summary_report.json";
const PAYLOAD_NAME: &str = "sweep_metrics_grid.json";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SweepManifest {
    schema_version: u32,
    axes: Vec<String>,
    runs: Vec<SweepMember>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SweepMember {
    path: PathBuf,
    coordinates: BTreeMap<String, Value>,
    #[serde(default)]
    labels: BTreeMap<String, String>,
}

pub fn run(experiment_dir: &Path) -> Result<()> {
    let manifest_path = experiment_dir.join(MANIFEST_NAME);
    let manifest: SweepManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("read sweep manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse sweep manifest {}", manifest_path.display()))?;
    validate_manifest(experiment_dir, &manifest)?;

    let domains = axis_domains(&manifest);
    let rows = manifest
        .runs
        .iter()
        .map(|member| collect_member(experiment_dir, member))
        .collect::<Result<Vec<_>>>()?;
    let metrics = metric_descriptors();
    let definitions = definitions();

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "available": !rows.is_empty(),
        "meta": {
            "experiment_dir": experiment_dir.display().to_string(),
            "num_axes": manifest.axes.len(),
            "num_runs": rows.len(),
        },
        "axes": manifest.axes,
        "domains": domains,
        "runs": rows,
        "definitions": definitions,
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "experiment_dir": experiment_dir.display().to_string(),
            "num_axes": manifest.axes.len(),
            "num_runs": rows.len(),
        },
        "axes": manifest.axes,
        "domains": domains,
        "metrics": metrics,
        "runs": rows,
        "definitions": definitions,
    });
    write_json(&report_path(experiment_dir, REPORT_NAME), &report)?;
    write_json(&payload_path(experiment_dir, PAYLOAD_NAME), &payload)?;
    Ok(())
}

fn validate_manifest(experiment_dir: &Path, manifest: &SweepManifest) -> Result<()> {
    if manifest.schema_version != SCHEMA_VERSION {
        bail!(
            "unsupported sweep manifest schema_version {}; expected {}",
            manifest.schema_version,
            SCHEMA_VERSION
        );
    }
    if manifest.axes.is_empty() {
        bail!("sweep manifest axes must not be empty");
    }
    if manifest.runs.is_empty() {
        bail!("sweep manifest runs must not be empty");
    }

    let mut axis_names = HashSet::new();
    for axis in &manifest.axes {
        if axis.is_empty() || !axis_names.insert(axis) {
            bail!("sweep manifest axes must be non-empty and unique");
        }
    }

    let canonical_experiment = experiment_dir
        .canonicalize()
        .with_context(|| format!("canonicalize experiment {}", experiment_dir.display()))?;
    let mut member_paths = HashSet::new();
    let mut coordinate_rows = HashSet::new();
    for member in &manifest.runs {
        validate_relative_member_path(&member.path)?;
        if !member_paths.insert(member.path.clone()) {
            bail!("duplicate sweep member path {}", member.path.display());
        }
        let coordinate_names = member.coordinates.keys().collect::<HashSet<_>>();
        let expected_names = manifest.axes.iter().collect::<HashSet<_>>();
        if coordinate_names != expected_names {
            bail!(
                "sweep member {} coordinates must match axes exactly",
                member.path.display()
            );
        }
        for (axis, value) in &member.coordinates {
            if !is_coordinate_value(value) {
                bail!(
                    "sweep member {} coordinate {axis} must be a scalar or scalar list",
                    member.path.display()
                );
            }
        }
        let coordinate_key = manifest
            .axes
            .iter()
            .map(|axis| serde_json::to_string(&member.coordinates[axis]))
            .collect::<Result<Vec<_>, _>>()?
            .join("\0");
        if !coordinate_rows.insert(coordinate_key) {
            bail!(
                "duplicate sweep coordinate row at member {}",
                member.path.display()
            );
        }

        let candidate = experiment_dir.join(&member.path);
        if candidate.exists() {
            let canonical_candidate = candidate
                .canonicalize()
                .with_context(|| format!("canonicalize sweep member {}", candidate.display()))?;
            if !canonical_candidate.starts_with(&canonical_experiment) {
                bail!(
                    "sweep member {} resolves outside experiment directory",
                    member.path.display()
                );
            }
        }
    }
    Ok(())
}

fn validate_relative_member_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!(
            "sweep member path must be a non-empty normalized relative path: {}",
            path.display()
        );
    }
    Ok(())
}

fn is_coordinate_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
    ) || value.as_array().is_some_and(|items| {
        items.iter().all(|item| {
            matches!(
                item,
                Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
            )
        })
    })
}

fn axis_domains(manifest: &SweepManifest) -> Map<String, Value> {
    manifest
        .axes
        .iter()
        .map(|axis| {
            let mut seen = HashSet::new();
            let values = manifest
                .runs
                .iter()
                .filter_map(|member| member.coordinates.get(axis))
                .filter(|value| seen.insert(serde_json::to_string(value).unwrap_or_default()))
                .cloned()
                .collect();
            (axis.clone(), Value::Array(values))
        })
        .collect()
}

fn collect_member(experiment_dir: &Path, member: &SweepMember) -> Result<Value> {
    let run_dir = experiment_dir.join(&member.path);
    let (lifecycle, metrics) = collect_run_scalars(&run_dir, true)?;
    Ok(json!({
        "path": member.path,
        "coordinates": member.coordinates,
        "labels": member.labels,
        "lifecycle": lifecycle,
        "metrics": metrics,
    }))
}

pub(crate) fn collect_singleton_row(run_dir: &Path, run_id: &str) -> Result<Value> {
    let (lifecycle, metrics) = collect_run_scalars(run_dir, false)?;
    Ok(json!({
        "run_id": run_id,
        "coordinates": {},
        "labels": {},
        "lifecycle": lifecycle,
        "metrics": metrics,
    }))
}

fn collect_run_scalars(
    run_dir: &Path,
    require_current_subject_timing: bool,
) -> Result<(Value, Map<String, Value>)> {
    let lifecycle = lifecycle(&run_dir);
    let summary = read_optional_json(&run_dir.join("summary.json"))?;
    let timing = read_optional_json(&run_dir.join("reports/analyzer_timing.json"))?;
    let slo = (!require_current_subject_timing || subject_ok(timing.as_ref(), "slo-general"))
        .then(|| read_optional_json(&run_dir.join("reports/slo_general_report.json")))
        .transpose()?
        .flatten()
        .filter(report_available);
    let throughput = (!require_current_subject_timing || subject_ok(timing.as_ref(), "throughput"))
        .then(|| read_optional_json(&run_dir.join("reports/throughput_report.json")))
        .transpose()?
        .flatten()
        .filter(report_available);
    let utilization = (!require_current_subject_timing
        || subject_ok(timing.as_ref(), "utilization"))
    .then(|| read_optional_json(&run_dir.join("reports/utilization_report.json")))
    .transpose()?
    .flatten()
    .filter(report_available);

    let mut metrics = Map::new();
    for percentile in ["mean", "p50", "p90", "p99"] {
        insert_number(
            &mut metrics,
            &format!("tpot_{percentile}_ms"),
            slo.as_ref(),
            &["metrics", "tpot", percentile],
        );
        insert_number(
            &mut metrics,
            &format!("ttft_{percentile}_ms"),
            slo.as_ref(),
            &["metrics", "ttft", percentile],
        );
    }
    for name in [
        "prefill_tps",
        "decode_tps",
        "total_tps",
        "total_tps_per_gpu",
    ] {
        insert_number(&mut metrics, name, throughput.as_ref(), &["totals", name]);
    }
    insert_number(
        &mut metrics,
        "gpu_utilization",
        utilization.as_ref(),
        &["totals", "overall_avg"],
    );
    insert_number(
        &mut metrics,
        "requests_finished",
        summary.as_ref(),
        &["requests_finished"],
    );
    insert_number(
        &mut metrics,
        "completed_req_s",
        summary.as_ref(),
        &["completed_req_s"],
    );

    Ok((lifecycle, metrics))
}

fn read_optional_json(path: &Path) -> Result<Option<Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("parse {}", path.display()))
}

fn report_available(report: &Value) -> bool {
    report.get("available").and_then(Value::as_bool) == Some(true)
}

fn subject_ok(timing: Option<&Value>, subject_name: &str) -> bool {
    timing
        .and_then(|value| value.get("subjects"))
        .and_then(Value::as_array)
        .is_some_and(|subjects| {
            subjects.iter().any(|subject| {
                subject.get("name").and_then(Value::as_str) == Some(subject_name)
                    && subject.get("status").and_then(Value::as_str) == Some("ok")
            })
        })
}

fn lifecycle(run_dir: &Path) -> Value {
    let simulation = if run_dir.join(".failed").is_file() {
        "failed"
    } else if run_dir.join(".complete").is_file() {
        "complete"
    } else if run_dir.join("raw/params.json").is_file() {
        "pending"
    } else {
        "not_started"
    };
    let analysis = if run_dir.join("reports/analyzer_timing.json").is_file() {
        "complete"
    } else {
        "not_started"
    };
    json!({"simulation": simulation, "analysis": analysis})
}

fn insert_number(
    metrics: &mut Map<String, Value>,
    name: &str,
    artifact: Option<&Value>,
    path: &[&str],
) {
    let value = artifact
        .and_then(|artifact| path.iter().try_fold(artifact, |node, key| node.get(*key)))
        .filter(|value| value.is_number())
        .cloned()
        .unwrap_or(Value::Null);
    metrics.insert(name.to_owned(), value);
}

pub(crate) fn metric_descriptors() -> Value {
    json!([
        {"key": "tpot_mean_ms", "label": "Mean TPOT", "unit": "ms/token", "group": "tpot", "objective": "minimize"},
        {"key": "tpot_p99_ms", "label": "P99 TPOT", "unit": "ms/token", "group": "tpot", "objective": "minimize"},
        {"key": "ttft_mean_ms", "label": "Mean TTFT", "unit": "ms", "group": "ttft", "objective": "minimize"},
        {"key": "ttft_p99_ms", "label": "P99 TTFT", "unit": "ms", "group": "ttft", "objective": "minimize"},
        {"key": "total_tps", "label": "Total throughput", "unit": "tok/s", "group": "throughput", "objective": "maximize"},
        {"key": "decode_tps", "label": "Decode throughput", "unit": "tok/s", "group": "throughput", "objective": "maximize"},
        {"key": "total_tps_per_gpu", "label": "Throughput / GPU", "unit": "tok/s", "group": "throughput_per_gpu", "objective": "maximize"},
        {"key": "gpu_utilization", "label": "GPU utilization", "unit": "%", "group": "utilization", "objective": "maximize"},
        {"key": "completed_req_s", "label": "Completed requests / s", "unit": "req/s", "group": "completion", "objective": "maximize"},
        {"key": "requests_finished", "label": "Finished requests", "unit": "requests", "group": "finished", "objective": "maximize"}
    ])
}

pub(crate) fn metric_objective(key: &str) -> Option<&'static str> {
    match key {
        "tpot_mean_ms" | "tpot_p99_ms" | "ttft_mean_ms" | "ttft_p99_ms" => Some("minimize"),
        "total_tps" | "decode_tps" | "total_tps_per_gpu" | "gpu_utilization"
        | "completed_req_s" | "requests_finished" => Some("maximize"),
        _ => None,
    }
}

pub(crate) fn definitions() -> Value {
    json!({
        "membership": "exactly the runs listed in sweep_manifest.json; adjacent directories are not scanned",
        "axis_order": "launcher DSL declaration order; the first two axes form plot x/y and later axes form facets",
        "latency": "copied from each ready run's slo_general_report.json",
        "throughput": "copied from each ready run's throughput_report.json totals",
        "utilization": "copied from each ready run's utilization_report.json totals.overall_avg",
        "completion": "copied from each run's simulator summary.json",
        "missing": "missing or unsuccessful source artifacts produce null metrics and remain visible through lifecycle"
    })
}

#[cfg(test)]
mod tests {
    use super::{run, validate_relative_member_path};
    use serde_json::json;
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    fn write_json(path: &Path, value: serde_json::Value) {
        fs::create_dir_all(path.parent().expect("parent")).expect("create parent");
        fs::write(path, serde_json::to_vec_pretty(&value).expect("serialize")).expect("write");
    }

    #[test]
    fn rejects_member_path_traversal() {
        assert!(validate_relative_member_path(Path::new("../other")).is_err());
        assert!(validate_relative_member_path(Path::new("/absolute")).is_err());
    }

    #[test]
    fn collects_ready_and_pending_members_without_scanning_neighbors() {
        let temp = tempdir().expect("tempdir");
        let root = temp.path();
        write_json(
            &root.join("sweep_manifest.json"),
            json!({
                "schema_version": 1,
                "axes": ["tp", "rate"],
                "runs": [
                    {"path": "tp2/r40", "coordinates": {"tp": 2, "rate": 40}, "labels": {}},
                    {"path": "tp4/r40", "coordinates": {"tp": 4, "rate": 40}, "labels": {}}
                ]
            }),
        );
        let ready = root.join("tp2/r40");
        write_json(&ready.join("raw/params.json"), json!({}));
        fs::write(ready.join(".complete"), "").expect("complete");
        write_json(
            &ready.join("summary.json"),
            json!({"requests_finished": 12, "completed_req_s": 3.0}),
        );
        write_json(
            &ready.join("reports/analyzer_timing.json"),
            json!({"subjects": [
                {"name": "slo-general", "status": "ok"},
                {"name": "throughput", "status": "ok"},
                {"name": "utilization", "status": "ok"}
            ]}),
        );
        write_json(
            &ready.join("reports/slo_general_report.json"),
            json!({"available": true, "metrics": {
                "tpot": {"mean": 2.0, "p50": 1.8, "p90": 2.5, "p99": 3.0},
                "ttft": {"mean": 10.0, "p50": 9.0, "p90": 12.0, "p99": 14.0}
            }}),
        );
        write_json(
            &ready.join("reports/throughput_report.json"),
            json!({"available": true, "totals": {
                "prefill_tps": 10.0, "decode_tps": 90.0, "total_tps": 100.0,
                "total_tps_per_gpu": 50.0
            }}),
        );
        write_json(
            &ready.join("reports/utilization_report.json"),
            json!({"available": true, "totals": {"overall_avg": 0.75}}),
        );
        write_json(&root.join("tp4/r40/raw/params.json"), json!({}));
        write_json(
            &root.join("neighbor/summary.json"),
            json!({"requests_finished": 999}),
        );

        run(root).expect("aggregate sweep");
        let report: serde_json::Value = serde_json::from_slice(
            &fs::read(root.join("reports/sweep_summary_report.json")).expect("report"),
        )
        .expect("parse report");
        assert_eq!(report["runs"].as_array().expect("runs").len(), 2);
        assert_eq!(report["runs"][0]["metrics"]["total_tps"], 100.0);
        assert!(report["runs"][1]["metrics"]["total_tps"].is_null());
        assert_eq!(report["runs"][1]["lifecycle"]["simulation"], "pending");
    }
}
