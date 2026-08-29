//! Protocol-v1 discovery and safe browser projection for launcher sweeps.
//!
//! The manifest remains the only membership authority. This module discovers
//! experiment envelopes, never neighboring runs, and replaces member paths
//! with opaque run ids before data crosses the HTTP boundary.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::artifact_kind::{read_artifact_kind, ArtifactKind};
use super::discovery::{
    discover_runs, ignored_directory_name, regular_file, timestamp, ConfiguredRoot, DiscoveredRun,
};
use super::{ArtifactNotFound, SweepNotFound, PROTOCOL_VERSION};
use crate::sweep::{collect_singleton_row, definitions, metric_descriptors, metric_objective};

const MANIFEST_NAME: &str = "sweep_manifest.json";
const EXPERIMENT_METADATA_NAME: &str = "experiment.meta.json";
const PAYLOAD_RELATIVE_PATH: &str = "payloads/sweep_metrics_grid.json";

#[derive(Debug, Deserialize)]
struct SweepManifest {
    schema_version: u32,
    axes: Vec<String>,
    runs: Vec<SweepManifestMember>,
}

#[derive(Debug, Deserialize)]
struct SweepManifestMember {
    path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ExperimentMetadata {
    schema_version: u32,
    experiment_id: String,
}

#[derive(Clone, Debug)]
pub(super) struct DiscoveredSweep {
    workspace_id: String,
    sweep_id: String,
    kind: AggregateKind,
    display_name: String,
    source: AggregateSource,
    axes: Vec<String>,
    num_runs: usize,
    status: SweepStatus,
    experiment_date: Option<String>,
    deployments: Vec<String>,
    traces: Vec<String>,
    updated_time: SystemTime,
}

#[derive(Clone, Debug)]
enum AggregateSource {
    Manifest(PathBuf),
    Singleton(DiscoveredRun),
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum AggregateKind {
    Sweep,
    Singleton,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SweepStatus {
    Ready,
    Pending,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SweepCatalogFilter {
    pub status: Option<SweepStatus>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub(super) struct SweepCatalog {
    protocol_version: u32,
    generated_at: String,
    sweeps: Vec<SweepCatalogEntry>,
}

#[derive(Debug, Serialize)]
struct SweepCatalogEntry {
    workspace_id: String,
    sweep_id: String,
    kind: AggregateKind,
    display_name: String,
    payload_href: String,
    axes: Vec<String>,
    num_runs: usize,
    status: SweepStatus,
    experiment_date: Option<String>,
    deployments: Vec<String>,
    traces: Vec<String>,
    updated_at: String,
    /// Present for singletons so a catalog consumer can open the run analyzer
    /// directly instead of hopping through the one-run aggregate page.
    #[serde(skip_serializing_if = "Option::is_none")]
    run_id: Option<String>,
}

#[cfg(test)]
pub(super) fn build_sweep_catalog(roots: &[ConfiguredRoot]) -> Result<SweepCatalog> {
    build_filtered_sweep_catalog(roots, SweepCatalogFilter::default())
}

pub(super) fn build_filtered_sweep_catalog(
    roots: &[ConfiguredRoot],
    filter: SweepCatalogFilter,
) -> Result<SweepCatalog> {
    let sweeps = canonical_sweeps(roots)?;
    let sweeps = sweeps
        .into_iter()
        .filter(|sweep| filter.status.is_none_or(|status| sweep.status == status))
        .take(filter.limit.unwrap_or(usize::MAX))
        .map(|sweep| SweepCatalogEntry {
            payload_href: format!("sweeps/{}/payload", sweep.sweep_id),
            workspace_id: sweep.workspace_id,
            sweep_id: sweep.sweep_id,
            kind: sweep.kind,
            display_name: sweep.display_name,
            axes: sweep.axes,
            num_runs: sweep.num_runs,
            status: sweep.status,
            experiment_date: sweep.experiment_date,
            deployments: sweep.deployments,
            traces: sweep.traces,
            updated_at: timestamp(sweep.updated_time),
            run_id: match &sweep.source {
                AggregateSource::Singleton(run) => Some(run.run_id.clone()),
                AggregateSource::Manifest(_) => None,
            },
        })
        .collect();
    Ok(SweepCatalog {
        protocol_version: PROTOCOL_VERSION,
        generated_at: timestamp(SystemTime::now()),
        sweeps,
    })
}

/** Resolve launcher metadata aliases once, before catalog filtering or payload
 * lookup. A copied experiment directory may intentionally retain its opaque
 * experiment id; exposing both copies would violate the route identity
 * contract. The normal catalog priority chooses the canonical copy, and both
 * discovery and payload lookup consume that same ordered set. */
fn canonical_sweeps(roots: &[ConfiguredRoot]) -> Result<Vec<DiscoveredSweep>> {
    let mut sweeps = discover_sweeps(roots)?;
    sweeps.sort_by(|left, right| {
        right
            .experiment_date
            .cmp(&left.experiment_date)
            .then_with(|| right.updated_time.cmp(&left.updated_time))
            .then_with(|| left.sweep_id.cmp(&right.sweep_id))
            .then_with(|| left.display_name.cmp(&right.display_name))
            .then_with(|| left.workspace_id.cmp(&right.workspace_id))
            .then_with(|| aggregate_source_path(left).cmp(aggregate_source_path(right)))
    });
    let mut seen_sweep_ids = HashSet::new();
    sweeps.retain(|sweep| seen_sweep_ids.insert(sweep.sweep_id.clone()));
    Ok(sweeps)
}

fn aggregate_source_path(sweep: &DiscoveredSweep) -> &Path {
    match &sweep.source {
        AggregateSource::Manifest(path) => path,
        AggregateSource::Singleton(run) => &run.path,
    }
}

pub(super) fn resolve_sweep(roots: &[ConfiguredRoot], sweep_id: &str) -> Result<DiscoveredSweep> {
    canonical_sweeps(roots)?
        .into_iter()
        .find(|sweep| sweep.sweep_id == sweep_id)
        .ok_or_else(|| SweepNotFound.into())
}

pub(super) fn read_sweep_payload(
    roots: &[ConfiguredRoot],
    sweep: &DiscoveredSweep,
) -> Result<Value> {
    match &sweep.source {
        AggregateSource::Manifest(path) => read_manifest_payload(roots, sweep, path),
        AggregateSource::Singleton(run) => read_singleton_payload(sweep, run),
    }
}

fn read_singleton_payload(sweep: &DiscoveredSweep, run: &DiscoveredRun) -> Result<Value> {
    let mut singleton_definitions = definitions();
    singleton_definitions["membership"] = Value::String(
        "one discovered run not claimed by any sweep manifest; siblings are never merged"
            .to_owned(),
    );
    singleton_definitions["axis_order"] =
        Value::String("singleton aggregates have no sweep axes".to_owned());
    Ok(serde_json::json!({
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": 1,
        "workspace_id": sweep.workspace_id,
        "sweep_id": sweep.sweep_id,
        "display_name": sweep.display_name,
        "meta": {"num_axes": 0, "num_runs": 1},
        "axes": [],
        "domains": {},
        "metrics": metric_descriptors(),
        "runs": [collect_singleton_row(&run.path, &run.run_id)?],
        "definitions": singleton_definitions,
    }))
}

fn read_manifest_payload(
    roots: &[ConfiguredRoot],
    sweep: &DiscoveredSweep,
    experiment_path: &Path,
) -> Result<Value> {
    let payload_path = experiment_path.join(PAYLOAD_RELATIVE_PATH);
    if !regular_file(&payload_path) {
        return Err(ArtifactNotFound.into());
    }
    let mut payload: Value = serde_json::from_slice(
        &fs::read(&payload_path)
            .with_context(|| format!("read sweep payload {}", payload_path.display()))?,
    )
    .with_context(|| format!("parse sweep payload {}", payload_path.display()))?;
    let payload_object = payload
        .as_object_mut()
        .context("sweep payload root must be an object")?;
    payload_object.insert("protocol_version".to_owned(), Value::from(PROTOCOL_VERSION));
    payload_object.insert(
        "workspace_id".to_owned(),
        Value::String(sweep.workspace_id.clone()),
    );
    payload_object.insert("sweep_id".to_owned(), Value::String(sweep.sweep_id.clone()));
    payload_object.insert(
        "display_name".to_owned(),
        Value::String(sweep.display_name.clone()),
    );
    if let Some(meta) = payload_object
        .get_mut("meta")
        .and_then(Value::as_object_mut)
    {
        meta.remove("experiment_dir");
    }
    let metrics = payload_object
        .get_mut("metrics")
        .and_then(Value::as_array_mut)
        .context("sweep payload metrics must be an array")?;
    for metric in metrics {
        let metric_object = metric
            .as_object_mut()
            .context("sweep payload metric must be an object")?;
        if metric_object.contains_key("objective") {
            continue;
        }
        let Some(objective) = metric_object
            .get("key")
            .and_then(Value::as_str)
            .and_then(metric_objective)
        else {
            continue;
        };
        metric_object.insert("objective".to_owned(), Value::String(objective.to_owned()));
    }

    let discovered_run_ids = discover_runs(roots)?
        .into_iter()
        .map(|run| (run.path, run.run_id))
        .collect::<HashMap<_, _>>();
    let canonical_experiment = experiment_path
        .canonicalize()
        .with_context(|| format!("canonicalize sweep {}", experiment_path.display()))?;
    let rows = payload_object
        .get_mut("runs")
        .and_then(Value::as_array_mut)
        .context("sweep payload runs must be an array")?;
    for row in rows {
        let row_object = row
            .as_object_mut()
            .context("sweep payload run row must be an object")?;
        let relative_path = row_object
            .remove("path")
            .and_then(|path| path.as_str().map(PathBuf::from))
            .context("sweep payload run row must contain a string path")?;
        validate_relative_member_path(&relative_path)?;
        let candidate = experiment_path.join(relative_path);
        let run_id = if candidate.exists() {
            let canonical_candidate = candidate
                .canonicalize()
                .with_context(|| format!("canonicalize sweep member {}", candidate.display()))?;
            if !canonical_candidate.starts_with(&canonical_experiment) {
                bail!("sweep member resolves outside its experiment envelope");
            }
            discovered_run_ids
                .get(&canonical_candidate)
                .cloned()
                .map(Value::String)
                .unwrap_or(Value::Null)
        } else {
            Value::Null
        };
        row_object.insert("run_id".to_owned(), run_id);
    }
    Ok(payload)
}

fn discover_sweeps(roots: &[ConfiguredRoot]) -> Result<Vec<DiscoveredSweep>> {
    let mut sweeps = Vec::new();
    let mut claimed_run_paths = HashSet::new();
    for root in roots {
        let root_path = configured_root_path(root);
        let mut pending = vec![root_path.to_path_buf()];
        while let Some(directory) = pending.pop() {
            let artifact_kind = match read_artifact_kind(&directory) {
                Ok(artifact_kind) => artifact_kind,
                Err(error) => {
                    eprintln!(
                        "warning: ignore invalid artifact marker under {}: {error:#}",
                        directory.display()
                    );
                    continue;
                }
            };
            let manifest_path = directory.join(MANIFEST_NAME);
            if artifact_kind == Some(ArtifactKind::SimulationSweep) && regular_file(&manifest_path)
            {
                let manifest: SweepManifest =
                    serde_json::from_slice(&fs::read(&manifest_path).with_context(|| {
                        format!("read sweep manifest {}", manifest_path.display())
                    })?)
                    .with_context(|| format!("parse sweep manifest {}", manifest_path.display()))?;
                validate_manifest_shape(&manifest)?;
                collect_claimed_run_paths(&directory, &manifest, &mut claimed_run_paths)?;
                let metadata = aggregate_metadata(
                    manifest
                        .runs
                        .iter()
                        .map(|member| directory.join(&member.path)),
                );
                let relative = directory
                    .strip_prefix(root_path)
                    .expect("discovery only queues paths below its configured root");
                let display_name = display_name(root_path, relative);
                let payload_path = directory.join(PAYLOAD_RELATIVE_PATH);
                sweeps.push(DiscoveredSweep {
                    workspace_id: root.workspace_id().to_owned(),
                    sweep_id: experiment_id(&directory)?
                        .unwrap_or_else(|| opaque_sweep_id(root.workspace_id(), relative)),
                    kind: AggregateKind::Sweep,
                    experiment_date: experiment_date(&display_name),
                    display_name,
                    source: AggregateSource::Manifest(directory.clone()),
                    axes: manifest.axes,
                    num_runs: manifest.runs.len(),
                    status: if regular_file(&payload_path) {
                        SweepStatus::Ready
                    } else {
                        SweepStatus::Pending
                    },
                    deployments: metadata.deployments,
                    traces: metadata.traces,
                    updated_time: latest_modified(&[&manifest_path, &payload_path]),
                });
            }

            if artifact_kind.is_some_and(|kind| !kind.can_contain_resources()) {
                continue;
            }
            let mut children = fs::read_dir(&directory)
                .with_context(|| format!("read logs directory {}", directory.display()))?
                .filter_map(|entry| entry.ok())
                .collect::<Vec<_>>();
            children.sort_by_key(|entry| entry.file_name());
            for entry in children.into_iter().rev() {
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if file_type.is_symlink()
                    || !file_type.is_dir()
                    || ignored_directory_name(&entry.file_name())
                {
                    continue;
                }
                pending.push(entry.path());
            }
        }
    }
    for run in discover_runs(roots)? {
        if claimed_run_paths.contains(&run.path) {
            continue;
        }
        let metadata = aggregate_metadata([run.path.clone()]);
        sweeps.push(DiscoveredSweep {
            workspace_id: run.workspace_id.clone(),
            sweep_id: experiment_id(&run.path)?.unwrap_or_else(|| opaque_singleton_id(&run.run_id)),
            kind: AggregateKind::Singleton,
            experiment_date: experiment_date(&run.display_name),
            display_name: run.display_name.clone(),
            source: AggregateSource::Singleton(run.clone()),
            axes: Vec::new(),
            num_runs: 1,
            status: SweepStatus::Ready,
            deployments: metadata.deployments,
            traces: metadata.traces,
            updated_time: run.updated_time,
        });
    }
    Ok(sweeps)
}

#[derive(Default)]
struct AggregateMetadata {
    deployments: Vec<String>,
    traces: Vec<String>,
}

fn aggregate_metadata(run_paths: impl IntoIterator<Item = PathBuf>) -> AggregateMetadata {
    let mut deployments = BTreeSet::new();
    let mut traces = BTreeSet::new();
    for run_path in run_paths {
        let Ok(bytes) = fs::read(run_path.join("raw/params.json")) else {
            continue;
        };
        let Ok(params) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        if let Some(deployment) = params.get("deployment").and_then(Value::as_str) {
            if !deployment.trim().is_empty() {
                deployments.insert(deployment.to_owned());
            }
        }
        let Some(trace_files) = params
            .get("workload")
            .and_then(|workload| workload.get("trace_files"))
            .and_then(Value::as_array)
        else {
            continue;
        };
        for trace_file in trace_files.iter().filter_map(Value::as_str) {
            let basename = Path::new(trace_file)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(trace_file);
            if !basename.trim().is_empty() {
                traces.insert(basename.to_owned());
            }
        }
    }
    AggregateMetadata {
        deployments: deployments.into_iter().collect(),
        traces: traces.into_iter().collect(),
    }
}

fn experiment_date(display_name: &str) -> Option<String> {
    let top_level_name = display_name.split('/').next()?;
    let compact_date = top_level_name.get(..8)?;
    NaiveDate::parse_from_str(compact_date, "%Y%m%d")
        .ok()
        .map(|date| date.format("%Y-%m-%d").to_string())
}

fn collect_claimed_run_paths(
    experiment_path: &Path,
    manifest: &SweepManifest,
    claimed_run_paths: &mut HashSet<PathBuf>,
) -> Result<()> {
    let canonical_experiment = experiment_path
        .canonicalize()
        .with_context(|| format!("canonicalize sweep {}", experiment_path.display()))?;
    for member in &manifest.runs {
        let candidate = experiment_path.join(&member.path);
        if !candidate.exists() {
            continue;
        }
        let canonical_candidate = candidate
            .canonicalize()
            .with_context(|| format!("canonicalize sweep member {}", candidate.display()))?;
        if !canonical_candidate.starts_with(&canonical_experiment) {
            bail!("sweep member resolves outside its experiment envelope");
        }
        claimed_run_paths.insert(canonical_candidate);
    }
    Ok(())
}

fn configured_root_path(root: &ConfiguredRoot) -> &Path {
    root.path()
}

fn validate_manifest_shape(manifest: &SweepManifest) -> Result<()> {
    if manifest.schema_version != 1 {
        bail!(
            "unsupported sweep manifest schema_version {}",
            manifest.schema_version
        );
    }
    if manifest.axes.is_empty() || manifest.runs.is_empty() {
        bail!("sweep manifest must contain axes and runs");
    }
    let mut axis_names = HashSet::new();
    if manifest
        .axes
        .iter()
        .any(|axis| axis.is_empty() || !axis_names.insert(axis))
    {
        bail!("sweep manifest axes must be non-empty and unique");
    }
    for member in &manifest.runs {
        validate_relative_member_path(&member.path)?;
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
            "sweep member path must be a normalized relative path: {}",
            path.display()
        );
    }
    Ok(())
}

fn latest_modified(paths: &[&Path]) -> SystemTime {
    paths
        .iter()
        .filter_map(|path| fs::metadata(path).ok()?.modified().ok())
        .max()
        .unwrap_or(UNIX_EPOCH)
}

fn opaque_sweep_id(workspace_id: &str, relative: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-analyzer-sweep-id-v2\0");
    hasher.update(workspace_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(relative.as_os_str().as_encoded_bytes());
    format!("s_{:x}", hasher.finalize())
}

fn experiment_id(directory: &Path) -> Result<Option<String>> {
    let metadata_path = directory.join(EXPERIMENT_METADATA_NAME);
    if !regular_file(&metadata_path) {
        return Ok(None);
    }
    let metadata: ExperimentMetadata = serde_json::from_slice(
        &fs::read(&metadata_path)
            .with_context(|| format!("read experiment metadata {}", metadata_path.display()))?,
    )
    .with_context(|| format!("parse experiment metadata {}", metadata_path.display()))?;
    if metadata.schema_version != 1 {
        bail!(
            "unsupported experiment metadata schema_version {} in {}",
            metadata.schema_version,
            metadata_path.display()
        );
    }
    if !valid_experiment_id(&metadata.experiment_id) {
        bail!(
            "invalid experiment_id {:?} in {}",
            metadata.experiment_id,
            metadata_path.display()
        );
    }
    Ok(Some(metadata.experiment_id))
}

fn valid_experiment_id(experiment_id: &str) -> bool {
    let mut characters = experiment_id.chars();
    characters.next() == Some('e')
        && characters.next() == Some('_')
        && characters.next().is_some()
        && characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
        && experiment_id.len() <= 80
}

fn opaque_singleton_id(run_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-analyzer-singleton-id-v1\0");
    hasher.update(run_id.as_bytes());
    format!("s_{:x}", hasher.finalize())
}

fn display_name(root: &Path, relative: &Path) -> String {
    let components = relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if components.is_empty() {
        root.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_owned())
    } else {
        components.join("/")
    }
}
