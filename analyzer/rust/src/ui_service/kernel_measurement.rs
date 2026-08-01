//! First-class ``kernel_measurement`` resource discovery, descriptor, summary, and
//! plot serving.
//!
//! ``kernel-measurement.meta.json`` (written by ``python -m profiling measure``) is
//! the discovery source of truth and declares the summary file + plot basenames. A
//! legacy measurement dir (summary.json with a complete, verifiable signature, no new
//! metadata) is discovered as ``km_legacy_<hash>``. Plot paths are validated to a
//! single component that must be exactly a declared plot name, so traversal or
//! undeclared files are impossible.

use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::artifact::read_json;
use super::discovery::{ignored_directory_name, regular_file, timestamp, ConfiguredRoot};
use super::{ArtifactNotFound, KernelMeasurementNotFound, PROTOCOL_VERSION};

const METADATA_FILE: &str = "kernel-measurement.meta.json";
const SUMMARY_FILE: &str = "summary.json";
const LEGACY_PLOTS: [&str; 2] = ["runtime_telemetry.png", "runtime_trend.png"];

#[derive(Clone, Debug, Deserialize)]
pub(super) struct MeasurementKernelGroup {
    kind: String,
    table: String,
    backend: String,
    metric_family: String,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct MeasurementGpu {
    cache_key: Option<String>,
    observed_name: Option<String>,
    count: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct MeasurementMetadata {
    schema_version: u32,
    measurement_id: String,
    kernel: MeasurementKernelGroup,
    gpu: MeasurementGpu,
    shape: Option<Value>,
    duration_s: Option<f64>,
    telemetry: Option<bool>,
    created_at: Option<String>,
    summary_file: String,
    plots: Vec<String>,
    #[serde(default)]
    artifacts: Vec<String>,
}

#[derive(Clone, Debug)]
pub(super) struct DiscoveredKernelMeasurement {
    pub(super) workspace_id: String,
    pub(super) measurement_id: String,
    pub(super) display_name: String,
    pub(super) path: PathBuf,
    pub(super) updated_time: SystemTime,
    pub(super) metadata: Option<MeasurementMetadata>,
    pub(super) legacy: bool,
}

impl DiscoveredKernelMeasurement {
    pub(super) fn measurement_id(&self) -> &str {
        &self.measurement_id
    }

    fn summary_file(&self) -> &str {
        self.metadata
            .as_ref()
            .map(|metadata| metadata.summary_file.as_str())
            .unwrap_or(SUMMARY_FILE)
    }

    fn plot_names(&self) -> Vec<String> {
        match &self.metadata {
            Some(metadata) => metadata
                .plots
                .iter()
                .filter(|name| regular_file(&self.path.join(name)))
                .cloned()
                .collect(),
            None => LEGACY_PLOTS
                .iter()
                .map(|name| name.to_string())
                .filter(|name| regular_file(&self.path.join(name)))
                .collect(),
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub(super) struct KernelMeasurementCatalog {
    protocol_version: u32,
    generated_at: String,
    pub(super) kernel_measurements: Vec<KernelMeasurementCatalogEntry>,
}

#[derive(Debug, serde::Serialize)]
pub(super) struct KernelMeasurementCatalogEntry {
    pub(super) workspace_id: String,
    measurement_id: String,
    kind: &'static str,
    pub(super) display_name: String,
    kernel_kind: String,
    table: String,
    backend: String,
    metric_family: String,
    gpu_cache_key: Option<String>,
    gpu_observed_name: Option<String>,
    legacy: bool,
    status: &'static str,
    descriptor_href: String,
    updated_at: String,
}

pub(super) fn build_kernel_measurement_catalog(
    roots: &[ConfiguredRoot],
) -> Result<KernelMeasurementCatalog> {
    let mut measurements = discover_kernel_measurements(roots)?;
    measurements.sort_by(|left, right| {
        right
            .updated_time
            .cmp(&left.updated_time)
            .then_with(|| left.measurement_id().cmp(right.measurement_id()))
    });
    let kernel_measurements = measurements.iter().map(catalog_entry).collect();
    Ok(KernelMeasurementCatalog {
        protocol_version: PROTOCOL_VERSION,
        generated_at: timestamp(SystemTime::now()),
        kernel_measurements,
    })
}

fn catalog_entry(measurement: &DiscoveredKernelMeasurement) -> KernelMeasurementCatalogEntry {
    let (kind, table, backend, family) = match &measurement.metadata {
        Some(metadata) => (
            metadata.kernel.kind.clone(),
            metadata.kernel.table.clone(),
            metadata.kernel.backend.clone(),
            metadata.kernel.metric_family.clone(),
        ),
        None => (String::new(), String::new(), String::new(), String::new()),
    };
    KernelMeasurementCatalogEntry {
        workspace_id: measurement.workspace_id.clone(),
        measurement_id: measurement.measurement_id.clone(),
        kind: "kernel_measurement",
        display_name: measurement.display_name.clone(),
        kernel_kind: kind,
        table,
        backend,
        metric_family: family,
        gpu_cache_key: measurement
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.gpu.cache_key.clone()),
        gpu_observed_name: measurement
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.gpu.observed_name.clone()),
        legacy: measurement.legacy,
        status: if regular_file(&measurement.path.join(measurement.summary_file())) {
            "ready"
        } else {
            "pending"
        },
        descriptor_href: format!(
            "kernel-measurements/{}/descriptor",
            measurement.measurement_id
        ),
        updated_at: timestamp(measurement.updated_time),
    }
}

pub(super) fn discover_kernel_measurements(
    roots: &[ConfiguredRoot],
) -> Result<Vec<DiscoveredKernelMeasurement>> {
    let mut measurements = Vec::new();
    let mut measurement_ids = HashSet::new();
    for root in roots {
        let mut pending = vec![root.path().to_path_buf()];
        while let Some(directory) = pending.pop() {
            if regular_file(&directory.join(METADATA_FILE)) {
                let metadata: MeasurementMetadata =
                    serde_json::from_value(read_json(&directory.join(METADATA_FILE))?)
                        .with_context(|| {
                            format!("decode {METADATA_FILE} under {}", directory.display())
                        })?;
                validate_metadata(&metadata)?;
                if !measurement_ids.insert(metadata.measurement_id.clone()) {
                    bail!(
                        "duplicate kernel measurement id: {}",
                        metadata.measurement_id
                    );
                }
                let relative = directory
                    .strip_prefix(root.path())
                    .expect("kernel measurement discovery stays below its configured root");
                measurements.push(DiscoveredKernelMeasurement {
                    workspace_id: root.workspace_id().to_owned(),
                    measurement_id: metadata.measurement_id.clone(),
                    display_name: display_name(root.path(), relative),
                    updated_time: measurement_updated_at(&directory),
                    path: directory,
                    metadata: Some(metadata),
                    legacy: false,
                });
                continue;
            }
            // Legacy measure dirs are discovered only when the summary signature is
            // complete and verifiable — never guessed from the host or a stray file.
            if summary_signature_complete(&directory.join(SUMMARY_FILE)) {
                let relative = directory
                    .strip_prefix(root.path())
                    .expect("kernel measurement discovery stays below its configured root");
                let legacy_id = legacy_measurement_id(root.workspace_id(), relative);
                if !measurement_ids.insert(legacy_id.clone()) {
                    bail!("duplicate legacy kernel measurement id: {legacy_id}");
                }
                measurements.push(DiscoveredKernelMeasurement {
                    workspace_id: root.workspace_id().to_owned(),
                    measurement_id: legacy_id,
                    display_name: display_name(root.path(), relative),
                    updated_time: measurement_updated_at(&directory),
                    path: directory,
                    metadata: None,
                    legacy: true,
                });
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
    Ok(measurements)
}

pub(super) fn resolve_kernel_measurement(
    roots: &[ConfiguredRoot],
    measurement_id: &str,
) -> Result<DiscoveredKernelMeasurement> {
    discover_kernel_measurements(roots)?
        .into_iter()
        .find(|measurement| measurement.measurement_id == measurement_id)
        .ok_or_else(|| KernelMeasurementNotFound.into())
}

pub(super) fn measurement_descriptor(measurement: &DiscoveredKernelMeasurement) -> Value {
    let summary_href = format!("kernel-measurements/{}/summary", measurement.measurement_id);
    let plots = measurement.plot_names();
    let plot_hrefs = plots
        .iter()
        .map(|name| {
            format!(
                "kernel-measurements/{}/plots/{name}",
                measurement.measurement_id
            )
        })
        .collect::<Vec<_>>();
    let (kernel, gpu, gpu_provenance) = match &measurement.metadata {
        Some(metadata) => (
            json!({
                "kind": metadata.kernel.kind,
                "table": metadata.kernel.table,
                "backend": metadata.kernel.backend,
                "metric_family": metadata.kernel.metric_family,
            }),
            json!({
                "cache_key": metadata.gpu.cache_key,
                "observed_name": metadata.gpu.observed_name,
                "count": metadata.gpu.count,
            }),
            json!({"source": "measurement"}),
        ),
        None => (
            json!({
                "kind": Value::Null,
                "table": Value::Null,
                "backend": Value::Null,
                "metric_family": Value::Null,
            }),
            json!({
                "cache_key": Value::Null,
                "observed_name": Value::Null,
                "count": Value::Null,
            }),
            // Legacy measurements lack reliable GPU provenance.
            json!({"source": "unavailable"}),
        ),
    };
    let shape = measurement
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.shape.clone());
    let duration_s = measurement
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.duration_s);
    let telemetry = measurement
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.telemetry);
    let created_at = measurement
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.created_at.clone());
    let artifacts = measurement
        .metadata
        .as_ref()
        .and_then(|metadata| {
            if metadata.artifacts.is_empty() {
                None
            } else {
                Some(json!(metadata.artifacts))
            }
        })
        .unwrap_or(Value::Null);
    json!({
        "schema_version": 1,
        "workspace_id": measurement.workspace_id,
        "measurement_id": measurement.measurement_id,
        "kind": "kernel_measurement",
        "legacy": measurement.legacy,
        "display_name": measurement.display_name,
        "created_at": created_at,
        "artifact_files": artifacts,
        "kernel": kernel,
        "gpu": gpu,
        "gpu_provenance": gpu_provenance,
        "shape": shape,
        "duration_s": duration_s,
        "telemetry": telemetry,
        "lifecycle": {
            "measurement": if regular_file(
                &measurement.path.join(measurement.summary_file())
            ) {
                "complete"
            } else {
                "pending"
            }
        },
        "resources": {
            "summary_href": summary_href,
            "plots": plot_hrefs,
        },
    })
}

pub(super) fn measurement_summary(measurement: &DiscoveredKernelMeasurement) -> Result<Value> {
    let summary = read_json(&measurement.path.join(measurement.summary_file()))
        .with_context(|| format!("read summary for {}", measurement.measurement_id))?;
    if summary.get("schema_version").and_then(Value::as_u64) != Some(1) {
        bail!(
            "measurement summary for {} must have schema_version 1",
            measurement.measurement_id
        );
    }
    Ok(summary)
}

pub(super) fn measurement_plot(
    measurement: &DiscoveredKernelMeasurement,
    plot_name: &str,
) -> Result<Vec<u8>> {
    validate_plot_name(plot_name)?;
    if !measurement
        .plot_names()
        .iter()
        .any(|name| name == plot_name)
    {
        return Err(ArtifactNotFound.into());
    }
    let path = measurement.path.join(plot_name);
    if !regular_file(&path) {
        return Err(ArtifactNotFound.into());
    }
    fs::read(&path)
        .with_context(|| format!("read plot {} for {}", plot_name, measurement.measurement_id))
}

fn validate_plot_name(plot_name: &str) -> Result<()> {
    let components = Path::new(plot_name).components().collect::<Vec<_>>();
    let is_basename = !plot_name.is_empty()
        && !plot_name.starts_with('.')
        && components.len() == 1
        && matches!(components[0], Component::Normal(_));
    if !is_basename {
        bail!("plot name must be a single basename component: {plot_name:?}");
    }
    Ok(())
}

fn summary_signature_complete(summary_path: &Path) -> bool {
    let Ok(summary) = read_json(summary_path) else {
        return false;
    };
    let Ok(schema) = summary
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or(())
    else {
        return false;
    };
    if schema != 1 {
        return false;
    }
    // A complete trend capture always carries the runtime statistics block.
    summary
        .get("runtime_ms")
        .and_then(Value::as_object)
        .and_then(|stats| stats.get("median"))
        .and_then(Value::as_f64)
        .is_some()
}

fn validate_metadata(metadata: &MeasurementMetadata) -> Result<()> {
    if metadata.schema_version != 1 {
        bail!(
            "unsupported kernel measurement metadata schema_version {}",
            metadata.schema_version
        );
    }
    if !valid_measurement_id(&metadata.measurement_id)
        || metadata.measurement_id.starts_with("km_legacy_")
    {
        bail!(
            "invalid kernel measurement id: {:?}",
            metadata.measurement_id
        );
    }
    if metadata.kernel.kind.is_empty()
        || metadata.kernel.table.is_empty()
        || metadata.kernel.backend.is_empty()
        || !matches!(metadata.kernel.metric_family.as_str(), "compute" | "comm")
    {
        bail!("kernel measurement metadata has incomplete kernel provenance");
    }
    if metadata.summary_file.is_empty() || !is_basename(&metadata.summary_file) {
        bail!(
            "invalid measurement summary_file: {:?}",
            metadata.summary_file
        );
    }
    if metadata.plots.iter().any(|name| !is_basename(name)) {
        bail!("measurement plots must be basenames");
    }
    if metadata.artifacts.iter().any(|name| !is_basename(name)) {
        bail!("measurement artifacts must be basenames");
    }
    Ok(())
}

fn is_basename(name: &str) -> bool {
    !name.is_empty() && !name.starts_with('.') && Path::new(name).components().count() == 1
}

fn valid_measurement_id(measurement_id: &str) -> bool {
    measurement_id.strip_prefix("km_").is_some_and(|suffix| {
        !suffix.is_empty()
            && suffix.len() <= 64
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    })
}

fn legacy_measurement_id(workspace_id: &str, relative: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-analyzer-kernel-measurement-legacy-v1\0");
    hasher.update(workspace_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(relative.as_os_str().as_encoded_bytes());
    format!("km_legacy_{:x}", hasher.finalize())
}

fn measurement_updated_at(path: &Path) -> SystemTime {
    [METADATA_FILE, SUMMARY_FILE, "runtimes.csv"]
        .into_iter()
        .filter_map(|relative| fs::metadata(path.join(relative)).ok()?.modified().ok())
        .max()
        .unwrap_or(UNIX_EPOCH)
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
