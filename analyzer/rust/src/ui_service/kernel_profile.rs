//! First-class ``kernel_profile`` resource discovery, descriptor, and curve.
//!
//! ``artifact.meta.json`` is the discovery source of truth. A marked legacy snapshot
//! without modern metadata but with the old ``job.meta.json`` + ``curve.json`` pair remains
//! readable as
//! ``kp_legacy_<hash>``. The curve endpoint enriches the immutable curve JSON with
//! GPU hardware ceiling lines (per-row, dtype-aware) instead of rewriting the
//! artifact, and never infers a missing GPU from the current host.

use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::artifact::read_json;
use super::artifact_kind::{read_artifact_kind, ArtifactKind};
use super::discovery::{ignored_directory_name, regular_file, timestamp, ConfiguredRoot};
use super::hardware::{resolve_gpu, ResolvedGpu};
use super::{KernelProfileNotFound, PROTOCOL_VERSION};

const METADATA_FILE: &str = "kernel-profile.meta.json";
const LEGACY_JOB_METADATA_FILE: &str = "job.meta.json";
const CURVE_FILE: &str = "curve.json";

#[derive(Clone, Debug, Deserialize)]
pub(super) struct KernelGroup {
    kind: String,
    table: String,
    backend: String,
    metric_family: String,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct GpuProvenance {
    cache_key: Option<String>,
    observed_name: Option<String>,
    count: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct ProvenanceFields {
    source: String,
    resolved_canonical_name: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct ProfileMetadata {
    schema_version: u32,
    profile_id: String,
    kernel: KernelGroup,
    gpu: GpuProvenance,
    provenance: ProvenanceFields,
    mode: Option<String>,
    // The submitted parameter list mirrors each curve row's ``args``.
    #[allow(dead_code)]
    args: Option<Value>,
    created_at: Option<String>,
    artifacts: Option<Value>,
}

#[derive(Clone, Debug)]
pub(super) struct DiscoveredKernelProfile {
    pub(super) workspace_id: String,
    pub(super) profile_id: String,
    pub(super) display_name: String,
    pub(super) path: PathBuf,
    pub(super) updated_time: SystemTime,
    pub(super) metadata: Option<ProfileMetadata>,
    pub(super) legacy: bool,
}

impl DiscoveredKernelProfile {
    pub(super) fn profile_id(&self) -> &str {
        &self.profile_id
    }

    fn curve_ready(&self) -> bool {
        regular_file(&self.path.join(CURVE_FILE))
    }
}

#[derive(Debug, serde::Serialize)]
pub(super) struct KernelProfileCatalog {
    protocol_version: u32,
    generated_at: String,
    pub(super) kernel_profiles: Vec<KernelProfileCatalogEntry>,
}

#[derive(Debug, serde::Serialize)]
pub(super) struct KernelProfileCatalogEntry {
    pub(super) workspace_id: String,
    pub(super) profile_id: String,
    kind: &'static str,
    pub(super) display_name: String,
    kernel_kind: String,
    table: String,
    backend: String,
    metric_family: String,
    gpu_cache_key: Option<String>,
    gpu_observed_name: Option<String>,
    provenance_source: String,
    legacy: bool,
    status: &'static str,
    descriptor_href: String,
    updated_at: String,
}

#[derive(Clone, Debug, Default)]
pub(super) struct KernelIdentity {
    kind: Option<String>,
    table: Option<String>,
    backend: Option<String>,
    metric_family: Option<String>,
}

/// Kernel identity from a profile's metadata, or reconstructed from a legacy
/// ``job.meta.json`` descriptor when the new metadata is absent.
fn kernel_identity(profile: &DiscoveredKernelProfile) -> KernelIdentity {
    if let Some(metadata) = &profile.metadata {
        return KernelIdentity {
            kind: Some(metadata.kernel.kind.clone()),
            table: Some(metadata.kernel.table.clone()),
            backend: Some(metadata.kernel.backend.clone()),
            metric_family: Some(metadata.kernel.metric_family.clone()),
        };
    }
    let job_metadata = read_json(&profile.path.join(LEGACY_JOB_METADATA_FILE)).ok();
    let descriptor = job_metadata
        .as_ref()
        .and_then(|value| value.get("descriptor"))
        .and_then(Value::as_object)
        .cloned();
    KernelIdentity {
        kind: descriptor
            .as_ref()
            .and_then(|d| d.get("kernelKind"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                descriptor
                    .as_ref()
                    .and_then(|d| d.get("table"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            }),
        table: descriptor
            .as_ref()
            .and_then(|d| d.get("table"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        backend: descriptor
            .as_ref()
            .and_then(|d| d.get("backend"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        metric_family: descriptor
            .as_ref()
            .and_then(|d| d.get("metricFamily"))
            .and_then(Value::as_str)
            .map(str::to_owned),
    }
}

impl KernelIdentity {
    fn is_p2p(&self) -> bool {
        self.kind
            .as_deref()
            .is_some_and(|kind| kind.starts_with("p2p"))
            || self
                .table
                .as_deref()
                .is_some_and(|table| table.starts_with("p2p"))
            || self
                .backend
                .as_deref()
                .is_some_and(|backend| backend == "p2p_inter" || backend == "p2p_intra")
    }
}

pub(super) fn build_kernel_profile_catalog(
    roots: &[ConfiguredRoot],
) -> Result<KernelProfileCatalog> {
    let mut profiles = discover_kernel_profiles(roots)?;
    profiles.sort_by(|left, right| {
        right
            .updated_time
            .cmp(&left.updated_time)
            .then_with(|| left.profile_id().cmp(right.profile_id()))
    });
    let kernel_profiles = profiles.iter().map(catalog_entry).collect();
    Ok(KernelProfileCatalog {
        protocol_version: PROTOCOL_VERSION,
        generated_at: timestamp(SystemTime::now()),
        kernel_profiles,
    })
}

fn catalog_entry(profile: &DiscoveredKernelProfile) -> KernelProfileCatalogEntry {
    let identity = kernel_identity(profile);
    let (kind, table, backend) = (
        identity.kind.clone().unwrap_or_default(),
        identity.table.clone().unwrap_or_default(),
        identity.backend.clone().unwrap_or_default(),
    );
    KernelProfileCatalogEntry {
        workspace_id: profile.workspace_id.clone(),
        profile_id: profile.profile_id.clone(),
        kind: "kernel_profile",
        display_name: profile.display_name.clone(),
        kernel_kind: kind,
        table,
        backend,
        metric_family: identity.metric_family.clone().unwrap_or_default(),
        gpu_cache_key: profile
            .metadata
            .as_ref()
            .and_then(|meta| meta.gpu.cache_key.clone()),
        gpu_observed_name: profile
            .metadata
            .as_ref()
            .and_then(|meta| meta.gpu.observed_name.clone()),
        provenance_source: profile
            .metadata
            .as_ref()
            .map(|meta| meta.provenance.source.clone())
            .unwrap_or_else(|| "unavailable".to_owned()),
        legacy: profile.legacy,
        status: if profile.curve_ready() {
            "ready"
        } else {
            "pending"
        },
        descriptor_href: format!("kernel-profiles/{}/descriptor", profile.profile_id),
        updated_at: timestamp(profile.updated_time),
    }
}

pub(super) fn discover_kernel_profiles(
    roots: &[ConfiguredRoot],
) -> Result<Vec<DiscoveredKernelProfile>> {
    let mut profiles = Vec::new();
    let mut profile_ids = HashSet::new();
    for root in roots {
        let mut pending = vec![root.path().to_path_buf()];
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
            if artifact_kind == Some(ArtifactKind::KernelProfile)
                && regular_file(&directory.join(METADATA_FILE))
            {
                let metadata: ProfileMetadata =
                    serde_json::from_value(read_json(&directory.join(METADATA_FILE))?)
                        .with_context(|| {
                            format!("decode {METADATA_FILE} under {}", directory.display())
                        })?;
                validate_metadata(&metadata)?;
                if !profile_ids.insert(metadata.profile_id.clone()) {
                    bail!("duplicate kernel profile id: {}", metadata.profile_id);
                }
                let relative = directory
                    .strip_prefix(root.path())
                    .expect("kernel profile discovery stays below its configured root");
                profiles.push(DiscoveredKernelProfile {
                    workspace_id: root.workspace_id().to_owned(),
                    profile_id: metadata.profile_id.clone(),
                    display_name: display_name(root.path(), relative),
                    updated_time: profile_updated_at(&directory),
                    path: directory,
                    metadata: Some(metadata),
                    legacy: false,
                });
                continue;
            }
            if artifact_kind == Some(ArtifactKind::KernelProfile)
                && regular_file(&directory.join(LEGACY_JOB_METADATA_FILE))
                && regular_file(&directory.join(CURVE_FILE))
            {
                let relative = directory
                    .strip_prefix(root.path())
                    .expect("kernel profile discovery stays below its configured root");
                let legacy_id = legacy_profile_id(root.workspace_id(), relative);
                if !profile_ids.insert(legacy_id.clone()) {
                    bail!("duplicate legacy kernel profile id: {legacy_id}");
                }
                profiles.push(DiscoveredKernelProfile {
                    workspace_id: root.workspace_id().to_owned(),
                    profile_id: legacy_id,
                    display_name: display_name(root.path(), relative),
                    updated_time: profile_updated_at(&directory),
                    path: directory,
                    metadata: None,
                    legacy: true,
                });
                continue;
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
    Ok(profiles)
}

pub(super) fn resolve_kernel_profile(
    roots: &[ConfiguredRoot],
    profile_id: &str,
) -> Result<DiscoveredKernelProfile> {
    discover_kernel_profiles(roots)?
        .into_iter()
        .find(|profile| profile.profile_id == profile_id)
        .ok_or_else(|| KernelProfileNotFound.into())
}

pub(super) fn profile_descriptor(profile: &DiscoveredKernelProfile) -> Value {
    let identity = kernel_identity(profile);
    let kind = identity.kind.clone().or_else(|| identity.table.clone());
    let resources = json!({
        "curve_href": format!("kernel-profiles/{}/curve", profile.profile_id()),
    });
    let metadata = profile.metadata.as_ref();
    let created_at = metadata.and_then(|meta| meta.created_at.clone());
    let mode = metadata.and_then(|meta| meta.mode.clone());
    let artifacts = metadata
        .and_then(|meta| meta.artifacts.clone())
        .unwrap_or(null_json());
    json!({
        "schema_version": 1,
        "workspace_id": profile.workspace_id,
        "profile_id": profile.profile_id,
        "kind": "kernel_profile",
        "legacy": profile.legacy,
        "display_name": profile.display_name,
        "created_at": created_at,
        "mode": mode,
        "artifact_files": artifacts,
        "kernel": {
            "kind": kind,
            "table": identity.table,
            "backend": identity.backend,
            "metric_family": identity.metric_family,
        },
        "gpu": metadata.map(|meta| json!({
            "cache_key": meta.gpu.cache_key,
            "observed_name": meta.gpu.observed_name,
            "count": meta.gpu.count,
        })),
        // Legacy snapshots carry no reliable GPU identity: data is shown but hardware
        // ceilings are never synthesized, and the current host is never inferred.
        "gpu_provenance": metadata.map(|meta| json!({
            "source": meta.provenance.source,
            "resolved_canonical_name": meta.provenance.resolved_canonical_name,
        })).unwrap_or_else(|| json!({
            "source": "unavailable",
            "resolved_canonical_name": Value::Null,
        })),
        "lifecycle": {
            "profile": if profile.curve_ready() { "complete" } else { "pending" },
        },
        "resources": resources,
    })
}

/// Serve the immutable curve with hardware ceiling enrichment. The artifact file is
/// left untouched; rows/series gain per-row limits resolved from the catalog.
pub(super) fn profile_curve(profile: &DiscoveredKernelProfile, repo_root: &Path) -> Result<Value> {
    let curve = read_json(&profile.path.join(CURVE_FILE))
        .with_context(|| format!("read curve for {}", profile.profile_id))?;
    if curve.get("schemaVersion").and_then(Value::as_u64) != Some(1) {
        bail!("curve schemaVersion must be 1");
    }
    let family = curve
        .get("metricFamily")
        .and_then(Value::as_str)
        .unwrap_or("compute")
        .to_owned();
    let identity = kernel_identity(profile);
    // A legacy identity may be absent; the curve itself declares its kernel kind.
    let is_p2p = identity.is_p2p() || curve_kind_is_p2p(&curve);

    // A measured profile resolves the physical GPU first (observed == cache key is a
    // job-time invariant); a cached-only profile resolves its cache key. Never infer a
    // GPU from the current host. Legacy profiles have no reliable GPU. The resolved
    // entry is owned here so the borrowed context outlives the enrichment call.
    let resolved_gpu = profile
        .metadata
        .as_ref()
        .and_then(|metadata| {
            metadata
                .gpu
                .observed_name
                .clone()
                .or_else(|| metadata.gpu.cache_key.clone())
        })
        .and_then(|name| resolve_gpu(repo_root, &name).ok().flatten());
    let context = match (&profile.legacy, resolved_gpu.as_ref()) {
        (true, _) => GpuContext::Unmatched("gpu_provenance_unavailable"),
        (false, Some(gpu)) => GpuContext::Matched(gpu),
        (false, None) => GpuContext::Unmatched("gpu_unmatched"),
    };
    Ok(enrich_curve(&curve, &family, is_p2p, context))
}

fn curve_kind_is_p2p(curve: &Value) -> bool {
    curve
        .get("kernelKind")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind.starts_with("p2p"))
}

#[derive(Clone, Copy, Debug)]
enum GpuContext<'a> {
    Matched(&'a ResolvedGpu),
    Unmatched(&'static str),
}

fn enrich_curve(curve: &Value, family: &str, is_p2p: bool, context: GpuContext<'_>) -> Value {
    let gpu = match context {
        GpuContext::Matched(gpu) => Some(gpu),
        GpuContext::Unmatched(_) => None,
    };
    let rows = curve.get("rows").and_then(Value::as_array);
    let enriched_rows = rows.map(|rows| {
        rows.iter()
            .map(|row| enrich_row(row, gpu, family, is_p2p, context))
            .collect::<Vec<_>>()
    });
    let mut output = Value::Null;
    if let Some(object) = curve.as_object() {
        let mut copy = object.clone();
        copy.remove("rows");
        if let Some(enriched_rows) = enriched_rows {
            copy.insert("rows".to_owned(), Value::Array(enriched_rows));
        }
        copy.insert(
            "hardware".to_owned(),
            hardware_block(gpu, family, is_p2p, context),
        );
        output = Value::Object(copy);
    }
    output
}

fn enrich_row(
    row: &Value,
    gpu: Option<&ResolvedGpu>,
    family: &str,
    is_p2p: bool,
    context: GpuContext<'_>,
) -> Value {
    let mut hardware = json!({});
    let dtype = row
        .get("args")
        .and_then(|args| args.get("dtype"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if family == "compute" {
        hardware["tflops_limit"] = row_metric(
            gpu.and_then(|gpu| row_dtype_tflops(row, gpu)),
            gpu.is_some(),
        );
        hardware["memory_bandwidth_gbps_limit"] =
            row_metric(gpu.and_then(|gpu| gpu.mem_bandwidth_gbps), gpu.is_some());
    } else if is_p2p {
        let one_way = gpu.and_then(|gpu| gpu.interconnect_bandwidth_gbps.map(|v| v / 2.0));
        hardware["algbw_gbps_limit"] = row_metric(one_way, gpu.is_some());
        hardware["busbw_gbps_limit"] = row_metric(one_way, gpu.is_some());
    } else {
        hardware["algbw_gbps_limit"] = json!({
            "limit": Value::Null,
            "available": false,
            "reason": "no_generic_theoretical_line_for_collective_algbw",
        });
        hardware["busbw_gbps_limit"] = row_metric(
            gpu.and_then(|gpu| gpu.interconnect_bandwidth_gbps),
            gpu.is_some(),
        );
    }
    hardware["dtype"] = json!(dtype);
    hardware["reason"] = match context {
        GpuContext::Matched(_) => json!(Value::Null),
        GpuContext::Unmatched(reason) => json!(reason),
    };
    let mut enriched = row.clone();
    if let Some(object) = enriched.as_object_mut() {
        object.insert("hardware".to_owned(), hardware);
    }
    enriched
}

fn row_dtype_tflops(row: &Value, gpu: &ResolvedGpu) -> Option<f64> {
    let dtype = row
        .get("args")
        .and_then(|args| args.get("dtype"))
        .and_then(Value::as_str)
        .unwrap_or("");
    gpu.dense_tflops(dtype)
}

fn row_metric(limit: Option<f64>, gpu_matched: bool) -> Value {
    match limit {
        Some(limit) => json!({"limit": limit, "available": true, "basis": "catalog_peak"}),
        None => json!({
            "limit": Value::Null,
            "available": false,
            "reason": if gpu_matched { "no_peak_for_dtype_or_device" } else { "gpu_unmatched" },
        }),
    }
}

fn hardware_block(
    gpu: Option<&ResolvedGpu>,
    family: &str,
    is_p2p: bool,
    context: GpuContext<'_>,
) -> Value {
    let reason = match context {
        GpuContext::Matched(_) => Value::Null,
        GpuContext::Unmatched(reason) => json!(reason),
    };
    let one_way = gpu.and_then(|gpu| gpu.interconnect_bandwidth_gbps.map(|v| v / 2.0));
    let two_way = gpu.and_then(|gpu| gpu.interconnect_bandwidth_gbps);
    let metric_limit = |limit: Option<f64>, basis: &str| match limit {
        Some(value) => json!({"available": true, "limit": value, "basis": basis}),
        None if gpu.is_some() => json!({"available": false, "reason": "no_peak_for_device"}),
        None => json!({"available": false, "reason": "gpu_unmatched"}),
    };
    let unavailable = |reason: &str| json!({"available": false, "reason": reason});
    let metrics = if family == "comm" {
        if is_p2p {
            json!({
                "algbw_gbps": metric_limit(one_way, "catalog_interconnect_one_way_peak"),
                "busbw_gbps": metric_limit(one_way, "catalog_interconnect_one_way_peak"),
                "time_ms": unavailable("no_theoretical_line"),
                "energy_j": unavailable("no_theoretical_line"),
            })
        } else {
            json!({
                "algbw_gbps": unavailable("no_generic_theoretical_line_for_collective_algbw"),
                "busbw_gbps": metric_limit(two_way, "catalog_interconnect_bidirectional_peak"),
                "time_ms": unavailable("no_theoretical_line"),
                "energy_j": unavailable("no_theoretical_line"),
            })
        }
    } else {
        let hbm = gpu.and_then(|gpu| gpu.mem_bandwidth_gbps);
        json!({
            "tflops": json!({
                "available": gpu.is_some(),
                "basis": "catalog_dense_peak",
                "per_row": true,
                "reason": if gpu.is_some() { Value::Null } else { json!(reason_str(context)) },
            }),
            "memory_bandwidth_gbps": metric_limit(hbm, "catalog_hbm_peak"),
            "time_ms": unavailable("no_theoretical_line"),
            "energy_j": unavailable("no_theoretical_line"),
        })
    };
    json!({
        "schema_version": 1,
        "matched": gpu.is_some(),
        "available": gpu.is_some(),
        "gpu": gpu.map(|gpu| json!({
            "canonical_name": gpu.canonical_name,
            "matched_alias": gpu.matched_alias,
            "provenance": "catalog",
        })),
        "reason": reason,
        "metrics": metrics,
    })
}

fn reason_str(context: GpuContext<'_>) -> &'static str {
    match context {
        GpuContext::Matched(_) => "",
        GpuContext::Unmatched(reason) => reason,
    }
}

fn null_json() -> Value {
    Value::Null
}

fn validate_metadata(metadata: &ProfileMetadata) -> Result<()> {
    if metadata.schema_version != 1 {
        bail!(
            "unsupported kernel profile metadata schema_version {}",
            metadata.schema_version
        );
    }
    if !valid_profile_id(&metadata.profile_id) || metadata.profile_id.starts_with("kp_legacy_") {
        bail!("invalid kernel profile id: {:?}", metadata.profile_id);
    }
    if metadata.kernel.kind.is_empty()
        || metadata.kernel.table.is_empty()
        || metadata.kernel.backend.is_empty()
        || !matches!(metadata.kernel.metric_family.as_str(), "compute" | "comm")
    {
        bail!("kernel profile metadata has incomplete kernel provenance");
    }
    if !matches!(
        metadata.provenance.source.as_str(),
        "measurement" | "cache_key"
    ) {
        bail!("kernel profile metadata has invalid provenance source");
    }
    if let Some(artifacts) = &metadata.artifacts {
        let Some(object) = artifacts.as_object() else {
            bail!("kernel profile artifact declaration must be a JSON object");
        };
        if object.values().any(|value| !value_filename(value)) {
            bail!("kernel profile artifact declaration values must be basenames");
        }
    }
    Ok(())
}

fn value_filename(value: &Value) -> bool {
    value.as_str().is_some_and(is_basename)
}

fn is_basename(name: &str) -> bool {
    !name.is_empty() && !name.starts_with('.') && Path::new(name).components().count() == 1
}

fn valid_profile_id(profile_id: &str) -> bool {
    profile_id.strip_prefix("kp_").is_some_and(|suffix| {
        !suffix.is_empty()
            && suffix.len() <= 64
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    })
}

fn legacy_profile_id(workspace_id: &str, relative: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-analyzer-kernel-profile-legacy-v1\0");
    hasher.update(workspace_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(relative.as_os_str().as_encoded_bytes());
    format!("kp_legacy_{:x}", hasher.finalize())
}

fn profile_updated_at(path: &Path) -> SystemTime {
    [METADATA_FILE, LEGACY_JOB_METADATA_FILE, CURVE_FILE]
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
