//! First-class timing-prediction discovery and case resources.
//!
//! A prediction owns cases and exact CostTrees, but no deployment topology.
//! The `predict/0` key present in parquet is resolved only as a private
//! [`CostLogSource`] and never crosses this module's public JSON boundary.

use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::artifact::read_json;
use super::discovery::{ignored_directory_name, regular_file, timestamp, ConfiguredRoot};
use super::worker_detail::{prediction_case_summary, CostLogSource, OperationIndexCache};
use super::{PredictionNotFound, PROTOCOL_VERSION};

const METADATA_FILE: &str = "prediction.meta.json";
const MAX_CASE_PAGE: usize = 128;

#[derive(Clone, Debug, Deserialize)]
pub(super) struct PredictionMetadata {
    schema_version: u32,
    prediction_id: String,
    selector: String,
    arch_type: String,
    gpu: String,
    gpu_count: usize,
    config_file: String,
    cases_file: String,
    case_count: usize,
}

#[derive(Clone, Debug)]
pub(super) struct DiscoveredPrediction {
    pub(super) workspace_id: String,
    pub(super) display_name: String,
    pub(super) path: PathBuf,
    pub(super) updated_time: SystemTime,
    pub(super) metadata: PredictionMetadata,
}

impl DiscoveredPrediction {
    pub(super) fn prediction_id(&self) -> &str {
        &self.metadata.prediction_id
    }

    pub(super) fn gpu_name(&self) -> &str {
        &self.metadata.gpu
    }

    pub(super) fn gpu_count(&self) -> usize {
        self.metadata.gpu_count
    }

    pub(super) fn cost_source(&self) -> Result<CostLogSource> {
        CostLogSource::unique_prediction_source(&self.path)
    }

    fn analysis_ready(&self) -> bool {
        regular_file(&self.path.join("reports/analyzer_timing.json"))
    }

    fn cost_ready(&self) -> bool {
        self.cost_source().is_ok()
    }
}

#[derive(Debug, Serialize)]
pub(super) struct PredictionCatalog {
    protocol_version: u32,
    generated_at: String,
    predictions: Vec<PredictionCatalogEntry>,
}

#[derive(Debug, Serialize)]
struct PredictionCatalogEntry {
    workspace_id: String,
    prediction_id: String,
    kind: &'static str,
    display_name: String,
    selector: String,
    arch_type: String,
    gpu: String,
    case_count: usize,
    status: &'static str,
    descriptor_href: String,
    updated_at: String,
}

pub(super) fn build_prediction_catalog(roots: &[ConfiguredRoot]) -> Result<PredictionCatalog> {
    let mut predictions = discover_predictions(roots)?;
    predictions.sort_by(|left, right| {
        right
            .updated_time
            .cmp(&left.updated_time)
            .then_with(|| left.prediction_id().cmp(right.prediction_id()))
    });
    let predictions = predictions
        .into_iter()
        .map(|prediction| {
            let status = if prediction.cost_ready() {
                "ready"
            } else {
                "pending"
            };
            PredictionCatalogEntry {
                descriptor_href: format!("predictions/{}/descriptor", prediction.prediction_id()),
                workspace_id: prediction.workspace_id,
                prediction_id: prediction.metadata.prediction_id,
                kind: "timing_predict",
                display_name: prediction.display_name,
                selector: prediction.metadata.selector,
                arch_type: prediction.metadata.arch_type,
                gpu: prediction.metadata.gpu,
                case_count: prediction.metadata.case_count,
                status,
                updated_at: timestamp(prediction.updated_time),
            }
        })
        .collect();
    Ok(PredictionCatalog {
        protocol_version: PROTOCOL_VERSION,
        generated_at: timestamp(SystemTime::now()),
        predictions,
    })
}

pub(super) fn discover_predictions(roots: &[ConfiguredRoot]) -> Result<Vec<DiscoveredPrediction>> {
    let mut predictions = Vec::new();
    let mut prediction_ids = HashSet::new();
    for root in roots {
        let mut pending = vec![root.path().to_path_buf()];
        while let Some(directory) = pending.pop() {
            if regular_file(&directory.join(METADATA_FILE)) {
                // A malformed metadata file (e.g. a stray pytest fixture) must
                // never abort the whole predictions catalog. Skip it with a warning
                // so one bad directory cannot hide every other prediction in every root.
                let parse = (|| -> Result<PredictionMetadata> {
                    let raw = read_json(&directory.join(METADATA_FILE))?;
                    let metadata: PredictionMetadata = serde_json::from_value(raw)?;
                    validate_metadata(&metadata)?;
                    Ok(metadata)
                })();
                let metadata = match parse {
                    Ok(metadata) => metadata,
                    Err(error) => {
                        eprintln!(
                            "[analyze] skipping malformed {METADATA_FILE} under {}: {error:#}",
                            directory.display()
                        );
                        continue;
                    }
                };
                if !prediction_ids.insert(metadata.prediction_id.clone()) {
                    eprintln!(
                        "[analyze] skipping duplicate prediction id {} under {}",
                        metadata.prediction_id,
                        directory.display()
                    );
                    continue;
                }
                let relative = directory
                    .strip_prefix(root.path())
                    .expect("prediction discovery stays below its configured root");
                predictions.push(DiscoveredPrediction {
                    workspace_id: root.workspace_id().to_owned(),
                    display_name: display_name(root.path(), relative),
                    updated_time: prediction_updated_at(&directory),
                    path: directory,
                    metadata,
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
    Ok(predictions)
}

/// Recognize the prediction that lives in `directory`, if one does.
///
/// This is [`discover_predictions`]'s test applied to a directory another
/// resource already names, rather than to one a walk arrived at. An alignment
/// bundle records the timing prediction it was paired against by path — that is
/// what its analysis was pointed at — and the client needs the opaque id the
/// prediction routes are addressed by. Walking every root again to translate one
/// known path would make a bundle's descriptor pay for the whole tree.
///
/// `None` means the directory is not a prediction this service would serve, and
/// the caller must then offer nothing: the walk that populates the catalog would
/// not have reached it either, so its id would resolve nowhere.
pub(super) fn prediction_at(
    roots: &[ConfiguredRoot],
    directory: &Path,
) -> Option<DiscoveredPrediction> {
    // The recorded path was written by an analysis, while roots are canonical.
    // Compare them in the same form or a root reached through a symlink never
    // matches the absolute path its own artifacts name.
    let directory = directory.canonicalize().ok()?;
    if !regular_file(&directory.join(METADATA_FILE)) {
        return None;
    }
    let (root, relative) = roots.iter().find_map(|root| {
        let relative = directory.strip_prefix(root.path()).ok()?;
        Some((root, relative))
    })?;
    // Discovery descends into neither dotted directories nor `old-logs`, so a
    // prediction under one is not servable however directly it is named.
    if !relative.components().all(|component| match component {
        Component::Normal(part) => !ignored_directory_name(part),
        Component::CurDir => true,
        _ => false,
    }) {
        return None;
    }
    let metadata: PredictionMetadata =
        serde_json::from_value(read_json(&directory.join(METADATA_FILE)).ok()?).ok()?;
    validate_metadata(&metadata).ok()?;
    Some(DiscoveredPrediction {
        workspace_id: root.workspace_id().to_owned(),
        display_name: display_name(root.path(), relative),
        updated_time: prediction_updated_at(&directory),
        path: directory,
        metadata,
    })
}

pub(super) fn resolve_prediction(
    roots: &[ConfiguredRoot],
    prediction_id: &str,
) -> Result<DiscoveredPrediction> {
    discover_predictions(roots)?
        .into_iter()
        .find(|prediction| prediction.prediction_id() == prediction_id)
        .ok_or_else(|| PredictionNotFound.into())
}

pub(super) fn prediction_descriptor(prediction: &DiscoveredPrediction) -> Value {
    json!({
        "schema_version": 1,
        "prediction_id": prediction.prediction_id(),
        "kind": "timing_predict",
        "display_name": prediction.display_name,
        "selector": prediction.metadata.selector,
        "arch": {"type": prediction.metadata.arch_type},
        "gpu": {
            "name": prediction.metadata.gpu,
            "count": prediction.metadata.gpu_count,
        },
        "case_count": prediction.metadata.case_count,
        "lifecycle": {
            "prediction": if prediction.cost_ready() { "complete" } else { "pending" },
            "analysis": if prediction.analysis_ready() { "complete" } else { "not_started" },
        },
        "resources": {
            "cases_href": format!("predictions/{}/cases", prediction.prediction_id()),
            "kernel_input_distribution_href": prediction.analysis_ready().then(|| format!(
                "predictions/{}/subjects/kernel-input-distribution/payload",
                prediction.prediction_id()
            )),
        },
    })
}

pub(super) async fn prediction_cases(
    prediction: &DiscoveredPrediction,
    cache: &OperationIndexCache,
    offset: usize,
    limit: usize,
    request_id: u64,
) -> Result<Value> {
    let cases = read_prediction_cases(prediction)?;
    let source = prediction.cost_source()?;
    let bounded_limit = limit.clamp(1, MAX_CASE_PAGE);
    let end = offset.saturating_add(bounded_limit).min(cases.len());
    let mut entries = Vec::with_capacity(end.saturating_sub(offset));
    for case_index in offset.min(end)..end {
        let case_id = u64::try_from(case_index).context("prediction case index exceeds u64")?;
        let mut summary = prediction_case_summary(cache, &source, case_id, request_id).await?;
        summary
            .as_object_mut()
            .context("prediction case summary must be an object")?
            .insert("input".to_owned(), cases[case_index].clone());
        entries.push(summary);
    }
    Ok(json!({
        "schema_version": 1,
        "prediction_id": prediction.prediction_id(),
        "range": {
            "offset": offset,
            "limit": bounded_limit,
            "returned": entries.len(),
            "total": cases.len(),
        },
        "cases": entries,
    }))
}

fn read_prediction_cases(prediction: &DiscoveredPrediction) -> Result<Vec<Value>> {
    let path = prediction.path.join(&prediction.metadata.cases_file);
    let cases = read_json(&path)?;
    let cases = cases
        .as_array()
        .cloned()
        .context("prediction cases snapshot must contain an array")?;
    if cases.len() != prediction.metadata.case_count {
        bail!(
            "prediction metadata declares {} cases but snapshot contains {}",
            prediction.metadata.case_count,
            cases.len()
        );
    }
    Ok(cases)
}

fn validate_metadata(metadata: &PredictionMetadata) -> Result<()> {
    if metadata.schema_version != 1 {
        bail!(
            "unsupported prediction metadata schema_version {}",
            metadata.schema_version
        );
    }
    if !valid_prediction_id(&metadata.prediction_id) {
        bail!("invalid prediction id: {:?}", metadata.prediction_id);
    }
    if !matches!(metadata.selector.as_str(), "iter" | "attn" | "ffn") {
        bail!("invalid prediction selector: {:?}", metadata.selector);
    }
    if metadata.arch_type.is_empty() || metadata.gpu.is_empty() || metadata.gpu_count == 0 {
        bail!("prediction metadata has incomplete architecture/GPU provenance");
    }
    validate_snapshot_name(&metadata.config_file)?;
    validate_snapshot_name(&metadata.cases_file)?;
    Ok(())
}

fn valid_prediction_id(prediction_id: &str) -> bool {
    prediction_id.strip_prefix("p_").is_some_and(|suffix| {
        !suffix.is_empty()
            && suffix.len() <= 64
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    })
}

fn validate_snapshot_name(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    if name.is_empty()
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
    {
        bail!("prediction snapshot must be a basename: {name:?}");
    }
    Ok(())
}


/// Prediction-only normalization: the per-kernel ladder's rungs are anchored by
/// the worker replica count (meta gpu_count, usually 1 -> per-rank GPU·s), but the
/// labeler's per-kernel necessary work is whole-model-on-one-GPU. Scale every kernel
/// necessary_work by `gpu_count / arch_gpus` so per-kernel necessary == hardware
/// limit (TP4/EP4 -> 52.2 == 52.2) instead of reporting a bogus 4x under-account.
/// Runs, whose rung anchor already matches their labeler scope, are untouched.
/// Effective TP/EP parallelism of the per-rank CostTree a prediction models, from
/// raw/params (e.g. attn_tp=4/ep=4 -> 4). Distinct from the meta replica count.
pub(super) fn prediction_worker_gpus(path: &Path) -> usize {
    let params = serde_json::from_str(
        &std::fs::read_to_string(path.join("raw/params.json")).unwrap_or_default(),
    )
    .unwrap_or(serde_json::Value::Null);
    let mut gpus = 1usize;
    if let Some(arch) = params
        .get("pools")
        .and_then(|p| p.as_object())
        .and_then(|pools| pools.values().next())
        .and_then(|pool| pool.get("groups"))
        .and_then(|groups| groups.as_array())
        .and_then(|groups| groups.first())
        .and_then(|group| group.get("arch"))
    {
        let attn = arch.get("attn_tp_size").and_then(|v| v.as_u64()).unwrap_or(1).max(1);
        let ep = arch.get("ep_size").and_then(|v| v.as_u64()).unwrap_or(1).max(1);
        gpus = (attn.max(ep) as usize).max(1);
    }
    gpus
}


pub(super) fn normalize_prediction_necessary(value: &mut Value, gpu_count: usize, arch_gpus: usize) {
    let factor = if arch_gpus > 0 && gpu_count != arch_gpus {
        gpu_count as f64 / arch_gpus as f64
    } else {
        1.0
    };
    if factor == 1.0 {
        return;
    }
    if let Some(kernels) = value.get_mut("kernels").and_then(Value::as_array_mut) {
        for kernel in kernels {
            let Some(work) = kernel.get_mut("necessary_work").and_then(Value::as_object_mut) else {
                continue;
            };
            for key in ["min_flops", "min_bytes", "redundant_gpu_s", "compute_gpu_s", "memory_gpu_s", "necessary_gpu_s", "wall_s"] {
                if key.starts_with("min_") {
                    continue;
                }
                if let Some(n) = work.get(key).and_then(Value::as_f64) {
                    work.insert(key.to_string(), json!(n * factor));
                }
            }
        }
    }
    // Iteration/level waterfall: the labeler's floor is WHOLE-model; scale it to the
    // per-rank rung footing so R6<R5 surfaces the real excess instead of clamping to
    // R5 (which reads as zero waste while R7 renders at the full iteration).
    if let Some(level) = value.get_mut("level").and_then(Value::as_object_mut) {
        if let Some(rungs) = level.get_mut("rungs").and_then(Value::as_object_mut) {
            for key in ["segmented_necessary", "hardware_necessary"] {
                if let Some(n) = rungs.get(key).and_then(Value::as_f64) {
                    rungs.insert(key.to_string(), json!((n * factor).max(0.0)));
                }
            }
        }
        let r5 = level.get("rungs").and_then(|r| r.get("hardware_limit")).and_then(Value::as_f64).unwrap_or(0.0);
        let seg = level.get("rungs").and_then(|r| r.get("segmented_necessary")).and_then(Value::as_f64).unwrap_or(0.0);
        let fus = level.get("rungs").and_then(|r| r.get("hardware_necessary")).and_then(Value::as_f64).unwrap_or(0.0);
        if let Some(buckets) = level.get_mut("buckets").and_then(Value::as_object_mut) {
            if let Some(k) = buckets.get_mut("excess_over_necessary").and_then(Value::as_object_mut) {
                k.insert("gpu_s".to_string(), json!((r5 - seg).max(0.0)));
            }
            if let Some(k) = buckets.get_mut("fusion").and_then(Value::as_object_mut) {
                k.insert("gpu_s".to_string(), json!((seg - fus).max(0.0)));
            }
            if let Some(k) = buckets.get_mut("hardware_necessary").and_then(Value::as_object_mut) {
                k.insert("gpu_s".to_string(), json!(fus.max(0.0)));
            }
        }
    }
}

fn prediction_updated_at(path: &Path) -> SystemTime {
    [
        METADATA_FILE,
        "raw/cost_log",
        "reports/analyzer_timing.json",
    ]
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
