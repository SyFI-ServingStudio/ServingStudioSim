//! Offline optimality for one CostTree subtree.
//!
//! This stays separate from the run-level ladder. A selected subtree has a
//! measured wall fold and a hardware roofline, but it does not have the
//! run-wide scheduler, load-balance, communication, or batching context needed
//! to publish those global rungs. Semantic R6/R7 is enabled only after the
//! exact location map and independent model.work label both reconcile.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use datafusion::prelude::SessionContext;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::io::{
    read_cost_manifests, read_run_meta, read_worker_gpu_counts, report_path, write_json,
    SCHEMA_VERSION,
};
use crate::kernel_query::repo_root;
use crate::session::{col, collect, require_columns, value_f32_list, value_f64, value_string};
use crate::trace::manifest::{
    fold_mean, node_time, FlatCostNode, Manifest, ManifestDoc, ManifestSection,
};

use super::floors::{self, SemanticWork};
use super::spec::{load_gpu_spec, GpuSpec};

type WorkerKey = (String, u16);
type OccurrenceKey = (String, u16, String);

#[derive(Clone, Debug)]
struct ParsedPath {
    section: Option<String>,
    ordinals: Vec<usize>,
}

#[derive(Clone, Debug)]
struct ParsedLabel {
    section: Option<String>,
    label: String,
}

#[derive(Clone, Debug)]
struct Selector {
    raw_path: Option<String>,
    raw_label: Option<String>,
    path: Option<ParsedPath>,
    label: Option<ParsedLabel>,
}

impl Selector {
    fn parse(path: Option<&str>, label: Option<&str>) -> Result<Self> {
        if path.is_none() && label.is_none() {
            bail!("one CostTree selector is required: pass --path and/or --label");
        }
        Ok(Self {
            raw_path: path.map(str::to_owned),
            raw_label: label.map(str::to_owned),
            path: path.map(parse_path).transpose()?,
            label: label.map(parse_label).transpose()?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NodeSignature {
    section: String,
    path: Vec<usize>,
    kind: String,
    label: Option<String>,
    leaves: Vec<String>,
}

#[derive(Clone, Debug)]
struct NodeMatch {
    section: String,
    node_idx: usize,
    ancestor_scale: f64,
    signature: NodeSignature,
}

#[derive(Clone, Debug)]
struct Occurrence {
    worker: WorkerKey,
    section: String,
    node_idx: usize,
    ancestor_scale: f64,
}

#[derive(Debug)]
struct Selection {
    occurrences: Vec<Occurrence>,
    occurrence_index: HashMap<OccurrenceKey, usize>,
    signature: NodeSignature,
}

#[derive(Clone, Debug)]
struct ScopeRow {
    slot_time_ms: Vec<f64>,
    slot_flops: Vec<f64>,
    slot_bytes: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RowRungs {
    r0_gpu_s: f64,
    r5_gpu_s: Option<f64>,
}

#[derive(Clone, Debug)]
struct HardwareRef {
    requested_name: String,
    canonical_name: String,
    spec: GpuSpec,
}

#[derive(Clone, Debug)]
struct HardwareInputs {
    by_worker: HashMap<WorkerKey, Option<HardwareRef>>,
    gpu_counts: HashMap<WorkerKey, f64>,
    defaulted_gpu_count: bool,
    params: Option<ParamsDocument>,
}

#[derive(Clone, Debug, Deserialize)]
struct ParamsDocument {
    #[serde(default)]
    pools: BTreeMap<String, PoolParams>,
}

#[derive(Clone, Debug, Deserialize)]
struct PoolParams {
    #[serde(default)]
    groups: Vec<GroupParams>,
}

#[derive(Clone, Debug, Deserialize)]
struct GroupParams {
    arch: ArchParams,
    #[serde(default)]
    gpu: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct ArchParams {
    #[serde(rename = "type")]
    arch_type: String,
    #[serde(default)]
    fp8: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct PoolSpec {
    arch_type: String,
    dtype: String,
}

#[derive(Clone, Debug, Deserialize)]
struct LocationMap {
    schema_version: u64,
    mapping_id: String,
    arch_types: Vec<String>,
    locations: Vec<LocationRule>,
}

#[derive(Clone, Debug, Deserialize)]
struct LocationRule {
    location: String,
    semantics: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct MeasuredScope {
    r0_gpu_s: f64,
    r5_gpu_s: f64,
    r5_available: bool,
    r5_reason: Option<String>,
    rows: u64,
}

#[derive(Clone, Debug)]
struct SemanticFloors {
    r6_gpu_s: f64,
    r7_gpu_s: f64,
    mapping_ids: BTreeSet<String>,
    semantic_names: BTreeSet<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ScopedReport {
    schema_version: u32,
    report_type: &'static str,
    log_dir: String,
    selection: SelectionReport,
    rungs: BTreeMap<String, RungReport>,
    omitted_rungs: Vec<OmittedRung>,
    provenance: Value,
}

#[derive(Debug, Serialize)]
struct SelectionReport {
    selector: SelectorReport,
    section: String,
    canonical_path: String,
    node_kind: String,
    node_label: Option<String>,
    matched_manifest_workers: usize,
    matched_cost_log_rows: u64,
    descendant_leaves: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SelectorReport {
    path: Option<String>,
    label: Option<String>,
}

#[derive(Debug, Serialize)]
struct RungReport {
    value_gpu_seconds: f64,
    unit: &'static str,
    definition: &'static str,
}

#[derive(Debug, Serialize)]
struct OmittedRung {
    rung: &'static str,
    reason: String,
}

/// Run the standalone analyze optimality-scoped command.
pub(crate) async fn run_scoped(
    ctx: &SessionContext,
    log_dir: &Path,
    path: Option<&str>,
    label: Option<&str>,
) -> Result<()> {
    let (report, slug) = compute_scoped(ctx, log_dir, path, label).await?;
    write_json(
        &report_path(log_dir, &format!("optimality_scoped_{slug}.json")),
        &report,
    )
}

/// Compute the scoped report without writing any file. Shared by the CLI
/// (which persists it under `reports/`) and the analyzer service endpoint
/// (which must stay read-only against the run directory).
pub(crate) async fn compute_scoped(
    ctx: &SessionContext,
    log_dir: &Path,
    path: Option<&str>,
    label: Option<&str>,
) -> Result<(ScopedReport, String)> {
    let selector = Selector::parse(path, label)?;
    let registered = crate::session::register_cost_log(ctx, log_dir).await?;
    if !registered {
        bail!(
            "cost_log/ is missing under {}; cannot analyze a scope",
            log_dir.display()
        );
    }
    require_columns(
        ctx,
        "cost_log",
        &[
            "pool_tag",
            "worker_id",
            "section",
            "slot_time_ms",
            "slot_flops",
            "slot_bytes",
        ],
    )
    .await?;

    let manifests = read_cost_manifests(log_dir)?;
    let selection = resolve_selection(&selector, &manifests)?;
    let hardware = load_hardware_inputs(log_dir, &selection);
    let measured = measure_scope(ctx, &selection, &manifests, &hardware).await?;

    let mut rungs = BTreeMap::new();
    rungs.insert(
        "r0_measured".to_owned(),
        RungReport {
            value_gpu_seconds: measured.r0_gpu_s,
            unit: "GPU-seconds",
            definition: "selected CostTree node wall fold multiplied by its worker GPU count",
        },
    );

    let mut omitted_rungs = undefined_scope_omissions();
    if measured.r5_available {
        rungs.insert(
            "r5_hardware_limit".to_owned(),
            RungReport {
                value_gpu_seconds: measured.r5_gpu_s,
                unit: "GPU-seconds",
                definition: "balanced selected-leaf fold of hardware compute/memory limits",
            },
        );
    } else {
        omitted_rungs.push(OmittedRung {
            rung: "r5_hardware_limit",
            reason: measured
                .r5_reason
                .unwrap_or_else(|| "hardware limit could not be established".to_owned()),
        });
    }

    let semantic_result =
        compute_semantic_floors(ctx, log_dir, &selection, &manifests, &hardware).await;
    let semantic_provenance;
    match semantic_result {
        Ok(floors) => {
            rungs.insert(
                "r6_segmented_necessary".to_owned(),
                RungReport {
                    value_gpu_seconds: floors.r6_gpu_s,
                    unit: "GPU-seconds",
                    definition:
                        "sum of per-semantic model.work hardware rooflines in the selected scope",
                },
            );
            rungs.insert(
                "r7_scope_fused_necessary".to_owned(),
                RungReport {
                    value_gpu_seconds: floors.r7_gpu_s,
                    unit: "GPU-seconds",
                    definition: "one hardware roofline over selected model.work FLOPs and bytes",
                },
            );
            semantic_provenance = json!({
                "status": "reconciled",
                "mapping_ids": floors.mapping_ids,
                "semantic_names": floors.semantic_names,
                "basis": "model.work full-work semantic segments mapped to exact CostTree locations; CostTree Scale is already represented by those segments and is not applied again"
            });
        }
        Err(error) => {
            let reason = format!("semantic scope unavailable: {error:#}");
            omitted_rungs.push(OmittedRung {
                rung: "r6_segmented_necessary",
                reason: reason.clone(),
            });
            omitted_rungs.push(OmittedRung {
                rung: "r7_scope_fused_necessary",
                reason,
            });
            semantic_provenance = json!({"status": "omitted", "reason": error.to_string()});
        }
    }

    let signature = &selection.signature;
    let report = ScopedReport {
        schema_version: SCHEMA_VERSION,
        report_type: "optimality_scoped_v1",
        log_dir: log_dir.display().to_string(),
        selection: SelectionReport {
            selector: SelectorReport {
                path: selector.raw_path.clone(),
                label: selector.raw_label.clone(),
            },
            section: signature.section.clone(),
            canonical_path: canonical_path(signature),
            node_kind: signature.kind.clone(),
            node_label: signature.label.clone(),
            matched_manifest_workers: selection.occurrences.len(),
            matched_cost_log_rows: measured.rows,
            descendant_leaves: signature.leaves.clone(),
        },
        rungs,
        omitted_rungs,
        provenance: json!({
            "cost_log": ["slot_time_ms", "slot_flops", "slot_bytes"],
            "cost_manifest": "raw/cost_manifest/worker_<pool>_<worker>.json",
            "hardware": {
                "source": "gpu/spec.json",
                "gpu_names": hardware_gpu_names(&hardware),
                "gpu_count_source": if hardware.defaulted_gpu_count { "default_one" } else { "run_meta.workers[].gpu_ids" }
            },
            "semantic": semantic_provenance,
            "global_rungs": "omitted because scheduler and cluster context are undefined for a CostTree subtree"
        }),
    };

    let slug = report_slug(&selector, signature);
    Ok((report, slug))
}

fn parse_path(raw: &str) -> Result<ParsedPath> {
    let raw = raw.trim();
    if raw.is_empty() {
        bail!("CostTree path cannot be empty");
    }
    let components: Vec<&str> = raw.split('/').collect();
    if components.iter().any(|component| component.is_empty()) {
        bail!("invalid CostTree path {raw:?}: empty path component");
    }
    let mut section = None;
    let mut start = 0;
    if components[0] != "root" && components[0].parse::<usize>().is_err() {
        section = Some(components[0].to_owned());
        start = 1;
    }
    let mut ordinals = Vec::new();
    for component in &components[start..] {
        if *component == "root" && ordinals.is_empty() {
            continue;
        }
        let ordinal = component.parse::<usize>().with_context(|| {
            format!("invalid CostTree path {raw:?}: {component:?} is not a child ordinal")
        })?;
        ordinals.push(ordinal);
    }
    Ok(ParsedPath { section, ordinals })
}

fn parse_label(raw: &str) -> Result<ParsedLabel> {
    if raw.is_empty() {
        bail!("CostTree label cannot be empty");
    }
    let (section, label) = match raw.split_once("::") {
        Some((section, label)) => {
            if section.is_empty() || label.is_empty() {
                bail!("qualified CostTree label must be section::exact label");
            }
            (Some(section.to_owned()), label.to_owned())
        }
        None => (None, raw.to_owned()),
    };
    Ok(ParsedLabel { section, label })
}

fn resolve_selection(
    selector: &Selector,
    manifests: &BTreeMap<WorkerKey, ManifestDoc>,
) -> Result<Selection> {
    let mut occurrences = Vec::new();
    let mut expected_signature = None;
    for (worker, doc) in manifests {
        let matches = resolve_in_doc(selector, doc).with_context(|| {
            format!(
                "resolve CostTree selector for worker {}/{}",
                worker.0, worker.1
            )
        })?;
        if matches.is_empty() {
            continue;
        }
        if matches.len() != 1 {
            bail!(
                "CostTree selector matched {} nodes for worker {}/{}; use a section-qualified selector",
                matches.len(),
                worker.0,
                worker.1
            );
        }
        let matched = matches.into_iter().next().unwrap();
        if let Some(expected) = &expected_signature {
            if expected != &matched.signature {
                bail!(
                    "CostTree selector is ambiguous across workers: {} does not match the first node signature",
                    canonical_path(&matched.signature)
                );
            }
        } else {
            expected_signature = Some(matched.signature.clone());
        }
        occurrences.push(Occurrence {
            worker: worker.clone(),
            section: matched.section,
            node_idx: matched.node_idx,
            ancestor_scale: matched.ancestor_scale,
        });
    }
    let signature = expected_signature
        .ok_or_else(|| anyhow!("CostTree selector matched no node in any cost manifest"))?;
    let mut occurrence_index = HashMap::new();
    for (index, occurrence) in occurrences.iter().enumerate() {
        let key = (
            occurrence.worker.0.clone(),
            occurrence.worker.1,
            occurrence.section.clone(),
        );
        if occurrence_index.insert(key, index).is_some() {
            bail!("CostTree selector matched duplicate worker/section occurrence");
        }
    }
    Ok(Selection {
        occurrences,
        occurrence_index,
        signature,
    })
}

fn resolve_in_doc(selector: &Selector, doc: &ManifestDoc) -> Result<Vec<NodeMatch>> {
    for section in &doc.sections {
        validate_manifest(&section.manifest)
            .with_context(|| format!("validate CostTree section {:?}", section.section))?;
    }
    let path_matches = selector
        .path
        .as_ref()
        .map(|path| resolve_path(doc, path))
        .transpose()?
        .unwrap_or_default();
    let label_matches = selector
        .label
        .as_ref()
        .map(|label| resolve_label(doc, label))
        .transpose()?
        .unwrap_or_default();

    match (selector.path.is_some(), selector.label.is_some()) {
        (true, true) => {
            if path_matches.len() > 1 || label_matches.len() > 1 {
                bail!("path and label selectors are individually ambiguous in this manifest");
            }
            match (path_matches.first(), label_matches.first()) {
                (Some(path_match), Some(label_match))
                    if path_match.signature == label_match.signature =>
                {
                    Ok(vec![path_match.clone()])
                }
                (Some(_), Some(_)) => {
                    bail!("--path and --label resolve to different CostTree nodes")
                }
                _ => Ok(Vec::new()),
            }
        }
        (true, false) => Ok(path_matches),
        (false, true) => Ok(label_matches),
        (false, false) => unreachable!(),
    }
}

fn resolve_path(doc: &ManifestDoc, path: &ParsedPath) -> Result<Vec<NodeMatch>> {
    let mut matches = Vec::new();
    for section in &doc.sections {
        if path
            .section
            .as_deref()
            .is_some_and(|expected| expected != section.section)
        {
            continue;
        }
        if let Some(matched) = resolve_path_in_section(section, &path.ordinals)? {
            matches.push(matched);
        }
    }
    Ok(matches)
}

fn resolve_path_in_section(
    section: &ManifestSection,
    ordinals: &[usize],
) -> Result<Option<NodeMatch>> {
    let manifest = &section.manifest;
    if manifest.nodes.is_empty() {
        return Ok(None);
    }
    let mut node_idx = 0;
    let mut ancestor_scale = 1.0;
    for ordinal in ordinals {
        let children = node_children(manifest, node_idx)?;
        let Some(child) = children.get(*ordinal) else {
            return Ok(None);
        };
        if let FlatCostNode::Scale { n, .. } = manifest.nodes[node_idx] {
            ancestor_scale *= f64::from(n);
        }
        node_idx = *child;
    }
    Ok(Some(make_match(
        section.section.clone(),
        manifest,
        node_idx,
        ordinals.to_vec(),
        ancestor_scale,
    )))
}

fn resolve_label(doc: &ManifestDoc, label: &ParsedLabel) -> Result<Vec<NodeMatch>> {
    let mut matches = Vec::new();
    for section in &doc.sections {
        if label
            .section
            .as_deref()
            .is_some_and(|expected| expected != section.section)
        {
            continue;
        }
        let mut section_matches = Vec::new();
        let mut path = Vec::new();
        walk_nodes(
            &section.manifest,
            0,
            &mut path,
            1.0,
            &mut |node_idx, path, ancestor_scale| {
                if section.manifest.node_labels[node_idx].as_deref() == Some(label.label.as_str()) {
                    section_matches.push(make_match(
                        section.section.clone(),
                        &section.manifest,
                        node_idx,
                        path,
                        ancestor_scale,
                    ));
                }
                Ok(())
            },
        )?;
        if section_matches.len() > 1 {
            bail!(
                "exact CostTree label {:?} occurs more than once in section {:?}",
                label.label,
                section.section
            );
        }
        matches.extend(section_matches);
    }
    Ok(matches)
}

fn walk_nodes<F: FnMut(usize, Vec<usize>, f64) -> Result<()>>(
    manifest: &Manifest,
    node_idx: usize,
    path: &mut Vec<usize>,
    ancestor_scale: f64,
    visit: &mut F,
) -> Result<()> {
    if node_idx >= manifest.nodes.len() {
        bail!("CostTree child index {node_idx} is outside the node array");
    }
    visit(node_idx, path.clone(), ancestor_scale)?;
    let children = node_children(manifest, node_idx)?;
    let child_scale = match manifest.nodes[node_idx] {
        FlatCostNode::Scale { n, .. } => ancestor_scale * f64::from(n),
        _ => ancestor_scale,
    };
    for (ordinal, child) in children.into_iter().enumerate() {
        path.push(ordinal);
        walk_nodes(manifest, child, path, child_scale, visit)?;
        path.pop();
    }
    Ok(())
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.nodes.is_empty() {
        bail!("CostTree has no root node");
    }
    if manifest.node_labels.len() != manifest.nodes.len() {
        bail!(
            "CostTree node_labels length {} does not match nodes length {}",
            manifest.node_labels.len(),
            manifest.nodes.len()
        );
    }
    for (node_idx, node) in manifest.nodes.iter().enumerate() {
        match node {
            FlatCostNode::Leaf(slot) if *slot >= manifest.slots.len() => {
                bail!("CostTree node {node_idx} references missing slot {slot}")
            }
            FlatCostNode::Scale { children, .. } if children.len() != 1 => {
                bail!("CostTree Scale node {node_idx} must have exactly one child")
            }
            FlatCostNode::Sum { children }
            | FlatCostNode::Max { children, .. }
            | FlatCostNode::Scale { children, .. }
                if children.start > children.end || children.end > manifest.nodes.len() =>
            {
                bail!("CostTree node {node_idx} has an invalid child range {children:?}")
            }
            _ => {}
        }
        for child in node_children(manifest, node_idx)? {
            if child <= node_idx {
                bail!("CostTree node {node_idx} is not parent-before-child");
            }
        }
    }
    walk_nodes(manifest, 0, &mut Vec::new(), 1.0, &mut |_, _, _| Ok(()))
}

fn node_children(manifest: &Manifest, node_idx: usize) -> Result<Vec<usize>> {
    let range = match &manifest.nodes[node_idx] {
        FlatCostNode::Leaf(_) => return Ok(Vec::new()),
        FlatCostNode::Sum { children }
        | FlatCostNode::Max { children, .. }
        | FlatCostNode::Scale { children, .. } => children.clone(),
    };
    if range.start > range.end || range.end > manifest.nodes.len() {
        bail!("CostTree node {node_idx} has invalid child range {range:?}");
    }
    Ok(range.collect())
}

fn make_match(
    section: String,
    manifest: &Manifest,
    node_idx: usize,
    path: Vec<usize>,
    ancestor_scale: f64,
) -> NodeMatch {
    NodeMatch {
        signature: NodeSignature {
            section: section.clone(),
            path: path.clone(),
            kind: node_kind(&manifest.nodes[node_idx]).to_owned(),
            label: manifest.node_labels[node_idx].clone(),
            leaves: descendant_leaves(manifest, node_idx),
        },
        section,
        node_idx,
        ancestor_scale,
    }
}

fn descendant_leaves(manifest: &Manifest, node_idx: usize) -> Vec<String> {
    let mut leaves = Vec::new();
    collect_descendant_leaves(manifest, node_idx, &mut leaves);
    leaves
}

fn collect_descendant_leaves(manifest: &Manifest, node_idx: usize, leaves: &mut Vec<String>) {
    match manifest.nodes.get(node_idx) {
        Some(FlatCostNode::Leaf(slot)) => {
            if let Some(leaf) = manifest.slots.get(*slot) {
                leaves.push(leaf.name.clone());
            }
        }
        Some(_) => {
            if let Ok(children) = node_children(manifest, node_idx) {
                for child in children {
                    collect_descendant_leaves(manifest, child, leaves);
                }
            }
        }
        None => {}
    }
}

fn node_kind(node: &FlatCostNode) -> &'static str {
    match node {
        FlatCostNode::Leaf(_) => "leaf",
        FlatCostNode::Sum { .. } => "sum",
        FlatCostNode::Max { .. } => "max",
        FlatCostNode::Scale { .. } => "scale",
    }
}

fn canonical_path(signature: &NodeSignature) -> String {
    if signature.path.is_empty() {
        format!("{}/root", signature.section)
    } else {
        format!(
            "{}/{}",
            signature.section,
            signature
                .path
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("/")
        )
    }
}

fn load_hardware_inputs(log_dir: &Path, selection: &Selection) -> HardwareInputs {
    let params = read_params(log_dir);
    let run_gpu_name =
        read_run_meta(log_dir).and_then(|(_, name)| (!name.trim().is_empty()).then_some(name));
    let mut gpu_counts = read_worker_gpu_counts(log_dir)
        .unwrap_or_default()
        .into_iter()
        .map(|(pool, worker, count)| ((pool, worker), count.max(1) as f64))
        .collect::<HashMap<_, _>>();
    let mut defaulted_gpu_count = false;
    let root = repo_root().ok();
    let mut by_worker = HashMap::new();
    for occurrence in &selection.occurrences {
        let gpu_name = params
            .as_ref()
            .and_then(|document| document.pools.get(&occurrence.worker.0))
            .and_then(|pool| pool.groups.first())
            .and_then(|group| group.gpu.clone())
            .or_else(|| run_gpu_name.clone())
            .unwrap_or_default();
        let hardware = root.as_deref().and_then(|root| {
            if gpu_name.trim().is_empty() {
                None
            } else {
                load_gpu_spec(root, &gpu_name).map(|(canonical_name, spec)| HardwareRef {
                    requested_name: gpu_name.clone(),
                    canonical_name,
                    spec,
                })
            }
        });
        by_worker.insert(occurrence.worker.clone(), hardware);
        if !gpu_counts.contains_key(&occurrence.worker) {
            defaulted_gpu_count = true;
            gpu_counts.insert(occurrence.worker.clone(), 1.0);
        }
    }
    HardwareInputs {
        by_worker,
        gpu_counts,
        defaulted_gpu_count,
        params,
    }
}

fn read_params(log_dir: &Path) -> Option<ParamsDocument> {
    let path = crate::io::resolve_artifact_path(log_dir, "params.json");
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn hardware_gpu_names(inputs: &HardwareInputs) -> Vec<String> {
    let mut names = BTreeSet::new();
    for hardware in inputs.by_worker.values().flatten() {
        names.insert(format!(
            "{} ({})",
            hardware.requested_name, hardware.canonical_name
        ));
    }
    names.into_iter().collect()
}

async fn measure_scope(
    ctx: &SessionContext,
    selection: &Selection,
    manifests: &BTreeMap<WorkerKey, ManifestDoc>,
    hardware: &HardwareInputs,
) -> Result<MeasuredScope> {
    let predicates = selection
        .occurrences
        .iter()
        .map(|occurrence| {
            format!(
                "(CAST(pool_tag AS VARCHAR) = {} AND worker_id = {} AND CAST(section AS VARCHAR) = {})",
                sql_quote(&occurrence.worker.0),
                occurrence.worker.1,
                sql_quote(&occurrence.section)
            )
        })
        .collect::<Vec<_>>()
        .join(" OR ");
    let sql = format!(
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, CAST(worker_id AS BIGINT) AS worker_id, CAST(section AS VARCHAR) AS section, slot_time_ms, slot_flops, slot_bytes FROM cost_log WHERE {predicates}"
    );
    let batches = collect(ctx, &sql).await?;
    let mut result = MeasuredScope {
        r5_available: true,
        ..MeasuredScope::default()
    };
    for batch in batches {
        let pool_column = col(&batch, "pool_tag")?;
        let worker_column = col(&batch, "worker_id")?;
        let section_column = col(&batch, "section")?;
        let times_column = col(&batch, "slot_time_ms")?;
        let flops_column = col(&batch, "slot_flops")?;
        let bytes_column = col(&batch, "slot_bytes")?;
        for row in 0..batch.num_rows() {
            let pool = value_string(pool_column, row)?;
            let worker_value = value_f64(worker_column, row)?;
            if !worker_value.is_finite()
                || worker_value.fract() != 0.0
                || worker_value < 0.0
                || worker_value > f64::from(u16::MAX)
            {
                bail!("cost_log worker_id is not a u16 integer at row {row}");
            }
            let worker = worker_value as u16;
            let section = value_string(section_column, row)?;
            let Some(&occurrence_index) =
                selection
                    .occurrence_index
                    .get(&(pool.clone(), worker, section.clone()))
            else {
                continue;
            };
            let occurrence = &selection.occurrences[occurrence_index];
            let manifest_doc = manifests
                .get(&occurrence.worker)
                .with_context(|| format!("selected worker {:?} disappeared", occurrence.worker))?;
            let manifest = manifest_doc.section(&occurrence.section).with_context(|| {
                format!("selected section {:?} disappeared", occurrence.section)
            })?;
            let scope_row = ScopeRow {
                slot_time_ms: value_f32_list(times_column, row)?,
                slot_flops: value_f32_list(flops_column, row)?,
                slot_bytes: value_f32_list(bytes_column, row)?,
            };
            validate_row(manifest, &scope_row)?;
            let gpu_count = hardware
                .gpu_counts
                .get(&occurrence.worker)
                .copied()
                .unwrap_or(1.0);
            let hardware_ref = hardware
                .by_worker
                .get(&occurrence.worker)
                .and_then(|value| value.as_ref());
            let row_rungs = compute_row_rungs(
                manifest,
                occurrence.node_idx,
                occurrence.ancestor_scale,
                &scope_row,
                gpu_count,
                hardware_ref,
            )?;
            result.r0_gpu_s += row_rungs.r0_gpu_s;
            match row_rungs.r5_gpu_s {
                Some(value) if result.r5_available => result.r5_gpu_s += value,
                Some(_) => {}
                None => {
                    result.r5_available = false;
                    if result.r5_reason.is_none() {
                        result.r5_reason = Some(if hardware_ref.is_none() {
                            format!(
                                "no gpu/spec.json entry for selected worker {}/{}",
                                occurrence.worker.0, occurrence.worker.1
                            )
                        } else {
                            "selected subtree contains a communication leaf; R5 is not a local hardware fold".to_owned()
                        });
                    }
                }
            }
            result.rows += 1;
        }
    }
    if result.rows == 0 {
        bail!("CostTree selector matched manifests but no matching cost_log rows");
    }
    Ok(result)
}

fn sql_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn validate_row(manifest: &Manifest, row: &ScopeRow) -> Result<()> {
    let expected = manifest.slots.len();
    for (name, values) in [
        ("slot_time_ms", &row.slot_time_ms),
        ("slot_flops", &row.slot_flops),
        ("slot_bytes", &row.slot_bytes),
    ] {
        if values.len() != expected {
            bail!(
                "selected CostTree row has {name} length {}, expected manifest slot count {expected}",
                values.len()
            );
        }
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            bail!("selected CostTree row has invalid {name} values");
        }
    }
    Ok(())
}

fn compute_row_rungs(
    manifest: &Manifest,
    node_idx: usize,
    ancestor_scale: f64,
    row: &ScopeRow,
    gpu_count: f64,
    hardware: Option<&HardwareRef>,
) -> Result<RowRungs> {
    let slot_ns = row
        .slot_time_ms
        .iter()
        .map(|value| (value * 1e6).round() as i64)
        .collect::<Vec<_>>();
    let r0_ms = ancestor_scale * node_time(manifest, node_idx, &slot_ns) as f64 / 1e6;
    let leaf_weights = selected_leaf_weights(manifest, node_idx, ancestor_scale);
    let has_communication = leaf_weights.iter().any(|(slot, _)| {
        manifest
            .slots
            .get(*slot)
            .is_some_and(|leaf| is_communication_kind(&leaf.kind))
    });
    let Some(hardware) = hardware else {
        return Ok(RowRungs {
            r0_gpu_s: r0_ms * gpu_count / 1e3,
            r5_gpu_s: None,
        });
    };
    if has_communication {
        return Ok(RowRungs {
            r0_gpu_s: r0_ms * gpu_count / 1e3,
            r5_gpu_s: None,
        });
    }
    let mut r5_ms = 0.0;
    for (slot, weight) in leaf_weights {
        let leaf = &manifest.slots[slot];
        let peak_tflops = hardware.spec.peak_tflops(leaf_dtype(&leaf.kernel_config));
        let bandwidth_gbps = hardware.spec.mem_bandwidth_gbps;
        if peak_tflops <= 0.0 || bandwidth_gbps <= 0.0 {
            return Ok(RowRungs {
                r0_gpu_s: r0_ms * gpu_count / 1e3,
                r5_gpu_s: None,
            });
        }
        let compute_ms = row.slot_flops[slot] / (peak_tflops * 1e12) * 1e3;
        let memory_ms = row.slot_bytes[slot] / (bandwidth_gbps * 1e9) * 1e3;
        // Do not let a stale/coarse hardware catalog exceed the measured leaf.
        r5_ms += weight * compute_ms.max(memory_ms).min(row.slot_time_ms[slot]);
    }
    Ok(RowRungs {
        r0_gpu_s: r0_ms * gpu_count / 1e3,
        r5_gpu_s: Some(r5_ms * gpu_count / 1e3),
    })
}

fn selected_leaf_weights(
    manifest: &Manifest,
    node_idx: usize,
    ancestor_scale: f64,
) -> Vec<(usize, f64)> {
    let mut weights = Vec::new();
    fold_mean(manifest, node_idx, ancestor_scale, &mut |slot, weight| {
        weights.push((slot, weight));
    });
    weights
}

fn leaf_dtype(config: &Value) -> &str {
    ["dtype", "q_dtype", "input_dtype", "kv_dtype"]
        .into_iter()
        .find_map(|key| config.get(key).and_then(Value::as_str))
        .unwrap_or("bf16")
}

fn is_communication_kind(kind: &str) -> bool {
    matches!(
        kind,
        "all_reduce"
            | "all_gather"
            | "reduce_scatter"
            | "all_to_all"
            | "broadcast"
            | "gather"
            | "scatter"
            | "send"
            | "recv"
    ) || kind.starts_with("p2p")
        || kind.starts_with("nccl")
        || kind.starts_with("comm")
}

async fn compute_semantic_floors(
    ctx: &SessionContext,
    log_dir: &Path,
    selection: &Selection,
    manifests: &BTreeMap<WorkerKey, ManifestDoc>,
    hardware: &HardwareInputs,
) -> Result<SemanticFloors> {
    let params = hardware
        .params
        .as_ref()
        .context("raw/params.json is required to prove semantic model.work scope")?;
    let root = repo_root().context("repository root is required for location maps")?;
    let maps = load_location_maps(&root)?;
    // Scoped analysis names its own streams, so nothing calibrated reaches here.
    let labels = floors::compute_saturated_run_labels(ctx, log_dir, 1, &HashSet::new()).await?;

    let mut mapping_ids = BTreeSet::new();
    let mut semantic_names = BTreeSet::new();
    let mut common_hardware: Option<(String, String, String, GpuSpec)> = None;
    let mut total_flops = 0.0;
    let mut total_bytes = 0.0;
    let mut segmented_gpu_s = 0.0;

    for occurrence in &selection.occurrences {
        let pool_spec = pool_spec(params, &occurrence.worker.0).with_context(|| {
            format!(
                "params has no model spec for selected pool {:?}",
                occurrence.worker.0
            )
        })?;
        let hardware_ref = hardware
            .by_worker
            .get(&occurrence.worker)
            .and_then(|value| value.as_ref())
            .with_context(|| {
                format!(
                    "GPU spec unavailable for selected worker {:?}",
                    occurrence.worker
                )
            })?;
        let manifest_doc = &manifests[&occurrence.worker];
        let map = select_location_map(&maps, &pool_spec.arch_type, manifest_doc)?;
        mapping_ids.insert(map.mapping_id.clone());
        let hardware_key = (
            pool_spec.arch_type.clone(),
            pool_spec.dtype.clone(),
            hardware_ref.canonical_name.clone(),
        );
        if let Some((arch, dtype, gpu, _)) = &common_hardware {
            if arch != &hardware_key.0 || dtype != &hardware_key.1 || gpu != &hardware_key.2 {
                bail!("selected semantic scope spans heterogeneous model, dtype, or GPU specs");
            }
        } else {
            common_hardware = Some((
                hardware_key.0.clone(),
                hardware_key.1.clone(),
                hardware_key.2.clone(),
                hardware_ref.spec,
            ));
        }

        let selected = selected_semantics(map, manifest_doc, occurrence)?;
        let composition = labels.workers.get(&occurrence.worker).with_context(|| {
            format!(
                "model.work label missing for selected worker {:?}",
                occurrence.worker
            )
        })?;
        let label = composition
            .labels
            .first()
            .context("model.work returned no worker label")?;
        let peak_tflops = hardware_ref.spec.peak_tflops(&pool_spec.dtype);
        let bandwidth_gbps = hardware_ref.spec.mem_bandwidth_gbps;
        if peak_tflops <= 0.0 || bandwidth_gbps <= 0.0 {
            bail!("selected semantic scope has no positive hardware peak or bandwidth");
        }
        let (occurrence_segmented_gpu_s, _) = semantic_floor_values(
            &label.label.segments,
            &selected,
            hardware_ref.spec,
            &pool_spec.dtype,
        )?;
        segmented_gpu_s += occurrence_segmented_gpu_s;
        for semantic in selected {
            semantic_names.insert(semantic.clone());
            let matching = label
                .label
                .segments
                .iter()
                .filter(|segment| segment.name == semantic)
                .collect::<Vec<_>>();
            let segment = match matching.as_slice() {
                [segment] => *segment,
                [] => bail!("model.work label has no semantic segment {semantic:?}"),
                _ => bail!("model.work label has ambiguous semantic segment {semantic:?}"),
            };
            if !segment.flops.is_finite()
                || !segment.bytes.is_finite()
                || segment.flops < 0.0
                || segment.bytes < 0.0
            {
                bail!("model.work semantic segment {semantic:?} has invalid work");
            }
            total_flops += segment.flops;
            total_bytes += segment.bytes;
        }
    }
    let (_, dtype, _, spec) = common_hardware.context("selected semantic scope has no workers")?;
    let peak_tflops = spec.peak_tflops(&dtype);
    if peak_tflops <= 0.0 || spec.mem_bandwidth_gbps <= 0.0 {
        bail!("selected semantic scope has no positive fused hardware ceiling");
    }
    let r7 =
        (total_flops / (peak_tflops * 1e12)).max(total_bytes / (spec.mem_bandwidth_gbps * 1e9));
    Ok(SemanticFloors {
        r6_gpu_s: segmented_gpu_s,
        r7_gpu_s: r7,
        mapping_ids,
        semantic_names,
    })
}

fn pool_spec(params: &ParamsDocument, pool: &str) -> Option<PoolSpec> {
    let group = params.pools.get(pool)?.groups.first()?;
    Some(PoolSpec {
        arch_type: group.arch.arch_type.clone(),
        dtype: if group.arch.fp8 { "fp8" } else { "bf16" }.to_owned(),
    })
}

fn load_location_maps(root: &Path) -> Result<Vec<LocationMap>> {
    let directory = root.join("model/work/location_maps");
    let mut maps = Vec::new();
    for entry in fs::read_dir(&directory)
        .with_context(|| format!("read location-map directory {}", directory.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        let map: LocationMap = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("parse location map {}", path.display()))?;
        if map.schema_version != 1 {
            bail!(
                "unsupported location-map schema {} in {}",
                map.schema_version,
                path.display()
            );
        }
        let mut names = BTreeSet::new();
        for location in &map.locations {
            if !names.insert(location.location.as_str()) {
                bail!(
                    "location map {:?} contains duplicate location",
                    map.mapping_id
                );
            }
        }
        maps.push(map);
    }
    Ok(maps)
}

fn select_location_map<'a>(
    maps: &'a [LocationMap],
    arch_type: &str,
    manifest_doc: &ManifestDoc,
) -> Result<&'a LocationMap> {
    let expected = manifest_doc
        .sections
        .iter()
        .flat_map(|section| section.manifest.slots.iter())
        .filter(|leaf| !is_communication_kind(&leaf.kind))
        .map(|leaf| leaf.name.as_str())
        .collect::<BTreeSet<_>>();
    let candidates = maps
        .iter()
        .filter(|map| {
            map.arch_types
                .iter()
                .any(|candidate| candidate == arch_type)
        })
        .filter(|map| {
            map.locations
                .iter()
                .map(|location| location.location.as_str())
                .collect::<BTreeSet<_>>()
                == expected
        })
        .collect::<Vec<_>>();
    match candidates.as_slice() {
        [map] => Ok(map),
        [] => bail!(
            "no exact semantic location map for arch type {arch_type:?} and selected run manifest"
        ),
        _ => bail!(
            "multiple exact semantic location maps for arch type {arch_type:?}: {:?}",
            candidates
                .iter()
                .map(|map| map.mapping_id.as_str())
                .collect::<Vec<_>>()
        ),
    }
}

fn selected_semantics(
    map: &LocationMap,
    manifest_doc: &ManifestDoc,
    occurrence: &Occurrence,
) -> Result<BTreeSet<String>> {
    let rules = map
        .locations
        .iter()
        .map(|rule| (rule.location.as_str(), rule))
        .collect::<HashMap<_, _>>();
    let manifest = manifest_doc
        .section(&occurrence.section)
        .context("selected section missing while applying location map")?;
    let leaf_names = descendant_leaves(manifest, occurrence.node_idx);
    let mut semantics = BTreeSet::new();
    for location in leaf_names {
        let rule = rules
            .get(location.as_str())
            .with_context(|| format!("location map has no rule for selected leaf {location:?}"))?;
        for semantic in &rule.semantics {
            if !semantics.insert(semantic.clone()) {
                bail!("selected CostTree subtree maps semantic row {semantic:?} more than once");
            }
        }
    }
    Ok(semantics)
}

fn undefined_scope_omissions() -> Vec<OmittedRung> {
    [
        ("idle", "scheduler idle requires whole-run holding spans"),
        (
            "imbalance",
            "cross-worker load imbalance requires a cluster or pool scope",
        ),
        (
            "communication",
            "communication attribution requires the run transfer context",
        ),
        ("balanced", "balanced R2 requires sibling worker timing"),
        ("busy", "busy time is a whole-run scheduler measure"),
        (
            "per_config_best",
            "batching requires a run-level batch distribution",
        ),
        (
            "ignore_network",
            "ignore-network is a whole-run communication counterfactual",
        ),
    ]
    .into_iter()
    .map(|(rung, reason)| OmittedRung {
        rung,
        reason: reason.to_owned(),
    })
    .collect()
}

fn report_slug(selector: &Selector, signature: &NodeSignature) -> String {
    let source = match (&selector.raw_path, &selector.raw_label) {
        (Some(path), Some(label)) => format!("path={path}_label={label}"),
        (Some(path), None) => format!("path={path}"),
        (None, Some(label)) => format!("label={label}"),
        (None, None) => canonical_path(signature),
    };
    let mut slug = source
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    while slug.contains("__") {
        slug = slug.replace("__", "_");
    }
    let digest = Sha256::digest(source.as_bytes());
    let suffix = format!(
        "_{}",
        digest[..6]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    let max_base = 80usize.saturating_sub(suffix.len());
    slug.truncate(max_base);
    format!("{}{}", slug.trim_matches('_'), suffix)
}

fn semantic_floor_values(
    segments: &[SemanticWork],
    selected: &BTreeSet<String>,
    spec: GpuSpec,
    dtype: &str,
) -> Result<(f64, f64)> {
    let peak = spec.peak_tflops(dtype);
    if peak <= 0.0 || spec.mem_bandwidth_gbps <= 0.0 {
        bail!("semantic hardware spec has no positive compute peak or bandwidth");
    }
    let mut flops = 0.0;
    let mut bytes = 0.0;
    let mut segmented = 0.0;
    for name in selected {
        let matches = segments
            .iter()
            .filter(|segment| &segment.name == name)
            .collect::<Vec<_>>();
        let segment = match matches.as_slice() {
            [segment] => *segment,
            [] => bail!("semantic segment {name:?} is missing"),
            _ => bail!("semantic segment {name:?} is ambiguous"),
        };
        flops += segment.flops;
        bytes += segment.bytes;
        segmented +=
            (segment.flops / (peak * 1e12)).max(segment.bytes / (spec.mem_bandwidth_gbps * 1e9));
    }
    let fused = (flops / (peak * 1e12)).max(bytes / (spec.mem_bandwidth_gbps * 1e9));
    Ok((segmented, fused))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::trace::manifest::LeafDesc;

    fn sample_manifest() -> Manifest {
        Manifest {
            slots: vec![
                LeafDesc {
                    name: "embedding".to_owned(),
                    kind: "elementwise".to_owned(),
                    kernel_config: json!({}),
                },
                LeafDesc {
                    name: "q_norm".to_owned(),
                    kind: "rms_norm".to_owned(),
                    kernel_config: json!({"dtype": "bf16"}),
                },
                LeafDesc {
                    name: "k_norm".to_owned(),
                    kind: "rms_norm".to_owned(),
                    kernel_config: json!({"dtype": "bf16"}),
                },
            ],
            nodes: vec![
                FlatCostNode::Sum { children: 1..3 },
                FlatCostNode::Leaf(0),
                FlatCostNode::Scale {
                    n: 2,
                    children: 3..4,
                },
                FlatCostNode::Sum { children: 4..6 },
                FlatCostNode::Leaf(1),
                FlatCostNode::Leaf(2),
            ],
            node_labels: vec![
                Some("root".to_owned()),
                Some("embedding".to_owned()),
                Some("layer".to_owned()),
                Some("qk".to_owned()),
                Some("q".to_owned()),
                Some("k".to_owned()),
            ],
        }
    }

    fn sample_doc() -> ManifestDoc {
        ManifestDoc {
            sections: vec![ManifestSection {
                section: "iter".to_owned(),
                manifest: sample_manifest(),
            }],
        }
    }

    fn test_spec() -> GpuSpec {
        let current = std::env::current_dir().unwrap();
        let root = current
            .ancestors()
            .find(|path| path.join("gpu/spec.json").is_file())
            .expect("find gpu/spec.json");
        load_gpu_spec(root, "H200").expect("H200 catalog entry").1
    }

    #[test]
    fn path_and_exact_label_select_the_same_node() {
        let manifests = BTreeMap::from([(("predict".to_owned(), 0), sample_doc())]);
        let path_selector = Selector::parse(Some("iter/1"), None).unwrap();
        let label_selector = Selector::parse(None, Some("layer")).unwrap();
        let path_selection = resolve_selection(&path_selector, &manifests).unwrap();
        let label_selection = resolve_selection(&label_selector, &manifests).unwrap();
        assert_eq!(path_selection.signature, label_selection.signature);
        assert_eq!(canonical_path(&path_selection.signature), "iter/1");
    }

    #[test]
    fn scale_selection_recurses_to_all_descendant_leaves_and_scales_weights() {
        let manifests = BTreeMap::from([(("predict".to_owned(), 0), sample_doc())]);
        let selector = Selector::parse(Some("iter/1"), None).unwrap();
        let selection = resolve_selection(&selector, &manifests).unwrap();
        let occurrence = &selection.occurrences[0];
        assert_eq!(selection.signature.leaves, ["q_norm", "k_norm"]);
        assert_eq!(occurrence.ancestor_scale, 1.0);
        let weights = selected_leaf_weights(
            &sample_manifest(),
            occurrence.node_idx,
            occurrence.ancestor_scale,
        );
        assert_eq!(weights, vec![(1, 2.0), (2, 2.0)]);
    }

    #[test]
    fn scoped_r0_and_r5_use_the_selected_scale_fold() {
        let manifest = sample_manifest();
        let row = ScopeRow {
            slot_time_ms: vec![1.0, 2.0, 3.0],
            slot_flops: vec![0.0, 0.0, 0.0],
            slot_bytes: vec![0.0, 4.8e9, 9.6e9],
        };
        let hardware = HardwareRef {
            requested_name: "test".to_owned(),
            canonical_name: "test".to_owned(),
            spec: test_spec(),
        };
        let rungs = compute_row_rungs(&manifest, 2, 1.0, &row, 2.0, Some(&hardware)).unwrap();
        assert!((rungs.r0_gpu_s - 0.020).abs() < 1e-12);
        assert!((rungs.r5_gpu_s.unwrap() - 0.012).abs() < 1e-12);
    }

    #[test]
    fn undefined_cluster_rungs_have_machine_readable_reasons() {
        let omissions = undefined_scope_omissions();
        let reasons = omissions
            .iter()
            .map(|omission| omission.rung)
            .collect::<BTreeSet<_>>();
        for rung in [
            "idle",
            "imbalance",
            "communication",
            "balanced",
            "busy",
            "ignore_network",
        ] {
            assert!(reasons.contains(rung));
        }
        assert!(omissions.iter().all(|omission| !omission.reason.is_empty()));
    }

    #[test]
    fn semantic_floors_reconcile_or_report_missing_work_explicitly() {
        let segments = vec![SemanticWork {
            name: "q_norm".to_owned(),
            flops: 1e9,
            bytes: 2e9,
            necessary_gpu_s: 0.0,
            compute_dtype: Some("bf16".to_owned()),
        }];
        let selected = BTreeSet::from(["q_norm".to_owned()]);
        let spec = test_spec();
        let (segmented, fused) = semantic_floor_values(&segments, &selected, spec, "bf16").unwrap();
        assert!(segmented >= fused);
        let missing = BTreeSet::from(["k_norm".to_owned()]);
        let error = semantic_floor_values(&segments, &missing, spec, "bf16").unwrap_err();
        assert!(error.to_string().contains("missing"));
    }

    #[test]
    fn ambiguous_label_is_rejected() {
        let mut doc = sample_doc();
        doc.sections[0].manifest.node_labels[1] = Some("duplicate".to_owned());
        doc.sections[0].manifest.node_labels[4] = Some("duplicate".to_owned());
        let manifests = BTreeMap::from([(("predict".to_owned(), 0), doc)]);
        let selector = Selector::parse(None, Some("duplicate")).unwrap();
        let error = resolve_selection(&selector, &manifests).unwrap_err();
        assert!(format!("{error:#}").contains("more than once"));
    }

    #[test]
    fn path_parser_accepts_section_root_and_rejects_empty_components() {
        let parsed = parse_path("iter/root/1").unwrap();
        assert_eq!(parsed.section.as_deref(), Some("iter"));
        assert_eq!(parsed.ordinals, vec![1]);
        assert!(parse_path("iter//1").is_err());
        assert!(parse_path("iter/nope").is_err());
    }
}
