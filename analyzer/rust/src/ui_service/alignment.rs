//! First-class alignment-bundle discovery and per-subject resources.
//!
//! An alignment bundle is not a run and not a prediction: it is a directory
//! holding a measured capture, the timing prediction of its shapes, and one or
//! two analyses of the pair. On disk that is
//!
//! ```text
//! <bundle>/profile/          nsys capture, parsed.json, host_timeline.json
//!          timing_predict/   the modelled side
//!          simulation/       the DES run, when e2e-align ran
//!          analysis_kernel/  alignment-iteration + alignment-timeline
//!          analysis_e2e/     alignment-workload + alignment-e2e
//! ```
//!
//! The two analysis halves are independent — a capture may have only the kernel
//! half — so the descriptor reports each separately rather than declaring the
//! bundle ready or not.
//!
//! **Iteration detail is served by byte range.** Both kernel-half payloads are
//! indexes naming a sibling `.jsonl`; this reads one record out of it per
//! request. That is the whole reason a 2,040-iteration capture is browsable: the
//! shard is 249 MB and no request ever reads more than one iteration of it.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::alignment_iteration::ZSTD_FRAMES;

use super::artifact::read_json;
use super::artifact_kind::{read_artifact_kind, ArtifactKind};
use super::discovery::{
    display_name, ignored_directory_name, regular_file, timestamp, ConfiguredRoot,
};
use super::prediction::{prediction_at, DiscoveredPrediction};
#[cfg(test)]
use super::AlignmentNotFound;
use super::{ArtifactNotFound, PROTOCOL_VERSION};

/// The manifest each analysis half writes. Its presence is what makes a
/// directory an analysis, and its parent a bundle.
const MANIFEST_FILE: &str = "alignment_manifest.json";
const KERNEL_ANALYSIS_CONFIG: &str = "analyze_kernel.yaml";
const KERNEL_ANALYSIS_DIR: &str = "analysis_kernel";
const E2E_ANALYSIS_DIR: &str = "analysis_e2e";

/// Subjects, and where each one's report and payload live. Named here rather
/// than assembled from the URL so no request can name a file.
const SUBJECTS: &[(&str, &str, &str)] = &[
    (
        "iteration",
        KERNEL_ANALYSIS_DIR,
        "alignment_iteration_report.json",
    ),
    (
        "timeline",
        KERNEL_ANALYSIS_DIR,
        "alignment_timeline_report.json",
    ),
    (
        "workload",
        E2E_ANALYSIS_DIR,
        "alignment_workload_report.json",
    ),
    ("e2e", E2E_ANALYSIS_DIR, "alignment_e2e_report.json"),
];

const PAYLOAD_NAMES: &[(&str, &str)] = &[
    ("iteration", "alignment_iteration_series.json"),
    ("timeline", "alignment_timeline.json"),
    ("workload", "alignment_workload_series.json"),
    ("e2e", "alignment_e2e_series.json"),
];

/// Which payload section names the shard, per subject that has one.
const DETAIL_SECTIONS: &[(&str, &str)] = &[
    ("iteration", "breakdown_detail"),
    ("timeline", "iteration_detail"),
];

#[derive(Clone, Debug)]
pub(super) struct DiscoveredAlignment {
    pub(super) workspace_id: String,
    pub(super) alignment_id: String,
    pub(super) display_name: String,
    pub(super) path: PathBuf,
    kernel_analysis_dir: Option<PathBuf>,
    pub(super) updated_time: SystemTime,
}

impl DiscoveredAlignment {
    fn analysis_dir(&self, name: &str) -> Option<PathBuf> {
        if name == KERNEL_ANALYSIS_DIR {
            self.kernel_analysis_dir.clone()
        } else {
            Some(self.path.join(name))
        }
    }

    /// A half is complete when its manifest and at least one of its reports are
    /// both on disk; a manifest alone means it was configured, not run.
    fn half_status(&self, directory: &str) -> &'static str {
        let Some(analysis_dir) = self.analysis_dir(directory) else {
            return "not_started";
        };
        if !regular_file(&analysis_dir.join(MANIFEST_FILE)) {
            return "not_started";
        }
        let has_report =
            SUBJECTS
                .iter()
                .filter(|(_, dir, _)| *dir == directory)
                .any(|(_, dir, report)| {
                    self.analysis_dir(dir)
                        .is_some_and(|path| regular_file(&path.join("reports").join(report)))
                });
        if has_report {
            "complete"
        } else {
            "pending"
        }
    }

    /// The timing prediction this bundle's modelled side came out of.
    ///
    /// Only the kernel manifest names it, because the modelled side is a
    /// kernel-half artifact, and it names it as a DIRECTORY — that is what the
    /// analysis was pointed at. A bundle routinely holds several
    /// `timing_predict*` directories from re-analyses of the same capture, so
    /// which one produced these numbers cannot be recovered from the bundle's
    /// layout; it has to be read out of the manifest that recorded the choice.
    ///
    /// A prediction under a different workspace is dropped: the alignment's
    /// resources are addressed within one workspace, and a cross-workspace id
    /// would resolve against the wrong root.
    fn paired_prediction(&self, roots: &[ConfiguredRoot]) -> Option<DiscoveredPrediction> {
        let manifest =
            read_json(&self.analysis_dir(KERNEL_ANALYSIS_DIR)?.join(MANIFEST_FILE)).ok()?;
        let predict_log_dir = manifest.get("predict_log_dir")?.as_str()?;
        prediction_at(roots, Path::new(predict_log_dir))
            .filter(|prediction| prediction.workspace_id == self.workspace_id)
    }
}

#[derive(Debug, Serialize)]
pub(super) struct AlignmentCatalog {
    protocol_version: u32,
    generated_at: String,
    alignments: Vec<AlignmentCatalogEntry>,
}

#[derive(Debug, Serialize)]
struct AlignmentCatalogEntry {
    workspace_id: String,
    alignment_id: String,
    kind: &'static str,
    display_name: String,
    kernel_analysis: &'static str,
    e2e_analysis: &'static str,
    updated_at: String,
}

pub(super) fn build_alignment_catalog(roots: &[ConfiguredRoot]) -> Result<AlignmentCatalog> {
    let mut alignments = discover_alignments(roots)?;
    alignments.sort_by(|left, right| {
        right
            .updated_time
            .cmp(&left.updated_time)
            .then_with(|| left.alignment_id.cmp(&right.alignment_id))
    });
    let alignments = alignments
        .into_iter()
        .map(|alignment| AlignmentCatalogEntry {
            kernel_analysis: alignment.half_status(KERNEL_ANALYSIS_DIR),
            e2e_analysis: alignment.half_status(E2E_ANALYSIS_DIR),
            kind: "alignment",
            updated_at: timestamp(alignment.updated_time),
            workspace_id: alignment.workspace_id,
            alignment_id: alignment.alignment_id,
            display_name: alignment.display_name,
        })
        .collect();
    Ok(AlignmentCatalog {
        protocol_version: PROTOCOL_VERSION,
        generated_at: timestamp(SystemTime::now()),
        alignments,
    })
}

/// Walk each configured root for explicitly typed alignment bundles.
///
/// The bundle is the analysis directory's PARENT, because that is what owns the
/// capture and the prediction both halves point at. Descent stops at a bundle
/// rather than continuing into its `profile/` and `simulation/` subtrees.
pub(super) fn discover_alignments(roots: &[ConfiguredRoot]) -> Result<Vec<DiscoveredAlignment>> {
    let mut alignments = Vec::new();
    let mut seen = HashSet::new();
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
            if artifact_kind == Some(ArtifactKind::AlignmentBundle) {
                let kernel_analysis_dir = selected_kernel_analysis_dir(&directory);
                if !alignment_has_analysis_half(&directory, kernel_analysis_dir.as_deref()) {
                    continue;
                }
                let relative = directory
                    .strip_prefix(root.path())
                    .expect("alignment discovery stays below its configured root");
                let alignment_id = opaque_alignment_id(root.workspace_id(), relative);
                if seen.insert(alignment_id.clone()) {
                    alignments.push(DiscoveredAlignment {
                        workspace_id: root.workspace_id().to_owned(),
                        alignment_id,
                        display_name: display_name(root.path(), relative),
                        updated_time: alignment_updated_at(
                            &directory,
                            kernel_analysis_dir.as_deref(),
                        ),
                        path: directory,
                        kernel_analysis_dir,
                    });
                }
                continue;
            }
            if artifact_kind.is_some_and(|kind| !kind.can_contain_resources()) {
                continue;
            }
            let Ok(entries) = fs::read_dir(&directory) else {
                continue;
            };
            let mut children = entries.filter_map(|entry| entry.ok()).collect::<Vec<_>>();
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
    Ok(alignments)
}

#[derive(Deserialize)]
struct KernelAnalysisConfig {
    schema_version: u32,
    log_dir: PathBuf,
}

/// Return the kernel analysis selected by the bundle's analyzer config.
///
/// Analysis generation already treats `analyze_kernel.yaml` as the authority;
/// discovery must not independently hardcode `analysis_kernel`. The legacy
/// directory applies only when no config exists. Once configured, a missing
/// output remains not-started and an invalid or external selection is withheld;
/// neither condition may silently publish stale legacy results.
fn selected_kernel_analysis_dir(bundle: &Path) -> Option<PathBuf> {
    let fallback = bundle.join(KERNEL_ANALYSIS_DIR);
    let config_path = bundle.join(KERNEL_ANALYSIS_CONFIG);
    let config_metadata = match fs::symlink_metadata(&config_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Some(fallback),
        Err(_) => return None,
    };
    if !config_metadata.file_type().is_file() {
        return None;
    }

    let config = fs::read(&config_path)
        .ok()
        .and_then(|bytes| serde_yaml::from_slice::<KernelAnalysisConfig>(&bytes).ok())
        .filter(|config| config.schema_version == 1)?;
    if config.log_dir.is_absolute()
        || config
            .log_dir
            .components()
            .any(|component| !matches!(component, Component::CurDir | Component::Normal(_)))
    {
        return None;
    }

    let selected = bundle.join(config.log_dir);
    if !selected.exists() {
        // A configured analysis may not have started yet. Relative paths with
        // no parent traversal remain within the bundle even before creation.
        return Some(selected);
    }
    let canonical_bundle = bundle.canonicalize().ok()?;
    let canonical_selected = selected.canonicalize().ok()?;
    canonical_selected
        .starts_with(canonical_bundle)
        .then_some(canonical_selected)
}

fn alignment_has_analysis_half(directory: &Path, kernel_analysis_dir: Option<&Path>) -> bool {
    regular_file(&directory.join(KERNEL_ANALYSIS_CONFIG))
        || kernel_analysis_dir.is_some_and(|path| regular_file(&path.join(MANIFEST_FILE)))
        || regular_file(&directory.join(E2E_ANALYSIS_DIR).join(MANIFEST_FILE))
}

fn alignment_updated_at(bundle: &Path, kernel_analysis_dir: Option<&Path>) -> SystemTime {
    kernel_analysis_dir
        .map(Path::to_path_buf)
        .into_iter()
        .chain([bundle.join(E2E_ANALYSIS_DIR)])
        .flat_map(|half| {
            [
                half.join("reports").join("analyzer_timing.json"),
                half.join(MANIFEST_FILE),
            ]
        })
        .filter_map(|path| fs::metadata(path).ok())
        .filter_map(|metadata| metadata.modified().ok())
        .max()
        .unwrap_or(UNIX_EPOCH)
}

fn opaque_alignment_id(workspace_id: &str, relative: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-analyzer-alignment-id-v1\0");
    hasher.update(workspace_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(relative.as_os_str().as_encoded_bytes());
    format!("al_{:x}", hasher.finalize())
}

#[cfg(test)]
pub(super) fn resolve_alignment(
    roots: &[ConfiguredRoot],
    alignment_id: &str,
) -> Result<DiscoveredAlignment> {
    discover_alignments(roots)?
        .into_iter()
        .find(|alignment| alignment.alignment_id == alignment_id)
        .ok_or_else(|| AlignmentNotFound.into())
}

pub(super) fn alignment_descriptor(
    alignment: &DiscoveredAlignment,
    roots: &[ConfiguredRoot],
) -> Value {
    let subjects: serde_json::Map<String, Value> = SUBJECTS
        .iter()
        .map(|(subject, directory, report)| {
            let available = alignment
                .analysis_dir(directory)
                .is_some_and(|path| regular_file(&path.join("reports").join(report)));
            (
                (*subject).to_owned(),
                json!({
                    "status": if available { "ready" } else { "not_generated" },
                    "views": if available { json!(["report", "payload"]) } else { json!([]) },
                    // Whether one iteration can be read on its own. It is a
                    // property of the subject, not a second address: the shape
                    // is `subjects/{name}/iterations/{id}` for every subject
                    // that has it.
                    "has_iteration_detail": available && detail_section(subject).is_some(),
                }),
            )
        })
        .collect();
    let prediction = alignment.paired_prediction(roots);
    json!({
        "schema_version": 1,
        "alignment_id": alignment.alignment_id,
        "kind": "alignment",
        "display_name": alignment.display_name,
        "lifecycle": {
            "kernel_analysis": alignment.half_status(KERNEL_ANALYSIS_DIR),
            "e2e_analysis": alignment.half_status(E2E_ANALYSIS_DIR),
        },
        // Null is a state, not an absence: a bundle analysed before the
        // manifest carried the prediction path, or one whose prediction this
        // service does not serve, must be distinguishable from one whose
        // prediction is simply not looked up yet.
        "prediction": prediction.map(|prediction| json!({
            "prediction_id": prediction.prediction_id(),
            "display_name": prediction.display_name,
        })),
        "subjects": subjects,
    })
}

fn subject_entry(subject: &str) -> Option<&'static (&'static str, &'static str, &'static str)> {
    SUBJECTS.iter().find(|(name, _, _)| *name == subject)
}

fn detail_section(subject: &str) -> Option<&'static str> {
    DETAIL_SECTIONS
        .iter()
        .find(|(name, _)| *name == subject)
        .map(|(_, section)| *section)
}

/// Resolve one artifact of one subject. The subject name is matched against the
/// table above, so a request never contributes a path component.
fn artifact_path(alignment: &DiscoveredAlignment, subject: &str, payload: bool) -> Result<PathBuf> {
    let (_, directory, report) = subject_entry(subject).ok_or(ArtifactNotFound)?;
    let analysis_dir = alignment.analysis_dir(directory).ok_or(ArtifactNotFound)?;
    let path = if payload {
        let name = PAYLOAD_NAMES
            .iter()
            .find(|(name, _)| *name == subject)
            .map(|(_, file)| *file)
            .ok_or(ArtifactNotFound)?;
        analysis_dir.join("payloads").join(name)
    } else {
        analysis_dir.join("reports").join(report)
    };
    if regular_file(&path) {
        Ok(path)
    } else {
        Err(ArtifactNotFound.into())
    }
}

#[cfg(test)]
pub(super) fn alignment_report(alignment: &DiscoveredAlignment, subject: &str) -> Result<Value> {
    read_json(&artifact_path(alignment, subject, false)?)
}

#[cfg(test)]
pub(super) fn alignment_payload(alignment: &DiscoveredAlignment, subject: &str) -> Result<Value> {
    read_json(&artifact_path(alignment, subject, true)?)
}

/// Return an alignment artifact without parsing and re-serializing it.
///
/// Reports and payloads are already canonical JSON written by the analyzer.
/// The UI service used to turn them into `serde_json::Value` and then encode
/// the same tree again for every request, which made a 4.5 MB iteration report
/// pay a full parse plus a full serialization before the browser could start
/// parsing it. Keep the Value helpers above for tests and descriptor logic;
/// HTTP handlers should use these bytes.
pub(super) fn alignment_artifact_bytes(
    alignment: &DiscoveredAlignment,
    subject: &str,
    payload: bool,
) -> Result<Vec<u8>> {
    let path = artifact_path(alignment, subject, payload)?;
    fs::read(&path).with_context(|| format!("read alignment artifact {}", path.display()))
}

/// The small index that points into one iteration-detail JSONL shard.
///
/// The payload contains this metadata beside the plot series. Parse it once
/// per service cache entry; reading one iteration must only seek the shard and
/// never parse the index document again.
#[derive(Clone)]
pub(super) struct AlignmentDetailIndex {
    pub(super) shard_path: PathBuf,
    pub(super) byte_ranges: HashMap<String, (u64, u64)>,
    /// Each record's decoded size when the shard is `zstd-frames`: every range
    /// is then one zstd frame holding the JSON line and its newline. `None` for
    /// a plain `.jsonl` shard written before the analyzer compressed them.
    pub(super) decoded_lengths: Option<HashMap<String, usize>>,
    /// The reference rank and drawable host threads are shared by every
    /// timeline detail row. Keeping them in the cached index lets the service
    /// build the UI's reference-lane projection without reopening the index.
    pub(super) reference_device_id: Option<i64>,
    pub(super) reference_host_thread_ids: HashSet<String>,
}

/// Byte ranges for folded sequence programs. The iteration series keeps only
/// browse metadata; opening the mapping board fetches the selected programs.
#[derive(Clone)]
pub(super) struct AlignmentSequenceIndex {
    pub(super) shard_path: PathBuf,
    pub(super) byte_ranges: HashMap<(String, String), (u64, u64)>,
}

pub(super) fn alignment_detail_index(
    alignment: &DiscoveredAlignment,
    subject: &str,
) -> Result<AlignmentDetailIndex> {
    let section = detail_section(subject).ok_or(ArtifactNotFound)?;
    let payload_path = artifact_path(alignment, subject, true)?;
    let payload: Value = read_json(&payload_path)?;
    let detail = payload.get(section).ok_or(ArtifactNotFound)?;
    let file = detail
        .get("file")
        .and_then(Value::as_str)
        .ok_or(ArtifactNotFound)?;
    let ranges = detail
        .get("byte_ranges")
        .and_then(Value::as_object)
        .ok_or(ArtifactNotFound)?;
    let byte_ranges = ranges
        .iter()
        .map(|(iteration_id, range)| {
            let range = range.as_array().ok_or(ArtifactNotFound)?;
            let offset = range
                .first()
                .and_then(Value::as_u64)
                .ok_or(ArtifactNotFound)?;
            let length = range
                .get(1)
                .and_then(Value::as_u64)
                .ok_or(ArtifactNotFound)?;
            Ok((iteration_id.clone(), (offset, length)))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    let decoded_lengths = match detail.get("encoding").and_then(Value::as_str) {
        Some(ZSTD_FRAMES) => Some(
            detail
                .get("decoded_lengths")
                .and_then(Value::as_object)
                .ok_or(ArtifactNotFound)?
                .iter()
                .map(|(iteration_id, length)| {
                    let length = length.as_u64().ok_or(ArtifactNotFound)?;
                    Ok((iteration_id.clone(), length as usize))
                })
                .collect::<Result<HashMap<_, _>>>()?,
        ),
        _ => None,
    };
    let reference_device_id = payload
        .get("meta")
        .and_then(|meta| meta.get("reference_device_id"))
        .and_then(Value::as_i64);
    let reference_host_thread_ids = payload
        .get("meta")
        .and_then(|meta| meta.get("host_timeline"))
        .and_then(|host| host.get("threads"))
        .and_then(Value::as_array)
        .map(|threads| {
            threads
                .iter()
                .enumerate()
                .filter(|(_, thread)| {
                    let device_id = thread.get("device_id").and_then(Value::as_i64);
                    device_id.is_none() || device_id == reference_device_id
                })
                .map(|(index, _)| index.to_string())
                .collect()
        })
        .unwrap_or_default();
    let shard_path = payload_path
        .parent()
        .ok_or(ArtifactNotFound)?
        .join(Path::new(file).file_name().ok_or(ArtifactNotFound)?);
    Ok(AlignmentDetailIndex {
        shard_path,
        byte_ranges,
        decoded_lengths,
        reference_device_id,
        reference_host_thread_ids,
    })
}

pub(super) fn read_alignment_iteration_detail(
    index: &AlignmentDetailIndex,
    iteration_id: &str,
) -> Result<Vec<u8>> {
    let (offset, length) = index
        .byte_ranges
        .get(iteration_id)
        .copied()
        .ok_or(ArtifactNotFound)?;
    let mut handle = fs::File::open(&index.shard_path)
        .with_context(|| format!("open alignment shard {}", index.shard_path.display()))?;
    handle.seek(SeekFrom::Start(offset))?;
    let mut buffer = vec![0u8; length as usize];
    handle.read_exact(&mut buffer)?;
    let Some(decoded_lengths) = &index.decoded_lengths else {
        return Ok(buffer);
    };
    let decoded_len = decoded_lengths
        .get(iteration_id)
        .copied()
        .ok_or(ArtifactNotFound)?;
    let mut line = zstd::bulk::decompress(&buffer, decoded_len).with_context(|| {
        format!(
            "decode iteration {iteration_id} of {}",
            index.shard_path.display()
        )
    })?;
    // The frame holds the line and its newline; serve the object alone, as a
    // plain shard's range does.
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    Ok(line)
}

pub(super) fn alignment_sequence_index(
    alignment: &DiscoveredAlignment,
) -> Result<AlignmentSequenceIndex> {
    let payload_path = artifact_path(alignment, "iteration", true)?;
    let payload: Value = read_json(&payload_path)?;
    let detail = payload.get("sequence_detail").ok_or(ArtifactNotFound)?;
    let file = detail
        .get("file")
        .and_then(Value::as_str)
        .ok_or(ArtifactNotFound)?;
    let phase_ranges = detail
        .get("byte_ranges")
        .and_then(Value::as_object)
        .ok_or(ArtifactNotFound)?;
    let mut byte_ranges = HashMap::new();
    for (phase, sequences) in phase_ranges {
        let sequences = sequences.as_object().ok_or(ArtifactNotFound)?;
        for (sequence_id, range) in sequences {
            let range = range.as_array().ok_or(ArtifactNotFound)?;
            let offset = range
                .first()
                .and_then(Value::as_u64)
                .ok_or(ArtifactNotFound)?;
            let length = range
                .get(1)
                .and_then(Value::as_u64)
                .ok_or(ArtifactNotFound)?;
            byte_ranges.insert((phase.clone(), sequence_id.clone()), (offset, length));
        }
    }
    let shard_path = payload_path
        .parent()
        .ok_or(ArtifactNotFound)?
        .join(Path::new(file).file_name().ok_or(ArtifactNotFound)?);
    Ok(AlignmentSequenceIndex {
        shard_path,
        byte_ranges,
    })
}

pub(super) fn read_alignment_sequence(
    index: &AlignmentSequenceIndex,
    phase: &str,
    sequence_id: &str,
) -> Result<Vec<u8>> {
    let (offset, length) = index
        .byte_ranges
        .get(&(phase.to_owned(), sequence_id.to_owned()))
        .copied()
        .ok_or(ArtifactNotFound)?;
    let mut handle = fs::File::open(&index.shard_path).with_context(|| {
        format!(
            "open alignment sequence shard {}",
            index.shard_path.display()
        )
    })?;
    handle.seek(SeekFrom::Start(offset))?;
    let mut buffer = vec![0u8; length as usize];
    handle.read_exact(&mut buffer)?;
    Ok(buffer)
}

/// Keep only what the wall-clock figure actually draws: reference-rank GPU
/// intervals, plus scheduler/reference-rank host rows.
///
/// The on-disk shard remains the lossless all-rank record. This projection is a
/// read-side view, so the timeline's definitions and aggregate metrics remain
/// unchanged while the browser no longer parses four ranks and every helper
/// thread just to discard them in `measuredLane` / `groupedHostLanes`.
pub(super) fn reference_lane_projection(
    index: &AlignmentDetailIndex,
    bytes: Vec<u8>,
) -> Result<Vec<u8>> {
    let Some(reference_device_id) = index.reference_device_id else {
        return Ok(bytes);
    };
    let mut document: Value = serde_json::from_slice(&bytes)
        .with_context(|| "parse alignment timeline iteration for reference-lane projection")?;
    if let Some(kernels) = document
        .get_mut("measured")
        .and_then(|measured| measured.get_mut("kernels"))
        .and_then(Value::as_array_mut)
    {
        for kernel in kernels {
            if let Some(intervals) = kernel.get_mut("iv").and_then(Value::as_array_mut) {
                intervals.retain(|interval| {
                    interval
                        .get(0)
                        .and_then(Value::as_i64)
                        .is_some_and(|device_id| device_id == reference_device_id)
                });
            }
        }
    }
    if let Some(host) = document.get_mut("host") {
        for lane_name in ["nvtx", "api"] {
            if let Some(lanes) = host.get_mut(lane_name).and_then(Value::as_object_mut) {
                lanes.retain(|thread_id, _| index.reference_host_thread_ids.contains(thread_id));
            }
        }
    }
    serde_json::to_vec(&document).with_context(|| "encode alignment reference-lane projection")
}

/// Keep only the fields §03 uses for its measured-vs-modelled operation stacks.
///
/// This is separate from the timeline projection: the operation split needs
/// kernel durations and names, but no first timestamp, device roster, or
/// analyzer bookkeeping fields. The full breakdown remains available at the
/// unprojected endpoint for other consumers.
pub(super) fn operation_split_projection(bytes: Vec<u8>) -> Result<Vec<u8>> {
    let mut document: Value = serde_json::from_slice(&bytes)
        .with_context(|| "parse alignment breakdown for operation-split projection")?;
    if let Some(kernels) = document
        .get_mut("measured_kernels")
        .and_then(Value::as_array_mut)
    {
        let source_kernels = std::mem::take(kernels);
        let mut folded_kernels: Vec<Value> = Vec::new();
        let mut folded_keys: Vec<String> = Vec::new();
        for source_kernel in source_kernels {
            let Some(source_kernel) = source_kernel.as_object() else {
                continue;
            };
            let phase = source_kernel
                .get("phase")
                .and_then(Value::as_str)
                .unwrap_or("");
            let operation = source_kernel
                .get("operation")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let name = source_kernel
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("");
            // Match the browser's foldMeasuredGroups key exactly: mapped rows
            // join by operation, while an unmapped row remains identifiable by
            // its kernel name. The first occurrence supplies the display name.
            let key = match &operation {
                Some(operation) => format!("{phase}\0operation\0{operation}"),
                None => format!("{phase}\0kernel\0{name}"),
            };
            let duration_ms = source_kernel
                .get("duration_ms")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let calls = source_kernel
                .get("calls")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if let Some(index) = folded_keys.iter().position(|existing| existing == &key) {
                let folded = folded_kernels[index]
                    .as_object_mut()
                    .expect("folded operation row is an object");
                let current_duration = folded
                    .get("duration_ms")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let current_calls = folded.get("calls").and_then(Value::as_u64).unwrap_or(0);
                folded.insert(
                    "duration_ms".to_owned(),
                    json!(current_duration + duration_ms),
                );
                folded.insert("calls".to_owned(), json!(current_calls + calls));
            } else {
                folded_keys.push(key);
                folded_kernels.push(json!({
                    "phase": phase,
                    "operation": operation,
                    "name": name,
                    "duration_ms": duration_ms,
                    "calls": calls,
                }));
            }
        }
        *kernels = folded_kernels;
    }
    if let Some(kernels) = document
        .get_mut("simulated_kernels")
        .and_then(Value::as_array_mut)
    {
        for kernel in kernels {
            if let Some(kernel) = kernel.as_object_mut() {
                kernel.retain(|name, _| {
                    matches!(
                        name.as_str(),
                        "slot_index"
                            | "name"
                            | "kind"
                            | "operation"
                            | "folded_ms"
                            | "critical_path_ms"
                            | "multiplicity"
                    )
                });
            }
        }
    }
    serde_json::to_vec(&document).with_context(|| "encode alignment operation-split projection")
}

/// One iteration's detail, read out of the subject's shard by byte range.
///
/// Returned as raw bytes: the shard already holds valid JSON for exactly this
/// record, so parsing and re-serializing it would cost megabytes of work per
/// request to produce the same bytes back.
#[cfg(test)]
pub(super) fn alignment_iteration_detail(
    alignment: &DiscoveredAlignment,
    subject: &str,
    iteration_id: &str,
) -> Result<Vec<u8>> {
    let index = alignment_detail_index(alignment, subject)?;
    read_alignment_iteration_detail(&index, iteration_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_subject_has_a_payload_and_only_the_sharded_ones_have_a_detail_section() {
        for (subject, _, _) in SUBJECTS {
            assert!(
                PAYLOAD_NAMES.iter().any(|(name, _)| name == subject),
                "subject {subject} has no payload"
            );
        }
        // The e2e half emits series, not per-iteration detail, so it has no
        // shard — and asking for one must be a 404, not an empty document.
        assert_eq!(detail_section("timeline"), Some("iteration_detail"));
        assert_eq!(detail_section("iteration"), Some("breakdown_detail"));
        assert_eq!(detail_section("e2e"), None);
        assert_eq!(detail_section("workload"), None);
    }

    #[test]
    fn an_unknown_subject_names_no_file() {
        assert!(subject_entry("../../etc/passwd").is_none());
        assert!(subject_entry("iteration").is_some());
    }

    #[test]
    fn the_opaque_id_depends_on_workspace_and_path_but_not_registry_order() {
        let left = opaque_alignment_id("w_one", Path::new("a/b"));
        assert_eq!(left, opaque_alignment_id("w_one", Path::new("a/b")));
        assert_ne!(left, opaque_alignment_id("w_two", Path::new("a/b")));
        assert_ne!(left, opaque_alignment_id("w_one", Path::new("a/c")));
        assert!(left.starts_with("al_"));
    }
}
