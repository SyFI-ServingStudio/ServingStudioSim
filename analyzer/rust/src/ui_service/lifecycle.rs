//! Analyzer generation lifecycle, descriptor assembly, and state projection.

use super::*;

pub(super) fn lifecycle(run: &RunRecord) -> Lifecycle {
    let pipeline = read_pipeline_state(run);
    let timing = read_timing(run);
    lifecycle_from(run, &pipeline, &timing)
}

pub(super) fn lifecycle_from(
    run: &RunRecord,
    pipeline: &PipelineStateRead,
    timing: &TimingState,
) -> Lifecycle {
    let simulation = if contained_marker(run, ".failed") {
        StageStatus::Failed
    } else if contained_marker(run, ".complete") {
        StageStatus::Complete
    } else {
        StageStatus::Pending
    };
    let analysis = analysis_status(pipeline, timing);
    Lifecycle {
        simulation,
        analysis,
    }
}

pub(super) fn analysis_status(pipeline: &PipelineStateRead, timing: &TimingState) -> StageStatus {
    match pipeline {
        PipelineStateRead::Valid(state) => {
            if state.stages.compute.status == StageStatus::Complete
                && !timing_matches_generation(timing, &state.generation_id)
            {
                StageStatus::Failed
            } else {
                state.status
            }
        }
        PipelineStateRead::Invalid { .. } => StageStatus::Failed,
        PipelineStateRead::Missing => match timing {
            TimingState::Valid(_) => StageStatus::Complete,
            TimingState::Invalid(_) => StageStatus::Failed,
            TimingState::Missing => StageStatus::NotStarted,
        },
    }
}

pub(super) fn contained_marker(run: &RunRecord, name: &str) -> bool {
    open_contained_artifact(run, Path::new(name), name).is_ok()
}

pub(super) fn read_pipeline_state(run: &RunRecord) -> PipelineStateRead {
    let relative = Path::new(PIPELINE_STATE_PATH);
    let (bytes, _) =
        match read_bounded_artifact(run, relative, MAX_PIPELINE_STATE_BYTES, "analyzer pipeline") {
            Ok(artifact) => artifact,
            Err(problem) if problem.code == "artifact_missing" => {
                return PipelineStateRead::Missing
            }
            Err(problem) => {
                return PipelineStateRead::Invalid {
                    code: problem.code,
                    reason: problem.detail,
                }
            }
        };
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            return PipelineStateRead::Invalid {
                code: "pipeline_state_invalid".to_string(),
                reason: format!("Analyzer pipeline state is not valid JSON: {error}"),
            }
        }
    };
    let schema_version = value.get("schema_version").and_then(Value::as_u64);
    if schema_version != Some(u64::from(PIPELINE_SCHEMA_VERSION)) {
        return PipelineStateRead::Invalid {
            code: "pipeline_state_incompatible".to_string(),
            reason: format!(
                "Analyzer pipeline schema {:?} is not supported by schema v{PIPELINE_SCHEMA_VERSION}.",
                schema_version
            ),
        };
    }
    let state: PipelineStateV1 = match serde_json::from_value(value) {
        Ok(state) => state,
        Err(error) => {
            return PipelineStateRead::Invalid {
                code: "pipeline_state_invalid".to_string(),
                reason: format!("Analyzer pipeline state has an invalid shape: {error}"),
            }
        }
    };
    match validate_pipeline_state(&state) {
        Ok(()) => PipelineStateRead::Valid(Box::new(state)),
        Err(reason) => PipelineStateRead::Invalid {
            code: "pipeline_state_invalid".to_string(),
            reason,
        },
    }
}

pub(super) fn validate_pipeline_state(state: &PipelineStateV1) -> std::result::Result<(), String> {
    if state.schema_version != PIPELINE_SCHEMA_VERSION {
        return Err("pipeline schema version changed during decoding".to_string());
    }
    for (name, value, maximum) in [
        ("generation_id", state.generation_id.as_str(), 128usize),
        (
            "artifact_revision",
            state.artifact_revision.as_str(),
            160usize,
        ),
    ] {
        if value.is_empty()
            || value.len() > maximum
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(format!(
                "pipeline {name} is not a bounded opaque identifier"
            ));
        }
    }
    for (name, timestamp_value) in [
        ("started_at", state.started_at.as_str()),
        ("updated_at", state.updated_at.as_str()),
    ] {
        DateTime::parse_from_rfc3339(timestamp_value)
            .map_err(|error| format!("pipeline {name} is not RFC3339: {error}"))?;
    }
    if let Some(completed_at) = &state.completed_at {
        DateTime::parse_from_rfc3339(completed_at)
            .map_err(|error| format!("pipeline completed_at is not RFC3339: {error}"))?;
    }
    for (name, value) in [
        ("producer.name", state.producer.name.as_str()),
        ("producer.version", state.producer.version.as_str()),
        ("producer.revision", state.producer.revision.as_str()),
        (
            "producer.binary_sha256",
            state.producer.binary_sha256.as_str(),
        ),
    ] {
        if value.is_empty() || value.len() > 160 {
            return Err(format!("pipeline {name} is missing or unbounded"));
        }
    }
    if state.stages.compute.status == StageStatus::Complete
        && (state.producer.version == "unavailable"
            || state.producer.revision == "unavailable"
            || !state.producer.binary_sha256.starts_with("sha256:"))
    {
        return Err("completed compute lacks a concrete producer identity".to_string());
    }
    if let Some(requested) = &state.requested_subjects {
        if requested.is_empty() {
            return Err("requested_subjects must be null or a non-empty list".to_string());
        }
        let mut unique = HashSet::new();
        for subject in requested {
            if !unique.insert(subject)
                || !SUBJECTS
                    .iter()
                    .any(|row| row.scope == Scope::Run && row.name == subject)
            {
                return Err(format!(
                    "requested_subjects contains duplicate or unknown token {subject:?}"
                ));
            }
        }
    }

    let compute = state.stages.compute.status;
    let render = state.stages.render.status;
    let trace = state.stages.trace.status;
    let render_terminal = matches!(render, StageStatus::Complete | StageStatus::Failed);
    let trace_terminal = matches!(trace, StageStatus::Complete | StageStatus::Failed);
    let shape_is_valid = match state.status {
        StageStatus::Pending => match compute {
            StageStatus::Pending => {
                render == StageStatus::NotStarted && trace == StageStatus::NotStarted
            }
            StageStatus::Complete => {
                render != StageStatus::NotStarted
                    && (!matches!(
                        trace,
                        StageStatus::Pending | StageStatus::Complete | StageStatus::Failed
                    ) || render_terminal)
            }
            StageStatus::NotStarted | StageStatus::Failed => false,
        },
        StageStatus::Complete => {
            compute == StageStatus::Complete && render_terminal && trace_terminal
        }
        StageStatus::Failed => {
            compute == StageStatus::Failed
                && render == StageStatus::NotStarted
                && trace == StageStatus::NotStarted
        }
        StageStatus::NotStarted => false,
    };
    if !shape_is_valid {
        return Err("pipeline and stage statuses do not form a valid transition".to_string());
    }
    if matches!(state.status, StageStatus::Complete | StageStatus::Failed)
        != state.completed_at.is_some()
    {
        return Err(
            "terminal pipeline state must have completed_at (and pending must not)".to_string(),
        );
    }
    for (name, stage) in [
        ("compute", &state.stages.compute),
        ("render", &state.stages.render),
        ("trace", &state.stages.trace),
    ] {
        if stage.status == StageStatus::Failed && stage.code.as_deref().is_none_or(str::is_empty) {
            return Err(format!("failed {name} stage lacks a stable code"));
        }
    }
    if state.stages.trace.status == StageStatus::Complete {
        let Some(artifact) = state.stages.trace.artifact.as_deref() else {
            return Err("complete trace stage lacks its exact artifact path".to_string());
        };
        let relative = Path::new(artifact);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            || !artifact.starts_with("traces/")
            || !(artifact.ends_with(".pftrace") || artifact.ends_with(".pftrace.gz"))
        {
            return Err("trace artifact is not a bounded trace-relative path".to_string());
        }
    }
    Ok(())
}

pub(super) fn timing_matches_generation(timing: &TimingState, generation_id: &str) -> bool {
    matches!(
        timing,
        TimingState::Valid(TimingInfo {
            generation_id: Some(observed),
            ..
        }) if observed == generation_id
    )
}

/// A reader fence for the fixed on-disk artifact names.
///
/// Versioned publishers expose their full pipeline and timing state. Legacy
/// runs have no publication pointer, so their content-derived revision is the
/// fence. Callers must capture this before and after every multi-file read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AnalysisSnapshot {
    pipeline: PipelineStateRead,
    timing: TimingState,
    legacy_revision: Option<String>,
}

impl AnalysisSnapshot {
    fn capture(run: &RunRecord) -> Self {
        let pipeline = read_pipeline_state(run);
        let timing = read_timing(run);
        let legacy_revision = matches!(&pipeline, PipelineStateRead::Missing)
            .then(|| legacy_analysis_revision(run, &timing));
        Self {
            pipeline,
            timing,
            legacy_revision,
        }
    }

    /// Only a compute-complete, timing-verified generation can own ready
    /// versioned JSON. Legacy artifacts use their content-derived revision.
    pub(super) fn artifact_revision(&self) -> Option<&str> {
        match &self.pipeline {
            PipelineStateRead::Valid(state)
                if state.stages.compute.status == StageStatus::Complete
                    && timing_matches_generation(&self.timing, &state.generation_id) =>
            {
                Some(&state.artifact_revision)
            }
            PipelineStateRead::Missing => self.legacy_revision.as_deref(),
            PipelineStateRead::Valid(_) | PipelineStateRead::Invalid { .. } => None,
        }
    }

    pub(super) fn pipeline(&self) -> &PipelineStateRead {
        &self.pipeline
    }

    pub(super) fn timing(&self) -> &TimingState {
        &self.timing
    }

    pub(super) fn require_revision(&self, requested_revision: &str) -> Result<(), ApiProblem> {
        if self.artifact_revision() == Some(requested_revision) {
            return Ok(());
        }
        Err(ApiProblem::artifact_generation_changed(format!(
            "Artifact revision {requested_revision:?} is no longer the current readable generation."
        )))
    }
}

/// Seqlock-style bounded read over atomically replaced generation artifacts.
/// A generation cutover retries the whole operation; repeated churn fails
/// closed instead of returning bytes assembled under a stale revision.
pub(super) fn with_consistent_analysis_snapshot<T>(
    run: &RunRecord,
    mut read: impl FnMut(&AnalysisSnapshot) -> Result<T, ApiProblem>,
) -> Result<T, ApiProblem> {
    for _ in 0..MAX_GENERATION_READ_ATTEMPTS {
        let before = AnalysisSnapshot::capture(run);
        let result = read(&before);
        let after = AnalysisSnapshot::capture(run);
        if before == after {
            return result;
        }
    }
    Err(ApiProblem::artifact_generation_changed(
        "The analyzer generation changed repeatedly while the resource was being read; retry the revision-linked request.",
    ))
}

pub(super) fn build_descriptor(run: &RunRecord) -> Result<RunDescriptor, ApiProblem> {
    with_consistent_analysis_snapshot(run, |snapshot| build_descriptor_at_snapshot(run, snapshot))
}

pub(super) fn build_descriptor_at_snapshot(
    run: &RunRecord,
    snapshot: &AnalysisSnapshot,
) -> Result<RunDescriptor, ApiProblem> {
    let params = read_json_value(run, Path::new("raw/params.json"), "params")?;
    let deployment = deployment_from_params(&params)?;
    let lifecycle = lifecycle_from(run, &snapshot.pipeline, &snapshot.timing);
    let run_meta = read_json_value_optional(run, Path::new("raw/run_meta.json"));
    let workers = run_meta
        .as_ref()
        .and_then(|run_meta| workers_from_run_meta(&deployment, run_meta));

    let subjects = SUBJECTS
        .iter()
        .filter(|subject| subject.scope == Scope::Run)
        .map(|subject| {
            (
                subject.name.to_string(),
                subject_state(
                    run,
                    subject,
                    &deployment,
                    lifecycle,
                    &snapshot.pipeline,
                    &snapshot.timing,
                    snapshot.artifact_revision(),
                    None,
                ),
            )
        })
        .collect();
    let mut details = BTreeMap::new();
    details.insert(
        "iteration-detail",
        serde_json::json!({
            "status": "not_generated",
            "reason": "No on-demand iteration detail endpoint is available."
        }),
    );
    details.insert(
        "worker-cost-tree",
        serde_json::json!({
            "status": "not_generated",
            "reason": "No versioned hierarchical worker CostTree endpoint is available."
        }),
    );
    details.insert(
        "worker-iteration-index",
        serde_json::json!({
            "status": "not_generated",
            "reason": "No paginated worker iteration index is available."
        }),
    );
    let mut traces = BTreeMap::new();
    traces.insert(
        "perfetto",
        trace_state(
            run,
            lifecycle,
            &snapshot.pipeline,
            &snapshot.timing,
            snapshot.artifact_revision(),
        ),
    );

    let analysis = analysis_identity(run, snapshot, &subjects, &traces);
    let generated_at = analysis
        .as_ref()
        .map(|identity| identity.generated_at.clone());
    let generator_version = analysis
        .as_ref()
        .map(|identity| identity.generator_version.clone())
        .unwrap_or_else(|| LEGACY_GENERATOR_VERSION.to_string());
    Ok(RunDescriptor {
        protocol_version: PROTOCOL_VERSION,
        run_id: run.run_id.clone(),
        kind: "simulation",
        display_name: run.display_name.clone(),
        model_name: model_name_from_params(&deployment, &params),
        deployment,
        lifecycle,
        summary: ArtifactRef {
            href: "summary",
            media_type: "application/json",
            schema_version: None,
        },
        topology: ArtifactRef {
            href: "topology",
            media_type: "application/json",
            schema_version: Some(TOPOLOGY_SCHEMA_VERSION),
        },
        workers,
        subjects,
        details,
        traces,
        analysis,
        provenance: AnalyzerProvenance {
            source: "analyzer",
            synthetic: false,
            generated_at,
            generator_version,
        },
    })
}

/// Cheap descriptor cache key. Analyzer publishers replace artifacts atomically,
/// so size + nanosecond mtime metadata changes before a new generation is
/// observable; unchanged polls avoid reparsing every report/payload JSON pair.
pub(super) fn descriptor_stamp(run: &RunRecord) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-ui-descriptor-stamp-v1\0");
    for relative in [
        Path::new("raw/params.json").to_path_buf(),
        Path::new("raw/run_meta.json").to_path_buf(),
        Path::new(".complete").to_path_buf(),
        Path::new(".failed").to_path_buf(),
        Path::new("reports/analyzer_timing.json").to_path_buf(),
        Path::new(PIPELINE_STATE_PATH).to_path_buf(),
    ] {
        stamp_path_metadata(&mut hasher, run, &relative);
    }
    for subject in SUBJECTS
        .iter()
        .filter(|subject| subject.scope == Scope::Run)
    {
        stamp_path_metadata(
            &mut hasher,
            run,
            &Path::new("reports").join(subject.report_name),
        );
        stamp_path_metadata(
            &mut hasher,
            run,
            &Path::new("payloads").join(subject.payload_name),
        );
    }
    let pipeline = read_pipeline_state(run);
    let selected_trace = match &pipeline {
        PipelineStateRead::Missing => select_trace(run),
        PipelineStateRead::Valid(_) | PipelineStateRead::Invalid { .. } => {
            select_trace_for_pipeline(run, &pipeline)
        }
    };
    match selected_trace {
        Ok(trace) => stamp_metadata(&mut hasher, &trace),
        Err(_) => hasher.update(b"\0trace-missing\0"),
    }
    hex(&hasher.finalize())
}

fn stamp_path_metadata(hasher: &mut Sha256, run: &RunRecord, relative: &Path) {
    hasher.update(relative.as_os_str().as_encoded_bytes());
    match resolve_contained_file(&run.root, &run.path, relative, "descriptor stamp") {
        Ok(path) => stamp_metadata(hasher, &path),
        Err(_) => hasher.update(b"\0missing\0"),
    }
}

fn stamp_metadata(hasher: &mut Sha256, path: &Path) {
    match path.metadata() {
        Ok(metadata) => {
            hasher.update(metadata.len().to_be_bytes());
            let modified = metadata
                .modified()
                .unwrap_or(UNIX_EPOCH)
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            hasher.update(modified.to_be_bytes());
        }
        Err(_) => hasher.update(b"\0metadata-error\0"),
    }
}

pub(super) fn read_deployment(run: &RunRecord) -> Result<String, ApiProblem> {
    let params = read_json_value(run, Path::new("raw/params.json"), "params")?;
    deployment_from_params(&params)
}

pub(super) fn deployment_from_params(params: &Value) -> Result<String, ApiProblem> {
    let Some(deployment) = params.get("deployment").and_then(Value::as_str) else {
        return Err(ApiProblem::artifact_incompatible(
            "params",
            "deployment is missing or is not a string",
        ));
    };
    if !matches!(deployment, "unified" | "pd" | "afd") {
        return Err(ApiProblem::artifact_incompatible(
            "params",
            format!("unsupported deployment {deployment:?}"),
        ));
    }
    Ok(deployment.to_string())
}

pub(super) fn pool_roles(deployment: &str) -> Option<&'static [&'static str]> {
    match deployment {
        "unified" => Some(&["main"]),
        "pd" => Some(&["prefill", "decode"]),
        "afd" => Some(&["attn", "ffn"]),
        _ => None,
    }
}

pub(super) fn model_name_from_params(deployment: &str, params: &Value) -> Option<String> {
    let roles = pool_roles(deployment)?;
    let pools = params.get("pools")?.as_object()?;
    let models = roles
        .iter()
        .filter_map(|role| pools.get(*role)?.get("groups")?.as_array())
        .flatten()
        .filter_map(|group| {
            group
                .get("arch")?
                .get("model_config")?
                .as_str()
                .map(str::to_owned)
        })
        .collect::<HashSet<_>>();
    if models.len() == 1 {
        models.into_iter().next()
    } else {
        None
    }
}

pub(super) fn workers_from_run_meta(deployment: &str, run_meta: &Value) -> Option<Vec<WorkerRef>> {
    let roles = pool_roles(deployment)?;
    let raw_workers = run_meta.get("workers")?.as_array()?;
    if raw_workers.is_empty() {
        return None;
    }
    let mut workers = Vec::with_capacity(raw_workers.len());
    let mut identities = HashSet::new();
    for raw_worker in raw_workers {
        let numeric_pool = usize::try_from(raw_worker.get("pool")?.as_u64()?).ok()?;
        let pool_tag = *roles.get(numeric_pool)?;
        let worker_id = raw_worker.get("worker_id")?.as_u64()?;
        if let Some(observed_tag) = raw_worker.get("pool_tag").and_then(Value::as_str) {
            if observed_tag != pool_tag {
                return None;
            }
        }
        if !identities.insert((pool_tag, worker_id)) {
            return None;
        }
        workers.push(WorkerRef {
            pool_tag: pool_tag.to_string(),
            worker_id,
        });
    }
    workers.sort_by_key(|worker| {
        (
            roles
                .iter()
                .position(|role| *role == worker.pool_tag)
                .unwrap_or(usize::MAX),
            worker.worker_id,
        )
    });
    Some(workers)
}

pub(super) fn read_timing(run: &RunRecord) -> TimingState {
    let relative = Path::new("reports/analyzer_timing.json");
    let (bytes, modified_at) =
        match read_bounded_artifact(run, relative, MAX_JSON_BYTES, "analyzer timing") {
            Ok(artifact) => artifact,
            Err(problem) if problem.code == "artifact_missing" => return TimingState::Missing,
            Err(problem) => return TimingState::Invalid(problem.detail),
        };
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            return TimingState::Invalid(format!("invalid analyzer timing JSON: {error}"))
        }
    };
    let generation_id = value
        .get("generation_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let Some(subjects) = value.get("subjects").and_then(Value::as_array) else {
        return TimingState::Invalid("analyzer timing has no subjects array".to_string());
    };
    let mut statuses = HashMap::new();
    for row in subjects {
        let (Some(name), Some(status)) = (
            row.get("name").and_then(Value::as_str),
            row.get("status").and_then(Value::as_str),
        ) else {
            return TimingState::Invalid(
                "analyzer timing subject row lacks string name/status".to_string(),
            );
        };
        if !matches!(status, "ok" | "failed") {
            return TimingState::Invalid(format!(
                "analyzer timing subject {name:?} has unknown status {status:?}"
            ));
        }
        if statuses
            .insert(name.to_string(), status.to_string())
            .is_some()
        {
            return TimingState::Invalid(format!("analyzer timing repeats subject {name:?}"));
        }
    }
    TimingState::Valid(TimingInfo {
        statuses,
        bytes,
        modified_at,
        generation_id,
    })
}

pub(super) fn subject_state(
    run: &RunRecord,
    subject: &Subject,
    deployment: &str,
    lifecycle: Lifecycle,
    pipeline: &PipelineStateRead,
    timing: &TimingState,
    artifact_revision: Option<&str>,
    artifact_bytes: Option<&mut SubjectArtifactBytes>,
) -> SubjectState {
    if !subject.applies.matches(Some(deployment)) {
        return SubjectState::Unavailable {
            reason: format!(
                "Analyzer subject {:?} does not apply to deployment {deployment:?}.",
                subject.name
            ),
            code: Some("subject_not_applicable".to_string()),
        };
    }
    match pipeline {
        PipelineStateRead::Invalid { code, reason } => {
            return SubjectState::Failed {
                code: code.clone(),
                reason: reason.clone(),
            }
        }
        PipelineStateRead::Valid(state) => {
            let requested = state
                .requested_subjects
                .as_ref()
                .is_none_or(|subjects| subjects.iter().any(|name| name == subject.name));
            if !requested {
                return SubjectState::NotGenerated {
                    reason: Some(format!(
                        "Analyzer subject {:?} was not requested for generation {:?}.",
                        subject.name, state.generation_id
                    )),
                };
            }
            match state.stages.compute.status {
                StageStatus::NotStarted | StageStatus::Pending => {
                    return SubjectState::Pending {
                        reason: Some(format!(
                            "Analyzer subject {:?} is waiting for current-generation compute.",
                            subject.name
                        )),
                    }
                }
                StageStatus::Failed => {
                    return SubjectState::Failed {
                        code: state
                            .stages
                            .compute
                            .code
                            .clone()
                            .unwrap_or_else(|| "compute_failed".to_string()),
                        reason: format!(
                            "Analyzer compute failed before subject {:?} became current.",
                            subject.name
                        ),
                    }
                }
                StageStatus::Complete => {
                    if !timing_matches_generation(timing, &state.generation_id) {
                        return SubjectState::Failed {
                            code: "analysis_generation_mismatch".to_string(),
                            reason: format!(
                                "Analyzer timing does not belong to current generation {:?}.",
                                state.generation_id
                            ),
                        };
                    }
                    if timing_status(timing, subject.name).is_none() {
                        return SubjectState::Failed {
                            code: "analysis_subject_unrecorded".to_string(),
                            reason: format!(
                                "Current generation did not record requested subject {:?}.",
                                subject.name
                            ),
                        };
                    }
                }
            }
        }
        PipelineStateRead::Missing => {}
    }
    if matches!(
        timing,
        TimingState::Valid(TimingInfo { statuses, .. })
            if statuses.get(subject.name).is_some_and(|status| status == "failed")
    ) {
        return SubjectState::Failed {
            code: "analysis_failed".to_string(),
            reason: format!(
                "Analyzer subject {:?} failed during generation.",
                subject.name
            ),
        };
    }

    let mut report = probe_json(
        run,
        &Path::new("reports").join(subject.report_name),
        "subject report",
    );
    let mut payload = probe_json(
        run,
        &Path::new("payloads").join(subject.payload_name),
        "subject payload",
    );
    if let JsonProbe::Invalid { code, reason } = &report {
        return SubjectState::Failed {
            code: code.clone(),
            reason: reason.clone(),
        };
    }
    if let JsonProbe::Invalid { code, reason } = &payload {
        return SubjectState::Failed {
            code: code.clone(),
            reason: reason.clone(),
        };
    }

    match (&mut report, &mut payload) {
        (
            JsonProbe::Valid {
                schema_version: report_version,
                value: report_value,
                bytes: report_bytes,
            },
            JsonProbe::Valid {
                schema_version: payload_version,
                value: payload_value,
                bytes: payload_bytes,
            },
        ) => {
            if report_version != payload_version {
                return SubjectState::Failed {
                    code: "artifact_incompatible".to_string(),
                    reason: format!(
                        "Analyzer subject {:?} report/payload schema versions disagree ({report_version} vs {payload_version}).",
                        subject.name
                    ),
                };
            }
            if artifact_available(report_value) == Some(false)
                || artifact_available(payload_value) == Some(false)
            {
                return SubjectState::Unavailable {
                    reason: artifact_reason(report_value)
                        .or_else(|| artifact_reason(payload_value))
                        .unwrap_or_else(|| {
                            format!(
                                "Analyzer subject {:?} reported no available data.",
                                subject.name
                            )
                        }),
                    code: artifact_code(report_value)
                        .or_else(|| artifact_code(payload_value))
                        .or_else(|| Some("subject_unavailable".to_string())),
                };
            }
            if let Some(artifact_bytes) = artifact_bytes {
                artifact_bytes.report = std::mem::take(report_bytes);
                artifact_bytes.payload = std::mem::take(payload_bytes);
            }
            let Some(artifact_revision) = artifact_revision else {
                return SubjectState::Failed {
                    code: "analysis_generation_mismatch".to_string(),
                    reason: format!(
                        "Analyzer subject {:?} has no stable artifact revision.",
                        subject.name
                    ),
                };
            };
            SubjectState::Ready {
                schema_version: *report_version,
                report_href: format!("revisions/{artifact_revision}/reports/{}", subject.name),
                payload_href: format!("revisions/{artifact_revision}/payloads/{}", subject.name),
            }
        }
        _ if timing_status(timing, subject.name) == Some("ok") => SubjectState::Failed {
            code: "artifact_missing".to_string(),
            reason: format!(
                "Analyzer subject {:?} completed but its report/payload pair is incomplete.",
                subject.name
            ),
        },
        _ if lifecycle.simulation == StageStatus::Pending
            || lifecycle.analysis == StageStatus::Pending =>
        {
            SubjectState::Pending {
                reason: Some(format!(
                    "Analyzer subject {:?} has not finished generating.",
                    subject.name
                )),
            }
        }
        _ if lifecycle.analysis == StageStatus::Failed => {
            let detail = match timing {
                TimingState::Invalid(detail) => detail.as_str(),
                TimingState::Missing | TimingState::Valid(_) => "unknown analyzer lifecycle error",
            };
            SubjectState::Failed {
                code: "analysis_state_invalid".to_string(),
                reason: format!(
                    "Analyzer lifecycle failed before subject {:?} produced a complete artifact pair: {detail}.",
                    subject.name
                ),
            }
        }
        _ => SubjectState::NotGenerated {
            reason: Some(format!(
                "Analyzer subject {:?} was not generated for this run.",
                subject.name
            )),
        },
    }
}

pub(super) fn timing_status<'a>(timing: &'a TimingState, subject: &str) -> Option<&'a str> {
    match timing {
        TimingState::Valid(info) => info.statuses.get(subject).map(String::as_str),
        TimingState::Missing | TimingState::Invalid(_) => None,
    }
}

pub(super) fn probe_json(run: &RunRecord, relative: &Path, resource: &str) -> JsonProbe {
    let (bytes, _) = match read_bounded_artifact(run, relative, MAX_JSON_BYTES, resource) {
        Ok(artifact) => artifact,
        Err(problem) if problem.code == "artifact_missing" => return JsonProbe::Missing,
        Err(problem) => {
            return JsonProbe::Invalid {
                code: problem.code,
                reason: problem.detail,
            }
        }
    };
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            return JsonProbe::Invalid {
                code: "artifact_incompatible".to_string(),
                reason: format!("The {resource} is not valid JSON: {error}"),
            }
        }
    };
    let Some(schema_version) = value.get("schema_version").and_then(Value::as_u64) else {
        return JsonProbe::Invalid {
            code: "artifact_incompatible".to_string(),
            reason: format!("The {resource} has no positive integer schema_version."),
        };
    };
    let Ok(schema_version) = u32::try_from(schema_version) else {
        return JsonProbe::Invalid {
            code: "artifact_incompatible".to_string(),
            reason: format!("The {resource} schema_version does not fit protocol-v1."),
        };
    };
    if schema_version == 0 {
        return JsonProbe::Invalid {
            code: "artifact_incompatible".to_string(),
            reason: format!("The {resource} schema_version must be positive."),
        };
    }
    JsonProbe::Valid {
        schema_version,
        value,
        bytes,
    }
}

pub(super) fn artifact_available(value: &Value) -> Option<bool> {
    value
        .get("available")
        .and_then(Value::as_bool)
        .or_else(|| value.get("meta")?.get("available")?.as_bool())
}

pub(super) fn artifact_reason(value: &Value) -> Option<String> {
    value
        .get("reason")
        .and_then(Value::as_str)
        .or_else(|| value.get("meta")?.get("reason")?.as_str())
        .filter(|reason| !reason.trim().is_empty())
        .map(str::to_owned)
}

pub(super) fn artifact_code(value: &Value) -> Option<String> {
    value
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| value.get("meta")?.get("code")?.as_str())
        .filter(|code| !code.trim().is_empty())
        .map(str::to_owned)
}

pub(super) fn trace_state(
    run: &RunRecord,
    lifecycle: Lifecycle,
    pipeline: &PipelineStateRead,
    timing: &TimingState,
    artifact_revision: Option<&str>,
) -> TraceState {
    match pipeline {
        PipelineStateRead::Invalid { code, reason } => TraceState::Failed {
            code: code.clone(),
            reason: reason.clone(),
        },
        PipelineStateRead::Valid(state) => {
            if state.stages.compute.status == StageStatus::Complete
                && !timing_matches_generation(timing, &state.generation_id)
            {
                return TraceState::Failed {
                    code: "analysis_generation_mismatch".to_string(),
                    reason: "Perfetto trace belongs to a pipeline whose compute generation cannot be verified."
                        .to_string(),
                };
            }
            match state.stages.trace.status {
                StageStatus::NotStarted | StageStatus::Pending => TraceState::Pending {
                    reason: Some(
                        "Current-generation Perfetto trace has not completed.".to_string(),
                    ),
                },
                StageStatus::Failed => TraceState::Failed {
                    code: state
                        .stages
                        .trace
                        .code
                        .clone()
                        .unwrap_or_else(|| "trace_failed".to_string()),
                    reason: "Current-generation Perfetto trace generation failed.".to_string(),
                },
                StageStatus::Complete => {
                    match (artifact_revision, select_trace_for_pipeline(run, pipeline)) {
                        (Some(artifact_revision), Ok(path)) => {
                            trace_ready_state(run, &path, artifact_revision)
                        }
                        (None, Ok(_)) => TraceState::Failed {
                            code: "analysis_generation_mismatch".to_string(),
                            reason: "Perfetto trace has no stable artifact revision.".to_string(),
                        },
                        (_, Err(problem)) => TraceState::Failed {
                            code: problem.code,
                            reason: problem.detail,
                        },
                    }
                }
            }
        }
        PipelineStateRead::Missing => match (artifact_revision, select_trace(run)) {
            (Some(artifact_revision), Ok(path)) => trace_ready_state(run, &path, artifact_revision),
            (None, Ok(_)) => TraceState::Failed {
                code: "analysis_generation_mismatch".to_string(),
                reason: "Legacy Perfetto trace has no stable artifact revision.".to_string(),
            },
            (_, Err(problem)) if problem.code == "artifact_missing" => {
                if lifecycle.simulation == StageStatus::Pending
                    || lifecycle.analysis == StageStatus::Pending
                {
                    TraceState::Pending {
                        reason: Some("Perfetto trace generation has not completed.".to_string()),
                    }
                } else {
                    TraceState::NotGenerated {
                        reason: Some("No Perfetto trace was generated for this run.".to_string()),
                    }
                }
            }
            (_, Err(problem)) => TraceState::Failed {
                code: problem.code,
                reason: problem.detail,
            },
        },
    }
}

pub(super) fn trace_ready_state(
    run: &RunRecord,
    path: &Path,
    artifact_revision: &str,
) -> TraceState {
    let relative = match path.strip_prefix(&run.path) {
        Ok(relative) => relative,
        Err(_) => {
            return TraceState::Failed {
                code: "artifact_outside_run".to_string(),
                reason: "Perfetto trace is outside its resolved run.".to_string(),
            }
        }
    };
    match open_contained_artifact(run, relative, "Perfetto trace").and_then(|(file, _)| {
        file.metadata().map_err(|error| {
            ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "Artifact read failed",
                format!("Perfetto trace metadata is unavailable: {error}"),
            )
        })
    }) {
        Ok(metadata) if metadata.len() <= MAX_TRACE_BYTES => TraceState::Ready {
            href: format!("revisions/{artifact_revision}/traces/perfetto"),
            media_type: trace_media_type(path),
            byte_length: metadata.len(),
        },
        Ok(_) => TraceState::Failed {
            code: "artifact_too_large".to_string(),
            reason: "The bounded Perfetto trace exceeds the service safety limit.".to_string(),
        },
        Err(error) => TraceState::Failed {
            code: error.code,
            reason: error.detail,
        },
    }
}

pub(super) fn trace_media_type(path: &Path) -> &'static str {
    if path.extension().and_then(|extension| extension.to_str()) == Some("gz") {
        "application/gzip"
    } else {
        "application/x-protobuf"
    }
}

pub(super) fn select_trace(run: &RunRecord) -> Result<PathBuf, ApiProblem> {
    select_trace_from(&run.root, &run.path)
}

pub(super) fn select_trace_for_pipeline(
    run: &RunRecord,
    pipeline: &PipelineStateRead,
) -> Result<PathBuf, ApiProblem> {
    match pipeline {
        PipelineStateRead::Missing => select_trace(run),
        PipelineStateRead::Invalid { code, reason } => Err(ApiProblem::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            code.clone(),
            "Analyzer pipeline is invalid",
            reason.clone(),
        )),
        PipelineStateRead::Valid(state) => {
            if state.stages.trace.status != StageStatus::Complete {
                return Err(ApiProblem::new(
                    StatusCode::NOT_FOUND,
                    "resource_not_ready",
                    "Resource is not ready",
                    "Current-generation Perfetto trace is not complete.",
                ));
            }
            let relative = state.stages.trace.artifact.as_deref().ok_or_else(|| {
                ApiProblem::artifact_incompatible(
                    "Perfetto trace",
                    "pipeline artifact path is missing",
                )
            })?;
            resolve_contained_file(&run.root, &run.path, Path::new(relative), "Perfetto trace")
        }
    }
}

pub(super) fn select_trace_from(root: &Path, run: &Path) -> Result<PathBuf, ApiProblem> {
    let trace_dir = run.join("traces");
    let entries = match fs::read_dir(&trace_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return Err(ApiProblem::artifact_missing("Perfetto trace"))
        }
        Err(error) => {
            return Err(ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "Artifact read failed",
                format!("Cannot enumerate the bounded Perfetto trace directory: {error}"),
            ))
        }
    };
    let mut candidates = Vec::new();
    let mut containment_error = None;
    for entry in entries.filter_map(|entry| entry.ok()) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".pftrace") && !name.ends_with(".pftrace.gz") {
            continue;
        }
        match resolve_contained_file(
            root,
            run,
            &Path::new("traces").join(&name),
            "Perfetto trace",
        ) {
            Ok(path) => {
                let modified_at = path
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .unwrap_or(UNIX_EPOCH);
                candidates.push((modified_at, name, path));
            }
            Err(problem) if problem.code == "artifact_missing" => {}
            Err(problem) => containment_error = Some(problem),
        }
    }
    candidates.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    if let Some((_, _, path)) = candidates.pop() {
        return Ok(path);
    }
    Err(containment_error.unwrap_or_else(|| ApiProblem::artifact_missing("Perfetto trace")))
}

pub(super) fn analysis_identity(
    run: &RunRecord,
    snapshot: &AnalysisSnapshot,
    subjects: &BTreeMap<String, SubjectState>,
    traces: &BTreeMap<&'static str, TraceState>,
) -> Option<AnalysisIdentity> {
    match &snapshot.pipeline {
        PipelineStateRead::Valid(state)
            if state.stages.compute.status == StageStatus::Complete
                && timing_matches_generation(&snapshot.timing, &state.generation_id) =>
        {
            let TimingState::Valid(timing) = &snapshot.timing else {
                unreachable!("generation match requires valid timing")
            };
            let revision = state.producer.revision.chars().take(12).collect::<String>();
            Some(AnalysisIdentity {
                revision: state.artifact_revision.clone(),
                generated_at: timestamp(timing.modified_at),
                generator_version: format!(
                    "{}-{}@{}",
                    state.producer.name, state.producer.version, revision
                ),
            })
        }
        PipelineStateRead::Valid(_) | PipelineStateRead::Invalid { .. } => None,
        PipelineStateRead::Missing => {
            let has_ready_artifact = subjects
                .values()
                .any(|state| matches!(state, SubjectState::Ready { .. }))
                || traces
                    .values()
                    .any(|state| matches!(state, TraceState::Ready { .. }));
            if !has_ready_artifact && !matches!(&snapshot.timing, TimingState::Valid(_)) {
                return None;
            }
            Some(AnalysisIdentity {
                revision: snapshot
                    .legacy_revision
                    .clone()
                    .expect("legacy analysis snapshot always has a revision"),
                generated_at: timestamp(legacy_analysis_generated_at(run, &snapshot.timing)),
                generator_version: LEGACY_GENERATOR_VERSION.to_string(),
            })
        }
    }
}

pub(super) fn legacy_analysis_revision(run: &RunRecord, timing: &TimingState) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-analyzer-legacy-analysis-revision-v1\0");
    if let TimingState::Valid(timing) = timing {
        hasher.update(&timing.bytes);
    }
    for subject in SUBJECTS
        .iter()
        .filter(|subject| subject.scope == Scope::Run)
    {
        fingerprint_artifact_contents(
            &mut hasher,
            run,
            &Path::new("reports").join(subject.report_name),
        );
        fingerprint_artifact_contents(
            &mut hasher,
            run,
            &Path::new("payloads").join(subject.payload_name),
        );
    }
    // Legacy has no state-recorded trace path. Bind the selected newest trace's
    // opened-file identity without scanning a potentially 512 MiB body.
    match select_trace(run).and_then(|path| {
        let relative = path.strip_prefix(&run.path).map_err(|_| {
            ApiProblem::new(
                StatusCode::FORBIDDEN,
                "artifact_outside_run",
                "Artifact escaped its run",
                "The selected legacy trace is outside its resolved run.",
            )
        })?;
        let (file, _) = open_contained_artifact(run, relative, "Perfetto trace")?;
        file.metadata().map_err(|error| {
            ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "Artifact read failed",
                format!("Cannot inspect the selected legacy trace: {error}"),
            )
        })
    }) {
        Ok(metadata) => {
            hasher.update(b"\0trace\0");
            hasher.update(trace_metadata_etag(&metadata).as_bytes());
        }
        Err(_) => hasher.update(b"\0trace-missing\0"),
    }
    format!("legacy-sha256-{}", hex(&hasher.finalize()))
}

pub(super) fn fingerprint_artifact_contents(hasher: &mut Sha256, run: &RunRecord, relative: &Path) {
    hasher.update(relative_label(relative).as_bytes());
    match read_bounded_artifact(run, relative, MAX_JSON_BYTES, "analysis artifact") {
        Ok((bytes, _)) => {
            hasher.update(bytes.len().to_be_bytes());
            hasher.update(bytes);
        }
        Err(_) => hasher.update(b"\0missing\0"),
    }
}

pub(super) fn legacy_analysis_generated_at(run: &RunRecord, timing: &TimingState) -> SystemTime {
    let mut generated_at = match timing {
        TimingState::Valid(timing) => timing.modified_at,
        TimingState::Missing | TimingState::Invalid(_) => UNIX_EPOCH,
    };
    for subject in SUBJECTS
        .iter()
        .filter(|subject| subject.scope == Scope::Run)
    {
        for relative in [
            Path::new("reports").join(subject.report_name),
            Path::new("payloads").join(subject.payload_name),
        ] {
            if let Ok((file, _)) = open_contained_artifact(run, &relative, "analysis artifact") {
                if let Ok(modified_at) = file.metadata().and_then(|metadata| metadata.modified()) {
                    generated_at = generated_at.max(modified_at);
                }
            }
        }
    }
    if let Ok(path) = select_trace(run) {
        if let Ok(modified_at) = path.metadata().and_then(|metadata| metadata.modified()) {
            generated_at = generated_at.max(modified_at);
        }
    }
    generated_at
}
